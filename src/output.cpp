#include "output.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>

#include "theme.h"

namespace fc {

namespace fs = std::filesystem;

// ---- AsyncWriter -------------------------------------------------------------------

AsyncWriter::AsyncWriter() : thread_([this] { run(); }) {}

AsyncWriter::~AsyncWriter() { finish(); }

void AsyncWriter::push(std::function<void()> job) {
    std::unique_lock<std::mutex> lock(mutex_);
    space_.wait(lock, [this] { return jobs_.size() < 8; });  // bounded: caps memory for queued frames
    jobs_.push_back(std::move(job));
    cv_.notify_one();
}

void AsyncWriter::finish() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void AsyncWriter::run() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return done_ || !jobs_.empty(); });
            if (jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        space_.notify_one();
        try {
            job();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "warning: output: %s\n", e.what());
        }
    }
}

// ---- helpers --------------------------------------------------------------------------

FieldStats computeStats(const std::vector<float>& T, const std::vector<double>& weights) {
    FieldStats s;
    s.tmin = std::numeric_limits<double>::infinity();
    s.tmax = -std::numeric_limits<double>::infinity();
    double sum = 0, wsum = 0;
    for (size_t i = 0; i < T.size(); ++i) {
        const double v = T[i];
        if (!std::isfinite(v)) {
            s.finite = false;
            continue;
        }
        s.tmin = std::min(s.tmin, v);
        s.tmax = std::max(s.tmax, v);
        sum += weights[i] * v;
        wsum += weights[i];
    }
    s.tmean = wsum > 0 ? sum / wsum : 0.0;
    if (!(s.tmin <= s.tmax)) s.tmin = s.tmax = 0.0;
    return s;
}

namespace {

void writeVtu(const std::string& path, const Mesh& m, const std::vector<float>& T, const std::vector<double>& k,
              double time) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    const size_t n = m.numNodes(), ne = m.numCells();
    const int npc = m.nodesPerCell();
    std::fprintf(f, "<?xml version=\"1.0\"?>\n");
    std::fprintf(f, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(f, "  <UnstructuredGrid>\n");
    std::fprintf(f, "    <FieldData>\n      <DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">%.10g</DataArray>\n    </FieldData>\n", time);
    std::fprintf(f, "    <Piece NumberOfPoints=\"%zu\" NumberOfCells=\"%zu\">\n", n, ne);
    std::fprintf(f, "      <PointData Scalars=\"temperature\">\n");
    std::fprintf(f, "        <DataArray type=\"Float32\" Name=\"temperature\" format=\"ascii\">\n");
    for (size_t i = 0; i < n; ++i) std::fprintf(f, "%.7g\n", T[i]);
    std::fprintf(f, "        </DataArray>\n      </PointData>\n");
    std::fprintf(f, "      <CellData Scalars=\"conductivity\">\n");
    std::fprintf(f, "        <DataArray type=\"Float32\" Name=\"conductivity\" format=\"ascii\">\n");
    for (size_t e = 0; e < ne; ++e) std::fprintf(f, "%.7g\n", k[e]);
    std::fprintf(f, "        </DataArray>\n      </CellData>\n");
    std::fprintf(f, "      <Points>\n        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (size_t i = 0; i < n; ++i) std::fprintf(f, "%.12g %.12g %.12g\n", m.x[i], m.y[i], m.z[i]);
    std::fprintf(f, "        </DataArray>\n      </Points>\n      <Cells>\n");
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (size_t e = 0; e < ne; ++e) {
        for (int k = 0; k < npc; ++k) std::fprintf(f, k ? " %u" : "%u", m.cells[e * npc + k]);
        std::fputc('\n', f);
    }
    std::fprintf(f, "        </DataArray>\n        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    for (size_t e = 0; e < ne; ++e) std::fprintf(f, "%zu\n", static_cast<size_t>(npc) * (e + 1));
    std::fprintf(f, "        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    const int vtkType = m.dim == 3 ? 10 : 5;  // VTK_TETRA : VTK_TRIANGLE
    for (size_t e = 0; e < ne; ++e) std::fprintf(f, "%d\n", vtkType);
    std::fprintf(f, "        </DataArray>\n      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
    std::fclose(f);
}

// Barycentric coordinates of point p in cell c; true if p lies inside (with a small tolerance).
bool barycentric(const Mesh& m, size_t c, double px, double py, double pz, double* lambda) {
    const int npc = m.nodesPerCell();
    const uint32_t* nd = &m.cells[c * npc];
    const double eps = -1e-9;
    if (m.dim == 2) {
        const uint32_t a = nd[0], b = nd[1], cc = nd[2];
        const double det = (m.x[b] - m.x[a]) * (m.y[cc] - m.y[a]) - (m.x[cc] - m.x[a]) * (m.y[b] - m.y[a]);
        lambda[0] = ((m.x[b] - px) * (m.y[cc] - py) - (m.x[cc] - px) * (m.y[b] - py)) / det;
        lambda[1] = ((m.x[cc] - px) * (m.y[a] - py) - (m.x[a] - px) * (m.y[cc] - py)) / det;
        lambda[2] = 1.0 - lambda[0] - lambda[1];
        return lambda[0] >= eps && lambda[1] >= eps && lambda[2] >= eps;
    }
    double e[3][3], q[3] = {px - m.x[nd[0]], py - m.y[nd[0]], pz - m.z[nd[0]]};
    for (int k = 0; k < 3; ++k) {
        e[k][0] = m.x[nd[k + 1]] - m.x[nd[0]];
        e[k][1] = m.y[nd[k + 1]] - m.y[nd[0]];
        e[k][2] = m.z[nd[k + 1]] - m.z[nd[0]];
    }
    auto cross = [](const double* a, const double* b, double* r) {
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
    };
    double r1[3], r2[3], r3[3];
    cross(e[1], e[2], r1);
    cross(e[2], e[0], r2);
    cross(e[0], e[1], r3);
    const double det = e[0][0] * r1[0] + e[0][1] * r1[1] + e[0][2] * r1[2];
    lambda[1] = (r1[0] * q[0] + r1[1] * q[1] + r1[2] * q[2]) / det;
    lambda[2] = (r2[0] * q[0] + r2[1] * q[1] + r2[2] * q[2]) / det;
    lambda[3] = (r3[0] * q[0] + r3[1] * q[1] + r3[2] * q[2]) / det;
    lambda[0] = 1.0 - lambda[1] - lambda[2] - lambda[3];
    return lambda[0] >= eps && lambda[1] >= eps && lambda[2] >= eps && lambda[3] >= eps;
}

}  // namespace

// ---- OutputManager ------------------------------------------------------------------

OutputManager::OutputManager(const ProblemDef& p, const Mesh& mesh, const FemSystem& sys)
    : mesh_(mesh), sys_(sys), dir_(p.output.directory), png_(p.output.png), gif_(p.output.gif), vtk_(p.output.vtk),
      csv_(p.output.csv), gifFps_(p.output.gifFps), hasExact_(p.hasExact),
      exact_(p.exact) {
    // Locate probes (barycentric weights in the containing element).
    const int npc = mesh.nodesPerCell();
    for (const ProbeDef& pd : p.output.probes) {
        bool found = false;
        for (size_t c = 0; c < mesh.numCells() && !found; ++c) {
            double lambda[4];
            if (!barycentric(mesh, c, pd.x, pd.y, pd.z, lambda)) continue;
            Probe pr{pd.name, {0, 0, 0, 0}, {0, 0, 0, 0}};
            for (int k = 0; k < npc; ++k) {
                pr.n[k] = mesh.cells[c * npc + k];
                pr.w[k] = lambda[k];
            }
            probes_.push_back(pr);
            found = true;
        }
        if (!found) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "probe '%s' at (%g, %g%s) lies outside the mesh", pd.name.c_str(), pd.x,
                          pd.y, mesh.dim == 3 ? (", " + std::to_string(pd.z)).c_str() : "");
            throw std::runtime_error(buf);
        }
    }

    if (!png_ && !gif_ && !vtk_ && !csv_) {
        enabled_ = false;
        return;
    }
    fs::create_directories(dir_);
    if (png_) fs::create_directories(fs::path(dir_) / "frames");
    if (vtk_) fs::create_directories(fs::path(dir_) / "vtk");
    if (csv_) {
        csvFile_.open((fs::path(dir_) / "timeseries.csv").string());
        if (!csvFile_) throw std::runtime_error("cannot write " + dir_ + "/timeseries.csv");
        csvFile_ << "time,step,T_min,T_max,T_mean";
        for (const Probe& pr : probes_) csvFile_ << ",probe_" << pr.name;
        if (hasExact_) csvFile_ << ",err_max,err_rms";
        csvFile_ << "\n";
    }
}

OutputManager::~OutputManager() { finish(); }

void OutputManager::write(double t, int step, const std::vector<float>& T, const std::vector<uint8_t>* rgba,
                          uint32_t width, uint32_t height) {
    // Error vs. the exact solution (computed even if no files are written).
    if (hasExact_) {
        double emax = 0, esum = 0, wsum = 0;
        for (size_t i = 0; i < T.size(); ++i) {
            const double e = static_cast<double>(T[i]) - exact_.eval(mesh_.x[i], mesh_.y[i], mesh_.z[i], t);
            emax = std::max(emax, std::fabs(e));
            esum += sys_.lumpedMass[i] * e * e;
            wsum += sys_.lumpedMass[i];
        }
        lastErrMax_ = emax;
        lastErrL2_ = wsum > 0 ? std::sqrt(esum / wsum) : 0.0;
        worstErrMax_ = std::max(worstErrMax_, emax);
    }
    if (!enabled_) return;

    const int frame = frame_++;
    if (csv_) {
        const FieldStats st = computeStats(T, sys_.lumpedMass);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%.10g,%d,%.7g,%.7g,%.7g", t, step, st.tmin, st.tmax, st.tmean);
        csvFile_ << buf;
        for (const Probe& pr : probes_) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k) v += pr.w[k] * T[pr.n[k]];
            std::snprintf(buf, sizeof(buf), ",%.7g", v);
            csvFile_ << buf;
        }
        if (hasExact_) {
            std::snprintf(buf, sizeof(buf), ",%.6g,%.6g", lastErrMax_, lastErrL2_);
            csvFile_ << buf;
        }
        csvFile_ << "\n";
    }
    char name[64];
    if (vtk_) {
        std::snprintf(name, sizeof(name), "field_%05d.vtu", frame);
        pvd_.emplace_back(t, std::string("vtk/") + name);
        const std::string path = (fs::path(dir_) / "vtk" / name).string();
        writer_.push([this, path, T, t] { writeVtu(path, mesh_, T, sys_.elementConductivity, t); });
    }
    if (rgba && (png_ || gif_)) {
        auto pixels = std::make_shared<std::vector<uint8_t>>(*rgba);
        if (png_) {
            std::snprintf(name, sizeof(name), "frame_%05d.png", frame);
            const std::string path = (fs::path(dir_) / "frames" / name).string();
            writer_.push([pixels, path, width, height] {
                if (!writePNG(path, pixels->data(), static_cast<int>(width), static_cast<int>(height)))
                    throw std::runtime_error("cannot write " + path);
            });
        }
        if (gif_) {
            const std::string path = (fs::path(dir_) / "animation.gif").string();
            writer_.push([this, pixels, path, width, height] {
                if (!gifWriter_.isOpen() &&
                    !gifWriter_.open(path, static_cast<int>(width), static_cast<int>(height), gifFps_,
                                     {{theme::kBackground[0], theme::kBackground[1], theme::kBackground[2]},
                                      {theme::kTitle[0], theme::kTitle[1], theme::kTitle[2]},
                                      {theme::kText[0], theme::kText[1], theme::kText[2]},
                                      {theme::kLabel[0], theme::kLabel[1], theme::kLabel[2]}}))
                    throw std::runtime_error("cannot write " + path);
                gifWriter_.addFrame(pixels->data());
            });
        }
    }
}

void OutputManager::finish() {
    writer_.finish();
    gifWriter_.close();
    if (csvFile_.is_open()) csvFile_.close();
    if (vtk_ && !pvd_.empty()) {
        const std::string path = (fs::path(dir_) / "field.pvd").string();
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (f) {
            std::fprintf(f, "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n  <Collection>\n");
            for (const auto& [t, file] : pvd_)
                std::fprintf(f, "    <DataSet timestep=\"%.10g\" group=\"\" part=\"0\" file=\"%s\"/>\n", t, file.c_str());
            std::fprintf(f, "  </Collection>\n</VTKFile>\n");
            std::fclose(f);
        }
        pvd_.clear();
    }
}

std::vector<std::string> OutputManager::writtenFiles() const {
    std::vector<std::string> out;
    if (!enabled_ && frame_ == 0) return out;
    if (csv_) out.push_back("timeseries.csv");
    if (gif_) out.push_back("animation.gif");
    if (png_) out.push_back("frames/frame_*.png");
    if (vtk_) out.push_back("field.pvd (+ vtk/*.vtu)");
    return out;
}

}  // namespace fc
