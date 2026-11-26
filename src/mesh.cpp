#include "mesh.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fc {

int Mesh::sideIndex(const std::string& name) const {
    for (size_t i = 0; i < sideNames.size(); ++i)
        if (sideNames[i] == name) return static_cast<int>(i);
    for (const auto& [alias, side] : sideAliases)
        if (alias == name) return side;
    return -1;
}

std::string Mesh::sideList() const {
    std::string s;
    for (const auto& n : sideNames) s += (s.empty() ? "" : " ") + n;
    s += " all";
    if (sideAliases.empty()) return s;
    s += " (aliases:";
    for (const auto& a : sideAliases) s += " " + a.first;
    return s + ")";
}

std::vector<uint32_t> Mesh::wireframeEdges() const {
    std::vector<uint64_t> keys;
    auto addEdge = [&](uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        keys.push_back((static_cast<uint64_t>(a) << 32) | b);
    };
    if (dim == 2) {
        keys.reserve(cells.size());
        for (size_t c = 0; c < numCells(); ++c)
            for (int k = 0; k < 3; ++k) addEdge(cells[3 * c + k], cells[3 * c + (k + 1) % 3]);
    } else {
        keys.reserve(boundary.size() * 3);
        for (const Face& f : boundary)
            for (int k = 0; k < 3; ++k) addEdge(f.n[k], f.n[(k + 1) % 3]);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<uint32_t> out;
    out.reserve(keys.size() * 2);
    for (uint64_t k : keys) {
        out.push_back(static_cast<uint32_t>(k >> 32));
        out.push_back(static_cast<uint32_t>(k & 0xffffffffu));
    }
    return out;
}

std::vector<uint32_t> Mesh::boundaryIndices() const {
    std::vector<uint32_t> out;
    out.reserve(boundary.size() * dim);
    for (const Face& f : boundary)
        for (int k = 0; k < dim; ++k) out.push_back(f.n[k]);
    return out;
}

Mesh makeRectangleMesh(double x0, double y0, double sx, double sy, int nx, int ny, DiagonalPattern pattern) {
    if (nx < 1 || ny < 1) throw std::runtime_error("mesh: nx and ny must be >= 1");
    if (!(sx > 0) || !(sy > 0)) throw std::runtime_error("mesh: the size must be > 0");
    const uint64_t nodes = static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny + 1);
    if (nodes > 50'000'000ull) throw std::runtime_error("mesh: too many nodes");

    Mesh m;
    m.dim = 2;
    m.sideNames = {"left", "right", "bottom", "top"};
    m.sideAliases = {{"xmin", 0}, {"xmax", 1}, {"ymin", 2}, {"ymax", 3}};
    m.xmin = x0;
    m.xmax = x0 + sx;
    m.ymin = y0;
    m.ymax = y0 + sy;
    m.x.resize(nodes);
    m.y.resize(nodes);
    m.z.assign(nodes, 0.0);
    auto id = [nx](int i, int j) { return static_cast<uint32_t>(j * (nx + 1) + i); };
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            // Exact end points avoid round-off at the far edges.
            m.x[id(i, j)] = (i == nx) ? x0 + sx : x0 + sx * i / nx;
            m.y[id(i, j)] = (j == ny) ? y0 + sy : y0 + sy * j / ny;
        }
    }
    m.cells.reserve(static_cast<size_t>(nx) * ny * 6);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const uint32_t n00 = id(i, j), n10 = id(i + 1, j), n01 = id(i, j + 1), n11 = id(i + 1, j + 1);
            bool rising;  // diagonal from (i,j) to (i+1,j+1)
            switch (pattern) {
                case DiagonalPattern::Right: rising = true; break;
                case DiagonalPattern::Left: rising = false; break;
                default: rising = ((i + j) % 2) == 0; break;
            }
            if (rising) {
                m.cells.insert(m.cells.end(), {n00, n10, n11, n00, n11, n01});
            } else {
                m.cells.insert(m.cells.end(), {n00, n10, n01, n10, n11, n01});
            }
        }
    }
    for (int j = 0; j < ny; ++j) m.boundary.push_back({{id(0, j + 1), id(0, j), 0}, 0});    // left
    for (int j = 0; j < ny; ++j) m.boundary.push_back({{id(nx, j), id(nx, j + 1), 0}, 1});  // right
    for (int i = 0; i < nx; ++i) m.boundary.push_back({{id(i, 0), id(i + 1, 0), 0}, 2});    // bottom
    for (int i = 0; i < nx; ++i) m.boundary.push_back({{id(i + 1, ny), id(i, ny), 0}, 3});  // top
    m.boxShaped = true;
    m.description = "rectangle " + std::to_string(nx) + " x " + std::to_string(ny) + " cells";
    return m;
}

Mesh makeBoxMesh(double x0, double y0, double z0, double sx, double sy, double sz, int nx, int ny, int nz) {
    if (nx < 1 || ny < 1 || nz < 1) throw std::runtime_error("mesh: nx, ny and nz must be >= 1");
    if (!(sx > 0) || !(sy > 0) || !(sz > 0)) throw std::runtime_error("mesh: the size must be > 0");
    const uint64_t nodes =
        static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz + 1);
    if (nodes > 20'000'000ull) throw std::runtime_error("mesh: too many nodes");

    Mesh m;
    m.dim = 3;
    m.sideNames = {"left", "right", "front", "back", "bottom", "top"};
    m.sideAliases = {{"xmin", 0}, {"xmax", 1}, {"ymin", 2}, {"ymax", 3}, {"zmin", 4}, {"zmax", 5}};
    m.xmin = x0;
    m.xmax = x0 + sx;
    m.ymin = y0;
    m.ymax = y0 + sy;
    m.zmin = z0;
    m.zmax = z0 + sz;
    m.x.resize(nodes);
    m.y.resize(nodes);
    m.z.resize(nodes);
    auto id = [nx, ny](int i, int j, int k) {
        return static_cast<uint32_t>((static_cast<uint64_t>(k) * (ny + 1) + j) * (nx + 1) + i);
    };
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                const uint32_t n = id(i, j, k);
                m.x[n] = (i == nx) ? x0 + sx : x0 + sx * i / nx;
                m.y[n] = (j == ny) ? y0 + sy : y0 + sy * j / ny;
                m.z[n] = (k == nz) ? z0 + sz : z0 + sz * k / nz;
            }

    // Kuhn subdivision: one tetrahedron per permutation of the axes, following
    // the path (0,0,0) -> +e_p0 -> +e_p1 -> (1,1,1). All six share the cube
    // diagonal, and every cube face is cut along its (low, low)-(high, high)
    // diagonal, so neighbouring cubes match.
    static const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    int corner[6][4][3];
    for (int p = 0; p < 6; ++p) {
        int c[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a) corner[p][0][a] = 0;
        for (int s = 0; s < 3; ++s) {
            c[perms[p][s]] = 1;
            for (int a = 0; a < 3; ++a) corner[p][s + 1][a] = c[a];
        }
        // Orientation: det[c1, c2, c3] (c0 = origin); make it positive.
        const int(&v)[4][3] = corner[p];
        const int det = v[1][0] * (v[2][1] * v[3][2] - v[2][2] * v[3][1]) -
                        v[1][1] * (v[2][0] * v[3][2] - v[2][2] * v[3][0]) +
                        v[1][2] * (v[2][0] * v[3][1] - v[2][1] * v[3][0]);
        if (det < 0)
            for (int a = 0; a < 3; ++a) std::swap(corner[p][1][a], corner[p][2][a]);
    }
    m.cells.reserve(static_cast<size_t>(nx) * ny * nz * 24);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                for (int p = 0; p < 6; ++p)
                    for (int v = 0; v < 4; ++v)
                        m.cells.push_back(id(i + corner[p][v][0], j + corner[p][v][1], k + corner[p][v][2]));

    // Boundary triangles, cut along the same face diagonals as the tetrahedra.
    auto quad = [&m](uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, int side) {
        m.boundary.push_back({{c00, c10, c11}, side});
        m.boundary.push_back({{c00, c11, c01}, side});
    };
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            quad(id(0, j, k), id(0, j + 1, k), id(0, j, k + 1), id(0, j + 1, k + 1), 0);
            quad(id(nx, j, k), id(nx, j + 1, k), id(nx, j, k + 1), id(nx, j + 1, k + 1), 1);
        }
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            quad(id(i, 0, k), id(i + 1, 0, k), id(i, 0, k + 1), id(i + 1, 0, k + 1), 2);
            quad(id(i, ny, k), id(i + 1, ny, k), id(i, ny, k + 1), id(i + 1, ny, k + 1), 3);
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            quad(id(i, j, 0), id(i + 1, j, 0), id(i, j + 1, 0), id(i + 1, j + 1, 0), 4);
            quad(id(i, j, nz), id(i + 1, j, nz), id(i, j + 1, nz), id(i + 1, j + 1, nz), 5);
        }
    m.boxShaped = true;
    m.description = "box " + std::to_string(nx) + " x " + std::to_string(ny) + " x " + std::to_string(nz) + " cells";
    return m;
}

Mesh makeTubeMesh(const TubeSpec& t) {
    constexpr double pi = 3.14159265358979323846;
    if (!(t.length > 0)) throw std::runtime_error("tube: length must be > 0");
    if (t.bend < 0) throw std::runtime_error("tube: bend must be >= 0 (0 = straight)");
    if (t.bend > 0 && t.length / t.bend > 1.9 * pi)
        throw std::runtime_error("tube: the bend closes on itself (length / bend must stay below ~6 rad)");
    if (t.cellsAlong < 1 || t.cellsAcross < 2) throw std::runtime_error("tube: needs >= 1 cell along and >= 2 across");
    if (!t.radius) throw std::runtime_error("tube: no radius profile");

    // Reference box: u in [0, 1] along the tube, (v, w) in [-1, 1]^2 across it.
    Mesh m = makeBoxMesh(0.0, -1.0, -1.0, 1.0, 2.0, 2.0, t.cellsAlong, t.cellsAcross, t.cellsAcross);
    double inner = 0.0;  // deepest reach towards the bend centre, relative to the bend radius
    for (size_t i = 0; i < m.numNodes(); ++i) {
        const double u = m.x[i], v = m.y[i], w = m.z[i];
        // Concentric square-to-disc map: concentric squares become circles,
        // so the cells stay well shaped (no degenerate corners).
        double px = 0.0, py = 0.0;
        if (v != 0.0 || w != 0.0) {
            double rr, th;
            if (std::fabs(v) >= std::fabs(w)) {
                rr = v;
                th = (pi / 4.0) * (w / v);
            } else {
                rr = w;
                th = pi / 2.0 - (pi / 4.0) * (v / w);
            }
            px = rr * std::cos(th);
            py = rr * std::sin(th);
        }
        const double angle = std::atan2(py, px);
        const double r = t.radius(u);
        const double sec = t.section ? t.section(angle) : 1.0;
        if (!(r > 0) || !std::isfinite(r))
            throw std::runtime_error("tube: radius(s) must be positive; radius(" + std::to_string(u) +
                                     ") = " + std::to_string(r));
        if (!(sec > 0) || !std::isfinite(sec))
            throw std::runtime_error("tube: section(angle) must be positive; section(" + std::to_string(angle) +
                                     ") = " + std::to_string(sec));
        const double a = px * r * sec, b = py * r * sec;  // offsets along N and B
        const double ell = (u - 0.5) * t.length;
        double C[3], T[3], N[3];
        if (t.bend > 0) {
            const double phi = ell / t.bend;
            C[0] = t.bend * std::sin(phi);
            C[1] = 0.0;
            C[2] = t.bend * (1.0 - std::cos(phi));
            T[0] = std::cos(phi);
            T[1] = 0.0;
            T[2] = std::sin(phi);
            N[0] = -std::sin(phi);  // towards the centre of the arc
            N[1] = 0.0;
            N[2] = std::cos(phi);
            inner = std::max(inner, a / t.bend);
        } else {
            C[0] = ell;
            C[1] = C[2] = 0.0;
            T[0] = 1.0;
            T[1] = T[2] = 0.0;
            N[0] = N[1] = 0.0;
            N[2] = 1.0;
        }
        const double B[3] = {T[1] * N[2] - T[2] * N[1], T[2] * N[0] - T[0] * N[2], T[0] * N[1] - T[1] * N[0]};
        m.x[i] = t.origin[0] + C[0] + a * N[0] + b * B[0];
        m.y[i] = t.origin[1] + C[1] + a * N[1] + b * B[1];
        m.z[i] = t.origin[2] + C[2] + a * N[2] + b * B[2];
    }
    if (inner >= 0.95)
        throw std::runtime_error("tube: too thick for its bend - the inside of the bend would fold over "
                                 "(make bend larger than the radius)");

    // The map is orientation-preserving up to one global sign; make every
    // tetrahedron positive and reject any that folded or collapsed.
    const size_t nc = m.numCells();
    std::vector<double> vol(nc);
    double maxAbs = 0.0;
    size_t negative = 0;
    for (size_t c = 0; c < nc; ++c) {
        const uint32_t* nd = &m.cells[4 * c];
        double e[3][3];
        for (int k = 0; k < 3; ++k) {
            e[k][0] = m.x[nd[k + 1]] - m.x[nd[0]];
            e[k][1] = m.y[nd[k + 1]] - m.y[nd[0]];
            e[k][2] = m.z[nd[k + 1]] - m.z[nd[0]];
        }
        vol[c] = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                 e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
        maxAbs = std::max(maxAbs, std::fabs(vol[c]));
        if (vol[c] < 0) ++negative;
    }
    const bool flip = negative > nc / 2;
    for (size_t c = 0; c < nc; ++c) {
        const double v = flip ? -vol[c] : vol[c];
        if (!(v > 1e-9 * maxAbs))
            throw std::runtime_error("tube: the mapping inverts or collapses elements; smooth the radius / section "
                                     "profiles or use more cells");
        if (flip) std::swap(m.cells[4 * c + 1], m.cells[4 * c + 2]);
    }

    m.sideNames = {"start", "end", "surface"};
    m.sideAliases.clear();
    for (Mesh::Face& f : m.boundary) f.side = f.side <= 1 ? f.side : 2;
    auto [x0, x1] = std::minmax_element(m.x.begin(), m.x.end());
    auto [y0, y1] = std::minmax_element(m.y.begin(), m.y.end());
    auto [z0, z1] = std::minmax_element(m.z.begin(), m.z.end());
    m.xmin = *x0;
    m.xmax = *x1;
    m.ymin = *y0;
    m.ymax = *y1;
    m.zmin = *z0;
    m.zmax = *z1;
    m.boxShaped = false;
    m.description = "tube " + std::to_string(t.cellsAlong) + " x " + std::to_string(t.cellsAcross) + " x " +
                    std::to_string(t.cellsAcross) + " cells";
    return m;
}

}  // namespace fc
