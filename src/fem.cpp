#include "fem.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace fc {

namespace {

std::string fmt(const char* f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

// Measure (area / volume), centroid and basis-function gradients of a P1 simplex.
struct CellGeometry {
    double measure = 0;
    double centroid[3] = {0, 0, 0};
    double grad[4][3] = {};
};

bool cellGeometry(const Mesh& m, size_t c, CellGeometry& g) {
    const int npc = m.nodesPerCell();
    const uint32_t* nd = &m.cells[c * npc];
    for (int k = 0; k < npc; ++k) {
        g.centroid[0] += m.x[nd[k]] / npc;
        g.centroid[1] += m.y[nd[k]] / npc;
        g.centroid[2] += m.z[nd[k]] / npc;
    }
    if (m.dim == 2) {
        const double xa = m.x[nd[0]], ya = m.y[nd[0]], xb = m.x[nd[1]], yb = m.y[nd[1]], xc = m.x[nd[2]], yc = m.y[nd[2]];
        const double det = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya);
        if (!(det > 0)) return false;
        g.measure = 0.5 * det;
        const double b[3] = {yb - yc, yc - ya, ya - yb};
        const double cc[3] = {xc - xb, xa - xc, xb - xa};
        for (int i = 0; i < 3; ++i) {
            g.grad[i][0] = b[i] / det;
            g.grad[i][1] = cc[i] / det;
            g.grad[i][2] = 0.0;
        }
        return true;
    }
    // Tetrahedron: J = [e1 e2 e3], e_k = p_k - p_0; the rows of J^-1 are the
    // gradients of the barycentric coordinates 1..3.
    double e[3][3];
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
    double c23[3], c31[3], c12[3];
    cross(e[1], e[2], c23);
    cross(e[2], e[0], c31);
    cross(e[0], e[1], c12);
    const double det = e[0][0] * c23[0] + e[0][1] * c23[1] + e[0][2] * c23[2];
    if (!(det > 0)) return false;
    g.measure = det / 6.0;
    for (int a = 0; a < 3; ++a) {
        g.grad[1][a] = c23[a] / det;
        g.grad[2][a] = c31[a] / det;
        g.grad[3][a] = c12[a] / det;
        g.grad[0][a] = -(g.grad[1][a] + g.grad[2][a] + g.grad[3][a]);
    }
    return true;
}

// Measure (length / area) and centroid of a boundary facet.
double facetGeometry(const Mesh& m, const Mesh::Face& f, double* centroid) {
    const int npf = m.nodesPerFace();
    centroid[0] = centroid[1] = centroid[2] = 0.0;
    for (int k = 0; k < npf; ++k) {
        centroid[0] += m.x[f.n[k]] / npf;
        centroid[1] += m.y[f.n[k]] / npf;
        centroid[2] += m.z[f.n[k]] / npf;
    }
    const double ax = m.x[f.n[1]] - m.x[f.n[0]], ay = m.y[f.n[1]] - m.y[f.n[0]], az = m.z[f.n[1]] - m.z[f.n[0]];
    if (npf == 2) return std::sqrt(ax * ax + ay * ay + az * az);
    const double bx = m.x[f.n[2]] - m.x[f.n[0]], by = m.y[f.n[2]] - m.y[f.n[0]], bz = m.z[f.n[2]] - m.z[f.n[0]];
    const double cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
    return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

}  // namespace

void LoadModel::evaluate(double t, std::vector<double>& F) const {
    F.assign(numNodes, 0.0);
    for (const Term& term : terms_) {
        if (!term.expr.usesSpace()) {
            const double v = term.expr.eval(0.0, 0.0, 0.0, t);
            for (size_t i = 0; i < term.node.size(); ++i) F[term.node[i]] += term.weight[i] * v;
        } else {
            for (size_t i = 0; i < term.node.size(); ++i)
                F[term.node[i]] += term.weight[i] * term.expr.eval(term.px[i], term.py[i], term.pz[i], t);
        }
    }
}

bool LoadModel::timeDependent() const {
    for (const Term& t : terms_)
        if (t.expr.usesT()) return true;
    return false;
}

void stepLoad(const FemSystem& s, const std::vector<double>& Fn, const std::vector<double>& Fn1, std::vector<double>& f) {
    f.resize(s.n);
    for (size_t i = 0; i < s.n; ++i)
        f[i] = s.dirichlet[i] ? 0.0 : s.dt * (s.theta * Fn1[i] + (1.0 - s.theta) * Fn[i]);
}

// Linear (P1) simplex elements in 2D (triangles) and 3D (tetrahedra):
//   stiffness  K_ij = k |T| grad(phi_i).grad(phi_j)
//   mass       M_ij = rho c |T| (1 + delta_ij) / ((d+1)(d+2))     (lumped: rho c |T| / (d+1))
//   facets     Robin h |F| (1 + delta_ij) / (d (d+1)),  loads q |F| / d per node
// In 2D every term is multiplied by the plate thickness.
Assembly assemble(const ProblemDef& p, const Mesh& m) {
    Assembly out;
    FemSystem& s = out.system;
    const size_t n = m.numNodes();
    const size_t nc = m.numCells();
    const int dim = m.dim, npc = m.nodesPerCell(), npf = m.nodesPerFace();
    const double d = dim == 2 ? p.mesh.thickness : 1.0;
    s.n = n;
    out.loads.numNodes = n;

    // ---- element material and centroids ----------------------------------------------
    std::vector<double> measure(nc), cx(nc), cy(nc), cz(nc), ek(nc), erc(nc);
    std::vector<size_t> matCount(p.materials.size() + 1, 0);
    for (size_t c = 0; c < nc; ++c) {
        CellGeometry g;
        if (!cellGeometry(m, c, g))
            throw std::runtime_error("element " + std::to_string(c) + " is degenerate or inverted");
        measure[c] = g.measure;
        cx[c] = g.centroid[0];
        cy[c] = g.centroid[1];
        cz[c] = g.centroid[2];
        const MaterialDef* mat = &p.baseMaterial;
        size_t mi = 0;
        for (size_t k = 0; k < p.materials.size(); ++k) {
            if (p.materials[k].region.contains(cx[c], cy[c], cz[c])) {
                mat = &p.materials[k];
                mi = k + 1;
            }
        }
        ++matCount[mi];
        ek[c] = mat->conductivity;
        erc[c] = mat->density * mat->specificHeat;
    }
    {
        const MaterialDef& b = p.baseMaterial;
        const double alpha = b.conductivity / (b.density * b.specificHeat);
        double L = std::min(m.xmax - m.xmin, m.ymax - m.ymin);
        if (dim == 3) L = std::min(L, m.zmax - m.zmin);
        out.notes.push_back("material base: k=" + fmt("%g", b.conductivity) + " rho=" + fmt("%g", b.density) +
                            " cp=" + fmt("%g", b.specificHeat) + "  (alpha=" + fmt("%.3g", alpha) +
                            " m^2/s, L^2/alpha=" + fmt("%.3g", L * L / alpha) + " s)  " +
                            std::to_string(matCount[0]) + " elements");
        for (size_t k = 0; k < p.materials.size(); ++k) {
            const MaterialDef& md = p.materials[k];
            out.notes.push_back("material " + md.name + ": k=" + fmt("%g", md.conductivity) + " rho=" +
                                fmt("%g", md.density) + " cp=" + fmt("%g", md.specificHeat) + "  " +
                                md.region.describe() + "  " + std::to_string(matCount[k + 1]) + " elements");
            if (matCount[k + 1] == 0)
                out.warnings.push_back("material " + md.name + " covers no element centroids (region too small?)");
        }
    }

    // ---- CSR sparsity pattern from node -> cell adjacency ------------------------------
    {
        std::vector<uint32_t> cellPtr(n + 1, 0), nodeCells(nc * npc);
        for (size_t k = 0; k < nc * npc; ++k) cellPtr[m.cells[k] + 1]++;
        for (size_t i = 0; i < n; ++i) cellPtr[i + 1] += cellPtr[i];
        std::vector<uint32_t> fill(cellPtr.begin(), cellPtr.end() - 1);
        for (size_t c = 0; c < nc; ++c)
            for (int k = 0; k < npc; ++k) nodeCells[fill[m.cells[c * npc + k]]++] = static_cast<uint32_t>(c);

        s.rowPtr.assign(n + 1, 0);
        s.col.clear();
        s.col.reserve(n * (dim == 3 ? 15 : 7));
        std::vector<uint32_t> row;
        for (size_t i = 0; i < n; ++i) {
            if (cellPtr[i + 1] == cellPtr[i]) throw std::runtime_error("node " + std::to_string(i) + " belongs to no element");
            row.clear();
            for (uint32_t k = cellPtr[i]; k < cellPtr[i + 1]; ++k)
                for (int v = 0; v < npc; ++v) row.push_back(m.cells[static_cast<size_t>(nodeCells[k]) * npc + v]);
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            s.col.insert(s.col.end(), row.begin(), row.end());
            s.rowPtr[i + 1] = static_cast<uint32_t>(s.col.size());
        }
    }
    const size_t nnz = s.col.size();
    auto find = [&](uint32_t i, uint32_t j) -> size_t {
        auto first = s.col.begin() + s.rowPtr[i], last = s.col.begin() + s.rowPtr[i + 1];
        return static_cast<size_t>(std::lower_bound(first, last, j) - s.col.begin());
    };

    // ---- mass and stiffness matrices ---------------------------------------------------
    const double theta = p.time.theta;
    bool lumped = p.time.mass == "lumped" || (p.time.mass == "auto" && theta < 0.5);
    if (theta < 0.5 && !lumped) {
        out.warnings.push_back("consistent mass with theta < 0.5 is not supported; using lumped mass");
        lumped = true;
    }
    const double massDen = static_cast<double>(npc) * (npc + 1);  // 12 (triangle), 20 (tetrahedron)
    const double hFace = dim == 2 ? p.faceH * p.faceCount : 0.0;
    std::vector<double> M(nnz, 0.0), A(nnz, 0.0), Mrow(n, 0.0);
    for (size_t c = 0; c < nc; ++c) {
        CellGeometry g;
        cellGeometry(m, c, g);
        const uint32_t* nd = &m.cells[c * npc];
        const double kd = ek[c] * d * g.measure;
        const double md = erc[c] * d * g.measure;
        const double fc = hFace * g.measure;  // 2D plate faces (consistent "mass-like" term)
        for (int i = 0; i < npc; ++i) {
            Mrow[nd[i]] += md / npc;
            for (int j = 0; j < npc; ++j) {
                const size_t k = find(nd[i], nd[j]);
                const double gg = g.grad[i][0] * g.grad[j][0] + g.grad[i][1] * g.grad[j][1] + g.grad[i][2] * g.grad[j][2];
                M[k] += lumped ? (i == j ? md / npc : 0.0) : md * (i == j ? 2.0 : 1.0) / massDen;
                A[k] += kd * gg + fc * (i == j ? 2.0 : 1.0) / massDen;
            }
        }
    }

    // ---- boundary conditions -------------------------------------------------------------
    std::vector<std::vector<bool>> sideMask(p.boundaries.size(), std::vector<bool>(m.sideNames.size(), false));
    for (size_t bi = 0; bi < p.boundaries.size(); ++bi) {
        for (const std::string& side : p.boundaries[bi].sides) {
            if (side == "all") {
                std::fill(sideMask[bi].begin(), sideMask[bi].end(), true);
                continue;
            }
            const int si = m.sideIndex(side);
            if (si < 0)
                throw std::runtime_error("[boundary." + p.boundaries[bi].name + "] unknown side '" + side +
                                         "' (known: " + m.sideList() + ")");
            sideMask[bi][si] = true;
        }
    }
    std::vector<int> faceBc(m.boundary.size(), -1);
    std::vector<size_t> bcFaces(p.boundaries.size(), 0);
    std::vector<double> faceMeasure(m.boundary.size()), fcx(m.boundary.size()), fcy(m.boundary.size()),
        fcz(m.boundary.size());
    for (size_t fi = 0; fi < m.boundary.size(); ++fi) {
        const Mesh::Face& f = m.boundary[fi];
        double cen[3];
        faceMeasure[fi] = facetGeometry(m, f, cen);
        fcx[fi] = cen[0];
        fcy[fi] = cen[1];
        fcz[fi] = cen[2];
        for (size_t bi = 0; bi < p.boundaries.size(); ++bi) {
            if (!sideMask[bi][f.side]) continue;
            if (p.boundaries[bi].where.valid() && p.boundaries[bi].where.eval(cen[0], cen[1], cen[2], 0.0) == 0.0)
                continue;
            faceBc[fi] = static_cast<int>(bi);
        }
        if (faceBc[fi] >= 0) bcFaces[faceBc[fi]]++;
    }

    std::vector<double> dirSum(n, 0.0);
    std::vector<int> dirCount(n, 0);
    std::vector<LoadModel::Term> bcTerms(p.boundaries.size());
    const double robinDen = static_cast<double>(npf) * (npf + 1);  // 6 (edge), 12 (triangle)
    for (size_t fi = 0; fi < m.boundary.size(); ++fi) {
        if (faceBc[fi] < 0) continue;
        const BoundaryDef& bd = p.boundaries[faceBc[fi]];
        LoadModel::Term& term = bcTerms[faceBc[fi]];
        const Mesh::Face& f = m.boundary[fi];
        const double area = d * faceMeasure[fi];
        switch (bd.type) {
            case BcType::Dirichlet:
                for (int k = 0; k < npf; ++k) {
                    const uint32_t node = f.n[k];
                    dirSum[node] += bd.temperature.eval(m.x[node], m.y[node], m.z[node], 0.0);
                    dirCount[node]++;
                }
                break;
            case BcType::Neumann:
                term.expr = bd.flux;
                for (int k = 0; k < npf; ++k) term.push(f.n[k], area / npf, fcx[fi], fcy[fi], fcz[fi]);
                break;
            case BcType::Robin: {
                const double h = bd.h.eval(fcx[fi], fcy[fi], fcz[fi], 0.0);
                if (h < 0) throw std::runtime_error("[boundary." + bd.name + "] h must be >= 0");
                for (int i = 0; i < npf; ++i)
                    for (int j = 0; j < npf; ++j) A[find(f.n[i], f.n[j])] += h * area * (i == j ? 2.0 : 1.0) / robinDen;
                term.expr = bd.ambient;
                for (int k = 0; k < npf; ++k) term.push(f.n[k], h * area / npf, fcx[fi], fcy[fi], fcz[fi]);
                break;
            }
            case BcType::Insulated:
                break;
        }
    }
    const char* facetWord = dim == 3 ? " faces" : " edges";
    for (size_t bi = 0; bi < p.boundaries.size(); ++bi) {
        const BoundaryDef& bd = p.boundaries[bi];
        std::string detail;
        if (bd.type == BcType::Dirichlet) detail = "T=" + bd.temperature.text();
        if (bd.type == BcType::Neumann) detail = "q=" + bd.flux.text();
        if (bd.type == BcType::Robin) detail = "h=" + bd.h.text() + ", T_inf=" + bd.ambient.text();
        out.notes.push_back("boundary " + bd.name + ": " + bcTypeName(bd.type) + (detail.empty() ? "" : " " + detail) +
                            "  (" + std::to_string(bcFaces[bi]) + facetWord + ")");
        if (bcFaces[bi] == 0) out.warnings.push_back("boundary " + bd.name + " matches no boundary" + facetWord);
        if (!bcTerms[bi].node.empty()) {
            bcTerms[bi].label = "boundary " + bd.name;
            out.loads.add(std::move(bcTerms[bi]));
        }
    }

    // ---- volumetric heat sources -----------------------------------------------------
    for (const SourceDef& src : p.sources) {
        LoadModel::Term term;
        term.label = "source " + src.name;
        term.expr = src.value;
        double regionMeasure = 0.0;
        std::vector<size_t> elems;
        for (size_t c = 0; c < nc; ++c) {
            if (src.region.contains(cx[c], cy[c], cz[c])) {
                elems.push_back(c);
                regionMeasure += measure[c];
            }
        }
        if (elems.empty())
            throw std::runtime_error("[source." + src.name + "] covers no element centroids: enlarge the region or refine the mesh");
        double power0 = 0.0;
        for (size_t c : elems) {
            const double w = src.total ? measure[c] / (npc * regionMeasure) : d * measure[c] / npc;
            for (int i = 0; i < npc; ++i) term.push(m.cells[c * npc + i], w, cx[c], cy[c], cz[c]);
            power0 += npc * w * src.value.eval(cx[c], cy[c], cz[c], 0.0);
        }
        out.notes.push_back("source " + src.name + ": " + src.region.describe() + "  " + std::to_string(elems.size()) +
                            " elements, " + (src.total ? "power=" : "power_density=") + src.value.text() +
                            "  (" + fmt("%.4g", power0) + " W at t=0)");
        out.loads.add(std::move(term));
    }

    // ---- 2D plate model: convection from the large faces --------------------------------
    if (hFace > 0) {
        LoadModel::Term term;
        term.label = "faces";
        term.expr = p.faceAmbient;
        for (size_t c = 0; c < nc; ++c)
            for (int i = 0; i < npc; ++i) term.push(m.cells[c * npc + i], hFace * measure[c] / npc, cx[c], cy[c], cz[c]);
        out.loads.add(std::move(term));
        out.notes.push_back("faces: h=" + fmt("%g", p.faceH) + " W/(m^2 K) on " + std::to_string(p.faceCount) +
                            " face(s), T_inf=" + p.faceAmbient.text());
    }

    // ---- initial condition (Dirichlet values override) --------------------------------
    s.dirichlet.assign(n, 0);
    s.T0.resize(n);
    for (size_t i = 0; i < n; ++i) {
        s.T0[i] = p.initial.eval(m.x[i], m.y[i], m.z[i], 0.0);
        if (dirCount[i] > 0) {
            s.dirichlet[i] = 1;
            s.T0[i] = dirSum[i] / dirCount[i];
            s.numDirichlet++;
        }
        if (!std::isfinite(s.T0[i])) throw std::runtime_error("initial temperature is not finite at node " + std::to_string(i));
    }

    // ---- time step ---------------------------------------------------------------------
    // Gershgorin bound on the largest eigenvalue of M_L^{-1} A (free rows) gives a
    // guaranteed-stable explicit step: dt <= 2 / ((1 - 2 theta) lambda_max).
    double lambda = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (s.dirichlet[i]) continue;
        double rs = 0.0;
        for (uint32_t k = s.rowPtr[i]; k < s.rowPtr[i + 1]; ++k) rs += std::fabs(A[k]);
        lambda = std::max(lambda, rs / Mrow[i]);
    }
    s.stableDt = (theta < 0.5 && lambda > 0) ? 2.0 / ((1.0 - 2.0 * theta) * lambda) : 0.0;

    double dt = p.time.dt;
    const double interval = p.output.interval;
    if (dt <= 0) {
        const double target = (theta < 0.5 && s.stableDt > 0) ? 0.9 * s.stableDt : p.time.end / 500.0;
        dt = interval / std::ceil(interval / target - 1e-9);
        out.notes.push_back("dt = auto -> " + fmt("%.6g", dt) + " s");
    } else if (theta < 0.5 && s.stableDt > 0 && dt > s.stableDt) {
        out.warnings.push_back("dt = " + fmt("%g", dt) + " s exceeds the explicit stability bound " +
                               fmt("%.4g", s.stableDt) + " s; the solution may blow up (try dt = auto)");
    }
    s.dt = dt;
    s.theta = theta;
    s.tEnd = p.time.end;
    s.nSteps = std::max(1, static_cast<int>(std::ceil(p.time.end / dt - 1e-9)));
    s.lumped = lumped;
    s.diagonal = (theta == 0.0 && lumped);

    // ---- system matrices with Dirichlet elimination --------------------------------------
    s.S.resize(nnz);
    s.dtA.resize(nnz);
    for (size_t k = 0; k < nnz; ++k) {
        s.S[k] = M[k] + theta * dt * A[k];
        s.dtA[k] = dt * A[k];
    }
    for (size_t i = 0; i < n; ++i) {
        for (uint32_t k = s.rowPtr[i]; k < s.rowPtr[i + 1]; ++k) {
            if (s.dirichlet[i]) {
                s.S[k] = (s.col[k] == i) ? 1.0 : 0.0;
                s.dtA[k] = 0.0;
            } else if (s.dirichlet[s.col[k]]) {
                s.S[k] = 0.0;  // keeps S symmetric; the coupling lives in dt*A*T
            }
        }
    }
    s.rowSumDtA.assign(n, 0.0);
    s.invDiag.resize(n);
    double diagSum = 0.0;
    size_t freeCount = 0;
    for (size_t i = 0; i < n; ++i) {
        double rs = 0.0;
        for (uint32_t k = s.rowPtr[i]; k < s.rowPtr[i + 1]; ++k) rs += s.dtA[k];
        s.rowSumDtA[i] = rs;
        const double diag = s.S[find(static_cast<uint32_t>(i), static_cast<uint32_t>(i))];
        if (!(diag > 0)) throw std::runtime_error("non-positive diagonal in the system matrix at node " + std::to_string(i));
        s.invDiag[i] = 1.0 / diag;
        if (!s.dirichlet[i]) {
            diagSum += diag;
            ++freeCount;
        }
    }
    s.lumpedMass = Mrow;
    s.elementConductivity = ek;

    // Below this residual the increment is far smaller than fp32 can resolve in T.
    double tScale = 1.0;
    for (double v : s.T0) tScale = std::max(tScale, std::fabs(v));
    const double meanDiag = freeCount ? diagSum / freeCount : 1.0;
    const double absTol = 1e-9 * meanDiag * tScale;
    s.absTolSq = std::max(absTol * absTol * static_cast<double>(std::max<size_t>(freeCount, 1)), 1e-36);
    return out;
}

}  // namespace fc
