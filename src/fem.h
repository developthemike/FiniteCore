#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mesh.h"
#include "problem.h"

namespace fc {

// Global load vector F(t) as a sum of terms
//   F[node[i]] += weight[i] * expr(px[i], py[i], pz[i], t)
// (Neumann flux, Robin ambient, face convection, volumetric sources).
class LoadModel {
public:
    struct Term {
        std::string label;
        Expr expr;
        std::vector<uint32_t> node;
        std::vector<double> weight, px, py, pz;
        void push(uint32_t n, double w, double x, double y, double z) {
            node.push_back(n);
            weight.push_back(w);
            px.push_back(x);
            py.push_back(y);
            pz.push_back(z);
        }
    };

    void add(Term term) { terms_.push_back(std::move(term)); }
    void evaluate(double t, std::vector<double>& F) const;  // F is resized and overwritten
    bool timeDependent() const;
    size_t numNodes = 0;

private:
    std::vector<Term> terms_;
};

// Fully assembled, time-discretised system (double precision; solvers convert).
//
//   M dT/dt + A T = F(t)
//   theta scheme in increment form:  S dT = dt (F_theta - A T^n),  S = M + theta dt A
//
// Dirichlet nodes: rows of dt*A and the load are zero, rows/columns of S are
// identity, so their increment is always 0 and T keeps the prescribed value.
struct FemSystem {
    size_t n = 0;
    std::vector<uint32_t> rowPtr, col;  // CSR pattern shared by S and dt*A
    std::vector<double> S, dtA;
    std::vector<double> rowSumDtA;  // for the difference form of dt*A*T
    std::vector<double> invDiag;    // 1 / S_ii (Jacobi preconditioner)
    std::vector<double> lumpedMass; // row sums of M (nodal weights for averages/norms)
    std::vector<uint8_t> dirichlet;
    std::vector<double> T0;
    std::vector<double> elementConductivity;

    double dt = 0, theta = 1, tEnd = 0;
    int nSteps = 0;
    bool lumped = false;
    bool diagonal = false;   // S is diagonal: explicit update, no linear solve
    double stableDt = 0;     // explicit stability bound (theta < 0.5), 0 if unconditionally stable
    double absTolSq = 0;     // absolute floor for |r|^2 in the iterative solver
    size_t numDirichlet = 0;

    size_t nnz() const { return col.size(); }
};

struct Assembly {
    FemSystem system;
    LoadModel loads;
    std::vector<std::string> notes;
    std::vector<std::string> warnings;
};

Assembly assemble(const ProblemDef& p, const Mesh& mesh);

// f = dt * (theta F(t_{n+1}) + (1-theta) F(t_n)), zero on Dirichlet nodes.
void stepLoad(const FemSystem& s, const std::vector<double>& Fn, const std::vector<double>& Fn1, std::vector<double>& f);

}  // namespace fc
