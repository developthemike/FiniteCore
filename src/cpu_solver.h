#pragma once

#include <vector>

#include "fem.h"

namespace fc {

struct StepStats {
    int iterations = 0;        // CG iterations of the last step (0 for explicit)
    double relResidual = 0.0;  // |r| / |r0| at exit
    bool converged = true;
    bool breakdown = false;
};

// Double-precision reference implementation of exactly the same algorithm as
// the GPU solver (increment form, Jacobi-preconditioned CG). Used by
// --compare-cpu to validate the GPU results and to measure the speed-up.
class CpuSolver {
public:
    CpuSolver(const FemSystem& sys, double tolerance, int maxIterations);
    void setLoad(const std::vector<double>& f) { f_ = f; }
    void setTemperature(const std::vector<double>& T) { T_ = T; }
    StepStats step();
    const std::vector<double>& temperature() const { return T_; }

private:
    void multiply(const std::vector<double>& vals, const std::vector<double>& in, std::vector<double>& out) const;
    const FemSystem& s_;
    double tolSq_;
    int maxIter_;
    std::vector<double> f_, T_, x_, r_, z_, p_, q_;
};

}  // namespace fc
