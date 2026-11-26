#include "cpu_solver.h"

#include <cmath>

namespace fc {

CpuSolver::CpuSolver(const FemSystem& sys, double tolerance, int maxIterations)
    : s_(sys), tolSq_(tolerance * tolerance), maxIter_(maxIterations) {
    const size_t n = sys.n;
    f_.assign(n, 0.0);
    T_ = sys.T0;
    x_.assign(n, 0.0);
    r_.assign(n, 0.0);
    z_.assign(n, 0.0);
    p_.assign(n, 0.0);
    q_.assign(n, 0.0);
}

void CpuSolver::multiply(const std::vector<double>& vals, const std::vector<double>& in, std::vector<double>& out) const {
    for (size_t i = 0; i < s_.n; ++i) {
        double acc = 0.0;
        for (uint32_t k = s_.rowPtr[i]; k < s_.rowPtr[i + 1]; ++k) acc += vals[k] * in[s_.col[k]];
        out[i] = acc;
    }
}

StepStats CpuSolver::step() {
    const size_t n = s_.n;
    StepStats st;
    // r0 = f - (dt A) T
    multiply(s_.dtA, T_, q_);
    for (size_t i = 0; i < n; ++i) r_[i] = f_[i] - q_[i];

    if (s_.diagonal) {
        for (size_t i = 0; i < n; ++i) T_[i] += s_.invDiag[i] * r_[i];
        return st;
    }

    double rz = 0.0, rr0 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x_[i] = 0.0;
        z_[i] = s_.invDiag[i] * r_[i];
        p_[i] = z_[i];
        rz += r_[i] * z_[i];
        rr0 += r_[i] * r_[i];
    }
    double rr = rr0;
    const double limit = std::fmax(tolSq_ * rr0, s_.absTolSq);
    st.converged = rr0 <= s_.absTolSq;
    while (!st.converged && st.iterations < maxIter_) {
        multiply(s_.S, p_, q_);
        double pq = 0.0;
        for (size_t i = 0; i < n; ++i) pq += p_[i] * q_[i];
        if (!(pq > 0.0)) {
            st.breakdown = true;
            break;
        }
        const double alpha = rz / pq;
        double rzNew = 0.0;
        rr = 0.0;
        for (size_t i = 0; i < n; ++i) {
            x_[i] += alpha * p_[i];
            r_[i] -= alpha * q_[i];
            z_[i] = s_.invDiag[i] * r_[i];
            rzNew += r_[i] * z_[i];
            rr += r_[i] * r_[i];
        }
        ++st.iterations;
        if (rr <= limit) {
            st.converged = true;
            break;
        }
        const double beta = rzNew / rz;
        rz = rzNew;
        for (size_t i = 0; i < n; ++i) p_[i] = z_[i] + beta * p_[i];
    }
    st.relResidual = rr0 > 0 ? std::sqrt(rr / rr0) : 0.0;
    for (size_t i = 0; i < n; ++i) T_[i] += x_[i];
    return st;
}

}  // namespace fc
