#include "simulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace fc {

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
}  // namespace

Simulation::Simulation(const FemSystem& sys, const LoadModel& loads, GpuSolver& gpu, CpuSolver* reference,
                       double outputInterval)
    : sys_(sys), loads_(loads), gpu_(gpu), ref_(reference), timeDependent_(loads.timeDependent()) {
    // Output at the first step at or after every multiple of the interval, plus the final step.
    for (int k = 0; k <= 10'000'000; ++k) {
        const double t = k * outputInterval;
        if (t > sys.tEnd * (1.0 + 1e-12)) break;
        const int s = std::min(static_cast<int>(std::ceil(t / sys.dt - 1e-6)), sys.nSteps);
        if (outputSteps_.empty() || s > outputSteps_.back()) outputSteps_.push_back(s);
    }
    if (outputSteps_.empty() || outputSteps_.back() != sys.nSteps) outputSteps_.push_back(sys.nSteps);

    if (!timeDependent_) {
        LoadCache c;
        stepLoadFor(c, 0);
        gpu_.setLoad(c.f);
        if (ref_) ref_->setLoad(c.f);
    }
}

void Simulation::reset() {
    step_ = 0;
    next_ = 0;
    gpuLoad_.cachedStep = -1;
    cpuLoad_.cachedStep = -1;
    gpu_.setTemperature(sys_.T0);
    if (ref_) ref_->setTemperature(sys_.T0);
}

void Simulation::stepLoadFor(LoadCache& c, int s) const {
    if (c.cachedStep != s) loads_.evaluate(s * sys_.dt, c.F);
    if (!timeDependent_) {
        stepLoad(sys_, c.F, c.F, c.f);
        c.cachedStep = s;
        return;
    }
    loads_.evaluate((s + 1) * sys_.dt, c.Fnext);
    stepLoad(sys_, c.F, c.Fnext, c.f);
    std::swap(c.F, c.Fnext);
    c.cachedStep = s + 1;
}

int Simulation::advance(int maxSteps) {
    if (finished() || maxSteps <= 0) return 0;
    int limit = sys_.nSteps;
    if (next_ < outputSteps_.size()) limit = std::min(limit, outputSteps_[next_]);
    if (limit <= step_) return 0;  // an output is pending at this step
    const int target = std::min(step_ + maxSteps, limit);
    const int first = step_;

    while (step_ < target) {
        int k = 1;
        if (timeDependent_) {
            stepLoadFor(gpuLoad_, step_);
            gpu_.setLoad(gpuLoad_.f);
        } else if (sys_.diagonal) {
            k = std::min(target - step_, 256);  // explicit steps batch into one submission
        }
        const auto t0 = Clock::now();
        last = gpu_.step(k);
        gpuSeconds += seconds(t0, Clock::now());
        if (!sys_.diagonal) {
            ++implicitSteps;
            totalIterations += last.iterations;
            maxIterations = std::max(maxIterations, last.iterations);
            if (!last.converged) {
                if (unconvergedSteps < 5)
                    std::fprintf(stderr, "warning: step %d: CG %s after %d iterations (|r|/|r0| = %.2e)\n", step_ + 1,
                                 last.breakdown ? "broke down" : "did not converge", last.iterations, last.relResidual);
                ++unconvergedSteps;
            }
        }
        step_ += k;
    }

    if (ref_) {
        for (int s = first; s < step_; ++s) {
            if (timeDependent_) {
                stepLoadFor(cpuLoad_, s);
                ref_->setLoad(cpuLoad_.f);
            }
            const auto t0 = Clock::now();
            ref_->step();
            cpuSeconds += seconds(t0, Clock::now());
        }
    }
    return step_ - first;
}

}  // namespace fc
