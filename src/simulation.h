#pragma once

#include <vector>

#include "cpu_solver.h"
#include "fem.h"
#include "gpu_solver.h"

namespace fc {

// Drives the time loop: builds the per-step load vector, advances the GPU
// solver (and optionally the CPU reference over the same steps) and stops
// exactly at the output steps so the caller can capture frames.
class Simulation {
public:
    Simulation(const FemSystem& sys, const LoadModel& loads, GpuSolver& gpu, CpuSolver* reference,
               double outputInterval);

    void reset();
    int step() const { return step_; }
    double time() const { return step_ * sys_.dt; }
    bool finished() const { return step_ >= sys_.nSteps; }
    // True while the current step is an output step that has not been written.
    bool outputPending() const { return next_ < outputSteps_.size() && outputSteps_[next_] == step_; }
    void markOutputWritten() { ++next_; }
    // Advances at most maxSteps steps, never past the next pending output.
    // The CPU reference (if any) then runs the same steps on its own, so each
    // processor is timed without the other interleaved.
    int advance(int maxSteps);

    size_t outputCount() const { return outputSteps_.size(); }

    // Statistics.
    long long totalIterations = 0;
    int maxIterations = 0;
    int unconvergedSteps = 0;
    int implicitSteps = 0;
    double gpuSeconds = 0, cpuSeconds = 0;
    StepStats last;

private:
    struct LoadCache {
        int cachedStep = -1;  // F holds F(t_cachedStep)
        std::vector<double> F, Fnext, f;
    };
    // f = dt * (theta F(t_{s+1}) + (1-theta) F(t_s)) for step s.
    void stepLoadFor(LoadCache& cache, int s) const;

    const FemSystem& sys_;
    const LoadModel& loads_;
    GpuSolver& gpu_;
    CpuSolver* ref_;
    bool timeDependent_;
    std::vector<int> outputSteps_;
    size_t next_ = 0;
    int step_ = 0;
    LoadCache gpuLoad_, cpuLoad_;
};

}  // namespace fc
