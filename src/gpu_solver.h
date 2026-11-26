#pragma once

#include <chrono>
#include <vector>

#include "cpu_solver.h"
#include "fem.h"
#include "vk_context.h"

namespace fc {

// Vulkan compute implementation of the time stepper.
//
// All matrices and vectors live in device-local storage buffers in fp32. An
// implicit step is a Jacobi-preconditioned conjugate-gradient solve recorded as
// chunks of iterations (3 dispatches each); the convergence test runs on the
// GPU and the host only reads a 16-byte status after each chunk. Iterations
// recorded past convergence turn into near-free no-ops.
class GpuSolver {
public:
    GpuSolver(VulkanContext& ctx, const FemSystem& sys, double tolerance, int maxIterations);
    ~GpuSolver();
    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    // Load vector f = dt * F_theta for the next step(s); uploaded with the next step.
    void setLoad(const std::vector<double>& f);
    void setTemperature(const std::vector<double>& T);
    // Advances `count` steps (count > 1 only batches for the explicit scheme with a static load).
    StepStats step(int count = 1);
    void download(std::vector<float>& T);

    VkBuffer temperatureBuffer() const { return T_.buffer; }

    struct Profile {
        long long submits = 0;
        long long dispatches = 0;
        double recordSeconds = 0;  // CPU time spent recording command buffers
        double submitSeconds = 0;  // submit + wait (GPU execution and latency)
    } profile;

private:
    struct Push {
        uint32_t n, numGroups, parity, force;
        float tolSq, absTolSq;
    };
    struct Status {
        uint32_t converged, iterations;
        float rr0, rr;
    };

    void begin();
    void submit();
    void dispatch(VkPipeline pipeline, uint32_t groups, uint32_t parity = 0, uint32_t force = 0);
    void recordLoadUpload();
    void recordIterations(int first, int count);
    void recordStatusCopy();
    Status readStatus() const;
    StepStats stepImplicit();
    StepStats stepExplicit(int count);

    VulkanContext& ctx_;
    const FemSystem& sys_;
    uint32_t n_ = 0, groups_ = 0;
    float tolSq_ = 0, absTolSq_ = 0;
    int maxIter_ = 0;
    int predicted_ = 16;
    bool loadDirty_ = false;

    Buffer rowPtr_, col_, valS_, valA_, rowSum_, invDiag_, load_, T_, x_, r_, z_, p_, q_, partials_, scalars_;
    Buffer upload_, download_, status_;

    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pExplicit_ = VK_NULL_HANDLE, pInit_ = VK_NULL_HANDLE, pInitCheck_ = VK_NULL_HANDLE,
               pSpmv_ = VK_NULL_HANDLE, pUpdate_ = VK_NULL_HANDLE, pDirection_ = VK_NULL_HANDLE,
               pApply_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    std::chrono::steady_clock::time_point recordStart_;
};

}  // namespace fc
