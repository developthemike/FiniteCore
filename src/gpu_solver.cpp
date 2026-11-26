#include "gpu_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "shaders/apply_delta_comp.h"
#include "shaders/cg_direction_comp.h"
#include "shaders/cg_init_check_comp.h"
#include "shaders/cg_init_comp.h"
#include "shaders/cg_spmv_comp.h"
#include "shaders/cg_update_comp.h"
#include "shaders/explicit_step_comp.h"

namespace fc {

namespace {

constexpr uint32_t kWorkgroupSize = 256;  // must match WG_SIZE in common.glsl
constexpr uint32_t kMaxGroups = 256;      // must match MAX_GROUPS
constexpr uint32_t kBindings = 15;

std::vector<float> toFloat(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

void computeBarrier(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

}  // namespace

GpuSolver::GpuSolver(VulkanContext& ctx, const FemSystem& sys, double tolerance, int maxIterations)
    : ctx_(ctx), sys_(sys) {
    n_ = static_cast<uint32_t>(sys.n);
    groups_ = std::min<uint32_t>((n_ + kWorkgroupSize - 1) / kWorkgroupSize, kMaxGroups);
    tolSq_ = static_cast<float>(tolerance * tolerance);
    absTolSq_ = static_cast<float>(std::max(sys.absTolSq, 1e-36));
    maxIter_ = maxIterations;

    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkDeviceSize nBytes = sizeof(float) * static_cast<VkDeviceSize>(n_);
    const VkDeviceSize nnzBytes = sizeof(float) * static_cast<VkDeviceSize>(sys.nnz());

    rowPtr_ = ctx_.createBuffer(sizeof(uint32_t) * (n_ + 1), storage, MemoryKind::Device);
    col_ = ctx_.createBuffer(sizeof(uint32_t) * sys.nnz(), storage, MemoryKind::Device);
    valS_ = ctx_.createBuffer(nnzBytes, storage, MemoryKind::Device);
    valA_ = ctx_.createBuffer(nnzBytes, storage, MemoryKind::Device);
    rowSum_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    invDiag_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    load_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    // T doubles as the vertex buffer of the renderer.
    T_ = ctx_.createBuffer(nBytes,
                           storage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           MemoryKind::Device);
    x_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    r_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    z_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    p_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    q_ = ctx_.createBuffer(nBytes, storage, MemoryKind::Device);
    partials_ = ctx_.createBuffer(sizeof(float) * 4 * kMaxGroups, storage, MemoryKind::Device);
    scalars_ = ctx_.createBuffer(sizeof(Status), storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Device);
    upload_ = ctx_.createBuffer(nBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Upload);
    download_ = ctx_.createBuffer(nBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);
    status_ = ctx_.createBuffer(sizeof(Status), VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);

    ctx_.uploadToBuffer(rowPtr_, sys.rowPtr.data(), sizeof(uint32_t) * sys.rowPtr.size());
    ctx_.uploadToBuffer(col_, sys.col.data(), sizeof(uint32_t) * sys.col.size());
    auto up = [&](const Buffer& b, const std::vector<double>& v) {
        std::vector<float> f = toFloat(v);
        ctx_.uploadToBuffer(b, f.data(), sizeof(float) * f.size());
    };
    up(valS_, sys.S);
    up(valA_, sys.dtA);
    up(rowSum_, sys.rowSumDtA);
    up(invDiag_, sys.invDiag);
    up(load_, std::vector<double>(sys.n, 0.0));
    up(T_, sys.T0);
    up(x_, std::vector<double>(sys.n, 0.0));
    const Status zero{0, 0, 0.0f, 0.0f};
    ctx_.uploadToBuffer(scalars_, &zero, sizeof(zero));
    std::vector<float> zeros(4 * kMaxGroups, 0.0f);
    ctx_.uploadToBuffer(partials_, zeros.data(), sizeof(float) * zeros.size());

    // Descriptor set: one storage buffer per binding, shared by all kernels.
    std::vector<VkDescriptorSetLayoutBinding> bindings(kBindings);
    for (uint32_t i = 0; i < kBindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = kBindings;
    lci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &setLayout_));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &pool_));

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &setLayout_;
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device, &ai, &set_));

    const Buffer* order[kBindings] = {&rowPtr_, &col_, &valS_, &valA_, &rowSum_, &invDiag_, &load_, &T_,
                                      &x_,      &r_,   &z_,    &p_,    &q_,      &partials_, &scalars_};
    std::vector<VkDescriptorBufferInfo> infos(kBindings);
    std::vector<VkWriteDescriptorSet> writes(kBindings);
    for (uint32_t i = 0; i < kBindings; ++i) {
        infos[i] = {order[i]->buffer, 0, VK_WHOLE_SIZE};
        writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx_.device, kBindings, writes.data(), 0, nullptr);

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(ctx_.device, &plci, nullptr, &layout_));

    auto makePipeline = [&](const uint32_t* code, size_t size) {
        VkShaderModule mod = ctx_.createShaderModule(code, size);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = layout_;
        VkPipeline pipe = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
        vkDestroyShaderModule(ctx_.device, mod, nullptr);
        return pipe;
    };
    pExplicit_ = makePipeline(spv_explicit_step_comp, sizeof(spv_explicit_step_comp));
    pInit_ = makePipeline(spv_cg_init_comp, sizeof(spv_cg_init_comp));
    pInitCheck_ = makePipeline(spv_cg_init_check_comp, sizeof(spv_cg_init_check_comp));
    pSpmv_ = makePipeline(spv_cg_spmv_comp, sizeof(spv_cg_spmv_comp));
    pUpdate_ = makePipeline(spv_cg_update_comp, sizeof(spv_cg_update_comp));
    pDirection_ = makePipeline(spv_cg_direction_comp, sizeof(spv_cg_direction_comp));
    pApply_ = makePipeline(spv_apply_delta_comp, sizeof(spv_apply_delta_comp));

    cmd_ = ctx_.allocateCommandBuffer();
}

GpuSolver::~GpuSolver() {
    vkDeviceWaitIdle(ctx_.device);
    for (VkPipeline p : {pExplicit_, pInit_, pInitCheck_, pSpmv_, pUpdate_, pDirection_, pApply_})
        vkDestroyPipeline(ctx_.device, p, nullptr);
    vkDestroyPipelineLayout(ctx_.device, layout_, nullptr);
    vkDestroyDescriptorPool(ctx_.device, pool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, setLayout_, nullptr);
    ctx_.freeCommandBuffer(cmd_);
    for (Buffer* b : {&rowPtr_, &col_, &valS_, &valA_, &rowSum_, &invDiag_, &load_, &T_, &x_, &r_, &z_, &p_, &q_,
                      &partials_, &scalars_, &upload_, &download_, &status_})
        ctx_.destroy(*b);
}

void GpuSolver::setLoad(const std::vector<double>& f) {
    float* dst = static_cast<float*>(upload_.mapped);
    for (uint32_t i = 0; i < n_; ++i) dst[i] = static_cast<float>(f[i]);
    loadDirty_ = true;
}

void GpuSolver::setTemperature(const std::vector<double>& T) {
    std::vector<float> f = toFloat(T);
    ctx_.uploadToBuffer(T_, f.data(), sizeof(float) * f.size());
    std::vector<float> zero(n_, 0.0f);
    ctx_.uploadToBuffer(x_, zero.data(), sizeof(float) * zero.size());
    predicted_ = 16;
}

void GpuSolver::begin() {
    recordStart_ = std::chrono::steady_clock::now();
    VK_CHECK(vkResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
    // Orders this submission after everything before it (incl. the renderer
    // reading T as a vertex buffer).
    fullBarrier(cmd_);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
}

void GpuSolver::submit() {
    VK_CHECK(vkEndCommandBuffer(cmd_));
    const auto t0 = std::chrono::steady_clock::now();
    ctx_.submitAndWait(cmd_);
    const auto t1 = std::chrono::steady_clock::now();
    profile.recordSeconds += std::chrono::duration<double>(t0 - recordStart_).count();
    profile.submitSeconds += std::chrono::duration<double>(t1 - t0).count();
    profile.submits++;
}

void GpuSolver::dispatch(VkPipeline pipeline, uint32_t groups, uint32_t parity, uint32_t force) {
    Push pc{n_, groups_, parity, force, tolSq_, absTolSq_};
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);
    vkCmdDispatch(cmd_, groups, 1, 1);
    computeBarrier(cmd_);
    profile.dispatches++;
}

void GpuSolver::recordLoadUpload() {
    if (!loadDirty_) return;
    VkBufferCopy region{0, 0, sizeof(float) * static_cast<VkDeviceSize>(n_)};
    vkCmdCopyBuffer(cmd_, upload_.buffer, load_.buffer, 1, &region);
    memoryBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    loadDirty_ = false;
}

void GpuSolver::recordIterations(int first, int count) {
    for (int j = first; j < first + count; ++j) {
        const uint32_t parity = static_cast<uint32_t>(j & 1);
        dispatch(pSpmv_, groups_, parity);
        dispatch(pUpdate_, groups_, parity);
        dispatch(pDirection_, groups_, parity);
    }
}

void GpuSolver::recordStatusCopy() {
    memoryBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy region{0, 0, sizeof(Status)};
    vkCmdCopyBuffer(cmd_, scalars_.buffer, status_.buffer, 1, &region);
    memoryBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                  VK_ACCESS_HOST_READ_BIT);
}

GpuSolver::Status GpuSolver::readStatus() const {
    Status st;
    std::memcpy(&st, status_.mapped, sizeof(st));
    return st;
}

StepStats GpuSolver::step(int count) {
    if (count < 1) return {};
    if (sys_.diagonal) return stepExplicit(count);
    StepStats last;
    for (int i = 0; i < count; ++i) last = stepImplicit();
    return last;
}

StepStats GpuSolver::stepExplicit(int count) {
    begin();
    recordLoadUpload();
    for (int s = 0; s < count; ++s) {
        dispatch(pExplicit_, groups_);
        dispatch(pApply_, groups_, 0, /*force=*/1);
    }
    submit();
    return StepStats{};
}

StepStats GpuSolver::stepImplicit() {
    // First submission: residual, status reset and a chunk sized from the
    // previous step's iteration count (consecutive steps need similar counts).
    const int chunk = std::clamp(predicted_, 1, maxIter_);
    begin();
    recordLoadUpload();
    dispatch(pInit_, groups_);
    dispatch(pInitCheck_, 1);
    recordIterations(0, chunk);
    dispatch(pApply_, groups_, 0, 0);
    recordStatusCopy();
    submit();
    Status st = readStatus();

    int recorded = chunk;
    while (st.converged == 0 && recorded < maxIter_) {
        const int more = std::min(std::max(4, chunk / 2), maxIter_ - recorded);
        begin();
        recordIterations(recorded, more);
        dispatch(pApply_, groups_, 0, 0);
        recordStatusCopy();
        submit();
        st = readStatus();
        recorded += more;
    }

    StepStats out;
    out.iterations = static_cast<int>(st.iterations);
    out.converged = st.converged == 1;
    out.breakdown = st.converged == 2;
    out.relResidual = st.rr0 > 0 ? std::sqrt(std::max(st.rr, 0.0f) / st.rr0) : 0.0;
    if (st.converged != 1) {
        // Max iterations or breakdown: keep the best increment we have.
        begin();
        dispatch(pApply_, groups_, 0, /*force=*/1);
        submit();
    }
    predicted_ = std::max(1, out.iterations + 1);
    return out;
}

void GpuSolver::download(std::vector<float>& T) {
    begin();
    VkBufferCopy region{0, 0, sizeof(float) * static_cast<VkDeviceSize>(n_)};
    vkCmdCopyBuffer(cmd_, T_.buffer, download_.buffer, 1, &region);
    memoryBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                  VK_ACCESS_HOST_READ_BIT);
    submit();
    T.resize(n_);
    std::memcpy(T.data(), download_.mapped, sizeof(float) * n_);
}

}  // namespace fc
