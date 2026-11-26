#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "font.h"
#include "shaders/field_frag.h"
#include "shaders/field_vert.h"
#include "shaders/line3d_frag.h"
#include "shaders/line3d_vert.h"
#include "shaders/line_frag.h"
#include "shaders/line_vert.h"
#include "shaders/scene_frag.h"
#include "shaders/slice_vert.h"
#include "shaders/surface_vert.h"
#include "shaders/text_frag.h"
#include "shaders/text_vert.h"
#include "theme.h"

namespace fc {

namespace {

constexpr uint32_t kMaxLineVertices = 1024;
constexpr uint32_t kMaxTextInstances = 4096;
constexpr int kLabelChars = 9;  // space reserved for colour-bar tick labels
constexpr double kPi = 3.14159265358979323846;

struct FieldPC {
    float scale[2], offset[2];
    float tmin, tmax;
    int32_t colormap, contours;
};
struct LinePC {
    float scale[2], offset[2];
    float color[4];
};
struct TextPC {
    float pixelToNdc[2];
};
struct DrawPC {  // matches `Draw` in scene.glsl
    float color[4];
    int32_t kind;
    int32_t pad[3];
};
struct SceneUBO {  // matches `Scene` in scene.glsl (std140)
    float viewProj[16];
    float eye[4], light[4], cut[4], side[4];
    int32_t mode[4];
    float range[4];
    int32_t style[4];
};

uint32_t packColor(int r, int g, int b, int a = 255) {
    return static_cast<uint32_t>(r) | (static_cast<uint32_t>(g) << 8) | (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}
uint32_t packColor(const uint8_t (&c)[3]) { return packColor(c[0], c[1], c[2]); }

const uint32_t kTitleColor = packColor(theme::kTitle);
const uint32_t kLineColor = packColor(theme::kText);
const uint32_t kLabelColor = packColor(theme::kLabel);

// "Nice" tick values covering [lo, hi].
std::vector<double> niceTicks(double lo, double hi, int target, double& step) {
    std::vector<double> ticks;
    const double range = hi - lo;
    if (!(range > 0)) {
        step = 1.0;
        ticks.push_back(lo);
        return ticks;
    }
    const double raw = range / target;
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double f = raw / mag;
    const double nice = f < 1.5 ? 1.0 : f < 3.0 ? 2.0 : f < 7.0 ? 5.0 : 10.0;
    step = nice * mag;
    for (double v = std::ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * range; v += step)
        ticks.push_back(std::fabs(v) < 1e-12 * step ? 0.0 : v);
    return ticks;
}

std::string formatTick(double v, double step) {
    char buf[32];
    const double a = std::fabs(v);
    if ((a >= 1e5 || step < 1e-4) && v != 0.0) {
        std::snprintf(buf, sizeof(buf), "%.3g", v);
    } else {
        const int decimals = std::max(0, static_cast<int>(std::ceil(-std::log10(step) - 1e-9)));
        std::snprintf(buf, sizeof(buf), "%.*f", std::min(decimals, 6), v);
    }
    return buf;
}

// Integer text scale for an image height (1 below ~540 px, 2 up to ~900 px, ...).
int textScale(uint32_t height) { return std::max(1, static_cast<int>(std::lround(height / 360.0))); }

// Title line + gap + two status lines, with padding (pixels).
float headerHeight(int ts) {
    const float pad = 8.0f * ts;
    return pad + 9.0f * (ts + 1) + 3.0f * ts + 2.0f * 11.0f * ts + pad;
}

float panelWidth(int ts) {
    const float pad = 8.0f * ts;
    return 2.0f * pad + 12.0f * ts + 7.0f * ts + kLabelChars * 6.0f * ts + 2.0f * pad;
}

int utf8Length(const std::string& s) {
    int n = 0;
    for (char ch : s)
        if ((static_cast<unsigned char>(ch) & 0xC0) != 0x80) ++n;  // count code points, not bytes
    return n;
}

// Column-major 4x4 matrices: m[col * 4 + row].
void matMul(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = s;
        }
}

void normalize3(double* v) {
    const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 0)
        for (int i = 0; i < 3; ++i) v[i] /= l;
}

}  // namespace

Renderer::Renderer(VulkanContext& ctx, const Mesh& mesh, VkBuffer temperature)
    : ctx_(ctx), mesh_(mesh), temperature_(temperature) {
    const VkSampleCountFlags counts =
        ctx_.properties.limits.framebufferColorSampleCounts & ctx_.properties.limits.framebufferDepthSampleCounts;
    samples_ = (counts & VK_SAMPLE_COUNT_4_BIT) ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(ctx_.physicalDevice, VK_FORMAT_D32_SFLOAT, &fp);
    depthFormat_ = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ? VK_FORMAT_D32_SFLOAT
                                                                                              : VK_FORMAT_D16_UNORM;

    // ---- static geometry --------------------------------------------------------------
    std::vector<float> pos(mesh.numNodes() * 3);
    for (size_t i = 0; i < mesh.numNodes(); ++i) {
        pos[3 * i] = static_cast<float>(mesh.x[i]);
        pos[3 * i + 1] = static_cast<float>(mesh.y[i]);
        pos[3 * i + 2] = static_cast<float>(mesh.z[i]);
    }
    positions_ = ctx_.createBuffer(sizeof(float) * pos.size(),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   MemoryKind::Device);
    ctx_.uploadToBuffer(positions_, pos.data(), sizeof(float) * pos.size());

    const VkBufferUsageFlags ib = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto indexBuffer = [&](const std::vector<uint32_t>& idx, Buffer& buf, uint32_t& count) {
        count = static_cast<uint32_t>(idx.size());
        if (idx.empty()) return;
        buf = ctx_.createBuffer(sizeof(uint32_t) * idx.size(), ib, MemoryKind::Device);
        ctx_.uploadToBuffer(buf, idx.data(), sizeof(uint32_t) * idx.size());
    };
    indexBuffer(mesh.wireframeEdges(), edgeIndices_, edgeIndexCount_);
    if (mesh.dim == 2) {
        indexBuffer(mesh.cells, cellIndices_, cellIndexCount_);
        indexBuffer(mesh.boundaryIndices(), outlineIndices_, outlineIndexCount_);
    } else {
        indexBuffer(mesh.boundaryIndices(), surfaceIndices_, surfaceIndexCount_);
        const float x0 = float(mesh.xmin), x1 = float(mesh.xmax), y0 = float(mesh.ymin), y1 = float(mesh.ymax),
                    z0 = float(mesh.zmin), z1 = float(mesh.zmax);
        const float box[24][3] = {{x0, y0, z0}, {x1, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x1, y1, z0}, {x0, y1, z0},
                                  {x0, y1, z0}, {x0, y0, z0}, {x0, y0, z1}, {x1, y0, z1}, {x1, y0, z1}, {x1, y1, z1},
                                  {x1, y1, z1}, {x0, y1, z1}, {x0, y1, z1}, {x0, y0, z1}, {x0, y0, z0}, {x0, y0, z1},
                                  {x1, y0, z0}, {x1, y0, z1}, {x1, y1, z0}, {x1, y1, z1}, {x0, y1, z0}, {x0, y1, z1}};
        boxLines_ = ctx_.createBuffer(sizeof(box), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      MemoryKind::Device);
        ctx_.uploadToBuffer(boxLines_, box, sizeof(box));
        cutLines_ = ctx_.createBuffer(sizeof(float) * 3 * 72, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, MemoryKind::Upload);
    }

    std::vector<uint32_t> fontRows(kGlyphCount * kGlyphRows);
    for (int g = 0; g < kGlyphCount; ++g)
        for (int r = 0; r < kGlyphRows; ++r) fontRows[g * kGlyphRows + r] = kFont5x9[g][r];
    font_ = ctx_.createBuffer(sizeof(uint32_t) * fontRows.size(),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Device);
    ctx_.uploadToBuffer(font_, fontRows.data(), sizeof(uint32_t) * fontRows.size());

    // ---- per-frame buffers ------------------------------------------------------------
    barPos_ = ctx_.createBuffer(sizeof(float) * 18, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, MemoryKind::Upload);
    barT_ = ctx_.createBuffer(sizeof(float) * 6, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, MemoryKind::Upload);
    lines_ = ctx_.createBuffer(sizeof(float) * 3 * kMaxLineVertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, MemoryKind::Upload);
    text_ = ctx_.createBuffer(sizeof(TextInstance) * kMaxTextInstances, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              MemoryKind::Upload);
    sceneUbo_ = ctx_.createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryKind::Upload);

    // ---- descriptor sets ----------------------------------------------------------------
    VkDescriptorSetLayoutBinding fontBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 1;
    lci.pBindings = &fontBinding;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &textSetLayout_));

    const VkShaderStageFlags vf = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding sceneBindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, vf, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vf, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vf, nullptr}};
    VkDescriptorSetLayoutCreateInfo sci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sci.bindingCount = 3;
    sci.pBindings = sceneBindings;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device, &sci, nullptr, &sceneSetLayout_));

    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 2;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &pool_));
    VkDescriptorSetLayout layouts[2] = {textSetLayout_, sceneSetLayout_};
    VkDescriptorSet sets[2];
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 2;
    ai.pSetLayouts = layouts;
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device, &ai, sets));
    textSet_ = sets[0];
    sceneSet_ = sets[1];
    VkDescriptorBufferInfo infos[4] = {{font_.buffer, 0, VK_WHOLE_SIZE},
                                       {sceneUbo_.buffer, 0, VK_WHOLE_SIZE},
                                       {positions_.buffer, 0, VK_WHOLE_SIZE},
                                       {temperature_, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; ++i) {
        writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = i == 0 ? textSet_ : sceneSet_;
        writes[i].dstBinding = i == 0 ? 0 : static_cast<uint32_t>(i - 1);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx_.device, 4, writes, 0, nullptr);

    // ---- pipeline layouts -----------------------------------------------------------------
    VkPushConstantRange pcRange{vf, 0, 32};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(ctx_.device, &plci, nullptr, &pcLayout_));
    VkPushConstantRange textRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(TextPC)};
    VkPipelineLayoutCreateInfo tlci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    tlci.setLayoutCount = 1;
    tlci.pSetLayouts = &textSetLayout_;
    tlci.pushConstantRangeCount = 1;
    tlci.pPushConstantRanges = &textRange;
    VK_CHECK(vkCreatePipelineLayout(ctx_.device, &tlci, nullptr, &textLayout_));
    VkPushConstantRange drawRange{vf, 0, sizeof(DrawPC)};
    VkPipelineLayoutCreateInfo slci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    slci.setLayoutCount = 1;
    slci.pSetLayouts = &sceneSetLayout_;
    slci.pushConstantRangeCount = 1;
    slci.pPushConstantRanges = &drawRange;
    VK_CHECK(vkCreatePipelineLayout(ctx_.device, &slci, nullptr, &sceneLayout_));

    // ---- pipelines --------------------------------------------------------------------------
    const VkVertexInputBindingDescription posBinding{0, 12, VK_VERTEX_INPUT_RATE_VERTEX};
    fieldPipeline_ = createPipeline({spv_field_vert, sizeof(spv_field_vert), spv_field_frag, sizeof(spv_field_frag),
                                     {posBinding, {1, 4, VK_VERTEX_INPUT_RATE_VERTEX}},
                                     {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 1, VK_FORMAT_R32_SFLOAT, 0}},
                                     VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, false, false, pcLayout_});
    linePipeline_ = createPipeline({spv_line_vert, sizeof(spv_line_vert), spv_line_frag, sizeof(spv_line_frag),
                                    {posBinding}, {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}},
                                    VK_PRIMITIVE_TOPOLOGY_LINE_LIST, true, false, false, pcLayout_});
    textPipeline_ = createPipeline({spv_text_vert, sizeof(spv_text_vert), spv_text_frag, sizeof(spv_text_frag),
                                    {{0, sizeof(TextInstance), VK_VERTEX_INPUT_RATE_INSTANCE}},
                                    {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                                     {1, 0, VK_FORMAT_R32_UINT, 8},
                                     {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 12},
                                     {3, 0, VK_FORMAT_R32_SFLOAT, 16}},
                                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, false, false, textLayout_});
    if (mesh.dim == 3) {
        surfacePipeline_ = createPipeline({spv_surface_vert, sizeof(spv_surface_vert), spv_scene_frag,
                                           sizeof(spv_scene_frag), {}, {}, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false,
                                           true, true, sceneLayout_});
        slicePipeline_ = createPipeline({spv_slice_vert, sizeof(spv_slice_vert), spv_scene_frag, sizeof(spv_scene_frag),
                                         {{0, sizeof(SliceVertex), VK_VERTEX_INPUT_RATE_VERTEX}},
                                         {{0, 0, VK_FORMAT_R32G32_UINT, 0}, {1, 0, VK_FORMAT_R32_SFLOAT, 8}},
                                         VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, true, true, sceneLayout_});
        line3dPipeline_ = createPipeline({spv_line3d_vert, sizeof(spv_line3d_vert), spv_line3d_frag,
                                          sizeof(spv_line3d_frag), {posBinding}, {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}},
                                          VK_PRIMITIVE_TOPOLOGY_LINE_LIST, true, true, false, sceneLayout_});
    }

    cmd_ = ctx_.allocateCommandBuffer();
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(ctx_.device);
    ctx_.freeCommandBuffer(cmd_);
    for (VkPipeline p : {fieldPipeline_, linePipeline_, textPipeline_, surfacePipeline_, slicePipeline_, line3dPipeline_})
        if (p) vkDestroyPipeline(ctx_.device, p, nullptr);
    for (VkPipelineLayout l : {pcLayout_, textLayout_, sceneLayout_}) vkDestroyPipelineLayout(ctx_.device, l, nullptr);
    vkDestroyDescriptorPool(ctx_.device, pool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, textSetLayout_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, sceneSetLayout_, nullptr);
    for (Buffer* b : {&positions_, &cellIndices_, &surfaceIndices_, &edgeIndices_, &outlineIndices_, &boxLines_, &font_,
                      &barPos_, &barT_, &lines_, &text_, &sceneUbo_, &cutLines_})
        ctx_.destroy(*b);
    for (Slice& s : slices_) {
        ctx_.destroy(s.buffer);
        ctx_.destroy(s.rim);
    }
}

VkPipeline Renderer::createPipeline(const PipelineDesc& d) {
    VkShaderModule vmod = ctx_.createShaderModule(d.vs, d.vsSize);
    VkShaderModule fmod = ctx_.createShaderModule(d.fs, d.fsSize);
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vmod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fmod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = static_cast<uint32_t>(d.bindings.size());
    vi.pVertexBindingDescriptions = d.bindings.data();
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(d.attributes.size());
    vi.pVertexAttributeDescriptions = d.attributes.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = d.topology;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = samples_;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = d.depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = d.depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (d.blend) {
        att.blendEnable = VK_TRUE;
        att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.colorBlendOp = VK_BLEND_OP_ADD;
        att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &att;
    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates = dyn;

    const VkFormat format = kFormat;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &format;
    rendering.depthAttachmentFormat = depthFormat_;

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &rendering;
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dynState;
    ci.layout = d.layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));
    vkDestroyShaderModule(ctx_.device, vmod, nullptr);
    vkDestroyShaderModule(ctx_.device, fmod, nullptr);
    return pipeline;
}

RenderTarget Renderer::createTarget(uint32_t width, uint32_t height, bool readback) {
    RenderTarget t;
    t.width = width;
    t.height = height;
    if (samples_ != VK_SAMPLE_COUNT_1_BIT)
        t.color = ctx_.createImage(width, height, kFormat,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, samples_);
    t.depth = ctx_.createImage(width, height, depthFormat_,
                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                               samples_);
    t.resolved = ctx_.createImage(width, height, kFormat,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  VK_SAMPLE_COUNT_1_BIT);
    if (readback)
        t.readback = ctx_.createBuffer(static_cast<VkDeviceSize>(width) * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       MemoryKind::Readback);
    return t;
}

void Renderer::destroyTarget(RenderTarget& t) {
    vkDeviceWaitIdle(ctx_.device);
    ctx_.destroy(t.color);
    ctx_.destroy(t.depth);
    ctx_.destroy(t.resolved);
    ctx_.destroy(t.readback);
    t = RenderTarget{};
}

uint32_t Renderer::preferredWidth(uint32_t height, double plotAspect) {
    const int ts = textScale(height);
    const float pad = 8.0f * ts;
    const float availH = std::max(16.0f, static_cast<float>(height) - headerHeight(ts) - 2.0f * pad);
    float w = 2.0f * pad + availH * static_cast<float>(plotAspect) + panelWidth(ts);
    w = std::clamp(w, 480.0f, 4096.0f);
    return (static_cast<uint32_t>(std::ceil(w)) + 1u) & ~1u;
}

Renderer::Layout Renderer::computeLayout(uint32_t width, uint32_t height) const {
    Layout L;
    const float W = static_cast<float>(width), H = static_cast<float>(height);
    L.ts = textScale(height);
    const float ts = static_cast<float>(L.ts);
    L.pad = 8.0f * ts;
    L.titleH = 9.0f * (ts + 1.0f);
    L.lineH = 11.0f * ts;
    const float header = headerHeight(L.ts);
    const float left = 2.0f * L.pad, gap = 2.0f * L.pad;
    const float panel = panelWidth(L.ts);
    L.barW = 12.0f * ts;
    const float availW = std::max(16.0f, W - left - panel);
    const float availH = std::max(16.0f, H - header - 2.0f * L.pad);
    if (mesh_.dim == 3) {
        // The 3D view uses the whole area; the camera fits the domain into it.
        L.plotX = left;
        L.plotY = std::round(header);
        L.plotW = std::round(availW);
        L.plotH = std::round(availH);
        L.barX = L.plotX + L.plotW + gap;
        L.barH = std::round(0.72f * availH);
        L.barY = std::round(L.plotY + 0.5f * (L.plotH - L.barH));
        return L;
    }
    const double dw = mesh_.xmax - mesh_.xmin, dh = mesh_.ymax - mesh_.ymin;
    L.scale = static_cast<float>(std::min(availW / dw, availH / dh));
    L.plotW = std::round(static_cast<float>(dw) * L.scale);
    L.plotH = std::round(static_cast<float>(dh) * L.scale);
    L.plotX = std::round(std::max(left, (W - (L.plotW + panel)) * 0.5f));
    L.plotY = std::round(header + (availH - L.plotH) * 0.5f);
    L.barX = L.plotX + L.plotW + gap;
    L.barH = std::round(std::min(availH, std::max(L.plotH, 0.6f * availH)));
    L.barY = std::round(L.plotY + 0.5f * (L.plotH - L.barH));
    return L;
}

Renderer::Camera Renderer::computeCamera(const Layout& L, const View3D& v) const {
    Camera cam{};
    const double center[3] = {0.5 * (mesh_.xmin + mesh_.xmax), 0.5 * (mesh_.ymin + mesh_.ymax),
                              0.5 * (mesh_.zmin + mesh_.zmax)};
    const double ext[3] = {mesh_.xmax - mesh_.xmin, mesh_.ymax - mesh_.ymin, mesh_.zmax - mesh_.zmin};
    const double R = 0.5 * std::sqrt(ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]);
    const double aspect = std::max(1.0f, L.plotW) / std::max(1.0f, L.plotH);
    const double tanHalf = std::tan(0.5 * 30.0 * kPi / 180.0);
    const double yaw = v.yaw * kPi / 180.0, pitch = std::clamp(v.pitch, -89.0, 89.0) * kPi / 180.0;
    const double dir[3] = {std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch)};
    const double f[3] = {-dir[0], -dir[1], -dir[2]};  // forward
    double s[3] = {f[1] * 1.0 - f[2] * 0.0, f[2] * 0.0 - f[0] * 1.0, 0.0};  // f x (0,0,1)
    normalize3(s);
    const double u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};

    // Fit the vertical cylinder around the box (independent of the yaw, so a
    // spinning camera keeps its zoom): a point p (relative to the centre) at
    // depth dist - p.dir stays inside the view if dist >= p.dir + |p.s| /
    // (m tan aspect) and dist >= p.dir + |p.u| / (m tan).
    const double rxy = 0.5 * std::hypot(ext[0], ext[1]), hz = 0.5 * ext[2], margin = 0.9;
    double fit = 0.0;
    for (int k = 0; k < 48; ++k) {
        const double ang = 2.0 * kPi * k / 48.0;
        for (double zz : {-hz, hz}) {
            const double p[3] = {rxy * std::cos(ang), rxy * std::sin(ang), zz};
            const double along = p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2];
            const double sx = std::fabs(p[0] * s[0] + p[1] * s[1] + p[2] * s[2]);
            const double uy = std::fabs(p[0] * u[0] + p[1] * u[1] + p[2] * u[2]);
            fit = std::max(fit, along + std::max(sx / (margin * tanHalf * aspect), uy / (margin * tanHalf)));
        }
    }
    const double dist = std::max(fit, 1e-6) / std::max(v.zoom, 1e-3);
    for (int a = 0; a < 3; ++a) cam.eye[a] = center[a] + dist * dir[a];
    for (int a = 0; a < 3; ++a) {
        cam.right[a] = s[a];
        cam.up[a] = u[a];
    }
    float view[16] = {};
    for (int a = 0; a < 3; ++a) {
        view[a * 4 + 0] = static_cast<float>(s[a]);
        view[a * 4 + 1] = static_cast<float>(u[a]);
        view[a * 4 + 2] = static_cast<float>(-f[a]);
    }
    view[12] = static_cast<float>(-(s[0] * cam.eye[0] + s[1] * cam.eye[1] + s[2] * cam.eye[2]));
    view[13] = static_cast<float>(-(u[0] * cam.eye[0] + u[1] * cam.eye[1] + u[2] * cam.eye[2]));
    view[14] = static_cast<float>(f[0] * cam.eye[0] + f[1] * cam.eye[1] + f[2] * cam.eye[2]);
    view[15] = 1.0f;

    const double zn = std::max(1e-4 * R, dist - 1.5 * R), zf = dist + 1.5 * R;
    float proj[16] = {};
    proj[0] = static_cast<float>(1.0 / (tanHalf * aspect));
    proj[5] = static_cast<float>(-1.0 / tanHalf);  // Vulkan clip space: y points down
    proj[10] = static_cast<float>(zf / (zn - zf));
    proj[11] = -1.0f;
    proj[14] = static_cast<float>(zn * zf / (zn - zf));
    matMul(proj, view, cam.viewProj);
    return cam;
}

// Marching tetrahedra for an axis-aligned plane: each cut tetrahedron yields a
// triangle (1-3 split) or a quad (2-2 split) whose vertices lie on its edges.
void Renderer::buildSlice(int axis, double at, std::vector<SliceVertex>& out) const {
    out.clear();
    const std::vector<double>& q = axis == 0 ? mesh_.x : axis == 1 ? mesh_.y : mesh_.z;
    const size_t nc = mesh_.numCells();
    for (size_t c = 0; c < nc; ++c) {
        const uint32_t* nd = &mesh_.cells[4 * c];
        double d[4];
        int pos[4], neg[4], np = 0, nn = 0;
        for (int k = 0; k < 4; ++k) {
            d[k] = q[nd[k]] - at;
            if (d[k] >= 0.0) {
                pos[np++] = k;
            } else {
                neg[nn++] = k;
            }
        }
        if (np == 0 || nn == 0) continue;
        auto vertex = [&](int a, int b) {  // a on the positive side, b on the negative side
            out.push_back({nd[a], nd[b], static_cast<float>(d[a] / (d[a] - d[b]))});
        };
        if (np == 1) {
            vertex(pos[0], neg[0]);
            vertex(pos[0], neg[1]);
            vertex(pos[0], neg[2]);
        } else if (np == 3) {
            vertex(pos[0], neg[0]);
            vertex(pos[1], neg[0]);
            vertex(pos[2], neg[0]);
        } else {  // quad (p0,n0) (p0,n1) (p1,n1) (p1,n0)
            vertex(pos[0], neg[0]);
            vertex(pos[0], neg[1]);
            vertex(pos[1], neg[1]);
            vertex(pos[0], neg[0]);
            vertex(pos[1], neg[1]);
            vertex(pos[1], neg[0]);
        }
    }
}

void Renderer::ensureSlice(int axis, double at) {
    Slice& s = slices_[axis];
    if (s.valid && s.at == at) return;
    buildSlice(axis, at, sliceScratch_);
    if (sliceScratch_.size() > s.capacity) {
        ctx_.destroy(s.buffer);  // safe: every submission has completed
        s.capacity = std::max<size_t>(sliceScratch_.size() * 3 / 2, 1024);
        s.buffer = ctx_.createBuffer(sizeof(SliceVertex) * s.capacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                     MemoryKind::Upload);
    }
    if (!sliceScratch_.empty())
        std::memcpy(s.buffer.mapped, sliceScratch_.data(), sizeof(SliceVertex) * sliceScratch_.size());
    s.count = static_cast<uint32_t>(sliceScratch_.size());

    // Curved shapes: outline the cut where the plane crosses the boundary triangles.
    if (!mesh_.boxShaped) {
        const std::vector<double>& q = axis == 0 ? mesh_.x : axis == 1 ? mesh_.y : mesh_.z;
        std::vector<float> rim;
        for (const Mesh::Face& f : mesh_.boundary) {
            const double d[3] = {q[f.n[0]] - at, q[f.n[1]] - at, q[f.n[2]] - at};
            float pts[6];
            int np = 0;
            for (int k = 0; k < 3 && np < 2; ++k) {
                const int a = k, b = (k + 1) % 3;
                if ((d[a] >= 0.0) == (d[b] >= 0.0)) continue;
                const double t = d[a] / (d[a] - d[b]);
                const uint32_t na = f.n[a], nb = f.n[b];
                pts[3 * np + 0] = static_cast<float>(mesh_.x[na] + t * (mesh_.x[nb] - mesh_.x[na]));
                pts[3 * np + 1] = static_cast<float>(mesh_.y[na] + t * (mesh_.y[nb] - mesh_.y[na]));
                pts[3 * np + 2] = static_cast<float>(mesh_.z[na] + t * (mesh_.z[nb] - mesh_.z[na]));
                ++np;
            }
            if (np == 2) rim.insert(rim.end(), pts, pts + 6);
        }
        const size_t verts = rim.size() / 3;
        if (verts > s.rimCapacity) {
            ctx_.destroy(s.rim);
            s.rimCapacity = std::max<size_t>(verts * 3 / 2, 256);
            s.rim = ctx_.createBuffer(sizeof(float) * 3 * s.rimCapacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                      MemoryKind::Upload);
        }
        if (!rim.empty()) std::memcpy(s.rim.mapped, rim.data(), sizeof(float) * rim.size());
        s.rimCount = static_cast<uint32_t>(verts);
    }
    s.at = at;
    s.valid = true;
}

// Edges that make the cut readable: the 12 edges of the removed corner box
// (octant), or the rectangle(s) where the cut plane(s) meet the domain.
void Renderer::buildCutLines(const View3D& v, const float side[3]) {
    const double lo[3] = {mesh_.xmin, mesh_.ymin, mesh_.zmin}, hi[3] = {mesh_.xmax, mesh_.ymax, mesh_.zmax};
    std::vector<float> pts;
    auto box = [&](const double* b0, const double* b1) {  // 12 edges of an axis-aligned box
        for (int a = 0; a < 3; ++a) {
            const int u = (a + 1) % 3, w = (a + 2) % 3;
            for (int k = 0; k < 4; ++k) {
                double p[3], q[3];
                p[a] = b0[a];
                q[a] = b1[a];
                p[u] = q[u] = (k & 1) ? b1[u] : b0[u];
                p[w] = q[w] = (k & 2) ? b1[w] : b0[w];
                for (int c = 0; c < 3; ++c) pts.push_back(static_cast<float>(p[c]));
                for (int c = 0; c < 3; ++c) pts.push_back(static_cast<float>(q[c]));
            }
        }
    };
    auto plane = [&](int a) {  // rectangle where the plane normal to `a` meets the domain
        double b0[3] = {lo[0], lo[1], lo[2]}, b1[3] = {hi[0], hi[1], hi[2]};
        b0[a] = b1[a] = std::clamp(v.cut[a], lo[a], hi[a]);
        const int u = (a + 1) % 3, w = (a + 2) % 3;
        const double corners[4][2] = {{b0[u], b0[w]}, {b1[u], b0[w]}, {b1[u], b1[w]}, {b0[u], b1[w]}};
        for (int k = 0; k < 4; ++k) {
            for (int e = 0; e < 2; ++e) {
                double p[3];
                p[a] = b0[a];
                p[u] = corners[(k + e) % 4][0];
                p[w] = corners[(k + e) % 4][1];
                for (int c = 0; c < 3; ++c) pts.push_back(static_cast<float>(p[c]));
            }
        }
    };
    if (v.mode == View3D::Octant) {
        double b0[3], b1[3];
        for (int a = 0; a < 3; ++a) {
            const double c = std::clamp(v.cut[a], lo[a], hi[a]);
            b0[a] = side[a] > 0 ? c : lo[a];
            b1[a] = side[a] > 0 ? hi[a] : c;
        }
        box(b0, b1);
    } else if (v.mode == View3D::Clip) {
        plane(v.axis);
    } else if (v.mode == View3D::Slices) {
        for (int a = 0; a < 3; ++a) plane(a);
    }
    cutLineCount_ = static_cast<uint32_t>(pts.size() / 3);
    if (!pts.empty()) std::memcpy(cutLines_.mapped, pts.data(), sizeof(float) * pts.size());
}

void Renderer::addText(float x, float y, const std::string& s, int scale, uint32_t color) {
    float cx = std::round(x);
    const float cy = std::round(y);
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int glyph;
        if (c == 0xC2 && i + 1 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0xB0) {
            glyph = kGlyphDegree;
            ++i;
        } else if (c >= 32 && c < 127) {
            glyph = c - 32;
        } else {
            if ((c & 0xC0) == 0x80) continue;  // skip other UTF-8 continuation bytes
            glyph = '?' - 32;
        }
        if (glyph != 0 && textScratch_.size() < kMaxTextInstances)
            textScratch_.push_back({cx, cy, static_cast<uint32_t>(glyph), color, static_cast<float>(scale)});
        cx += 6.0f * scale;
    }
}

void Renderer::addLine(float ax, float ay, float bx, float by) {
    if (lineScratch_.size() + 6 > 3 * kMaxLineVertices) return;
    lineScratch_.insert(lineScratch_.end(), {ax, ay, 0.0f, bx, by, 0.0f});
}

void Renderer::buildOverlay(const Layout& L, const FrameInfo& info, const Camera* cam) {
    textScratch_.clear();
    lineScratch_.clear();
    const int ts = L.ts;

    // Header text.
    addText(2.0f * L.pad, L.pad, info.title, ts + 1, kTitleColor);
    for (size_t i = 0; i < info.lines.size() && i < 2; ++i)
        addText(2.0f * L.pad, L.pad + L.titleH + 3.0f * ts + static_cast<float>(i) * L.lineH, info.lines[i], ts,
                kLineColor);

    // Colour bar quad: bottom = rangeMin, top = rangeMax.
    const float x0 = L.barX, x1 = L.barX + L.barW, y0 = L.barY, y1 = L.barY + L.barH;
    const float pos[18] = {x0, y1, 0, x1, y1, 0, x1, y0, 0, x0, y1, 0, x1, y0, 0, x0, y0, 0};
    const float tv[6] = {info.rangeMin, info.rangeMin, info.rangeMax, info.rangeMin, info.rangeMax, info.rangeMax};
    std::memcpy(barPos_.mapped, pos, sizeof(pos));
    std::memcpy(barT_.mapped, tv, sizeof(tv));

    // Frame around the bar (pixel centres at +0.5 for crisp 1-px lines) and ticks.
    const float ox0 = x0 - 0.5f, ox1 = x1 + 0.5f, oy0 = y0 - 0.5f, oy1 = y1 + 0.5f;
    addLine(ox0, oy0, ox1, oy0);
    addLine(ox1, oy0, ox1, oy1);
    addLine(ox1, oy1, ox0, oy1);
    addLine(ox0, oy1, ox0, oy0);
    double step = 1.0;
    const int target = std::max(3, static_cast<int>(L.barH / (22.0f * ts)));
    const std::vector<double> ticks = niceTicks(info.rangeMin, info.rangeMax, std::min(target, 8), step);
    const double span = static_cast<double>(info.rangeMax) - info.rangeMin;
    for (double v : ticks) {
        const float frac = span > 0 ? static_cast<float>((v - info.rangeMin) / span) : 0.5f;
        const float y = std::round(y1 - frac * L.barH) + 0.5f;
        addLine(x1, y, x1 + 5.0f * ts, y);
        addText(x1 + 7.0f * ts, y - 3.5f * ts, formatTick(v, step), ts, kLabelColor);
    }
    const float captionW = static_cast<float>(utf8Length(info.unit)) * 6.0f * ts;
    addText(x0 + 0.5f * L.barW - 0.5f * captionW, y0 - 7.0f * ts - L.pad, info.unit, ts, kLabelColor);

    // Axis triad (3D): the world axes as seen by the camera, bottom-left of the view.
    triadFirst_ = static_cast<uint32_t>(lineScratch_.size() / 3);
    if (cam) {
        const float len = 22.0f * ts;
        const float ox = L.plotX + 12.0f * ts + len, oy = L.plotY + L.plotH - 12.0f * ts - len;
        const char* names[3] = {"x", "y", "z"};
        for (int a = 0; a < 3; ++a) {
            const float dx = static_cast<float>(cam->right[a]), dy = static_cast<float>(-cam->up[a]);
            addLine(ox, oy, ox + len * dx, oy + len * dy);
            addText(ox + (len + 6.0f * ts) * dx - 2.5f * ts, oy + (len + 6.0f * ts) * dy - 3.5f * ts, names[a], ts,
                    kLabelColor);
        }
    }

    lineVertexCount_ = static_cast<uint32_t>(lineScratch_.size() / 3);
    textCount_ = static_cast<uint32_t>(textScratch_.size());
    if (!lineScratch_.empty()) std::memcpy(lines_.mapped, lineScratch_.data(), sizeof(float) * lineScratch_.size());
    if (!textScratch_.empty())
        std::memcpy(text_.mapped, textScratch_.data(), sizeof(TextInstance) * textScratch_.size());
}

void Renderer::record2D(VkCommandBuffer cmd, const Layout& L, const RenderTarget& t, const FrameInfo& info) {
    const float W = static_cast<float>(t.width), H = static_cast<float>(t.height);
    // Domain -> NDC (y up in the domain, y down in Vulkan NDC).
    const float s = L.scale;
    const float dScale[2] = {2.0f * s / W, -2.0f * s / H};
    const float dOffset[2] = {(L.plotX - static_cast<float>(mesh_.xmin) * s) * 2.0f / W - 1.0f,
                              (L.plotY + L.plotH + static_cast<float>(mesh_.ymin) * s) * 2.0f / H - 1.0f};
    const VkShaderStageFlags pcStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkDeviceSize zero = 0;

    FieldPC fpc{};
    std::memcpy(fpc.scale, dScale, sizeof(dScale));
    std::memcpy(fpc.offset, dOffset, sizeof(dOffset));
    fpc.tmin = info.rangeMin;
    fpc.tmax = info.rangeMax;
    fpc.colormap = info.colormap;
    fpc.contours = info.contours;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fieldPipeline_);
    vkCmdPushConstants(cmd, pcLayout_, pcStages, 0, sizeof(fpc), &fpc);
    VkBuffer fieldBuffers[2] = {positions_.buffer, temperature_};
    VkDeviceSize offsets[2] = {0, 0};
    vkCmdBindVertexBuffers(cmd, 0, 2, fieldBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, cellIndices_.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, cellIndexCount_, 1, 0, 0, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline_);
    LinePC lpc{};
    std::memcpy(lpc.scale, dScale, sizeof(dScale));
    std::memcpy(lpc.offset, dOffset, sizeof(dOffset));
    vkCmdBindVertexBuffers(cmd, 0, 1, &positions_.buffer, &zero);
    if (info.showMesh && edgeIndexCount_ > 0) {
        const float c[4] = {0.0f, 0.0f, 0.0f, 0.35f};
        std::memcpy(lpc.color, c, sizeof(c));
        vkCmdPushConstants(cmd, pcLayout_, pcStages, 0, sizeof(lpc), &lpc);
        vkCmdBindIndexBuffer(cmd, edgeIndices_.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, edgeIndexCount_, 1, 0, 0, 0);
    }
    if (outlineIndexCount_ > 0) {
        const float c[4] = {0.85f, 0.86f, 0.9f, 0.9f};
        std::memcpy(lpc.color, c, sizeof(c));
        vkCmdPushConstants(cmd, pcLayout_, pcStages, 0, sizeof(lpc), &lpc);
        vkCmdBindIndexBuffer(cmd, outlineIndices_.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, outlineIndexCount_, 1, 0, 0, 0);
    }
}

void Renderer::record3D(VkCommandBuffer cmd, const Layout& L, const FrameInfo& info) {
    const View3D& v = info.view;
    VkViewport viewport{L.plotX, L.plotY, L.plotW, L.plotH, 0.0f, 1.0f};
    VkRect2D scissor{{static_cast<int32_t>(L.plotX), static_cast<int32_t>(L.plotY)},
                     {static_cast<uint32_t>(L.plotW), static_cast<uint32_t>(L.plotH)}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneLayout_, 0, 1, &sceneSet_, 0, nullptr);
    const VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkDeviceSize zero = 0;
    DrawPC pc{};

    if (v.mode != View3D::Slices && surfaceIndexCount_ > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, surfacePipeline_);
        pc.kind = 0;
        vkCmdPushConstants(cmd, sceneLayout_, stages, 0, sizeof(pc), &pc);
        vkCmdBindIndexBuffer(cmd, surfaceIndices_.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, surfaceIndexCount_, 1, 0, 0, 0);
    }
    if (v.mode != View3D::Surface) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, slicePipeline_);
        for (int a = 0; a < 3; ++a) {
            if (v.mode == View3D::Clip && a != v.axis) continue;
            const Slice& s = slices_[a];
            if (!s.valid || s.count == 0) continue;
            pc.kind = 1 + a;
            vkCmdPushConstants(cmd, sceneLayout_, stages, 0, sizeof(pc), &pc);
            vkCmdBindVertexBuffers(cmd, 0, 1, &s.buffer.buffer, &zero);
            vkCmdDraw(cmd, s.count, 1, 0, 0);
        }
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line3dPipeline_);
    if (info.showMesh && edgeIndexCount_ > 0 && v.mode != View3D::Slices) {
        DrawPC wire{{0.0f, 0.0f, 0.0f, 0.35f}, 0, {0, 0, 0}};
        vkCmdPushConstants(cmd, sceneLayout_, stages, 0, sizeof(wire), &wire);
        vkCmdBindVertexBuffers(cmd, 0, 1, &positions_.buffer, &zero);
        vkCmdBindIndexBuffer(cmd, edgeIndices_.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, edgeIndexCount_, 1, 0, 0, 0);
    }
    DrawPC outline{{0.85f, 0.86f, 0.9f, 0.85f}, 4, {0, 0, 0}};
    if (mesh_.boxShaped) {
        // The domain is its bounding box: draw its frame and the edges of the cut.
        vkCmdPushConstants(cmd, sceneLayout_, stages, 0, sizeof(outline), &outline);
        vkCmdBindVertexBuffers(cmd, 0, 1, &boxLines_.buffer, &zero);
        vkCmdDraw(cmd, 24, 1, 0, 0);
        if (cutLineCount_ > 0) {
            vkCmdBindVertexBuffers(cmd, 0, 1, &cutLines_.buffer, &zero);
            vkCmdDraw(cmd, cutLineCount_, 1, 0, 0);
        }
    } else if (v.mode != View3D::Surface) {
        // Curved shapes: outline each visible cut where it meets the surface.
        for (int a = 0; a < 3; ++a) {
            if (v.mode == View3D::Clip && a != v.axis) continue;
            const Slice& s = slices_[a];
            if (!s.valid || s.rimCount == 0) continue;
            outline.kind = 1 + a;
            vkCmdPushConstants(cmd, sceneLayout_, stages, 0, sizeof(outline), &outline);
            vkCmdBindVertexBuffers(cmd, 0, 1, &s.rim.buffer, &zero);
            vkCmdDraw(cmd, s.rimCount, 1, 0, 0);
        }
    }
}

void Renderer::record(VkCommandBuffer cmd, RenderTarget& t, const FrameInfo& info) {
    const Layout L = computeLayout(t.width, t.height);
    const bool is3d = mesh_.dim == 3;
    Camera cam{};
    if (is3d) {
        const View3D& v = info.view;
        cam = computeCamera(L, v);
        if (v.mode == View3D::Clip) ensureSlice(v.axis, v.cut[v.axis]);
        if (v.mode == View3D::Octant || v.mode == View3D::Slices)
            for (int a = 0; a < 3; ++a) ensureSlice(a, v.cut[a]);

        SceneUBO u{};
        std::memcpy(u.viewProj, cam.viewProj, sizeof(u.viewProj));
        double light[3];
        const double toEye[3] = {cam.eye[0] - 0.5 * (mesh_.xmin + mesh_.xmax), cam.eye[1] - 0.5 * (mesh_.ymin + mesh_.ymax),
                                 cam.eye[2] - 0.5 * (mesh_.zmin + mesh_.zmax)};
        for (int a = 0; a < 3; ++a) light[a] = toEye[a];
        normalize3(light);
        for (int a = 0; a < 3; ++a) light[a] = 0.8 * light[a] + 0.5 * cam.up[a] - 0.3 * cam.right[a];
        normalize3(light);
        for (int a = 0; a < 3; ++a) {
            u.eye[a] = static_cast<float>(cam.eye[a]);
            u.light[a] = static_cast<float>(light[a]);
            u.cut[a] = static_cast<float>(v.cut[a]);
            u.side[a] = cam.eye[a] >= v.cut[a] ? 1.0f : -1.0f;  // cut away the part facing the camera
        }
        u.light[3] = 0.55f;  // ambient
        buildCutLines(v, u.side);
        u.mode[0] = v.mode;
        u.mode[1] = v.axis;
        u.range[0] = info.rangeMin;
        u.range[1] = info.rangeMax;
        u.style[0] = info.colormap;
        u.style[1] = info.contours;
        std::memcpy(sceneUbo_.mapped, &u, sizeof(u));
    }
    buildOverlay(L, info, is3d ? &cam : nullptr);
    const float W = static_cast<float>(t.width), H = static_cast<float>(t.height);

    fullBarrier(cmd);
    const VkPipelineStageFlags colorStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkPipelineStageFlags depthStages =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    if (samples_ != VK_SAMPLE_COUNT_1_BIT)
        imageBarrier(cmd, t.color.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, colorStage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    imageBarrier(cmd, t.resolved.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, colorStage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    imageBarrier(cmd, t.depth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, depthStages,
                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT);

    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.clearValue.color = {{theme::kBackground[0] / 255.0f, theme::kBackground[1] / 255.0f,
                               theme::kBackground[2] / 255.0f, 1.0f}};
    if (samples_ != VK_SAMPLE_COUNT_1_BIT) {
        color.imageView = t.color.view;
        color.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        color.resolveImageView = t.resolved.view;
        color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else {
        color.imageView = t.resolved.view;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView = t.depth.view;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {t.width, t.height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd, &ri);

    const VkViewport full{0.0f, 0.0f, W, H, 0.0f, 1.0f};
    const VkRect2D fullScissor{{0, 0}, {t.width, t.height}};
    vkCmdSetViewport(cmd, 0, 1, &full);
    vkCmdSetScissor(cmd, 0, 1, &fullScissor);
    if (is3d) {
        record3D(cmd, L, info);
        vkCmdSetViewport(cmd, 0, 1, &full);
        vkCmdSetScissor(cmd, 0, 1, &fullScissor);
    } else {
        record2D(cmd, L, t, info);
    }

    // ---- overlay (pixel coordinates) -----------------------------------------------------
    const float pScale[2] = {2.0f / W, 2.0f / H};
    const float pOffset[2] = {-1.0f, -1.0f};
    const VkShaderStageFlags pcStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkDeviceSize zero = 0;
    VkDeviceSize offsets[2] = {0, 0};

    FieldPC bpc{};
    std::memcpy(bpc.scale, pScale, sizeof(pScale));
    std::memcpy(bpc.offset, pOffset, sizeof(pOffset));
    bpc.tmin = info.rangeMin;
    bpc.tmax = info.rangeMax;
    bpc.colormap = info.colormap;
    bpc.contours = info.contours;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fieldPipeline_);
    vkCmdPushConstants(cmd, pcLayout_, pcStages, 0, sizeof(bpc), &bpc);
    VkBuffer barBuffers[2] = {barPos_.buffer, barT_.buffer};
    vkCmdBindVertexBuffers(cmd, 0, 2, barBuffers, offsets);
    vkCmdDraw(cmd, 6, 1, 0, 0);

    if (lineVertexCount_ > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, &lines_.buffer, &zero);
        LinePC lpc{};
        std::memcpy(lpc.scale, pScale, sizeof(pScale));
        std::memcpy(lpc.offset, pOffset, sizeof(pOffset));
        const float frame[4] = {0.85f, 0.86f, 0.9f, 1.0f};
        std::memcpy(lpc.color, frame, sizeof(frame));
        vkCmdPushConstants(cmd, pcLayout_, pcStages, 0, sizeof(lpc), &lpc);
        vkCmdDraw(cmd, triadFirst_, 1, 0, 0);
        const float axisColors[3][4] = {{0.92f, 0.38f, 0.33f, 1.0f}, {0.45f, 0.80f, 0.45f, 1.0f}, {0.42f, 0.62f, 0.96f, 1.0f}};
        for (uint32_t a = 0; a < 3 && triadFirst_ + 2 * a + 1 < lineVertexCount_; ++a) {
            std::memcpy(lpc.color, axisColors[a], sizeof(lpc.color));
            vkCmdPushConstants(cmd, pcLayout_, pcStages, 0, sizeof(lpc), &lpc);
            vkCmdDraw(cmd, 2, 1, triadFirst_ + 2 * a, 0);
        }
    }

    if (textCount_ > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, textLayout_, 0, 1, &textSet_, 0, nullptr);
        TextPC tpc{{2.0f / W, 2.0f / H}};
        vkCmdPushConstants(cmd, textLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(tpc), &tpc);
        vkCmdBindVertexBuffers(cmd, 0, 1, &text_.buffer, &zero);
        vkCmdDraw(cmd, 6, textCount_, 0, 0);
    }

    vkCmdEndRendering(cmd);
    imageBarrier(cmd, t.resolved.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 colorStage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT);
}

void Renderer::renderToMemory(RenderTarget& t, const FrameInfo& info, std::vector<uint8_t>& rgba) {
    if (!t.readback.buffer) throw std::runtime_error("render target has no readback buffer");
    VK_CHECK(vkResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
    record(cmd_, t, info);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {t.width, t.height, 1};
    vkCmdCopyImageToBuffer(cmd_, t.resolved.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.readback.buffer, 1, &region);
    memoryBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                  VK_ACCESS_HOST_READ_BIT);
    VK_CHECK(vkEndCommandBuffer(cmd_));
    ctx_.submitAndWait(cmd_);
    rgba.resize(static_cast<size_t>(t.width) * t.height * 4);
    std::memcpy(rgba.data(), t.readback.mapped, rgba.size());
}

}  // namespace fc
