#pragma once

#include <string>
#include <vector>

#include "mesh.h"
#include "vk_context.h"

namespace fc {

struct RenderTarget {
    uint32_t width = 0, height = 0;
    Image color;      // multisampled colour attachment (unused with 1 sample)
    Image depth;      // depth attachment (used by 3D scenes)
    Image resolved;   // single-sample result, TRANSFER_SRC_OPTIMAL after record()
    Buffer readback;  // optional host copy of `resolved`
};

// Camera and cut-away settings for 3D meshes.
struct View3D {
    enum Mode { Surface = 0, Clip = 1, Octant = 2, Slices = 3 };
    int mode = Octant;
    int axis = 0;                  // clip axis (0 x, 1 y, 2 z)
    double cut[3] = {0, 0, 0};     // cut point
    double yaw = -60, pitch = 25;  // degrees; z is up
    double zoom = 1.0;
};

struct FrameInfo {
    std::string title;
    std::vector<std::string> lines;  // status lines under the title
    std::string unit;                // colour-bar caption
    float rangeMin = 0.0f, rangeMax = 1.0f;
    int colormap = 0;
    int contours = 10;
    bool showMesh = false;
    View3D view;  // 3D meshes only
};

// Draws the FEM temperature field straight from the solver's T buffer, plus
// isotherms, mesh, colour bar and text, into an offscreen image shared by the
// window presenter and the frame exporter.
//   2D: the triangles, coloured per node (exactly the P1 field).
//   3D: the exterior surface and planar cuts through the tetrahedra, with an
//       orbit camera, depth buffer and lighting; cut-away modes reveal the
//       interior.
class Renderer {
public:
    Renderer(VulkanContext& ctx, const Mesh& mesh, VkBuffer temperature);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    RenderTarget createTarget(uint32_t width, uint32_t height, bool readback);
    void destroyTarget(RenderTarget& t);

    // Records a frame into target.resolved. Per-frame geometry goes into
    // host-visible buffers, so the previous frame must have completed
    // (FiniteCore always waits for each submission).
    void record(VkCommandBuffer cmd, RenderTarget& target, const FrameInfo& info);
    // Renders and copies the pixels (RGBA8) to host memory. Blocking.
    void renderToMemory(RenderTarget& target, const FrameInfo& info, std::vector<uint8_t>& rgba);

    // Image width for the given height when the plot area has this aspect (width/height).
    static uint32_t preferredWidth(uint32_t height, double plotAspect);
    static constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

private:
    struct Layout {
        int ts = 1;  // integer text scale
        float pad = 8, titleH = 14, lineH = 10;
        float plotX = 0, plotY = 0, plotW = 0, plotH = 0, scale = 1;
        float barX = 0, barY = 0, barW = 0, barH = 0;
    };
    struct TextInstance {
        float x, y;
        uint32_t glyph;
        uint32_t color;
        float scale;
    };
    struct SliceVertex {
        uint32_t a, b;  // mesh edge
        float s;        // position along the edge
    };
    struct Slice {
        Buffer buffer;
        size_t capacity = 0;
        uint32_t count = 0;
        // Outline where the plane meets the boundary surface (non-box meshes).
        Buffer rim;
        size_t rimCapacity = 0;
        uint32_t rimCount = 0;
        double at = 0;
        bool valid = false;
    };
    struct Camera {
        float viewProj[16];
        double eye[3], right[3], up[3];
    };
    struct PipelineDesc {
        const uint32_t* vs;
        size_t vsSize;
        const uint32_t* fs;
        size_t fsSize;
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        bool blend = false;
        bool depthTest = false;
        bool depthWrite = false;
        VkPipelineLayout layout = VK_NULL_HANDLE;
    };

    Layout computeLayout(uint32_t width, uint32_t height) const;
    Camera computeCamera(const Layout& L, const View3D& v) const;
    void buildOverlay(const Layout& L, const FrameInfo& info, const Camera* cam);
    void addText(float x, float y, const std::string& s, int scale, uint32_t color);
    void addLine(float ax, float ay, float bx, float by);
    void buildSlice(int axis, double at, std::vector<SliceVertex>& out) const;
    void ensureSlice(int axis, double at);
    void buildCutLines(const View3D& v, const float side[3]);
    void record2D(VkCommandBuffer cmd, const Layout& L, const RenderTarget& t, const FrameInfo& info);
    void record3D(VkCommandBuffer cmd, const Layout& L, const FrameInfo& info);
    VkPipeline createPipeline(const PipelineDesc& d);

    VulkanContext& ctx_;
    const Mesh& mesh_;
    VkBuffer temperature_;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkFormat depthFormat_ = VK_FORMAT_D32_SFLOAT;

    // Static mesh geometry.
    Buffer positions_;      // xyz per node (vertex + storage buffer)
    Buffer cellIndices_;    // 2D triangles
    Buffer surfaceIndices_; // 3D boundary triangles
    Buffer edgeIndices_;    // wireframe
    Buffer outlineIndices_; // 2D boundary edges
    Buffer boxLines_;       // 3D bounding box (24 vertices)
    Buffer font_;
    uint32_t cellIndexCount_ = 0, surfaceIndexCount_ = 0, edgeIndexCount_ = 0, outlineIndexCount_ = 0;

    // Per-frame data (host-visible).
    Buffer barPos_, barT_, lines_, text_, sceneUbo_;
    Buffer cutLines_;  // 3D: outline of the cut-away region / cut planes
    uint32_t cutLineCount_ = 0;
    uint32_t lineVertexCount_ = 0, textCount_ = 0;
    uint32_t triadFirst_ = 0;  // first UI-line vertex of the axis triad (3D)
    std::vector<float> lineScratch_;
    std::vector<TextInstance> textScratch_;
    Slice slices_[3];
    std::vector<SliceVertex> sliceScratch_;

    VkDescriptorSetLayout textSetLayout_ = VK_NULL_HANDLE, sceneSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet textSet_ = VK_NULL_HANDLE, sceneSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pcLayout_ = VK_NULL_HANDLE, textLayout_ = VK_NULL_HANDLE, sceneLayout_ = VK_NULL_HANDLE;
    VkPipeline fieldPipeline_ = VK_NULL_HANDLE, linePipeline_ = VK_NULL_HANDLE, textPipeline_ = VK_NULL_HANDLE;
    VkPipeline surfacePipeline_ = VK_NULL_HANDLE, slicePipeline_ = VK_NULL_HANDLE, line3dPipeline_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
};

}  // namespace fc
