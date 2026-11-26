#pragma once

#include <functional>
#include <string>
#include <vector>

#include "vk_common.h"

namespace fc {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // persistently mapped for host-visible kinds
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

enum class MemoryKind {
    Device,    // device-local, not mappable
    Upload,    // host-visible + coherent (CPU writes, GPU reads)
    Readback,  // host-visible + coherent, cached when available (GPU writes, CPU reads)
};

// Instance, device, a single graphics+compute queue and small helpers.
// Everything in FiniteCore runs on that one queue, which keeps the
// compute -> graphics hand-off of the temperature buffer trivial.
class VulkanContext {
public:
    struct Options {
        int deviceIndex = -1;  // -1: pick automatically (discrete > integrated > others)
        bool validation = false;
        std::vector<std::string> instanceExtensions;            // e.g. surface extensions from GLFW
        std::function<VkSurfaceKHR(VkInstance)> createSurface;  // empty: headless
    };

    explicit VulkanContext(const Options& options);
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    static void listDevices();

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind);
    void destroy(Buffer& b);
    Image createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                      VkSampleCountFlagBits samples);
    void destroy(Image& img);

    // Copies host data into a (device-local) buffer through a temporary staging buffer. Blocking.
    void uploadToBuffer(const Buffer& dst, const void* data, VkDeviceSize size, VkDeviceSize dstOffset = 0);

    VkCommandBuffer allocateCommandBuffer();
    void freeCommandBuffer(VkCommandBuffer cmd);
    // Submits a recorded command buffer and blocks until the GPU is done with it.
    void submitAndWait(VkCommandBuffer cmd);

    VkShaderModule createShaderModule(const uint32_t* code, size_t sizeBytes);

    std::string deviceName() const { return properties.deviceName; }
    std::string deviceTypeName() const;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};

private:
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) const;

    VkFence fence_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

// Convenience barriers (synchronization 1).
void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
// Orders everything recorded/submitted before against everything after.
void fullBarrier(VkCommandBuffer cmd);
void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                  VkAccessFlags dstAccess, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

}  // namespace fc
