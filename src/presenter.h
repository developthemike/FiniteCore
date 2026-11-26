#pragma once

#ifdef FC_HAVE_GLFW

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>

#include "renderer.h"
#include "vk_context.h"

namespace fc {

// Swapchain + per-frame submission for the live window. The renderer draws
// into an offscreen target which is then blitted to the swapchain image.
class Presenter {
public:
    Presenter(VulkanContext& ctx, Renderer& renderer, GLFWwindow* window);
    ~Presenter();
    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    // Draws and presents one frame (blocking). Returns false when skipped
    // (window minimised or swapchain being recreated).
    bool present(const FrameInfo& info);
    VkExtent2D extent() const { return extent_; }

private:
    void recreate();
    void destroySwapchain(bool keepHandle);

    VulkanContext& ctx_;
    Renderer& renderer_;
    GLFWwindow* window_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{0, 0};
    std::vector<VkImage> images_;
    std::vector<VkSemaphore> renderDone_;
    VkSemaphore acquired_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    RenderTarget target_;
    bool dirty_ = true;
};

}  // namespace fc

#endif  // FC_HAVE_GLFW
