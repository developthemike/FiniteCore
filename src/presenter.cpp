#ifdef FC_HAVE_GLFW

#include "presenter.h"

#include <algorithm>
#include <stdexcept>

namespace fc {

Presenter::Presenter(VulkanContext& ctx, Renderer& renderer, GLFWwindow* window)
    : ctx_(ctx), renderer_(renderer), window_(window) {
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(ctx_.device, &sci, nullptr, &acquired_));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(ctx_.device, &fci, nullptr, &fence_));
    cmd_ = ctx_.allocateCommandBuffer();
    recreate();
}

Presenter::~Presenter() {
    vkDeviceWaitIdle(ctx_.device);
    destroySwapchain(false);
    ctx_.freeCommandBuffer(cmd_);
    vkDestroyFence(ctx_.device, fence_, nullptr);
    vkDestroySemaphore(ctx_.device, acquired_, nullptr);
}

void Presenter::destroySwapchain(bool keepHandle) {
    for (VkSemaphore s : renderDone_) vkDestroySemaphore(ctx_.device, s, nullptr);
    renderDone_.clear();
    images_.clear();
    if (target_.resolved.image) renderer_.destroyTarget(target_);
    if (!keepHandle && swapchain_) {
        vkDestroySwapchainKHR(ctx_.device, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void Presenter::recreate() {
    vkDeviceWaitIdle(ctx_.device);
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window_, &fbw, &fbh);
    if (fbw <= 0 || fbh <= 0) return;

    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.physicalDevice, ctx_.surface, &caps));
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        throw std::runtime_error("the window surface does not support TRANSFER_DST swapchain images");

    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physicalDevice, ctx_.surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(nf);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physicalDevice, ctx_.surface, &nf, formats.data());
    auto blittable = [&](VkFormat f) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(ctx_.physicalDevice, f, &fp);
        return (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
    };
    // Prefer a UNORM format: the renderer already produces display-encoded colours.
    const VkSurfaceFormatKHR* chosen = nullptr;
    for (const auto& f : formats)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) && blittable(f.format)) {
            chosen = &f;
            break;
        }
    if (!chosen)
        for (const auto& f : formats)
            if (blittable(f.format)) {
                chosen = &f;
                break;
            }
    if (!chosen) throw std::runtime_error("no blittable swapchain format");
    format_ = chosen->format;

    if (caps.currentExtent.width != UINT32_MAX) {
        extent_ = caps.currentExtent;
    } else {
        extent_.width = std::clamp(static_cast<uint32_t>(fbw), caps.minImageExtent.width, caps.maxImageExtent.width);
        extent_.height = std::clamp(static_cast<uint32_t>(fbh), caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent_.width == 0 || extent_.height == 0) return;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);
    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & alpha)) {
        for (VkCompositeAlphaFlagBitsKHR a : {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                                              VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                                              VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            if (caps.supportedCompositeAlpha & a) {
                alpha = a;
                break;
            }
    }

    VkSwapchainKHR old = swapchain_;
    destroySwapchain(true);

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = ctx_.surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosen->format;
    ci.imageColorSpace = chosen->colorSpace;
    ci.imageExtent = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = alpha;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // vsync; always available
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = old;
    VK_CHECK(vkCreateSwapchainKHR(ctx_.device, &ci, nullptr, &swapchain_));
    if (old) vkDestroySwapchainKHR(ctx_.device, old, nullptr);

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(ctx_.device, swapchain_, &n, nullptr);
    images_.resize(n);
    vkGetSwapchainImagesKHR(ctx_.device, swapchain_, &n, images_.data());
    renderDone_.resize(n);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& s : renderDone_) VK_CHECK(vkCreateSemaphore(ctx_.device, &sci, nullptr, &s));
    target_ = renderer_.createTarget(extent_.width, extent_.height, false);
    dirty_ = false;
}

bool Presenter::present(const FrameInfo& info) {
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window_, &fbw, &fbh);
    if (fbw <= 0 || fbh <= 0) return false;
    if (dirty_ || !swapchain_ || static_cast<uint32_t>(fbw) != extent_.width ||
        static_cast<uint32_t>(fbh) != extent_.height) {
        dirty_ = true;
        recreate();
        if (dirty_ || !swapchain_ || !target_.resolved.image) return false;
    }

    uint32_t index = 0;
    VkResult r = vkAcquireNextImageKHR(ctx_.device, swapchain_, UINT64_MAX, acquired_, VK_NULL_HANDLE, &index);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        dirty_ = true;
        return false;
    }
    if (r == VK_SUBOPTIMAL_KHR) {
        dirty_ = true;
    } else if (r != VK_SUCCESS) {
        VK_CHECK(r);
    }

    VK_CHECK(vkResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
    renderer_.record(cmd_, target_, info);
    imageBarrier(cmd_, images_[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<int32_t>(target_.width), static_cast<int32_t>(target_.height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(extent_.width), static_cast<int32_t>(extent_.height), 1};
    vkCmdBlitImage(cmd_, target_.resolved.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[index],
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    imageBarrier(cmd_, images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    VK_CHECK(vkEndCommandBuffer(cmd_));

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &acquired_;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &renderDone_[index];
    VK_CHECK(vkResetFences(ctx_.device, 1, &fence_));
    VK_CHECK(vkQueueSubmit(ctx_.queue, 1, &si, fence_));
    VK_CHECK(vkWaitForFences(ctx_.device, 1, &fence_, VK_TRUE, UINT64_MAX));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderDone_[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &index;
    r = vkQueuePresentKHR(ctx_.queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        dirty_ = true;
    } else if (r != VK_SUCCESS) {
        VK_CHECK(r);
    }
    return true;
}

}  // namespace fc

#endif  // FC_HAVE_GLFW
