#pragma once

#include <vulkan/vulkan.h>

namespace fc {

const char* vkResultName(VkResult r);
[[noreturn]] void vkCheckFailed(VkResult r, const char* expr, const char* file, int line);

}  // namespace fc

#define VK_CHECK(expr)                                                     \
    do {                                                                   \
        VkResult fc_vk_result_ = (expr);                                   \
        if (fc_vk_result_ != VK_SUCCESS)                                   \
            ::fc::vkCheckFailed(fc_vk_result_, #expr, __FILE__, __LINE__); \
    } while (0)
