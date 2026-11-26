#include "vk_context.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fc {

const char* vkResultName(VkResult r) {
    switch (r) {
#define FC_CASE(x) \
    case x:        \
        return #x;
        FC_CASE(VK_SUCCESS)
        FC_CASE(VK_NOT_READY)
        FC_CASE(VK_TIMEOUT)
        FC_CASE(VK_INCOMPLETE)
        FC_CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
        FC_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        FC_CASE(VK_ERROR_INITIALIZATION_FAILED)
        FC_CASE(VK_ERROR_DEVICE_LOST)
        FC_CASE(VK_ERROR_MEMORY_MAP_FAILED)
        FC_CASE(VK_ERROR_LAYER_NOT_PRESENT)
        FC_CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
        FC_CASE(VK_ERROR_FEATURE_NOT_PRESENT)
        FC_CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
        FC_CASE(VK_ERROR_TOO_MANY_OBJECTS)
        FC_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
        FC_CASE(VK_ERROR_SURFACE_LOST_KHR)
        FC_CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)
        FC_CASE(VK_SUBOPTIMAL_KHR)
        FC_CASE(VK_ERROR_OUT_OF_DATE_KHR)
#undef FC_CASE
        default:
            return "VK_ERROR_(unknown)";
    }
}

void vkCheckFailed(VkResult r, const char* expr, const char* file, int line) {
    throw std::runtime_error(std::string("Vulkan call failed: ") + expr + " -> " + vkResultName(r) + " (" + file +
                             ":" + std::to_string(line) + ")");
}

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) std::fprintf(stderr, "[vulkan] %s\n", data->pMessage);
    return VK_FALSE;
}

bool hasInstanceLayer(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers)
        if (std::strcmp(l.layerName, name) == 0) return true;
    return false;
}

bool hasInstanceExtension(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

bool hasDeviceExtension(VkPhysicalDevice dev, const char* name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, exts.data());
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

const char* typeName(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU (software)";
        default: return "other";
    }
}

int typeScore(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
        default: return 0;
    }
}

struct DeviceInfo {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    int family = -1;
    bool suitable = false;
    std::string reason;
};

DeviceInfo inspectDevice(VkPhysicalDevice dev, VkSurfaceKHR surface) {
    DeviceInfo info;
    info.device = dev;
    vkGetPhysicalDeviceProperties(dev, &info.props);
    if (info.props.apiVersion < VK_API_VERSION_1_3) {
        info.reason = "requires Vulkan 1.3";
        return info;
    }
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(dev, &f2);
    if (!f13.dynamicRendering) {
        info.reason = "dynamicRendering not supported";
        return info;
    }
    if (surface != VK_NULL_HANDLE && !hasDeviceExtension(dev, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        info.reason = "no swapchain support";
        return info;
    }
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qn, qprops.data());
    for (uint32_t q = 0; q < qn; ++q) {
        const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((qprops[q].queueFlags & need) != need) continue;
        if (surface != VK_NULL_HANDLE) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, q, surface, &present);
            if (!present) continue;
        }
        info.family = static_cast<int>(q);
        break;
    }
    if (info.family < 0) {
        info.reason = surface ? "no graphics+compute queue that can present to the window"
                              : "no graphics+compute queue";
        return info;
    }
    info.suitable = true;
    return info;
}

VkInstance createInstance(const std::vector<const char*>& exts, const std::vector<const char*>& layers) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "FiniteCore";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.pEngineName = "FiniteCore";
    app.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    VkInstance instance = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
    return instance;
}

}  // namespace

std::string VulkanContext::deviceTypeName() const { return typeName(properties.deviceType); }

void VulkanContext::listDevices() {
    VkInstance inst = createInstance({}, {});
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());
    std::printf("Vulkan devices:\n");
    for (uint32_t i = 0; i < count; ++i) {
        DeviceInfo info = inspectDevice(devs[i], VK_NULL_HANDLE);
        std::printf("  [%u] %s (%s, Vulkan %u.%u)%s%s\n", i, info.props.deviceName, typeName(info.props.deviceType),
                    VK_API_VERSION_MAJOR(info.props.apiVersion), VK_API_VERSION_MINOR(info.props.apiVersion),
                    info.suitable ? "" : "  -- unsuitable: ", info.suitable ? "" : info.reason.c_str());
    }
    if (count == 0) std::printf("  (none)\n");
    vkDestroyInstance(inst, nullptr);
}

VulkanContext::VulkanContext(const Options& opt) {
    std::vector<const char*> exts;
    for (const auto& e : opt.instanceExtensions) exts.push_back(e.c_str());
    std::vector<const char*> layers;
    bool debugUtils = false;
    if (opt.validation) {
        if (hasInstanceLayer("VK_LAYER_KHRONOS_validation")) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            if (hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                debugUtils = true;
            }
        } else {
            std::fprintf(stderr, "warning: --validation requested but VK_LAYER_KHRONOS_validation is not installed\n");
        }
    }
    instance = createInstance(exts, layers);

    if (debugUtils) {
        VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mci.pfnUserCallback = debugCallback;
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (fn) VK_CHECK(fn(instance, &mci, nullptr, &messenger_));
    }

    if (opt.createSurface) surface = opt.createSurface(instance);

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("no Vulkan devices found");
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance, &count, devs.data());

    DeviceInfo chosen;
    if (opt.deviceIndex >= 0) {
        if (opt.deviceIndex >= static_cast<int>(count))
            throw std::runtime_error("--device " + std::to_string(opt.deviceIndex) + " is out of range (" +
                                     std::to_string(count) + " devices; see --list-devices)");
        chosen = inspectDevice(devs[opt.deviceIndex], surface);
        if (!chosen.suitable)
            throw std::runtime_error(std::string("device ") + chosen.props.deviceName + " is unsuitable: " + chosen.reason);
    } else {
        int best = -1;
        for (uint32_t i = 0; i < count; ++i) {
            DeviceInfo info = inspectDevice(devs[i], surface);
            if (!info.suitable) continue;
            int score = typeScore(info.props.deviceType);
            if (score > best) {
                best = score;
                chosen = info;
            }
        }
        if (best < 0) throw std::runtime_error("no suitable Vulkan 1.3 device found (see --list-devices)");
    }
    physicalDevice = chosen.device;
    properties = chosen.props;
    queueFamily = static_cast<uint32_t>(chosen.family);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;

    std::vector<const char*> devExts;
    if (surface != VK_NULL_HANDLE) devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();
    VK_CHECK(vkCreateDevice(physicalDevice, &dci, nullptr, &device));
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &commandPool));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence_));
}

VulkanContext::~VulkanContext() {
    if (device) {
        vkDeviceWaitIdle(device);
        vkDestroyFence(device, fence_, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (messenger_) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(instance, messenger_, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                                       VkMemoryPropertyFlags preferred) const {
    for (int pass = 0; pass < 2; ++pass) {
        VkMemoryPropertyFlags want = pass == 0 ? (required | preferred) : required;
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & want) == want) return i;
        }
    }
    throw std::runtime_error("no suitable Vulkan memory type");
}

Buffer VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size > 0 ? size : 4;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &ci, nullptr, &b.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    VkMemoryPropertyFlags required = 0, preferred = 0;
    switch (kind) {
        case MemoryKind::Device: required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; break;
        case MemoryKind::Upload:
            required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case MemoryKind::Readback:
            required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            break;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, required, preferred);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    if (kind != MemoryKind::Device) VK_CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    return b;
}

void VulkanContext::destroy(Buffer& b) {
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);  // implicitly unmaps
    b = Buffer{};
}

Image VulkanContext::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                                 VkSampleCountFlagBits samples) {
    Image img;
    img.format = format;
    img.width = width;
    img.height = height;
    img.samples = samples;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = samples;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &ci, nullptr, &img.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, img.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(device, img.image, img.memory, 0));

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    const bool isDepth = format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM ||
                         format == VK_FORMAT_X8_D24_UNORM_PACK32;
    vci.subresourceRange = {static_cast<VkImageAspectFlags>(isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                            0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &vci, nullptr, &img.view));
    return img;
}

void VulkanContext::destroy(Image& img) {
    if (img.view) vkDestroyImageView(device, img.view, nullptr);
    if (img.image) vkDestroyImage(device, img.image, nullptr);
    if (img.memory) vkFreeMemory(device, img.memory, nullptr);
    img = Image{};
}

VkCommandBuffer VulkanContext::allocateCommandBuffer() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
    return cmd;
}

void VulkanContext::freeCommandBuffer(VkCommandBuffer cmd) { vkFreeCommandBuffers(device, commandPool, 1, &cmd); }

void VulkanContext::submitAndWait(VkCommandBuffer cmd) {
    VK_CHECK(vkResetFences(device, 1, &fence_));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence_));
    VK_CHECK(vkWaitForFences(device, 1, &fence_, VK_TRUE, UINT64_MAX));
}

void VulkanContext::uploadToBuffer(const Buffer& dst, const void* data, VkDeviceSize size, VkDeviceSize dstOffset) {
    if (size == 0) return;
    Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Upload);
    std::memcpy(staging.mapped, data, size);
    VkCommandBuffer cmd = allocateCommandBuffer();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    fullBarrier(cmd);
    VkBufferCopy region{0, dstOffset, size};
    vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
    fullBarrier(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
    submitAndWait(cmd);
    freeCommandBuffer(cmd);
    destroy(staging);
}

VkShaderModule VulkanContext::createShaderModule(const uint32_t* code, size_t sizeBytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sizeBytes;
    ci.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &m));
    return m;
}

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void fullBarrier(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                  VkAccessFlags dstAccess, VkImageAspectFlags aspect) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace fc
