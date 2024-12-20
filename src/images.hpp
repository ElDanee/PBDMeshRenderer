#pragma once

#include <vulkan/vulkan.h>

namespace vkutil {

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout, PFN_vkVoidFunction pVkCmdPipelineBarrier2KHR, bool depth = false);

void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize, PFN_vkVoidFunction pVkCmdBlitImage2KHR, bool depth = false);
}

