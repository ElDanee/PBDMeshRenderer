#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include "vk_mem_alloc.h"

#include "fmt/core.h"

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
             fmt::print("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)

struct AllocatedImage {
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct VertPushConstants{
    glm::mat4 pad; //Model Matrix if needed
    VkDeviceAddress vertexBuffer; //Used for vertex pulling
};

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
    glm::vec4 eyePosition;
    float time;
};

struct ComputeSceneUniformData{
    glm::vec3 acceleration;
    float timeStep;
};

struct InstanceData{
    uint32_t v1;
    uint32_t v2;
    uint32_t v3;
    uint32_t pad;
};

// holds the resources needed for a mesh
struct GPUMeshBuffers {
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

struct VertexData{
    glm::vec3 pos;
    float pad; //memory alignment, hold invMass if needed
};

struct VertexDynamics{
    glm::vec3 velocity;
    float invMass;
};

struct Correction{
    glm::vec3 dP;
    float pad;
};

struct EdgeConstraint{
    float length;
    uint32_t v1;
    uint32_t v2;
    float elasticity;
};

struct VolumeConstraint{
    uint32_t v1;
    uint32_t v2;
    uint32_t v3;
    uint32_t v4;
    glm::vec3 areas; //Also, padding for memory alignment
    float volume;
};

struct SurfaceConstraint{
    uint32_t v1;
    uint32_t v2;
    uint32_t v3;
    float area;
};
