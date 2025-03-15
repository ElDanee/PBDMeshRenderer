#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "types.h"
#include "descriptors.hpp"
#include "camera.hpp"
#include "mesh.hpp"


struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function) {
        deletors.push_back(function);
    }

    void flush() {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)(); //call functors
        }

        deletors.clear();
    }
};

struct FrameData {
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;
    VkSemaphore _swapchainSemaphore;
    VkSemaphore _renderSemaphore;
    VkFence _renderFence;
    
    AllocatedBuffer _gpuSceneDataBuffer;
    AllocatedBuffer _computeSceneDataBuffer;
    
    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:
    
    VkInstance _instance;// Vulkan library handle
    VkDebugUtilsMessengerEXT _debug_messenger;// Vulkan debug output handle
    VkPhysicalDevice _chosenGPU;// GPU chosen as the default device
    VkDevice _device; // Vulkan device for commands
    VkSurfaceKHR _surface;// Vulkan window surface
    
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;
    
    FrameData _frames[FRAME_OVERLAP];

    FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    bool _isInitialized{false};
    int _frameNumber {0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{2560, 1600};
    bool resize_requested {false};

    struct GLFWwindow* _window{};

    DeletionQueue _mainDeletionQueue;
    
    VmaAllocator _allocator;
    
    //draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;
    
    VkExtent2D _drawExtent;
    float renderScale = 1.f;
    
    DescriptorAllocatorGrowable globalDescriptorAllocator;

    GPUSceneData sceneData;

    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
    VkDescriptorSet _globalDescriptor;

    VkDescriptorSetLayout _singleImageDescriptorLayout;
    
    VkPipelineLayout planePLayout;
    VkPipeline planePipeline;
    GPUMeshBuffers planeMesh;
    
    AllocatedBuffer instanceBuffer;
    VkDescriptorSetLayout _softBodyDSLayout;
    VkDescriptorSet _softBodyDSet;
    VkPipelineLayout _softBodyPLayout;
    VkPipeline _softBodyPipeline;
    VkPipeline _softBodyEdgePipeline;
    VkPipeline _softBodyPointPipeline;
    bool useEdgePipeline = false;
    
    GPUMeshBuffers loadedMesh;
    AllocatedBuffer vertexDataBuffer;
    AllocatedBuffer vertexIntermediateDataBuffer;
    AllocatedBuffer correctionBuffer;
    AllocatedBuffer vertexDynamicsBuffer;
    AllocatedBuffer edgeConstraintsBuffer;
    uint32_t numIndices;
    uint32_t numVertices;
    
    std::vector<EdgeConstraint> constraints;
    std::vector<std::vector<EdgeConstraint>> coloring_constraints;
    
    VkDescriptorSetLayout _uniformComputeDSLayout;
    VkDescriptorSet _uniformComputeDSet;
    ComputeSceneUniformData uniformComputeData;
    
    VkPipelineLayout computePLayout;
    VkPipeline computeConstraintsPipeline;
    VkPipeline computePreSolvePipeline;
    
    VkPipelineLayout computePostSolvePLayout;
    VkPipeline computePostSolvePipeline;
    
    VkDescriptorSetLayout computeDSLayout;
    std::vector<VkDescriptorSet> computeConstraintsDSet;
    VkDescriptorSet computePreSolveDSet;
    
    VkDescriptorSetLayout computePostSolveDSLayout;
    VkDescriptorSet computePostSolveDSet;
    
    PBDMesh* meshPBD;
    int solverIterations = 20;
    float timeStepScale = 1.f;
    glm::vec3 acceleration;
    bool solveGPU = true;
    bool snapshotSection = false;
    bool showEdges = true;
    bool freeCamera = false;
    bool recordFPS = false;
    bool recordStretch = false;
    bool recordVolume = false;
    bool squash = false;
    
    bool use3D;
    
    int numSubdivisions = 10;
    
    std::ofstream outputCSV;
    
    Camera mainCamera;
    
    static VulkanEngine& Get();

    //initializes everything in the engine
    void init(int subdivisions, int startingIterations, bool use3D);
    void init();

    //shuts down the engine
    void cleanup();

    //draw loop
    void draw_background(VkCommandBuffer cmd);
    void draw();
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
    
    void update_scene();
    void solve_constraints(VkCommandBuffer cmd);
    
    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
    
    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<VertexData> vertices);
    GPUMeshBuffers finalizeMesh(std::span<uint32_t> indices, std::span<VertexData> vertices, int detailLevel);
    GPUMeshBuffers tessellate_square_mesh(float sideSize, int tessellationLevel);
    GPUMeshBuffers tessellate_cube_mesh(float sideSize, int tessellationLevel);
    GPUMeshBuffers tessellate_cube_mesh_surface(float sideSize, int tessellationLevel);
    
    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    void copy_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, void *dataSrc, AllocatedBuffer dstBuffer);
    void copy_buffer_from_device(size_t dataSize, void *dataDst, AllocatedBuffer srcBuffer);
    AllocatedBuffer create_copy_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, void *dataSrc);
    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage createTextureImage(char const *path);
    void destroy_buffer(const AllocatedBuffer& buffer);
    void destroy_image(const AllocatedImage& img);
    //run main loop
    void run();

private:

    void init_vulkan();
    void init_imgui();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void init_descriptors();
    void init_pipelines();
    void init_default_data();
    void resize_swapchain();
    void init_data_buffers();
    
    void build_compute_pipeline();
    void build_soft_body_pipeline();
    void build_plane_pipeline();
    void prepare_graphics_rendering(VkCommandBuffer cmd);
};

