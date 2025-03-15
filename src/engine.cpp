#include "engine.hpp"
#include "pipelines.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "initializers.hpp"
#include "types.h"
#include "images.hpp"

//bootstrap library
#include "VkBootstrap.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <chrono>
#include <thread>

using namespace vkinit;

constexpr bool bUseValidationLayers = true;

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

//Use this callback to manage any window change (X-Ray, colors, etc)
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_X && action == GLFW_PRESS){
        VulkanEngine::Get().useEdgePipeline = !VulkanEngine::Get().useEdgePipeline;
    }
}

void VulkanEngine::init(int subdivisions, int startingIterations, bool use3D){
    numSubdivisions = subdivisions;
    solverIterations = startingIterations;
    this->use3D = use3D;
    init();
}

void VulkanEngine::init(){
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;
    
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    _window = glfwCreateWindow(_windowExtent.width,
                                          _windowExtent.height,
                                          "Mesh Renderer",
                                          nullptr,
                                          nullptr);
    
    glfwSetKeyCallback(_window, key_callback);
    
    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();
    init_data_buffers();
    init_pipelines();
    init_imgui();
    init_default_data();
    
    outputCSV.open("../sampleFPSIter.csv");
    if(!outputCSV.is_open()){
        fmt::print("Failed to open file!");
        exit(1);
    } else outputCSV << "Timestamp;Iterations;FPS\n";
    
    //everything went fine
    _isInitialized = true;
    fmt::print("\nEngine init: {}\n", _isInitialized);
}

void VulkanEngine::init_vulkan()
{
    
    vkb::InstanceBuilder builder;
    //make the vulkan instance
    auto inst_ret = builder.set_app_name("PDB Mesh App")
        .request_validation_layers(bUseValidationLayers)
        .enable_extension("VK_KHR_surface")
        .enable_extension("VK_EXT_swapchain_colorspace")
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();
    
    vkb::Instance vkb_inst = inst_ret.value();

    //grab the instance
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    glfwCreateWindowSurface(_instance, _window, NULL, &_surface);
    
    //vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features.dynamicRendering = true;
    features.synchronization2 = true;

    //vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    {
        features12.bufferDeviceAddress = true;
        features12.descriptorIndexing = true;
        features12.scalarBlockLayout = true;
        features12.descriptorIndexing = true;
        features12.shaderStorageBufferArrayNonUniformIndexing = true;
        features12.shaderStorageImageArrayNonUniformIndexing = true;
        features12.shaderUniformBufferArrayNonUniformIndexing = true;
        features12.shaderSampledImageArrayNonUniformIndexing = true;
    }
    
    VkPhysicalDeviceFeatures required_features{};
    {
        required_features.tessellationShader = VK_TRUE;
        required_features.fillModeNonSolid = VK_TRUE;
        required_features.samplerAnisotropy = VK_TRUE;
    }
    
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shader_features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    {
        shader_features.shaderBufferFloat32AtomicAdd = true;
    }

    //use vkbootstrap to select a gpu.
    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 2)
        .add_required_extension("VK_KHR_synchronization2")
        .add_required_extension("VK_KHR_copy_commands2")
        .add_required_extension("VK_KHR_dynamic_rendering")
        .add_required_extension("VK_KHR_format_feature_flags2")
        .add_required_extension("VK_KHR_spirv_1_4")
        .add_required_extension("VK_EXT_shader_atomic_float")
        .add_required_extension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME)
        .add_required_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
        .set_required_features(required_features)
        .add_required_extension_features(shader_features)
        .set_required_features_12(features12)
        .set_required_features_13(features)
        .set_surface(_surface)
        .select()
        .value();
    
    fmt::print("{}\n", physicalDevice.properties.deviceName);

    
    //create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{ physicalDevice };

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;
    
    // use vkbootstrap to get a Graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
    
    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
    });
}


void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU,_device,_surface };

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        //.use_default_format_selection()
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }) //Normal
        //use vsync present mode
        .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;
    //store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    //draw image size will match the window
    VkExtent3D drawImageExtent = {
        _windowExtent.width,//2560
        _windowExtent.height, //1600
        1
    };
    VkFormat drawFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    _drawImage = create_image(drawImageExtent, drawFormat, drawImageUsages);
    
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    
    _depthImage = create_image(drawImageExtent, depthFormat, depthImageUsages);
    
    //add to deletion queues
    _mainDeletionQueue.push_function([=, this]() {
        destroy_image(_drawImage);
        destroy_image(_depthImage);
    });
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    // destroy swapchain resources
    for (int i = 0; i < _swapchainImageViews.size(); i++) {
        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
}


void VulkanEngine::init_commands()
{
    //create a command pool for commands submitted to the graphics queue.
    //we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));
        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }
    
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));
    _mainDeletionQueue.push_function([=, this]() {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });
}

void VulkanEngine::init_sync_structures()
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
    }
    //Immediate submit fence
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=, this]() { vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::init_descriptors()
{
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
    };

    globalDescriptorAllocator.init(_device, 10, sizes);
    
    //descriptor set layout for global uniform
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_ALL);
    }
    //Allocate global descriptor set;
    _globalDescriptor = globalDescriptorAllocator.allocate(_device, _gpuSceneDataDescriptorLayout);
    
    //uniform descriptor set for constraints solver
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _uniformComputeDSLayout = builder.build(_device, VK_SHADER_STAGE_ALL);
    }
    _uniformComputeDSet = globalDescriptorAllocator.allocate(_device, _uniformComputeDSLayout);
    
    //descriptor set layout for vertex data
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        _softBodyDSLayout = builder.build(_device, VK_SHADER_STAGE_ALL);
    }
    _softBodyDSet = globalDescriptorAllocator.allocate(_device, _softBodyDSLayout);
    
    
    //both the descriptor allocator and the new layout get cleaned up properly
    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pools(_device);
        vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _uniformComputeDSLayout, nullptr); 
        vkDestroyDescriptorSetLayout(_device, _softBodyDSLayout, nullptr);
        
    });
    
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);
    
        _mainDeletionQueue.push_function([&, i]() {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }
}

void VulkanEngine::init_data_buffers(){
    for(int i = 0; i<FRAME_OVERLAP; i++){
        //allocate a new uniform buffer for the scene data
        _frames[i]._gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        _frames[i]._computeSceneDataBuffer = create_buffer(sizeof(ComputeSceneUniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        
        _mainDeletionQueue.push_function([=, this]() {
            destroy_buffer(_frames[i]._gpuSceneDataBuffer);
            destroy_buffer(_frames[i]._computeSceneDataBuffer);
        });
    }
};

void VulkanEngine::init_pipelines()
{
    build_soft_body_pipeline();
    build_plane_pipeline();
}

void VulkanEngine::build_plane_pipeline(){
    VkShaderModule vertShader;
    VkShaderModule fragShader;
    
    { //Load shader modules
        if (!vkutil::load_shader_module("../shaders/plane.vert.spv", _device, &vertShader)) {
            fmt::println("Error when building the triangle mesh shader module");
        }
        if (!vkutil::load_shader_module("../shaders/base.frag.spv", _device, &fragShader)) {
            fmt::println("Error when building the triangle fragment shader module");
        }
    }
    
    {//PipelineLayout creation
        VkDescriptorSetLayout layouts[] = {_gpuSceneDataDescriptorLayout};
        //Push Constants
        VkPushConstantRange matrixRange{};
        matrixRange.offset = 0;
        matrixRange.size = sizeof(VertPushConstants);
        matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        
        VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
        mesh_layout_info.setLayoutCount = 1;
        mesh_layout_info.pSetLayouts = layouts;
        mesh_layout_info.pPushConstantRanges = &matrixRange;
        mesh_layout_info.pushConstantRangeCount = 1;
        
        VK_CHECK(vkCreatePipelineLayout(_device, &mesh_layout_info, nullptr, &planePLayout));
        _mainDeletionQueue.push_function([=, this]() {
            vkDestroyPipelineLayout(_device, planePLayout, nullptr);
        });
    }
    
    {//Pipeline creation
        PipelineBuilder pipelineBuilder;
        pipelineBuilder.set_shaders(vertShader, fragShader);
        
        pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
        pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pipelineBuilder.set_multisampling_none();
        pipelineBuilder.disable_blending();
        pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
        
        //render format
        pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
        pipelineBuilder.set_depth_format(_depthImage.imageFormat);
        
        pipelineBuilder._pipelineLayout = planePLayout;
        
        planePipeline = pipelineBuilder.build_pipeline(_device);
        
        _mainDeletionQueue.push_function([=, this]() {
            vkDestroyPipeline(_device, planePipeline, nullptr);
        });
    }
    
    vkDestroyShaderModule(_device, vertShader, nullptr);
    vkDestroyShaderModule(_device, fragShader, nullptr);
    
    std::vector<uint32_t> indices = {0,1,2,0,2,3};
    
    std::vector<VertexData> vertices;
    VertexData newVtx;
    vertices.push_back(newVtx);
    planeMesh = uploadMesh(indices, vertices);
    
}

void VulkanEngine::build_soft_body_pipeline(){
    VkShaderModule vertShader;
    VkShaderModule vertWireShader;
    VkShaderModule fragShader;
    { //Load shader modules
        if (!vkutil::load_shader_module("../shaders/base.vert.spv", _device, &vertShader)) {
            fmt::println("Error when building the triangle mesh vertex shader module");
        }
        if (!vkutil::load_shader_module("../shaders/baseWire.vert.spv", _device, &vertWireShader)) {
            fmt::println("Error when building the triangle wire vertex shader module");
        }
        if (!vkutil::load_shader_module("../shaders/base.frag.spv", _device, &fragShader)) {
            fmt::println("Error when building the triangle mesh fragment shader module");
        }
    }
    
    {//PipelineLayout creation
        VkDescriptorSetLayout layouts[] = {_gpuSceneDataDescriptorLayout, _softBodyDSLayout};
        //Push Constants
        VkPushConstantRange matrixRange{};
        matrixRange.offset = 0;
        matrixRange.size = sizeof(VertPushConstants);
        matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        
        VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
        mesh_layout_info.setLayoutCount = 2;
        mesh_layout_info.pSetLayouts = layouts;
        mesh_layout_info.pPushConstantRanges = &matrixRange;
        mesh_layout_info.pushConstantRangeCount = 1;
        
        VK_CHECK(vkCreatePipelineLayout(_device, &mesh_layout_info, nullptr, &_softBodyPLayout));
        _mainDeletionQueue.push_function([=, this]() {
            vkDestroyPipelineLayout(_device, _softBodyPLayout, nullptr);
        });
    }
    
    {//Pipeline creation
        PipelineBuilder pipelineBuilder;
        pipelineBuilder.set_shaders(vertShader, fragShader);
        
        pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
        pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pipelineBuilder.set_multisampling_none();
        pipelineBuilder.disable_blending();
        pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
        
        //render format
        pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
        pipelineBuilder.set_depth_format(_depthImage.imageFormat);
        
        pipelineBuilder._pipelineLayout = _softBodyPLayout;
        
        _softBodyPipeline = pipelineBuilder.build_pipeline(_device);
        
        _mainDeletionQueue.push_function([=, this]() {
            vkDestroyPipeline(_device, _softBodyPipeline, nullptr);
        });
    }
    {//Edge Pipeline creation
        PipelineBuilder pipelineBuilder;
        pipelineBuilder.set_shaders(vertShader, fragShader);
        
        pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_LINE);
        pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pipelineBuilder.set_multisampling_none();
        pipelineBuilder.disable_blending();
        pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
        
        //render format
        pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
        pipelineBuilder.set_depth_format(_depthImage.imageFormat);
        
        pipelineBuilder._pipelineLayout = _softBodyPLayout;
        
        _softBodyEdgePipeline = pipelineBuilder.build_pipeline(_device);
        
        _mainDeletionQueue.push_function([=, this]() {
            vkDestroyPipeline(_device, _softBodyEdgePipeline, nullptr);
        });
    }
    {//Point Pipeline creation
        PipelineBuilder pipelineBuilder;
        pipelineBuilder.set_shaders(vertWireShader, fragShader);
        
        pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
        pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_POINT);
        pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pipelineBuilder.set_multisampling_none();
        pipelineBuilder.disable_blending();
        pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
        
        //render format
        pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
        pipelineBuilder.set_depth_format(_depthImage.imageFormat);
        
        pipelineBuilder._pipelineLayout = _softBodyPLayout;
        
        _softBodyPointPipeline = pipelineBuilder.build_pipeline(_device);
        
        _mainDeletionQueue.push_function([=, this]() {
            vkDestroyPipeline(_device, _softBodyPointPipeline, nullptr);
        });
    }
    vkDestroyShaderModule(_device, vertShader, nullptr);
    vkDestroyShaderModule(_device, vertWireShader, nullptr);
    vkDestroyShaderModule(_device, fragShader, nullptr);
}

void VulkanEngine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, as per ImGui demo
    VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    
    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // 2: initialize imgui library
    // this initializes the core structures of imgui
    ImGui::CreateContext();
    // this initializes imgui for SDL
    ImGui_ImplGlfw_InitForVulkan(_window, true);
    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    //dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    // add the destroy the imgui created structures
    _mainDeletionQueue.push_function([=, this]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
    });
}

void VulkanEngine::init_default_data(){
    if(use3D)
        meshPBD = new PBDMesh3D(this, 8.f, numSubdivisions);
    else
        meshPBD = new PBDMesh2D(this, 8.f, numSubdivisions);
    _mainDeletionQueue.push_function([=, this](){
        meshPBD->clear_resources();
    });
    
    { //Init mesh vertex and instance buffers descriptor set
        DescriptorWriter writer;
        writer.write_buffer(0, meshPBD->get_loaded_mesh().vertexBuffer.buffer, sizeof(VertexData) * meshPBD->get_vertices(), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, meshPBD->get_instance_buffer().buffer, sizeof(InstanceData) * meshPBD->get_indices()/3, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(_device, _softBodyDSet);
    }
    
    acceleration = glm::vec3(0, 0, 0);
    
    
    mainCamera.position = glm::vec3(-19,-10,-14);
    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.window = _window;
}

void VulkanEngine::prepare_graphics_rendering(VkCommandBuffer cmd){
    //set dynamic viewport and scissor
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = _drawExtent.width;
    viewport.height = _drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _drawExtent.width;
    scissor.extent.height = _drawExtent.height;

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    //write the buffer on current frame allocated memory
    GPUSceneData* sceneUniformData = (GPUSceneData*)get_current_frame()._gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = sceneData;

    //update global descriptor set
    DescriptorWriter writer;
    writer.write_buffer(0, get_current_frame()._gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, _globalDescriptor);
}

void VulkanEngine::update_scene(){
    static auto startTime = std::chrono::high_resolution_clock::now();
    static float lastTime = 0.0f;
    static float timeElapsed = 0.0f;
    
    
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::milliseconds::period>
                (currentTime - startTime).count();
    float deltaT = time - lastTime;
    timeElapsed+= deltaT;
    lastTime = time;
    
    glm::mat4 view;
    
    mainCamera.update();
    if(freeCamera)
        view = mainCamera.getViewMatrix();
    else view = mainCamera.getLookAtMatrix(meshPBD->get_vertex_zero(), glm::vec3(0,-1,0));
    sceneData.eyePosition = glm::vec4(mainCamera.position, 1.0f);
    
    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.f, 0.1f);

    // invert the Y direction on projection matrix, more similar to opengl and gltf axis
    projection[1][1] *= -1;

    sceneData.view = view;
    sceneData.proj = projection;
    sceneData.viewproj = projection * view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(0.3f, 0.6f, 0.3f, 1.0f);
    glm::vec3 dir = glm::vec3(1, 1, 0);
    sceneData.sunlightDirection = glm::vec4(dir, 1.0f);
    sceneData.time = timeElapsed;
    
    uniformComputeData.acceleration = acceleration;
    uniformComputeData.timeStep = 1.f/60;
}

void VulkanEngine::solve_constraints(VkCommandBuffer cmd){
    const int nSteps = solverIterations;
    static bool GPU = true;
    uniformComputeData.timeStep /= nSteps;
    
    ComputeSceneUniformData* sceneUniformData = (ComputeSceneUniformData*)get_current_frame()._computeSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = uniformComputeData;

    //update global descriptor set
    DescriptorWriter writer;
    writer.write_buffer(0, get_current_frame()._computeSceneDataBuffer.buffer, sizeof(ComputeSceneUniformData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, _uniformComputeDSet);
    
    if(solveGPU || !(use3D)) //Not implemented yet for 2D
        meshPBD->solve_constraints(cmd, _uniformComputeDSet, nSteps);
    else
        meshPBD->solve_constraints_sequential(nSteps, uniformComputeData.timeStep, uniformComputeData.acceleration);
    if(GPU != solveGPU){
        GPU = solveGPU;
        meshPBD->log_data("0;0;0;0;0;0;0");
    }
}

void VulkanEngine::draw()
{
    update_scene();
    
    PFN_vkVoidFunction pVkCmdPipelineBarrier2KHR = vkGetDeviceProcAddr(_device, "vkCmdPipelineBarrier2KHR");
    PFN_vkVoidFunction pVkCmdBlitImage2KHR = vkGetDeviceProcAddr(_device, "vkCmdBlitImage2KHR");
    
    // wait until the gpu has finished rendering the last frame. Timeout of 1 second
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000000));

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);
    
    //increase the number of frames drawn
    _frameNumber++;
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));
    
    //request image from the swapchain
    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return ;
    }
    
    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    _drawExtent.width= std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;
        
    //naming it cmd for shorter writing
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    
    _drawExtent.width = _drawImage.imageExtent.width;
    _drawExtent.height = _drawImage.imageExtent.height;
    
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    
    //Solve constraints XPBD
    solve_constraints(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, pVkCmdPipelineBarrier2KHR);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,  pVkCmdPipelineBarrier2KHR);

    // GEOMETRY DRAWS
    //begin a render pass  connected to our draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_GENERAL);
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    
    static PFN_vkVoidFunction pVkCmdBeginRenderingKHR = vkGetDeviceProcAddr(_device, "vkCmdBeginRenderingKHR");
    ((PFN_vkCmdBeginRenderingKHR)(pVkCmdBeginRenderingKHR))(cmd, &renderInfo);
                     
    prepare_graphics_rendering(cmd);
    
    //Rendering
    VertPushConstants pushC;
    pushC.vertexBuffer = meshPBD->get_loaded_mesh().vertexBufferAddress;
    vkCmdPushConstants(cmd, _softBodyPLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VertPushConstants), &pushC);
    VkDescriptorSet descriptors[] = {_globalDescriptor, _softBodyDSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _softBodyPLayout, 0, 2, descriptors, 0, nullptr);
    
    if(useEdgePipeline){
        if(showEdges){
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _softBodyEdgePipeline);
            vkCmdBindIndexBuffer(cmd, meshPBD->get_loaded_mesh().indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, 3, meshPBD->get_indices()/3, 0, 0, 0);
        }
        
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _softBodyPointPipeline);
        vkCmdDraw(cmd, meshPBD->get_vertices(), 1, 0, 0);
        
    }
    else{
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _softBodyPipeline);
        vkCmdBindIndexBuffer(cmd, meshPBD->get_loaded_mesh().indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, 3, meshPBD->get_indices()/3, 0, 0, 0);
    }
    
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, planePLayout, 0, 1, &_globalDescriptor, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, planePipeline);
    vkCmdBindIndexBuffer(cmd, planeMesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    
    static PFN_vkVoidFunction pVkCmdEndRenderingKHR = vkGetDeviceProcAddr(_device, "vkCmdEndRenderingKHR");
    
    ((PFN_vkCmdEndRenderingKHR)(pVkCmdEndRenderingKHR))(cmd);
    
    //transtion the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pVkCmdPipelineBarrier2KHR);
    
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, pVkCmdPipelineBarrier2KHR);

    // execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent, pVkCmdBlitImage2KHR);
    
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, pVkCmdPipelineBarrier2KHR);

    //draw imgui into the swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);
    
    // set swapchain image layout to Present so we can show it on the screen
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, pVkCmdPipelineBarrier2KHR);

    //finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));
    
    //prepare the submission to the queue.
    //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //we will signal the _renderSemaphore, to signal that rendering has finished

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);
    
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo,&signalInfo,&waitInfo);
    
    
    //submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    PFN_vkVoidFunction pVkQueueSubmit2KHR = vkGetDeviceProcAddr(_device, "vkQueueSubmit2KHR");
    VK_CHECK(((PFN_vkQueueSubmit2KHR)(pVkQueueSubmit2KHR))(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));
    //VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that,
    // as its necessary that drawing commands have finished before the image is displayed to the user
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);
    
    static PFN_vkVoidFunction pVkCmdBeginRenderingKHR = vkGetDeviceProcAddr(_device, "vkCmdBeginRenderingKHR");
    ((PFN_vkCmdBeginRenderingKHR)(pVkCmdBeginRenderingKHR))(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    static PFN_vkVoidFunction pVkCmdEndRenderingKHR = vkGetDeviceProcAddr(_device, "vkCmdEndRenderingKHR");
    ((PFN_vkCmdEndRenderingKHR)(pVkCmdEndRenderingKHR))(cmd);
}

void VulkanEngine::run(){
    bool printStats = false;
    
    while (!glfwWindowShouldClose(_window)){
        glfwPollEvents();
        mainCamera.processEvent();
        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        if (ImGui::Begin("Settings")) {
            ImGui::Text("FPS : %f", ImGui::GetIO().Framerate);
            ImGui::Checkbox("Show Wireframe", &useEdgePipeline);
            ImGui::Checkbox("Show Edges", &showEdges);
            ImGui::SliderInt("Solver Iterations:", &solverIterations, 1, 50);
            ImGui::SliderFloat("TimeStep Scaler (Higher = shorter time step):", &timeStepScale, 0.1, 100);
            ImGui::Checkbox("Solve with GPU", &solveGPU);
            ImGui::InputFloat3("Acceleration",(float*)& acceleration);
            ImGui::Checkbox("Snapshot section", &snapshotSection);
            ImGui::Checkbox("Record stretch", &recordStretch);
            ImGui::Checkbox("Record volume", &recordVolume);
            ImGui::Checkbox("Squash Mesh", &squash);
            ImGui::Checkbox("Record FPS", &recordFPS);
            ImGui::Checkbox("Free camera", &freeCamera);
            ImGui::Text("Lock camera: C");
            ImGui::Text("Toggle wireframe: X");
            ImGui::Text("Move camera: WASD");

        }
        ImGui::End();
        
        static bool squashed = false;
        if(use3D){ //Not implemented yet for 2D
            if(squash){
                acceleration = glm::vec3(0,100000,0);
                squashed = true;
            }
        }
        //Main draw loop
        ImGui::Render();
        //draw function
        draw();
        
        //Logging section
        if(use3D){ //Not implemented yet for 2D
            if(!squash && squashed){
                acceleration = glm::vec3(0);
                squashed = false;
            }
            static float FPS = 0;
            static int frameToQuit = 1000;
            if (recordFPS) {
                FPS += ImGui::GetIO().Framerate;
                
                outputCSV << 1000 - frameToQuit << ";" << solverIterations << ";" << ImGui::GetIO().Framerate <<"\n";
                frameToQuit--;
            }
            
            if(snapshotSection){
                
                meshPBD->copy_from_device();
                //if(frame % 50 == 0)
                    meshPBD -> log_section();
                //frame++;
                snapshotSection = false;
            } //else frame = 0;*/
            
            if(recordStretch){
                meshPBD->copy_from_device();
                meshPBD->log_stretch();
                frameToQuit--;
                if(frameToQuit%150 == 0 && frameToQuit>250)
                    acceleration.x -=200;
            }
            
            if(recordVolume){
                meshPBD->copy_from_device();
                meshPBD->log_volume();
                frameToQuit--;
                if(frameToQuit == 500){
                    acceleration = glm::vec3(0);
                }
            }
            
            if(frameToQuit == 0){
                if(recordFPS) {
                    fmt::print("solver iteration : {}, mesh resolution : {}, Average FPS on 1000 frames: {}", solverIterations, numSubdivisions, FPS/1000);
                }
                exit(1);
            }
            
            //Time first N frames
            static auto startTime = std::chrono::high_resolution_clock::now();
            static float lastTime = 0.0f;
            static int frameCapture = 10;
            if(frameCapture>=0){
                auto currentTime = std::chrono::high_resolution_clock::now();
                float time = std::chrono::duration<float, std::chrono::milliseconds::period>
                (currentTime - startTime).count();
                fmt::print("Time elapsed: {} milliseconds\n", time-lastTime);
                lastTime =+ time;
            }
            if(frameCapture == 0)
            {
                auto currentTime = std::chrono::high_resolution_clock::now();
                float time = std::chrono::duration<float, std::chrono::milliseconds::period>
                (currentTime - startTime).count();
                fmt::print("Total time elapsed: {} milliseconds\n", time);
            }
            
            frameCapture--;
        }//end logging section
    }
}

void VulkanEngine::cleanup(){
    if (_isInitialized) {
        // make sure the gpu has stopped doing its things
        vkDeviceWaitIdle(_device);
        
        //free per-frame structures and deletion queue
        for (int i = 0; i < FRAME_OVERLAP; i++) {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
            //destroy sync objects
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

            _frames[i]._deletionQueue.flush();
        }
        _mainDeletionQueue.flush();
        
        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);
        
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        glfwDestroyWindow(_window);
        outputCSV.close();
    }
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    
    PFN_vkVoidFunction pVkQueueSubmit2KHR = vkGetDeviceProcAddr(_device, "vkQueueSubmit2KHR");
    
    VK_CHECK(((PFN_vkQueueSubmit2KHR)(pVkQueueSubmit2KHR))(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

AllocatedBuffer VulkanEngine::create_copy_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, void* dataSrc){
    AllocatedBuffer staging = create_buffer(allocSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging.allocation->GetMappedData();
    
    memcpy(data, dataSrc, allocSize);
    
    AllocatedBuffer result = create_buffer(allocSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                   memoryUsage);
    
    immediate_submit([&](VkCommandBuffer cmd){
        VkBufferCopy dataCopy{ 0 };
        dataCopy.dstOffset = 0;
        dataCopy.srcOffset = 0;
        dataCopy.size = allocSize;
        
        vkCmdCopyBuffer(cmd, staging.buffer, result.buffer, 1, &dataCopy);
    });
    
    destroy_buffer(staging);
    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(result);
    });
    
    return result;
}

void VulkanEngine::copy_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, void* dataSrc, AllocatedBuffer dstBuffer){
    AllocatedBuffer staging = create_buffer(allocSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging.allocation->GetMappedData();
    
    memcpy(data, dataSrc, allocSize);
    
    immediate_submit([&](VkCommandBuffer cmd){
        VkBufferCopy dataCopy{ 0 };
        dataCopy.dstOffset = 0;
        dataCopy.srcOffset = 0;
        dataCopy.size = allocSize;
        
        vkCmdCopyBuffer(cmd, staging.buffer, dstBuffer.buffer, 1, &dataCopy);
    });
    
    destroy_buffer(staging);
}

void VulkanEngine::copy_buffer_from_device(size_t dataSize, void *dataDst, AllocatedBuffer srcBuffer){
    AllocatedBuffer staging = create_buffer(dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    
    immediate_submit([&](VkCommandBuffer cmd){
        VkBufferCopy dataCopy{ 0 };
        dataCopy.dstOffset = 0;
        dataCopy.srcOffset = 0;
        dataCopy.size = dataSize;
        
        vkCmdCopyBuffer(cmd, srcBuffer.buffer, staging.buffer, 1, &dataCopy);
    });
    
    void* data = staging.allocation->GetMappedData();
    memcpy(dataDst, data, dataSize);
    
    destroy_buffer(staging);
    return;
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // allocate buffer
    VkBufferCreateInfo bufferInfo = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation,
        &newBuffer.info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    // if the format is a depth format, we will need to have it use the correct
    // aspect flag
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // build a image-view for the image
    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));
    return newImage;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    PFN_vkVoidFunction pVkCmdPipelineBarrier2KHR = vkGetDeviceProcAddr(_device, "vkCmdPipelineBarrier2KHR");
    
    size_t data_size = size.depth * size.width * size.height * 4;
    
    AllocatedBuffer uploadbuffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, pVkCmdPipelineBarrier2KHR);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            &copyRegion);

        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, pVkCmdPipelineBarrier2KHR);
        });

    destroy_buffer(uploadbuffer);

    return new_image;
}

void VulkanEngine::destroy_image(const AllocatedImage& img)
{
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<VertexData> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(VertexData);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    //create vertex buffer
    newSurface.vertexBuffer = create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VMA_MEMORY_USAGE_GPU_ONLY);

    //find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer = newSurface.vertexBuffer.buffer };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    //create index buffer
    newSurface.indexBuffer = create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    ///TODO: divide staging vertex and staging indices
    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);
    
    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(newSurface.indexBuffer);
        destroy_buffer(newSurface.vertexBuffer);
    });

    return newSurface;
}
