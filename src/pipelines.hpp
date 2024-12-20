#pragma once
#include "types.h"

#define VK_INPUT_RATE_NO { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO }


class PipelineBuilder {
public:
    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
   
    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineColorBlendAttachmentState _colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo _multisampling;
    VkPipelineLayout _pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo _depthStencil;
    VkPipelineRenderingCreateInfo _renderInfo;
    VkFormat _colorAttachmentformat;
    VkPipelineTessellationStateCreateInfo _tessellationState;
    
    
    PipelineBuilder(){ clear(); }

    void clear();

    VkPipeline build_pipeline(VkDevice device, VkPipelineVertexInputStateCreateInfo vInputInfo = VK_INPUT_RATE_NO);
    
    void set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader,
                     VkShaderModule texControlShader = nullptr, VkShaderModule texEvalShader = nullptr);
    void set_mesh_shaders(VkShaderModule taskShader, VkShaderModule meshShader, VkShaderModule fragShader);
    void set_input_topology(VkPrimitiveTopology topology);
    void set_polygon_mode(VkPolygonMode mode);
    void set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace);
    void set_multisampling_none();
    void disable_blending();
    void enable_blending_additive();
    void enable_blending_alphablend();
    void enable_tesselation_state(int patchControlPoints = 1);

    void set_color_attachment_format(VkFormat format);
    void set_depth_format(VkFormat format);
    void disable_depthtest();
    void enable_depthtest(bool depthWriteEnable,VkCompareOp op);
};

namespace vkutil {
bool load_shader_module(const char* filePath,
                        VkDevice device,
                        VkShaderModule* outShaderModule);
};

