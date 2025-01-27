#include "mesh.hpp"
#include "engine.hpp"
#include "pipelines.hpp"

void PBDMesh::init_solver_pipelines(){
    VkShaderModule compPreSolveShader;
    if (!vkutil::load_shader_module("../shaders/preSolve.comp.spv",engine-> _device, &compPreSolveShader)) {
        fmt::print("Error when building the compute shader \n");
    }
    VkShaderModule compPostSolveShader;
    if (!vkutil::load_shader_module("../shaders/postSolve.comp.spv", engine->_device, &compPostSolveShader)) {
        fmt::print("Error when building the compute shader \n");
    }
    {
        VkDescriptorSetLayout layouts[] = {uniformComputeDSLayout, computePSolveDSLayout};
        VkPipelineLayoutCreateInfo computeLayout{};
        computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        computeLayout.pNext = nullptr;
        computeLayout.pSetLayouts = layouts;
        computeLayout.setLayoutCount = 2;

        VK_CHECK(vkCreatePipelineLayout(engine->_device, &computeLayout, nullptr, &computeSolvePLayout));
        
        VkPipelineShaderStageCreateInfo stageinfo{};
        stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageinfo.pNext = nullptr;
        stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageinfo.module = compPostSolveShader;
        stageinfo.pName = "main";

        VkComputePipelineCreateInfo computePipelineCreateInfo{};
        computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        computePipelineCreateInfo.pNext = nullptr;
        computePipelineCreateInfo.layout = computeSolvePLayout;
        computePipelineCreateInfo.stage = stageinfo;

        VK_CHECK(vkCreateComputePipelines(engine->_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &computePostSolvePipeline));
        
        computePipelineCreateInfo.stage.module = compPreSolveShader;
        VK_CHECK(vkCreateComputePipelines(engine->_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &computePreSolvePipeline));
        
        vkDestroyShaderModule(engine->_device, compPreSolveShader, nullptr);
        vkDestroyShaderModule(engine->_device, compPostSolveShader, nullptr);
    }
}

void PBDMesh::init_solver_descriptor_sets(){
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        computePSolveDSLayout = builder.build(engine->_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    
    computePreSolveDSet = engine->globalDescriptorAllocator.allocate(engine->_device, computePSolveDSLayout);
    computePostSolveDSet = engine->globalDescriptorAllocator.allocate(engine->_device, computePSolveDSLayout);
    
    //uniform descriptor set for constraints solver
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        uniformComputeDSLayout = builder.build(engine->_device, VK_SHADER_STAGE_ALL);
    }
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        computeSolveDSLayout = builder.build(engine->_device, VK_SHADER_STAGE_ALL);
    }
    
    computeSolveDSet = engine->globalDescriptorAllocator.allocate(engine->_device, computeSolveDSLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, vertexIntermediateBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, computeSolveDSet);
    }
    
}

void IConstraint2D::create_edge_descriptor_sets(VulkanEngine* engine){
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        compute2DDSLayout = builder.build(engine->_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    
    compute2DConstraintsDSet.resize(coloringConstraints.size());
    for(int i = 0; i<compute2DConstraintsDSet.size(); i++)
        compute2DConstraintsDSet[i] = engine->globalDescriptorAllocator.allocate(engine->_device, compute2DDSLayout);
}

void IConstraint2D::create_edge_pipeline(VulkanEngine* engine){
    VkShaderModule compConstraintsShader;
    if (!vkutil::load_shader_module("../shaders/constraints.comp.spv", engine->_device, &compConstraintsShader)) {
        fmt::print("Error when building the compute shader \n");
    }
    
    VkDescriptorSetLayout layouts[] = {compute2DDSLayout};
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = layouts;
    computeLayout.setLayoutCount = 1;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &computeLayout, nullptr, &compute2DPLayout));
    
    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = compConstraintsShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = compute2DPLayout;
    computePipelineCreateInfo.stage = stageinfo;

    VK_CHECK(vkCreateComputePipelines(engine->_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &compute2DPipeline));
    
    vkDestroyShaderModule(engine->_device, compConstraintsShader, nullptr);
}

void IConstraint2D::solve_edge_constraints(VkCommandBuffer cmd){
    //Solve
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute2DPipeline);
    for(int c = 0; c<coloringConstraints.size(); c++){
        VkDescriptorSet dSet = compute2DConstraintsDSet[c];
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute2DPLayout, 0, 1, &dSet, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(coloringConstraints[c].size() / 32.0), 1, 1);
    }
}

PBDMesh2D::PBDMesh2D(VulkanEngine *engine, float sideSize, int subdivisions){
    this->engine = engine;
    init_mesh(sideSize, subdivisions);
    init_descriptors();
    init_pipelines();
}

void PBDMesh2D::clear_resources(){
    vkDestroyDescriptorSetLayout(engine->_device, compute2DDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, computePSolveDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, computeSolveDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, uniformComputeDSLayout, nullptr);
    vkDestroyPipelineLayout(engine->_device, compute2DPLayout, nullptr);
    vkDestroyPipeline(engine->_device, compute2DPipeline, nullptr);
    vkDestroyPipelineLayout(engine->_device, computeSolvePLayout, nullptr);
    vkDestroyPipeline(engine->_device, computePreSolvePipeline, nullptr);
    vkDestroyPipeline(engine->_device, computePostSolvePipeline, nullptr);
}

void PBDMesh2D::init_pipelines(){
    init_solver_pipelines();
    create_edge_pipeline(engine);
}

void PBDMesh2D::init_descriptors()
{
    create_edge_descriptor_sets(engine);
    init_solver_descriptor_sets();
    
    int offset = 0;
    for (int i = 0; i < coloringConstraints.size(); i++)
    {
        DescriptorWriter writer;
        writer.write_buffer(0, edgeConstraintsBuffer.buffer, sizeof(EdgeConstraint) * coloringConstraints[i].size(), offset, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        offset += sizeof(EdgeConstraint) * coloringConstraints[i].size();
        writer.write_buffer(1, vertexIntermediateBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, compute2DConstraintsDSet[i]);
    }
    {
        DescriptorWriter writer;
        writer.write_buffer(0, loadedMesh.vertexBuffer.buffer, sizeof(VertexData) *numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, vertexIntermediateBuffer.buffer, sizeof(VertexData) *numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, vertexDynamicsBuffer.buffer, sizeof(VertexDynamics) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(3, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, computePreSolveDSet);
    }
    {
        DescriptorWriter writer;
        writer.write_buffer(0, vertexIntermediateBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, loadedMesh.vertexBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, vertexDynamicsBuffer.buffer, sizeof(VertexDynamics) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(3, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, computePostSolveDSet);
    }
}

void PBDMesh2D::init_mesh(float sideSize, int subdivisions){
    std::vector<uint32_t> indices;
    std::vector<VertexData> vertices;
    
    float displ = sideSize/subdivisions;
    float halfSide = sideSize/2.f;
    for(int i = 0; i < subdivisions; i++){
        for(int j = 0; j < subdivisions; j++){
            VertexData newVtx;
            newVtx.pos = glm::vec3(displ * i, 0, displ * j) - glm::vec3(halfSide);
            newVtx.pad = 1;
            vertices.push_back(newVtx);
        }
    }
    
    std::vector<InstanceData> instances;
    
    for(int i = 0; i < subdivisions - 1; i++){
        uint32_t i0 = i;
        uint32_t i1 = i + 1;
        for(int j = 0; j < subdivisions - 1; j++){
            uint32_t v0 = (i0 * subdivisions) + j;
            uint32_t v1 = (i0 * subdivisions) + j + 1;
            uint32_t v2 = (i1) * subdivisions + j;
            uint32_t v3 = (i1) * subdivisions + j + 1;
            
            {
                indices.push_back(v0);
                indices.push_back(v1);
                indices.push_back(v2);
                instances.push_back(InstanceData{v0,v1,v2});
            }
            {
                indices.push_back(v2);
                indices.push_back(v1);
                indices.push_back(v3);
                instances.push_back(InstanceData{v2,v1,v3});
            }
        }
    }
    
    instanceBuffer = engine->create_copy_buffer(instances.size() * sizeof(InstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, instances.data());
    
    std::vector<glm::vec3> mapping;
    
    coloringConstraints.resize(4);
    
    for(int i = 0; i < subdivisions; i++){
        for(int j = 0; j < subdivisions; j++){
            uint32_t v0 = ((i * subdivisions) + j);
            
            if(j<subdivisions-1)
            {//Constraint along y direction
                uint32_t v2 = ((i * subdivisions) + j + 1);
                EdgeConstraint nC{};
                nC.length = glm::length(vertices[v0].pos - vertices[v2].pos);
                nC.elasticity = 2.f;
                nC.v1 = v0;
                nC.v2 = v2;
                mapping.push_back(glm::vec3(constraints.size(), 0+2*(j%2), coloringConstraints[0+2*(j%2)].size()));
                constraints.push_back(nC);
                coloringConstraints[0+2*(j%2)].push_back(nC);
            }
            if(i<subdivisions-1)
            {//Constraint along x direction
                uint32_t v4 = (((i+1) * subdivisions) + j);
                EdgeConstraint nC{};
                nC.length = glm::length(vertices[v0].pos - vertices[v4].pos);
                nC.elasticity = 2.f;
                nC.v1 = v0;
                nC.v2 = v4;
                mapping.push_back(glm::vec3(constraints.size(), 1+2*(i%2), coloringConstraints[1+2*(i%2)].size()));
                constraints.push_back(nC);
                coloringConstraints[1+2*(i%2)].push_back(nC);
            }
        }
    }
    
    numVertices = vertices.size();
    
    numIndices = indices.size();
    
    indices = {0,1,2};
    std::vector<VertexDynamics> vertexDynBuffer;
    vertexDynBuffer.resize(numVertices);
    for(int i = 0; i< numVertices; i++){
        VertexDynamics newVtxDyn;
        newVtxDyn.mass = 1.f;
        newVtxDyn.velocity = glm::vec3(0, 0.01f, 0);
        vertexDynBuffer[i] = newVtxDyn;
    }
    
    vertexDynamicsBuffer = engine->create_copy_buffer(numVertices * sizeof(VertexDynamics), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, vertexDynBuffer.data());
    
    int t = 0;
    for(int i = 0; i<coloringConstraints.size(); i++){
        fmt::print("{} constraints in bucket {}\n", coloringConstraints[i].size(), i);
        for(int j = 0; j < coloringConstraints[i].size(); j++){
            constraints[t] = coloringConstraints[i][j];
            t++;
        }
    }
    
    edgeConstraintsBuffer = engine->create_copy_buffer(constraints.size() * sizeof(EdgeConstraint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, constraints.data());
    
    for (int i = 0; i< mapping.size(); i++){
        constraints[i] = coloringConstraints[mapping[i].y][mapping[i].z];
    }
    
    correctionBuffer = engine->create_buffer(numVertices * sizeof(Correction), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    
    vertexIntermediateBuffer = engine->create_buffer(numVertices * sizeof(VertexData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    
    engine->_mainDeletionQueue.push_function([=, this]() {
        engine->destroy_buffer(correctionBuffer);
        engine->destroy_buffer(vertexIntermediateBuffer);
    });
    
    loadedMesh = engine->uploadMesh(indices, vertices);
    
    fmt::print("2D Mesh loaded, {} vertices, {} indices, {} constraints\n", numVertices, numIndices, constraints.size());
}

void PBDMesh2D::solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps){
    VkDescriptorSet descriptorSets[] = {uniformSceneDSet, computePreSolveDSet};
    
    for (int i = 0; i<solverSteps; i++){
        //PreSolve
        descriptorSets[1] = computePreSolveDSet;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePreSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 32.0), 1, 1);
        
        solve_edge_constraints(cmd);
        
        //PostSolve
        descriptorSets[1] = computePostSolveDSet;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePostSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 32.0), 1, 1);
    }
}

void PBDMesh3D::init_mesh(float sideSize, int subdivisions){
    std::vector<uint32_t> indices;
    std::vector<VertexData> vertices;
    
    if(subdivisions < 3){
        subdivisions = 3;
    }
    if(subdivisions%2 == 0){
        subdivisions+=1;
    }
    
    float displ = sideSize/(subdivisions-1);
    float halfSide = sideSize/2.f;
    fmt::print("subdivisions = {}, displ = {}, hSide = {}\n", subdivisions, displ, halfSide);
    for(int i = 0; i < subdivisions; i++){
        for(int j = 0; j < subdivisions; j++){
            for(int k = 0; k < subdivisions; k++){
                VertexData newVtx;
                newVtx.pos = glm::vec3(displ * i, displ * j, displ * k) - glm::vec3(halfSide);
                newVtx.pad = 1;
                vertices.push_back(newVtx);
            }
        }
    }
    
    std::vector<InstanceData> instances;
    
    for(int i = 0; i < subdivisions; i += subdivisions-1){
        for(int j = 0; j < subdivisions - 1; j++){
            uint32_t j0 = j;
            uint32_t j1 = j + 1;
            for(int k = 0; k < subdivisions - 1; k++){
                uint32_t k0 = k;
                uint32_t k1 = k + 1;
                
                uint32_t v0 = ((i * subdivisions) + j0) * subdivisions + k0;
                uint32_t v1 = ((i * subdivisions) + j0) * subdivisions + k1;
                uint32_t v2 = ((i * subdivisions) + j1) * subdivisions + k0;
                uint32_t v3 = ((i * subdivisions) + j1) * subdivisions + k1;
                
                {//F1
                    indices.push_back(v0);
                    indices.push_back(v1);
                    indices.push_back(v2);
                    instances.push_back(InstanceData{v0,v1,v2});
                }
                {
                    indices.push_back(v2);
                    indices.push_back(v1);
                    indices.push_back(v3);
                    instances.push_back(InstanceData{v2,v1,v3});
                }
            }
        }
    }
    
    
    for(int j = 0; j < subdivisions; j += subdivisions-1){
        for(int i = 0; i < subdivisions - 1; i++){
            uint32_t i0 = i;
            uint32_t i1 = i + 1;
            for(int k = 0; k < subdivisions - 1; k++){
                uint32_t k0 = k;
                uint32_t k1 = k + 1;
                
                uint32_t v0 = ((i0 * subdivisions) + j) * subdivisions + k0;
                uint32_t v1 = ((i0 * subdivisions) + j) * subdivisions + k1;
                uint32_t v2 = ((i1 * subdivisions) + j) * subdivisions + k0;
                uint32_t v3 = ((i1 * subdivisions) + j) * subdivisions + k1;
                
                {//F1
                    indices.push_back(v0);
                    indices.push_back(v1);
                    indices.push_back(v2);
                    instances.push_back(InstanceData{v0,v1,v2});
                }
                {
                    indices.push_back(v2);
                    indices.push_back(v1);
                    indices.push_back(v3);
                    instances.push_back(InstanceData{v2,v1,v3});
                }
            }
        }
    }
    
    for(int k = 0; k < subdivisions; k += subdivisions-1){
        for(int i = 0; i < subdivisions - 1; i++){
            uint32_t i0 = i;
            uint32_t i1 = i + 1;
            for(int j = 0; j < subdivisions - 1; j++){
                uint32_t j0 = j;
                uint32_t j1 = j + 1;
                
                uint32_t v0 = ((i0 * subdivisions) + j0) * subdivisions + k;
                uint32_t v1 = ((i1 * subdivisions) + j0) * subdivisions + k;
                uint32_t v2 = ((i0 * subdivisions) + j1) * subdivisions + k;
                uint32_t v3 = ((i1 * subdivisions) + j1) * subdivisions + k;
                
                {//F1
                    indices.push_back(v0);
                    indices.push_back(v1);
                    indices.push_back(v2);
                    instances.push_back(InstanceData{v0,v1,v2});
                }
                {
                    indices.push_back(v2);
                    indices.push_back(v1);
                    indices.push_back(v3);
                    instances.push_back(InstanceData{v2,v1,v3});
                }
            }
        }
    }
    
    int volumeUnitSize = 1;
    coloringVConstraints.resize(10);
    
    int numVolumeConstraints = 0;
    for(int i = 0; i < subdivisions - volumeUnitSize; i += volumeUnitSize){
        int idx;
        uint32_t i0 = i;
        uint32_t i1 = i0 + volumeUnitSize;
        for(int j = 0; j < subdivisions - volumeUnitSize; j += volumeUnitSize){
            uint32_t j0 = j;
            uint32_t j1 = j0 + volumeUnitSize;
            for(int k = 0; k < subdivisions - volumeUnitSize; k += volumeUnitSize){
                idx = ((i+j+k)/volumeUnitSize)%2;
                idx*=5;
                
                uint32_t k0 = k;
                uint32_t k1 = k0 + volumeUnitSize;
                
                uint32_t v0 = ((i0 * subdivisions) + j0) * subdivisions + k0;
                uint32_t v1 = ((i0 * subdivisions) + j0) * subdivisions + k1;
                uint32_t v2 = ((i0 * subdivisions) + j1) * subdivisions + k0;
                uint32_t v3 = ((i0 * subdivisions) + j1) * subdivisions + k1;
                uint32_t v4 = ((i1 * subdivisions) + j0) * subdivisions + k0;
                uint32_t v5 = ((i1 * subdivisions) + j0) * subdivisions + k1;
                uint32_t v6 = ((i1 * subdivisions) + j1) * subdivisions + k0;
                uint32_t v7 = ((i1 * subdivisions) + j1) * subdivisions + k1;
                
                {//V1
                    VolumeConstraint vC{};
                    vC.v1 = v1;
                    vC.v2 = v0;
                    vC.v3 = v2;
                    vC.v4 = v4;
                    vC.volume = glm::dot(glm::cross((vertices[vC.v2].pos - vertices[vC.v1].pos),
                                                    (vertices[vC.v3].pos - vertices[vC.v1].pos)),
                                         (vertices[vC.v4].pos - vertices[vC.v1].pos))/6 ;
                    coloringVConstraints[idx++].push_back(vC);
                    numVolumeConstraints++;
                }
                {//V2
                    VolumeConstraint vC{};
                    vC.v1 = v3;
                    vC.v2 = v1;
                    vC.v3 = v2;
                    vC.v4 = v7;
                    vC.volume = glm::dot(glm::cross((vertices[vC.v2].pos - vertices[vC.v1].pos),
                                                    (vertices[vC.v3].pos - vertices[vC.v1].pos)),
                                         (vertices[vC.v4].pos - vertices[vC.v1].pos))/6 ;
                    coloringVConstraints[idx++].push_back(vC);
                    numVolumeConstraints++;
                }
                {//V3
                    VolumeConstraint vC{};
                    vC.v1 = v4;
                    vC.v2 = v5;
                    vC.v3 = v7;
                    vC.v4 = v1;
                    vC.volume = glm::dot(glm::cross((vertices[vC.v2].pos - vertices[vC.v1].pos),
                                                    (vertices[vC.v3].pos - vertices[vC.v1].pos)),
                                         (vertices[vC.v4].pos - vertices[vC.v1].pos))/6 ;
                    coloringVConstraints[idx++].push_back(vC);
                    numVolumeConstraints++;
                }
                {//V4
                    VolumeConstraint vC{};
                    vC.v1 = v6;
                    vC.v2 = v4;
                    vC.v3 = v7;
                    vC.v4 = v2;
                    vC.volume = glm::dot(glm::cross((vertices[vC.v2].pos - vertices[vC.v1].pos),
                                                    (vertices[vC.v3].pos - vertices[vC.v1].pos)),
                                         (vertices[vC.v4].pos - vertices[vC.v1].pos))/6 ;
                    coloringVConstraints[idx++].push_back(vC);
                    numVolumeConstraints++;
                }
                {//V5
                    VolumeConstraint vC{};
                    vC.v1 = v2;
                    vC.v2 = v1;
                    vC.v3 = v4;
                    vC.v4 = v7;
                    vC.volume = glm::dot(glm::cross((vertices[vC.v2].pos - vertices[vC.v1].pos),
                                                    (vertices[vC.v3].pos - vertices[vC.v1].pos)),
                                         (vertices[vC.v4].pos - vertices[vC.v1].pos))/6 ;
                    coloringVConstraints[idx++].push_back(vC);
                    numVolumeConstraints++;
                
                }
            }
        }
    }
    
    int count = 0;
    float vTot = 0;
    volumeIdxList.resize(numVolumeConstraints * 4);
    for(int coloring = 0; coloring < coloringVConstraints.size(); coloring++){
        int bucketSize = coloringVConstraints[coloring].size();
        for(int vC = 0; vC < bucketSize; vC++){
            volumeIdxList[count*4 + vC] = coloringVConstraints[coloring][vC].v1;
            volumeIdxList[count*4 + vC + bucketSize * 1] = coloringVConstraints[coloring][vC].v2;
            volumeIdxList[count*4 + vC + bucketSize * 2] = coloringVConstraints[coloring][vC].v3;
            volumeIdxList[count*4 + vC + bucketSize * 3] = coloringVConstraints[coloring][vC].v4;
            volumeList.push_back(coloringVConstraints[coloring][vC].volume);
            vTot+= coloringVConstraints[coloring][vC].volume;
        }
        count += bucketSize;
        while(volumeList.size()%4!=0){
            volumeList.push_back(0.f);
        }
    }
    
    fmt::print("Total constrained volume : {}, Total volume: {}\n", vTot, sideSize*sideSize*sideSize);
    
    volumeValBuffer = engine->create_copy_buffer(volumeList.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, volumeList.data());
    
    volumeIdxBuffer = engine->create_copy_buffer(volumeIdxList.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, volumeIdxList.data());
    
    instanceBuffer = engine->create_copy_buffer(instances.size() * sizeof(InstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, instances.data());
    
    std::vector<glm::vec3> mapping;
    
    coloringConstraints.resize(6);

    for(int i = 0; i < subdivisions; i++){
        for(int j = 0; j < subdivisions; j++){
            for(int k = 0; k < subdivisions; k++){
                uint32_t v0 = ((i * subdivisions) + j) * subdivisions + k;
                
                if(k<subdivisions-1)
                {//Constraint along z direction
                    uint32_t v1 = ((i * subdivisions) + j) * subdivisions + k + 1;
                    EdgeConstraint nC{};
                    nC.length = glm::length(vertices[v0].pos - vertices[v1].pos);
                    nC.elasticity = 2.f;
                    nC.v1 = v0;
                    nC.v2 = v1;
                    mapping.push_back(glm::vec3(constraints.size(), 0+3*(k%2), coloringConstraints[0+3*(k%2)].size()));
                    constraints.push_back(nC);
                    coloringConstraints[0+3*(k%2)].push_back(nC);
                }
                
                if(j<subdivisions-1)
                {//Constraint along y direction
                    uint32_t v2 = ((i * subdivisions) + j + 1) * subdivisions + k;
                    EdgeConstraint nC{};
                    nC.length = glm::length(vertices[v0].pos - vertices[v2].pos);
                    nC.elasticity = 2.f;
                    nC.v1 = v0;
                    nC.v2 = v2;
                    mapping.push_back(glm::vec3(constraints.size(), 1+3*(j%2), coloringConstraints[1+3*(j%2)].size()));
                    constraints.push_back(nC);
                    coloringConstraints[1+3*(j%2)].push_back(nC);
                }
                if(i<subdivisions-1)
                {//Constraint along x direction
                    uint32_t v4 = (((i+1) * subdivisions) + j) * subdivisions + k;
                    EdgeConstraint nC{};
                    nC.length = glm::length(vertices[v0].pos - vertices[v4].pos);
                    nC.elasticity = 2.f;
                    nC.v1 = v0;
                    nC.v2 = v4;
                    mapping.push_back(glm::vec3(constraints.size(), 2+3*(i%2), coloringConstraints[2+3*(i%2)].size()));
                    constraints.push_back(nC);
                    coloringConstraints[2+3*(i%2)].push_back(nC);
                }
            }
        }
    }
    numVertices = vertices.size();
    
    numIndices = indices.size();
    
    indices = {0,1,2};
    std::vector<VertexDynamics> vertexDynBuffer;
    //vertexDynBuffer.resize(numVertices);
    for(int i = 0; i< numVertices; i++){
        VertexDynamics newVtxDyn;
        newVtxDyn.mass = 1.f;
        newVtxDyn.velocity = glm::vec3(0, 0., 0);
        vertexDynBuffer.push_back(newVtxDyn);//[i] = newVtxDyn;
    }
    
    vertexDynamicsBuffer = engine->create_copy_buffer(numVertices * sizeof(VertexDynamics), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, vertexDynBuffer.data());
    
    int t = 0;
    for(int i = 0; i<coloringConstraints.size(); i++){
        fmt::print("{} constraints in bucket {}\n", coloringConstraints[i].size(), i);
        for(int j = 0; j < coloringConstraints[i].size(); j++){
            constraints[t] = coloringConstraints[i][j];
            t++;
        }
    }
    
    edgeConstraintsBuffer = engine->create_copy_buffer(constraints.size() * sizeof(EdgeConstraint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, constraints.data());
    
    for (int i = 0; i< mapping.size(); i++){
        constraints[i] = coloringConstraints[mapping[i].y][mapping[i].z];
    }
    
    correctionBuffer = engine->create_buffer(numVertices * sizeof(Correction), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    
    vertexIntermediateBuffer = engine->create_buffer(numVertices * sizeof(VertexData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    
    engine->_mainDeletionQueue.push_function([=, this]() {
        engine->destroy_buffer(correctionBuffer);
        engine->destroy_buffer(vertexIntermediateBuffer);
    });
    
    loadedMesh = engine->uploadMesh(indices, vertices);
    
    fmt::print("3D Mesh loaded, {} vertices, {} indices, {} edge constraints, {} volume constraints\n", numVertices, numIndices, constraints.size(), volumeList.size());
}

void PBDMesh3D::init_pipelines(){
    init_solver_pipelines();
    create_edge_pipeline(engine);
    create_volume_pipeline(engine);
}

void IConstraint3D::create_volume_pipeline(VulkanEngine* engine){
    VkShaderModule compConstraintsShader;
    if (!vkutil::load_shader_module("../shaders/volumeConstraints.comp.spv", engine->_device, &compConstraintsShader)) {
        fmt::print("Error when building the compute shader \n");
    }
    
    VkDescriptorSetLayout dsLayout, uboLayout;
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        dsLayout = builder.build(engine->_device, VK_SHADER_STAGE_ALL);
    }
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        uboLayout = builder.build(engine->_device, VK_SHADER_STAGE_ALL);
    }
    
    VkDescriptorSetLayout layouts[] = {compute3DDSLayout, dsLayout, uboLayout};
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = layouts;
    computeLayout.setLayoutCount = 3;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &computeLayout, nullptr, &compute3DPLayout));
    
    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = compConstraintsShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = compute3DPLayout;
    computePipelineCreateInfo.stage = stageinfo;

    VK_CHECK(vkCreateComputePipelines(engine->_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &compute3DPipeline));
    
    vkDestroyShaderModule(engine->_device, compConstraintsShader, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, dsLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, uboLayout, nullptr);
}

void IConstraint3D::create_volume_descriptor_sets(VulkanEngine* engine){
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        compute3DDSLayout = builder.build(engine->_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    
    compute3DConstraintsDSet.resize(coloringVConstraints.size());
    for(int i = 0; i<compute3DConstraintsDSet.size(); i++)
        compute3DConstraintsDSet[i] = engine->globalDescriptorAllocator.allocate(engine->_device, compute3DDSLayout);
}

void PBDMesh3D::init_descriptors()
{
    create_volume_descriptor_sets(engine);
    create_edge_descriptor_sets(engine);
    init_solver_descriptor_sets();
        
    int offset = 0;
    for (int i = 0; i < coloringConstraints.size(); i++)
    {
        DescriptorWriter writer;
        writer.write_buffer(0, edgeConstraintsBuffer.buffer, sizeof(EdgeConstraint) * coloringConstraints[i].size(), offset, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        offset += sizeof(EdgeConstraint) * coloringConstraints[i].size();
        writer.write_buffer(1, vertexIntermediateBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, compute2DConstraintsDSet[i]);
    }
    offset = 0;
    for (int i = 0; i < coloringVConstraints.size(); i++)
    {
        DescriptorWriter writer;
        writer.write_buffer(0, volumeValBuffer.buffer, sizeof(float) * coloringVConstraints[i].size(), ceil(offset/4)*4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, volumeIdxBuffer.buffer, sizeof(uint32_t) * coloringVConstraints[i].size() * 4, offset * 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        offset += sizeof(float) * coloringVConstraints[i].size();
        writer.update_set(engine->_device, compute3DConstraintsDSet[i]);
    }
    {
        DescriptorWriter writer;
        writer.write_buffer(0, loadedMesh.vertexBuffer.buffer, sizeof(VertexData) *numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, vertexIntermediateBuffer.buffer, sizeof(VertexData) *numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, vertexDynamicsBuffer.buffer, sizeof(VertexDynamics) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(3, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, computePreSolveDSet);
    }
    {
        DescriptorWriter writer;
        writer.write_buffer(0, vertexIntermediateBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, loadedMesh.vertexBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, vertexDynamicsBuffer.buffer, sizeof(VertexDynamics) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(3, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, computePostSolveDSet);
    }
}

void PBDMesh3D::solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps){
    VkDescriptorSet descriptorSets[] = {uniformSceneDSet, computePreSolveDSet};
    
    for (int i = 0; i<solverSteps; i++){
        //PreSolve
        descriptorSets[1] = computePreSolveDSet;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePreSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 32.0), 1, 1);
        
        solve_volume_constraints(cmd, computeSolveDSet, uniformSceneDSet);
        solve_edge_constraints(cmd);
        
        //PostSolve
        descriptorSets[1] = computePostSolveDSet;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePostSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 32.0), 1, 1);
    }
}

void IConstraint3D::solve_volume_constraints(VkCommandBuffer cmd, VkDescriptorSet computeDSet, VkDescriptorSet uniformDSet){
    //Solve
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute3DPipeline);
    for(int c = 0; c<coloringVConstraints.size(); c++){
        VkDescriptorSet sets[] = {compute3DConstraintsDSet[c], computeDSet, uniformDSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute3DPLayout, 0, 3, sets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(coloringVConstraints[c].size() / 32.0), 1, 1);
    }
    
    
}

PBDMesh3D::PBDMesh3D(VulkanEngine *engine, float sideSize, int subdivisions){
    this->engine = engine;
    init_mesh(sideSize, subdivisions);
    init_descriptors();
    init_pipelines();
}

void PBDMesh3D::clear_resources(){
    vkDestroyDescriptorSetLayout(engine->_device, compute2DDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, compute3DDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, computePSolveDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, computeSolveDSLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, uniformComputeDSLayout, nullptr);
    vkDestroyPipelineLayout(engine->_device, compute2DPLayout, nullptr);
    vkDestroyPipeline(engine->_device, compute2DPipeline, nullptr);
    vkDestroyPipelineLayout(engine->_device, compute3DPLayout, nullptr);
    vkDestroyPipeline(engine->_device, compute3DPipeline, nullptr);
    vkDestroyPipelineLayout(engine->_device, computeSolvePLayout, nullptr);
    vkDestroyPipeline(engine->_device, computePreSolvePipeline, nullptr);
    vkDestroyPipeline(engine->_device, computePostSolvePipeline, nullptr);
}
