#include "mesh.hpp"
#include "engine.hpp"
#include "pipelines.hpp"
#include "initializers.hpp"

#include <algorithm>
#include <random>

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
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        computeSolveDSLayout = builder.build(engine->_device, VK_SHADER_STAGE_ALL);
    }
    
    computeSolveDSet = engine->globalDescriptorAllocator.allocate(engine->_device, computeSolveDSLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, vertexIntermediateBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, vertexDynamicsBuffer.buffer, sizeof(VertexDynamics) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(2, correctionBuffer.buffer, sizeof(Correction) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(engine->_device, computeSolveDSet);
    }
    
}

void IConstraint2D::create_edge_descriptor_sets(VulkanEngine* engine){
    /*{
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        compute2DDSLayout = builder.build(engine->_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }*/
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
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
    VkDescriptorSetLayout uboLayout, dsLayout;
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        dsLayout = builder.build(engine->_device, VK_SHADER_STAGE_ALL);
    }
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        uboLayout = builder.build(engine->_device, VK_SHADER_STAGE_ALL);
    }
    
    VkDescriptorSetLayout layouts[] = {compute2DDSLayout, dsLayout, uboLayout};
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = layouts;
    computeLayout.setLayoutCount = 3;
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
    vkDestroyDescriptorSetLayout(engine->_device, dsLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->_device, uboLayout, nullptr);
}

void IConstraint2D::solve_edge_constraints(VkCommandBuffer cmd, VkDescriptorSet computeSolveDSet, VkDescriptorSet uniformDSet, int solverStep){
    //Solve
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute2DPipeline);
    for(int c = 0; c<coloringConstraints.size(); c++){
        VkDescriptorSet sets[] = {compute2DConstraintsDSet[c], computeSolveDSet, uniformDSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute2DPLayout, 0, 3, sets, 0, nullptr);
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
        writer.write_buffer(0, distanceValBuffer.buffer, sizeof(float) * coloringConstraints[i].size(), offset, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, distanceIdxBuffer.buffer, sizeof(uint32_t) * coloringConstraints[i].size() * 2, offset*2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        
        offset += sizeof(float) * ceil(coloringConstraints[i].size()/4.f)*4;
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
    
    numVertices = vertices.size();
    
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
    
    
    numIndices = indices.size();
    indices = {0,1,2}; //Instanced triangle
    
    std::vector<VertexDynamics> vertexDynBuffer;
    vertexDynBuffer.resize(numVertices);
    for(int i = 0; i< numVertices; i++){
        VertexDynamics newVtxDyn;
        newVtxDyn.invMass = 1/(1000.f/numVertices);
        newVtxDyn.velocity = glm::vec3(0);
        vertexDynBuffer[i] = newVtxDyn;
    }
    
    vertexDynamicsBuffer = engine->create_copy_buffer(numVertices * sizeof(VertexDynamics), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, vertexDynBuffer.data());
    
    auto defDConst = [&] (uint32_t v1, uint32_t v2) ->EdgeConstraint{
        EdgeConstraint nC{};
        nC.length = glm::length(vertices[v1].pos - vertices[v2].pos);
        nC.elasticity = 0.0000000001;
        nC.v1 = v1;
        nC.v2 = v2;
        totalDConst++;
        return nC;
    };
    
    coloringConstraints.resize(4);
    for(int i = 0; i < subdivisions; i++){
        for(int j = 0; j < subdivisions; j++){
            uint32_t v0 = ((i * subdivisions) + j);
            
            if(j<subdivisions-1)
            {//Constraint along y direction
                uint32_t v1 = ((i * subdivisions) + j + 1);
                coloringConstraints[0+2*(j%2)].push_back(defDConst(v0, v1));
            }
            if(i<subdivisions-1)
            {//Constraint along x direction
                uint32_t v1 = (((i+1) * subdivisions) + j);
                coloringConstraints[1+2*(i%2)].push_back(defDConst(v0, v1));
            }
        }
    }
    
    auto rng = std::default_random_engine {}; //shuffle seed
    
    distanceIdxList.resize(totalDConst * 2); //two indices for each constraint
    constraints.resize(totalDConst);
    int count = 0;
    int t = 0;
    for(int i = 0; i<coloringConstraints.size(); i++){
        std::shuffle(std::begin(coloringConstraints[i]), std::end(coloringConstraints[i]), rng);
        int bucketSize = coloringConstraints[i].size();
        for(int j = 0; j < bucketSize; j++){
            distanceIdxList[count*2 + j] = coloringConstraints[i][j].v1;
            distanceIdxList[count*2 + j + bucketSize] = coloringConstraints[i][j].v2;
            distanceList.push_back(coloringConstraints[i][j].length);
            
            constraints[t] = coloringConstraints[i][j];
            t++;
        }
        count += bucketSize;
        while(distanceList.size()%4!=0){ //Memory alignment
            distanceList.push_back(0.f);
            distanceIdxList[count*2]=0;
            distanceIdxList[count*2+1]=0;
            count++;
            distanceIdxList.resize(distanceIdxList.size()+2);
        }
    }
    
    distanceValBuffer = engine->create_copy_buffer(distanceList.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, distanceList.data());
    
    distanceIdxBuffer = engine->create_copy_buffer(distanceIdxList.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, distanceIdxList.data());
    
    edgeConstraintsBuffer = engine->create_copy_buffer(constraints.size() * sizeof(EdgeConstraint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, constraints.data());
    
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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePreSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 64.0), 1, 1);
        
        VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(vertexIntermediateBuffer.buffer, engine->_graphicsQueueFamily);
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        
        solve_edge_constraints(cmd, computeSolveDSet, uniformSceneDSet, i);

        barrier = vkinit::buffer_barrier(vertexIntermediateBuffer.buffer, engine->_graphicsQueueFamily);
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

        //PostSolve
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePostSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 64.0), 1, 1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }
}

void PBDMesh3D::init_mesh(float sideSize, int subdivisions){
    std::vector<uint32_t> indices;
    
    Mesh::subdivisions = subdivisions;
    
    if(subdivisions < 3){
        subdivisions = 3;
    }
    if(subdivisions%2 == 0){
        subdivisions+=1;
    }
    
    bool cylinderMesh = false;
    float displ = sideSize/(subdivisions-1);
    float halfSide = sideSize/2.f;
    fmt::print("subdivisions = {}, displ = {}, hSide = {}\n", subdivisions, displ, halfSide);
    
    if(cylinderMesh)
        displ = 0.00845f;// hollow cylinder with 2.56 mˆ2 cross area
        
    for(int i = 0; i < subdivisions; i++){
        for(int j = 0; j < subdivisions; j++){
            for(int k = 0; k < subdivisions; k++){
                VertexData newVtx;
                if(!cylinderMesh)
                    newVtx.pos = glm::vec3(2*displ * i,  displ/5 * j, displ/5 * k) - glm::vec3(3*halfSide);// - glm::vec3(0,10*sin(M_PI/subdivisions*i),0);
                else
                    newVtx.pos = glm::vec3(4*0.25f * i,
                                           (displ * j + 1.32) * cos(2*M_PI * k/(subdivisions-1)),
                                           (displ * j + 1.32) * sin(2*M_PI * k/(subdivisions-1))) - glm::vec3(4*halfSide,0,0);
                newVtx.pad = 1;
                vertices.push_back(newVtx);
            }
        }
    }
    
    /* POC: sphere lattice
    for(int i = subdivisions; i > 0; i--){
        float r = displ * i;
        for(int j = 0; j < subdivisions; j++){
            float theta = j * M_PI / (subdivisions-1);
            for(int k = 0; k < subdivisions; k++){
                float phi = 2 * k * M_PI / (subdivisions-1);
                VertexData newVtx;
                newVtx.pos = r * glm::vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));;// - glm::vec3(0,20,0);
                newVtx.pad = 1;
                vertices.push_back(newVtx);
            }
        }
    }*/
    
    numVertices = vertices.size();
    previousPositions.resize(numVertices);
    
    for(int i = 0; i< numVertices; i++){
        VertexDynamics newVtxDyn;
        newVtxDyn.invMass = 1.0f / (1000.f / numVertices);
        newVtxDyn.velocity = glm::vec3(0, 0, 0);
        vertexDynBuffer.push_back(newVtxDyn);
    }
    vertexDynamicsBuffer = engine->create_copy_buffer(numVertices * sizeof(VertexDynamics), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, vertexDynBuffer.data());
    
    
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
    
    if(!cylinderMesh)
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
    
    
    numIndices = indices.size();
    indices = {0,1,2}; //Instanced triangle
    
    instanceBuffer = engine->create_copy_buffer(instances.size() * sizeof(InstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, instances.data());
    
    auto defVConst = [&] (uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4) ->VolumeConstraint{
        VolumeConstraint vC{};
        vC.v1 = v1;
        vC.v2 = v2;
        vC.v3 = v3;
        vC.v4 = v4;
        vC.volume = glm::dot(glm::cross((vertices[vC.v2].pos - vertices[vC.v1].pos),
                                        (vertices[vC.v3].pos - vertices[vC.v1].pos)),
                             (vertices[vC.v4].pos - vertices[vC.v1].pos))/6 ;
        totalVConst++;
        return vC;
    };
    
    int volumeUnitSize = 1;
    coloringVConstraints.resize(10);
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
                    coloringVConstraints[idx++].push_back(defVConst(v1, v0, v2, v4));
                }
                {//V2
                    coloringVConstraints[idx++].push_back(defVConst(v3, v1, v2, v7));
                }
                {//V3
                    coloringVConstraints[idx++].push_back(defVConst(v4, v5, v7, v1));
                }
                {//V4
                    coloringVConstraints[idx++].push_back(defVConst(v6, v4, v7, v2));
                }
                {//V5
                    coloringVConstraints[idx++].push_back(defVConst(v2, v1, v4, v7));
                }
            }
        }
    }
    
    auto rng = std::default_random_engine {}; //shuffle seed
    int count = 0;
    volumeIdxList.resize(totalVConst * 4);
    for(int coloring = 0; coloring < coloringVConstraints.size(); coloring++){
        int bucketSize = coloringVConstraints[coloring].size();
        std::shuffle(std::begin(coloringVConstraints[coloring]), std::end(coloringVConstraints[coloring]), rng);
        
        for(int vC = 0; vC < bucketSize; vC++){
            vConstraints.push_back(coloringVConstraints[coloring][vC]);
            volumeIdxList[count*4 + vC] = coloringVConstraints[coloring][vC].v1;
            volumeIdxList[count*4 + vC + bucketSize * 1] = coloringVConstraints[coloring][vC].v2;
            volumeIdxList[count*4 + vC + bucketSize * 2] = coloringVConstraints[coloring][vC].v3;
            volumeIdxList[count*4 + vC + bucketSize * 3] = coloringVConstraints[coloring][vC].v4;
            volumeList.push_back(coloringVConstraints[coloring][vC].volume);
        }
        count += bucketSize;
        while(volumeList.size()%4!=0){ //Memory alignment
            volumeList.push_back(0.f);
        }
    }
    
    volumeValBuffer = engine->create_copy_buffer(volumeList.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, volumeList.data());
    
    volumeIdxBuffer = engine->create_copy_buffer(volumeIdxList.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, volumeIdxList.data());
    
    coloringConstraints.resize(6);
    
    
    auto defDConst = [&] (uint32_t v1, uint32_t v2) ->EdgeConstraint{
        EdgeConstraint nC{};
        nC.length = glm::length(vertices[v1].pos - vertices[v2].pos);
        nC.elasticity = 0.0000000001;
        nC.v1 = v1;
        nC.v2 = v2;
        totalDConst++;
        return nC;
    };

    for(int i = 0; i < subdivisions; i++){
        for(int j = 0; j < subdivisions; j++){
            for(int k = 0; k < subdivisions; k++){
                uint32_t v0 = ((i * subdivisions) + j) * subdivisions + k;
                
                if(k < subdivisions-1)
                {//Constraint along z direction
                    uint32_t v1 = ((i * subdivisions) + j) * subdivisions + k + 1;
                    coloringConstraints[0+3*(k%2)].push_back(defDConst(v0, v1));
                }
                if(k==subdivisions-1 && cylinderMesh)
                {
                    uint32_t v1 = ((i * subdivisions) + j) * subdivisions;
                    coloringConstraints[0+3*(k%2)].push_back(defDConst(v0, v1));
                }
                
                if(j < subdivisions-1)
                {//Constraint along y direction
                    uint32_t v1 = ((i * subdivisions) + j + 1) * subdivisions + k;
                    coloringConstraints[1+3*(j%2)].push_back(defDConst(v0, v1));
                    totalDConst++;
                }
                
                if(i < subdivisions-1)
                {//Constraint along x direction
                    uint32_t v1 = (((i+1) * subdivisions) + j) * subdivisions + k;
                    coloringConstraints[2+3*(i%2)].push_back(defDConst(v0, v1));
                    totalDConst++;
                }
            }
        }
    }
    
    distanceIdxList.resize(totalDConst * 2); //two indices for each constraint
    constraints.resize(totalDConst);
    count = 0;
    int t = 0;
    for(int i = 0; i<coloringConstraints.size(); i++){
        std::shuffle(std::begin(coloringConstraints[i]), std::end(coloringConstraints[i]), rng);
        int bucketSize = coloringConstraints[i].size();
        for(int j = 0; j < bucketSize; j++){
            distanceIdxList[count*2 + j] = coloringConstraints[i][j].v1;
            distanceIdxList[count*2 + j + bucketSize] = coloringConstraints[i][j].v2;
            distanceList.push_back(coloringConstraints[i][j].length);
            
            constraints[t] = coloringConstraints[i][j];
            t++;
        }
        count += bucketSize;
        while(distanceList.size()%4!=0){ //Memory alignment
            distanceList.push_back(0.f);
            distanceIdxList[count*2]=0;
            distanceIdxList[count*2+1]=0;
            count++;
            distanceIdxList.resize(distanceIdxList.size()+2);
        }
    }
    
    distanceValBuffer = engine->create_copy_buffer(distanceList.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, distanceList.data());
    
    distanceIdxBuffer = engine->create_copy_buffer(distanceIdxList.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, distanceIdxList.data());
    
    edgeConstraintsBuffer = engine->create_copy_buffer(constraints.size() * sizeof(EdgeConstraint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, constraints.data());
    
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
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
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
        writer.write_buffer(0, distanceValBuffer.buffer, sizeof(float) * coloringConstraints[i].size(), offset, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, distanceIdxBuffer.buffer, sizeof(uint32_t) * coloringConstraints[i].size() * 2, offset*2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        
        offset += sizeof(float) * ceil(coloringConstraints[i].size()/4.f)*4;
        writer.update_set(engine->_device, compute2DConstraintsDSet[i]);
    }
    
    offset = 0;
    for (int i = 0; i < coloringVConstraints.size(); i++)
    {
        DescriptorWriter writer;
        writer.write_buffer(0, volumeValBuffer.buffer, sizeof(float) * coloringVConstraints[i].size(), offset, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, volumeIdxBuffer.buffer, sizeof(uint32_t) * coloringVConstraints[i].size() * 4, offset * 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        offset += sizeof(float) * ceil(coloringVConstraints[i].size()/4)*4;
        writer.update_set(engine->_device, compute3DConstraintsDSet[i]);
    }
    {
        DescriptorWriter writer;
        writer.write_buffer(0, loadedMesh.vertexBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, vertexIntermediateBuffer.buffer, sizeof(VertexData) * numVertices, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePreSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 64.0), 1, 1);
        
        VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(vertexIntermediateBuffer.buffer, engine->_graphicsQueueFamily);
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        
        solve_edge_constraints(cmd, computeSolveDSet, uniformSceneDSet, i);
        solve_volume_constraints(cmd, computeSolveDSet, uniformSceneDSet, i);

        barrier = vkinit::buffer_barrier(vertexIntermediateBuffer.buffer, engine->_graphicsQueueFamily);
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

        //PostSolve
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePostSolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeSolvePLayout, 0, 2, descriptorSets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(numVertices / 64.0), 1, 1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }
}

void IConstraint3D::solve_volume_constraints(VkCommandBuffer cmd, VkDescriptorSet computeDSet, VkDescriptorSet uniformDSet, int solverStep){
    //Solve
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute3DPipeline);
    for(int c = 0; c<coloringVConstraints.size(); c++){
        VkDescriptorSet sets[] = {compute3DConstraintsDSet[c], computeDSet, uniformDSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute3DPLayout, 0, 3, sets, 0, nullptr);
        vkCmdDispatch(cmd, std::ceil(coloringVConstraints[c].size() / 64.0), 1, 1);
    }
}

PBDMesh3D::PBDMesh3D(VulkanEngine *engine, float sideSize, int subdivisions){
    std::string modelCode = std::to_string(subdivisions);
    outputPosCSV.open("../samplePosition" + modelCode + ".csv");
    if(!outputPosCSV.is_open()){
        fmt::print("Failed to open file!");
        exit(1);
    } else outputPosCSV << "TIMESTEP;Xa;Ya;Za;Xb;Yb;Zb\n";
    
    outputSectionCSV.open("../sampleSection" + modelCode + ".csv");
    if(!outputSectionCSV.is_open()){
        fmt::print("Failed to open file!");
        exit(1);
    } else outputSectionCSV << "Snapshot;X;Y;Z\n";
    
    outputStretchCSV.open("../sampleStretch" + modelCode + ".csv");
    
    if(!outputStretchCSV.is_open()){
        fmt::print("Failed to open file!");
        exit(1);
    } else outputStretchCSV << "Snapshot;Stretch\n";
    
    outputVolumeCSV.open("../sampleVolume" + modelCode + ".csv");
    if(!outputVolumeCSV.is_open()){
        fmt::print("Failed to open file!");
        exit(1);
    } else outputVolumeCSV << "Snapshot;Volume\n";
    
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
    outputPosCSV.close();
    outputSectionCSV.close();
    outputStretchCSV.close();
    outputVolumeCSV.close();
}

void PBDMesh2D::solve_constraints_sequential(int solverSteps, float timeStep, glm::vec3 acceleration){
    //static std::vector<glm::vec3> positions[numVertices];
    
    for(int i = 0; i<numVertices; i++){
        ;//vertexDynBuffer.velocity
    }
    
}

void PBDMesh3D::solve_constraints_sequential(int solverSteps, float timeStep, glm::vec3 acceleration){
    //static std::vector<glm::vec3> positions[numVertices];
    std::vector<glm::vec3> corrections;
    corrections.resize(numVertices);
    
    for(int s = 0; s<solverSteps; s++){
        //pre solve
        for(int i = 0; i<numVertices; i++){
            //Velocity update
            vertexDynBuffer[i].velocity += acceleration * timeStep;
            
            //Position update
            previousPositions[i] = vertices[i].pos;
            vertices[i].pos = vertices[i].pos + vertexDynBuffer[i].velocity * timeStep;
        }
        
        
        //edge constraints
        for(int e = 0; e<constraints.size(); e++){
            VertexData v1 = vertices[constraints[e].v1];
            VertexData v2 = vertices[constraints[e].v2];
            
            float w1 = vertexDynBuffer[constraints[e].v1].invMass;
            float w2 = vertexDynBuffer[constraints[e].v2].invMass;
            
            glm::vec3 d = v1.pos - v2.pos;
            float l = glm::length(d);
            d = glm::normalize(d);
            
            float dP = -(l - constraints[e].length);
            if(std::abs(dP) > std::pow(10, -6)){
                float K = w1 + w2;
                float alfa = constraints[e].elasticity / timeStep / timeStep;
                float gamma = alfa * (10 * timeStep * timeStep);
                
                K *= 1 + gamma/timeStep;
                K += alfa;
                
                vertices[constraints[e].v1].pos += (dP - gamma * glm::dot(d, vertexDynBuffer[constraints[e].v1].velocity)) * d/K * w1;
                vertices[constraints[e].v2].pos -= (dP - gamma * glm::dot(-d, vertexDynBuffer[constraints[e].v2].velocity)) * d/K * w2;
            }
        }
        
        //volume constraints
        for(int i = 0; i<vConstraints.size(); i++){
            VertexData v1 = vertices[vConstraints[i].v1];
            VertexData v2 = vertices[vConstraints[i].v2];
            VertexData v3 = vertices[vConstraints[i].v3];
            VertexData v4 = vertices[vConstraints[i].v4];
            
            
            float w1 = vertexDynBuffer[vConstraints[i].v1].invMass;
            float w2 = vertexDynBuffer[vConstraints[i].v2].invMass;
            float w3 = vertexDynBuffer[vConstraints[i].v3].invMass;
            float w4 = vertexDynBuffer[vConstraints[i].v4].invMass;
            
            float v = glm::dot(glm::cross(v2.pos - v1.pos, v3.pos - v1.pos), (v4.pos - v1.pos))/6;
            float dP = -(v - vConstraints[i].volume);
            
            if(std::abs(dP) > std::pow(10, -6)){
                glm::vec3 grad2 = glm::cross(v3.pos - v1.pos, v4.pos - v1.pos);
                glm::vec3 grad3 = glm::cross(v4.pos - v1.pos, v2.pos - v1.pos);
                glm::vec3 grad4 = glm::cross(v2.pos - v1.pos, v3.pos - v1.pos);
                glm::vec3 grad1 = -(grad2 + grad3 + grad4);
                
                float K = w1 * glm::dot(grad1, grad1) +
                            w2 * glm::dot(grad2, grad2) +
                            w3 * glm::dot(grad3, grad3) +
                            w4 * glm::dot(grad4, grad4);
                
                float alfa = std::pow(10,-7) / timeStep / timeStep;
                float gamma = alfa * (pow(10.f, 2.f) * timeStep * timeStep);
                
                K *= 1 + gamma/timeStep;
                K += alfa;
                
                vertices[vConstraints[i].v1].pos += (dP - gamma * glm::dot(grad1, vertexDynBuffer[vConstraints[i].v1].velocity)) * grad1/K * w1;
                vertices[vConstraints[i].v2].pos += (dP - gamma * glm::dot(grad2, vertexDynBuffer[vConstraints[i].v2].velocity)) * grad2/K * w2;
                vertices[vConstraints[i].v3].pos += (dP - gamma * glm::dot(grad3, vertexDynBuffer[vConstraints[i].v3].velocity)) * grad3/K * w3;
                vertices[vConstraints[i].v4].pos += (dP - gamma * glm::dot(grad4, vertexDynBuffer[vConstraints[i].v4].velocity)) * grad4/K * w4;
            }
        }
        
        //post solve
        for(int i = 0; i<numVertices; i++){
            if(vertices[i].pos.y >= 10.f){
                vertexDynBuffer[i].velocity = glm::vec3(0);
                vertices[i].pos.y = 10.f;
            }
            else{
                vertexDynBuffer[i].velocity = (vertices[i].pos - previousPositions[i]) / timeStep;
            }
        }
    }
    engine->copy_buffer(numVertices * sizeof(VertexData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, vertices.data(), loadedMesh.vertexBuffer);
}


//Logging methods
void PBDMesh::log_position(){
    static int iteration = 0;
    
    outputPosCSV << iteration++<< ";" << vertices[numVertices-1].pos.x << ";" << vertices[numVertices-1].pos.y << ";" << vertices[numVertices-1].pos.z << ";";
    
    outputPosCSV << "\n";
    return;
}

void PBDMesh::log_section(){
    static int sectionSnap = 0;
    std::ofstream file;
    file.open("../sampleSection" + std::to_string(sectionSnap) + ".csv");
    if(!file.is_open()){
        fmt::print("Failed to open file!");
        exit(1);
    } else file << "Snapshot;X;Y;Z\n";
    
    for(int y = 0; y <= subdivisions; y+=1){
        for(int z = 0; z <= subdivisions; z+=1){
            int index = 7 * (subdivisions+1) * (subdivisions+1) + y * (subdivisions+1) + z ;
            file << sectionSnap << ";" << vertices[index].pos.x << ";" << vertices[index].pos.y << ";" << vertices[index].pos.z << "\n";
        }
    }
    sectionSnap++;
    file.close();
}

void PBDMesh2D::log_stretch(){
    return;
}

void PBDMesh2D::log_volume(){
    return;
}

void PBDMesh3D::log_stretch(){
    static int stretchSnap = 0;
    VertexData v1 = vertices[coloringConstraints[2][0].v1];
    VertexData v2 = vertices[coloringConstraints[2][0].v2];
    
    glm::vec3 d = v1.pos - v2.pos;
    float l = glm::length(d);
    
    float dP = (l - coloringConstraints[2][0].length);
    
    outputStretchCSV << stretchSnap++ <<";"<< dP/coloringConstraints[2][0].length * 100 << "\n";
    return;
}

void PBDMesh3D::log_volume(){
    static int volumeSnap = 0;
    VertexData v1 = vertices[vConstraints[0].v1];
    VertexData v2 = vertices[vConstraints[0].v2];
    VertexData v3 = vertices[vConstraints[0].v3];
    VertexData v4 = vertices[vConstraints[0].v4];
    
    float v = glm::dot(glm::cross(v2.pos - v1.pos, v3.pos - v1.pos), (v4.pos - v1.pos))/6;
    float dP = (v - vConstraints[0].volume);
    
    outputVolumeCSV << volumeSnap++ <<";"<< dP/vConstraints[0].volume * 100 << "\n";
    return;
}

void PBDMesh::log_data(char* data){
    outputPosCSV << data <<"\n";
}

void PBDMesh::copy_from_device(){
    engine->copy_buffer_from_device(numVertices * sizeof(VertexData), vertices.data(), loadedMesh.vertexBuffer);
    return;
}

