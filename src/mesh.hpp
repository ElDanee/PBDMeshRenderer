#pragma once

#include "mesh.hpp"
#include "types.h"

#include <iostream>
#include <fstream>

#include <stdio.h>

class VulkanEngine;

class Mesh{
protected:
    int numIndices;
    int numVertices;
    int subdivisions;
    std::vector<VertexData> vertices;
    GPUMeshBuffers loadedMesh; //Mesh indices and vertices
    AllocatedBuffer vertexBuffer; //Vertex pulling buffer
    AllocatedBuffer instanceBuffer; //Buffer to hold surface instances(render normals)
    VulkanEngine* engine;
    
public:
    GPUMeshBuffers get_loaded_mesh(){return loadedMesh;}
    AllocatedBuffer get_vertex_buffer(){return vertexBuffer;}
    AllocatedBuffer get_instance_buffer(){return instanceBuffer;}
    int get_vertices(){return numVertices;}
    int get_indices(){return numIndices;}
    glm::vec3 get_vertex_zero(){return vertices[0].pos;}
};

class PBDMesh : public Mesh{
protected:
    std::ofstream outputPosCSV;
    std::ofstream outputSectionCSV;
    std::ofstream outputStretchCSV;
    std::ofstream outputVolumeCSV;
    
    std::vector<VertexDynamics> vertexDynBuffer;
    std::vector<float> invMass;
    AllocatedBuffer vertexIntermediateBuffer; //Interemediate solver results
    AllocatedBuffer vertexDynamicsBuffer; //Vertex velocity and mass
    AllocatedBuffer correctionBuffer; //Jacobi solver correction buffer
    
    std::vector<glm::vec3> previousPositions;
    
    VkDescriptorSetLayout computePSolveDSLayout;
    VkDescriptorSetLayout uniformComputeDSLayout;
    VkDescriptorSetLayout computeSolveDSLayout;
    
    VkDescriptorSet computePreSolveDSet;
    VkDescriptorSet computePostSolveDSet;
    VkDescriptorSet computeSolveDSet;
    
    VkPipelineLayout computeSolvePLayout;
    
    VkPipeline computePreSolvePipeline;
    VkPipeline computePostSolvePipeline;
    
    void init_solver_pipelines();
    void init_solver_descriptor_sets();
    
    
public:
    virtual void solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps)=0;
    virtual void solve_constraints_sequential(int solverSteps, float timeStep, glm::vec3 acceleration)=0;
    PBDMesh(){};
    virtual ~PBDMesh(){};
    virtual void clear_resources(){};
    void log_position();
    void log_section();
    void log_data(char* data);
    void copy_from_device();
    virtual void log_stretch()=0;
    virtual void log_volume()=0;
    
};

class IConstraint2D{
public:
    AllocatedBuffer edgeConstraintsBuffer;
    std::vector<EdgeConstraint> constraints;
    std::vector<std::vector<EdgeConstraint>> coloringConstraints;
    
    AllocatedBuffer distanceValBuffer;
    AllocatedBuffer distanceIdxBuffer;
    std::vector<uint32_t> distanceIdxList;
    std::vector<float> distanceList;
    
    int totalDConst = 0;
    
    VkDescriptorSetLayout compute2DDSLayout;
    std::vector<VkDescriptorSet> compute2DConstraintsDSet;
    
    VkPipelineLayout compute2DPLayout;
    VkPipeline compute2DPipeline;
    
    void create_edge_descriptor_sets(VulkanEngine* engine);
    void create_edge_pipeline(VulkanEngine* engine);
    void solve_edge_constraints(VkCommandBuffer cmd, VkDescriptorSet computeSolveDSet, VkDescriptorSet uniformDSet, int solverStep);
};

class IConstraint3D{
public:
    AllocatedBuffer volumeValBuffer;
    AllocatedBuffer volumeIdxBuffer;
    std::vector<VolumeConstraint> vConstraints;
    std::vector<std::vector<VolumeConstraint>> coloringVConstraints;
    std::vector<uint32_t> volumeIdxList;
    std::vector<float> volumeList;
    
    int totalVConst = 0;
    
    VkDescriptorSetLayout compute3DDSLayout;
    std::vector<VkDescriptorSet> compute3DConstraintsDSet;
    
    VkPipelineLayout compute3DPLayout;
    VkPipeline compute3DPipeline;
    
    void create_volume_descriptor_sets(VulkanEngine* engine);
    void create_volume_pipeline(VulkanEngine* engine);
    void solve_volume_constraints(VkCommandBuffer cmd, VkDescriptorSet computeDSet, VkDescriptorSet uniformDSet, int solverStep);
};

class PBDMesh2D : public PBDMesh, public IConstraint2D{
public:
    virtual void solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps) override;
    virtual void solve_constraints_sequential(int solverSteps, float timeStep, glm::vec3 acceleration) override;
    PBDMesh2D(){};
    ~PBDMesh2D(){clear_resources();};
    PBDMesh2D(VulkanEngine *engine, float sideSize = 10.f, int subdivisions = 10); //create square mesh
    void clear_resources() override;
    
    virtual void log_stretch() override;
    virtual void log_volume() override;
    
private:
    void init_pipelines();
    void init_descriptors();
    void init_mesh(float sideSize = 10.f, int subdivisions = 10);
    
};

class PBDMesh3D : public PBDMesh, public IConstraint2D, public IConstraint3D{
public:
    virtual void solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps) override;
    virtual void solve_constraints_sequential(int solverSteps, float timeStep, glm::vec3 acceleration) override;
    PBDMesh3D(){};
    ~PBDMesh3D(){clear_resources();};
    PBDMesh3D(VulkanEngine *engine, float sideSize = 10.f, int subdivisions = 10); //create cube mesh
    void clear_resources() override;
    
    virtual void log_stretch() override;
    virtual void log_volume() override;
    
private:
    void init_pipelines();
    void init_descriptors();
    void init_mesh(float sideSize = 10.f, int subdivisions = 10);
    
};


