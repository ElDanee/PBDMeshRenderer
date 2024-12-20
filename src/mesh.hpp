#pragma once

#include "mesh.hpp"
#include "types.h"

#include <stdio.h>

class VulkanEngine;

class Mesh{
protected:
    int numIndices;
    int numVertices;
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
};

class PBDMesh : public Mesh{
protected:
    AllocatedBuffer vertexIntermediateBuffer; //Interemediate solver results
    AllocatedBuffer vertexDynamicsBuffer; //Vertex velocity and mass
    AllocatedBuffer correctionBuffer; //Jacobi solver correction buffer
    
    VkDescriptorSetLayout computeSolveDSLayout;
    VkDescriptorSetLayout uniformComputeDSLayout;
    
    VkPipelineLayout computeSolvePLayout;
    
    VkDescriptorSet computePreSolveDSet;
    VkDescriptorSet computePostSolveDSet;

    VkPipeline computePreSolvePipeline;
    VkPipeline computePostSolvePipeline;
    
    void init_solver_pipelines();
    void init_solver_descriptor_sets();
    
public:
    virtual void solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps)=0;
    PBDMesh(){};
    virtual ~PBDMesh(){};
    virtual void clear_resources(){};
};

class IConstraint2D{
public:
    AllocatedBuffer edgeConstraintsBuffer;
    std::vector<EdgeConstraint> constraints;
    std::vector<std::vector<EdgeConstraint>> coloringConstraints;
    
    VkDescriptorSetLayout compute2DDSLayout;
    std::vector<VkDescriptorSet> compute2DConstraintsDSet;
    
    VkPipelineLayout compute2DPLayout;
    VkPipeline compute2DPipeline;
    
    void create_edge_descriptor_sets(VulkanEngine* engine);
    void create_edge_pipeline(VulkanEngine* engine);
    void solve_edge_constraints(VkCommandBuffer cmd);
};

class IConstraint3D{
public:
    AllocatedBuffer volumeConstraintsBuffer;
    std::vector<VolumeConstraint> vConstraints;
    std::vector<std::vector<VolumeConstraint>> coloringVConstraints;
    
    VkDescriptorSetLayout compute3DDSLayout;
    std::vector<VkDescriptorSet> compute3DConstraintsDSet;
    
    VkPipelineLayout compute3DPLayout;
    VkPipeline compute3DPipeline;
    
    void create_volume_descriptor_sets(VulkanEngine* engine);
    void create_volume_pipeline(VulkanEngine* engine);
    void solve_volume_constraints(VkCommandBuffer cmd);
};

class PBDMesh2D : public PBDMesh, public IConstraint2D{
public:
    virtual void solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps) override;
    PBDMesh2D(){};
    ~PBDMesh2D(){clear_resources();};
    PBDMesh2D(VulkanEngine *engine, float sideSize = 10.f, int subdivisions = 10); //create square mesh
    void clear_resources() override;
    
private:
    void init_pipelines();
    void init_descriptors();
    void init_mesh(float sideSize = 10.f, int subdivisions = 10);
};

class PBDMesh3D : public PBDMesh, public IConstraint2D, public IConstraint3D{
public:
    virtual void solve_constraints(VkCommandBuffer cmd, VkDescriptorSet uniformSceneDSet, int solverSteps) override;
    PBDMesh3D(){};
    ~PBDMesh3D(){clear_resources();};
    PBDMesh3D(VulkanEngine *engine, float sideSize = 10.f, int subdivisions = 10); //create cube mesh
    void clear_resources() override;
    
private:
    void init_pipelines();
    void init_descriptors();
    void init_mesh(float sideSize = 10.f, int subdivisions = 10);
};


