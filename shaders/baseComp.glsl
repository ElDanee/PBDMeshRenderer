#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require

#include "types.glsl"

//size of a workgroup for compute
layout (local_size_x = 32) in;

layout(set = 0, binding = 0) readonly buffer ConstraintsVal{
    float distances[];
} constraintsVal;

layout(set = 0, binding = 1) readonly buffer ConstraintsIdx{
    uint indices[];
} constraintsIdx;

layout(set = 1, binding = 0) buffer Vertices{
    VertexData pos[];
} vertices;

layout(set = 1, binding = 1) buffer Velocities{
    VertexDynamics vel[];
} velocities;

layout(set = 1, binding = 2) buffer Corrections{
    corrStruct data[];
} corrections;

layout(set = 2, binding = 0) uniform UniformSceneData{
    vec3 acceleration;
    float timeStep;
} uData;

void main(){
    
    int texelCoord = int(gl_GlobalInvocationID.x);
    uint stride = constraintsVal.distances.length();
    
    if (texelCoord < stride){
        //EdgeConstraint edConst = constraints.edges[texelCoord];
        VertexData v1 = vertices.pos[constraintsIdx.indices[texelCoord]];
        VertexData v2 = vertices.pos[constraintsIdx.indices[texelCoord + stride]];
        
        vec3 d = v1.pos - v2.pos;
        float l = length(d);
        vec3 n = normalize(d);
        
        float dP = -(l - constraintsVal.distances[texelCoord]);
        if(abs(dP)<pow(10,-6)) return;
        
        float w1 = v1.invMass;
        float w2 = v2.invMass;
        
        float alfa = (pow(10, -10)/uData.timeStep)/uData.timeStep;//(edConst.elasticity/uData.timeStep)/uData.timeStep/1000;
        float beta = pow(10, 1) * uData.timeStep * uData.timeStep;
        float gamma = alfa * beta;
        
        float K = (w1 + w2);
        K*= 1+ (gamma/uData.timeStep);
        K += alfa;
        
        if(K <= pow(10,-6)) return;
        
        vertices.pos[constraintsIdx.indices[texelCoord]].pos +=
            (dP - gamma * dot(n, velocities.vel[constraintsIdx.indices[texelCoord]].velocity)) * n/K *w1;
        
        vertices.pos[constraintsIdx.indices[texelCoord + stride]].pos +=
            (dP - gamma * dot(-n, velocities.vel[constraintsIdx.indices[texelCoord + stride]].velocity)) * -n/K *w2;
        
        //Undampened
        //vertices.pos[constraintsIdx.indices[texelCoord + stride]].pos += dP * n * w1;
        //vertices.pos[constraintsIdx.indices[texelCoord + stride]].pos += -dP * n * w2;
    }
}
