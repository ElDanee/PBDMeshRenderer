#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require

#include "types.glsl"

//size of a workgroup for compute
layout (local_size_x = 64) in;

layout(set = 0, binding = 0) uniform UniformSceneData{
    vec3 acceleration;
    float timeStep;
} uData;


layout(set = 1, binding = 0) buffer VerticesF{
    VertexData pos[];
} currentPositions;

layout(set = 1, binding = 1) readonly buffer VerticesB{
    VertexData pos[];
} intermediatePositions;

layout(set = 1, binding = 2) buffer Velocities{
    VertexDynamics data[];
} velocities;

layout(set = 1, binding = 3) buffer Corrections{
    corrStruct data[];
} corrections;

const vec3 sCenter1 = vec3(0,2.5,0);
const float radius1 = 1.f;
const vec3 sCenter2 = vec3(0,5,0);
const float radius2 = 3.f;


void main(){
    int texelCoord = int(gl_GlobalInvocationID.x);
    
    if(texelCoord < currentPositions.pos.length()){
        //GS Solver
        vec3 f = intermediatePositions.pos[texelCoord].pos;
        //Jacobi solver
        //vec3 f = intermediatePositions.pos[texelCoord].pos + corrections.data[texelCoord].dP * 0.2;
        
        //Sphere check
        bool sphereCheck = true;
        if(sphereCheck && length(f - sCenter1) <= radius1){
            f = sCenter1 + normalize(f- sCenter1) * (radius1);
            currentPositions.pos[texelCoord].pos = f;
        }
        if(sphereCheck && length(f - sCenter2) <= radius2){
            f = sCenter2 + normalize(f- sCenter2) * (radius2);
            currentPositions.pos[texelCoord].pos = f;
        }
        
        //Plane check
        if(true && f.y >= 10.f){
            velocities.data[texelCoord].velocity = vec3(0);
            f.y = 10.f;
        }
        else{
            velocities.data[texelCoord].velocity = (f - currentPositions.pos[texelCoord].pos) /(uData.timeStep);
        }
        currentPositions.pos[texelCoord].pos = f;
    }
}
