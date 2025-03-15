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

layout(set = 1, binding = 0) buffer VerticesB{
    VertexData pos[];
} basePositions;

layout(set = 1, binding = 1) buffer VerticesF{
    VertexData pos[];
} finalPositions;

layout(set = 1, binding = 2) buffer Velocities{
    VertexDynamics data[];
} velocities;

layout(set = 1, binding = 3) writeonly buffer Corrections{
    corrStruct data[];
} corrections;

void main(){
    int texelCoord = int(gl_GlobalInvocationID.x);
    if(texelCoord < basePositions.pos.length()){
        corrections.data[texelCoord].dP = vec3(0.f); //reset
        corrections.data[texelCoord].pad = 0; //reset
        //Stick to place
        bool stick = true;
        if(stick && (texelCoord <  100 )){ //1089 //16335 //19602
            velocities.data[texelCoord].velocity = vec3(0);
            finalPositions.pos[texelCoord].invMass = 0.f;
            finalPositions.pos[texelCoord].pos = basePositions.pos[texelCoord].pos;
        } else {
            //RK4 integration
            finalPositions.pos[texelCoord].invMass = velocities.data[texelCoord].invMass;
            
            vec3 k1_v = velocities.data[texelCoord].velocity + uData.acceleration.xyz * uData.timeStep;
            
            vec3 k2_v = velocities.data[texelCoord].velocity + uData.acceleration.xyz * uData.timeStep/2;
            
            vec3 k3_v = k2_v + uData.acceleration.xyz * uData.timeStep/2;
            
            vec3 k4_v = k3_v + uData.acceleration.xyz * uData.timeStep;
            
            velocities.data[texelCoord].velocity = (k1_v + 2*k2_v + 2*k3_v + k4_v)/6;
            
            //Simple integration
            //velocities.data[texelCoord].velocity += uData.acceleration.xyz * uData.timeStep;
            
            //Position estimate
            finalPositions.pos[texelCoord].pos = basePositions.pos[texelCoord].pos + velocities.data[texelCoord].velocity * uData.timeStep;
        }
    }
}

