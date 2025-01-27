#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require

//size of a workgroup for compute
layout (local_size_x = 32) in;

struct VertexData{
    vec3 pos;
    float pad;
};

struct VertexDynamics{
    vec3 velocity;
    float mass;
};

struct corrStruct{
    vec3 dP;
    float pad;
};

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
        corrections.data[texelCoord].pad = 1; //reset
        
        //Stick to place
        bool stick = false;
        if(stick && (texelCoord < 900)){
            velocities.data[texelCoord].velocity = vec3(0);
            finalPositions.pos[texelCoord].pad = 1000000000.f;
            finalPositions.pos[texelCoord].pos = basePositions.pos[texelCoord].pos;
        } else {
            finalPositions.pos[texelCoord].pad = 1;// velocities.data[texelCoord].mass;
            velocities.data[texelCoord].velocity += uData.acceleration.xyz * uData.timeStep;
            finalPositions.pos[texelCoord].pos = basePositions.pos[texelCoord].pos + uData.timeStep * velocities.data[texelCoord].velocity;
            //finalPositions.pos[texelCoord].pos = basePositions.pos[texelCoord].pos + uData.timeStep * velocities.data[texelCoord].velocity + 0.5f * uData.acceleration * pow(uData.timeStep, 2);
        }
        
        /*velocities.data[texelCoord].velocity += uData.acceleration * uData.timeStep;
        basePositions.pos[texelCoord].pos+= uData.timeStep * velocities.data[texelCoord].velocity;*/
        
        //finalPositions.pos[texelCoord].pos = basePositions.pos[texelCoord].pos + uData.timeStep * velocities.data[texelCoord].velocity + 0.5f * uData.acceleration * pow(uData.timeStep, 2);
        
        //velocities.data[texelCoord].velocity += uData.acceleration * uData.timeStep;
        //finalPositions.pos[texelCoord].pos = basePositions.pos[texelCoord].pos + uData.timeStep * velocities.data[texelCoord].velocity;
    }
}

