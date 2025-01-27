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

layout(set = 0, binding = 0) uniform UniformSceneData{
    vec3 acceleration;
    float timeStep;
} uData;

layout(set = 1, binding = 0) readonly buffer VerticesB{
    VertexData pos[];
} basePositions;

layout(set = 1, binding = 1) buffer VerticesF{
    VertexData pos[];
} finalPositions;

layout(set = 1, binding = 2) buffer Velocities{
    VertexDynamics data[];
} velocities;

struct corrStruct{
    vec3 dP;
    float pad;
};

layout(set = 1, binding = 3) buffer Corrections{
    corrStruct data[];
} corrections;

const vec3 sCenter = vec3(5,0,0);
const float radius = 2.f;

void main(){
    int texelCoord = int(gl_GlobalInvocationID.x);
    if(texelCoord < basePositions.pos.length()){
        vec3 f = basePositions.pos[texelCoord].pos + (corrections.data[texelCoord].dP * .25f);// 1.f/corrections.data[texelCoord].pad);//1.f/corrections.data[texelCoord].pad);
        
        //Sphere check
        /*if(length(f - sCenter) <= radius){
            velocities.data[texelCoord].velocity = vec3(0.f);
            f = sCenter + normalize(f - sCenter) * radius;
        }*/
        
        //Plane check
        if(f.y > 10){
            f.y = 10;
            velocities.data[texelCoord].velocity = vec3(0.f);
        } else {
            velocities.data[texelCoord].velocity += corrections.data[texelCoord].dP * .25f;// / (uData.timeStep);
            //velocities.data[texelCoord].velocity = (f - finalPositions.pos[texelCoord].pos) * .25f / (uData.timeStep);
        }
        
        finalPositions.pos[texelCoord].pos = f;
    }
}
