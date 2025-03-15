#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require

#include "types.glsl"


//size of a workgroup for compute
layout (local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer ConstraintsVal{
    float volumes[];
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
    
    uint stride = constraintsVal.volumes.length();
    
    if (texelCoord < stride){
        VertexData v1 = vertices.pos[constraintsIdx.indices[texelCoord]];
        VertexData v2 = vertices.pos[constraintsIdx.indices[texelCoord + stride]];
        VertexData v3 = vertices.pos[constraintsIdx.indices[texelCoord + stride * 2]];
        VertexData v4 = vertices.pos[constraintsIdx.indices[texelCoord + stride * 3]];
        
        float v = dot(cross(v2.pos - v1.pos,v3.pos - v1.pos), (v4.pos - v1.pos))/6;
        float dP = - (v - constraintsVal.volumes[texelCoord]);
        if(abs(dP)<pow(10,-6)) return;
        
        float w1 = v1.invMass;
        float w2 = v2.invMass;
        float w3 = v3.invMass;
        float w4 = v4.invMass;
        
        float alfa = (pow(10,-8)/uData.timeStep)/uData.timeStep;
        float beta = pow(10.f, 3.f) * uData.timeStep * uData.timeStep;
        float gamma = alfa * beta;
        
        vec3 grad2 = cross(v3.pos - v1.pos, v4.pos - v1.pos);
        vec3 grad3 = cross(v4.pos - v1.pos, v2.pos - v1.pos);
        vec3 grad4 = cross(v2.pos - v1.pos, v3.pos - v1.pos);
        vec3 grad1 = -(grad2 + grad3 + grad4);
        
        float K = w1 * dot(grad1, grad1) +
                    w2 * dot(grad2,grad2) +
                    w3 * dot(grad3, grad3) +
                    w4 * dot(grad4, grad4);
        
        K *= 1 + (gamma/uData.timeStep);
        K += alfa;
        if(K <= pow(10,-6)) return;
        
        vertices.pos[constraintsIdx.indices[texelCoord]].pos +=
        (dP - gamma * dot(grad1, velocities.vel[constraintsIdx.indices[texelCoord]].velocity)) * grad1/K * w1;
        vertices.pos[constraintsIdx.indices[texelCoord + stride]].pos +=
        (dP - gamma * dot(grad2, velocities.vel[constraintsIdx.indices[texelCoord + stride]].velocity))* grad2/K * w2;
        vertices.pos[constraintsIdx.indices[texelCoord + stride * 2]].pos +=
        (dP - gamma * dot(grad3, velocities.vel[constraintsIdx.indices[texelCoord + stride * 2]].velocity))* grad3/K * w3;
        vertices.pos[constraintsIdx.indices[texelCoord + stride * 3]].pos +=
        (dP - gamma * dot(grad4, velocities.vel[constraintsIdx.indices[texelCoord + stride * 3]].velocity)) * grad4/K * w4;
        
        //Undampened
        /*vertices.pos[constraintsIdx.indices[texelCoord]].pos += dP * grad1 * w1;
        vertices.pos[constraintsIdx.indices[texelCoord + stride]].pos += dP * grad2 * w2;
        vertices.pos[constraintsIdx.indices[texelCoord + stride * 2]].pos += dP * grad3 * w3;
        vertices.pos[constraintsIdx.indices[texelCoord + stride * 3]].pos += dP * grad4 * w4;*/
    }
}
