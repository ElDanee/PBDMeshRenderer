#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require

//size of a workgroup for compute
layout (local_size_x = 32) in;

layout(set = 0, binding = 0) readonly buffer ConstraintsVal{
    float volumes[];
} constraintsVal;

layout(set = 0, binding = 1) readonly buffer ConstraintsIdx{
    uint indices[];
} constraintsIdx;

struct VertexData{
    vec3 pos;
    float mass;
};

layout(set = 1, binding = 0) buffer Vertices{
    VertexData pos[];
} vertices;

struct corrStruct{
    vec3 dP;
    float pad;
};

layout(set = 1, binding = 1) buffer Corrections{
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
        
        float v = dot(cross((v2.pos - v1.pos), (v3.pos - v1.pos)), (v4.pos - v1.pos))/6;
        
        float w1 = 1/v1.mass;
        float w2 = 1/v2.mass;
        float w3 = 1/v3.mass;
        float w4 = 1/v4.mass;
        
        vec3 grad1 = cross(v4.pos - v2.pos, v3.pos - v2.pos);
        vec3 grad2 = cross(v3.pos - v1.pos, v4.pos - v1.pos);
        vec3 grad3 = cross(v4.pos - v1.pos, v2.pos - v1.pos);
        vec3 grad4 = cross(v2.pos - v1.pos, v3.pos - v1.pos);
        
        float dP = -6*(v - constraintsVal.volumes[texelCoord]);
        dP /=  w1 * pow(length(grad1),2) +
                w2 * pow(length(grad2),2) +
                w3 * pow(length(grad3),2) +
                w4 * pow(length(grad4),2);
        
        
        
        corrections.data[constraintsIdx.indices[texelCoord]].dP += dP * w1 * grad1;
        //corrections.data[constraintsIdx.indices[texelCoord]].pad += 1;
        corrections.data[constraintsIdx.indices[texelCoord + stride]].dP += dP * w2 * grad2;
        //corrections.data[constraintsIdx.indices[texelCoord + stride]].pad += 1;
        corrections.data[constraintsIdx.indices[texelCoord + stride * 2]].dP += dP * w3 * grad3;
        //corrections.data[constraintsIdx.indices[texelCoord + stride * 2]].pad += 1;
        corrections.data[constraintsIdx.indices[texelCoord + stride * 3]].dP += dP * w4 * grad4;
        //corrections.data[constraintsIdx.indices[texelCoord + stride * 3]].pad += 1;
    }
}
