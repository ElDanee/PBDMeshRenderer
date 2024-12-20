#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require

//size of a workgroup for compute
layout (local_size_x = 32) in;

struct VolumeConstraint{
    uint v1;
    uint v2;
    uint v3;
    uint v4;
    vec3 areas;
    float volume;
};

layout(set = 0, binding = 0) readonly buffer Constraints{
    VolumeConstraint tets[];
} constraints;

struct VertexData{
    vec3 pos;
    float mass;
};

layout(set = 0, binding = 1) buffer Vertices{
    VertexData pos[];
} vertices;

struct corrStruct{
    vec3 dP;
    float pad;
};

layout(set = 0, binding = 2) buffer Corrections{
    corrStruct data[];
} corrections;

void main(){
    int texelCoord = int(gl_GlobalInvocationID.x);
    
    if (texelCoord < constraints.tets.length()){
        VolumeConstraint vConst = constraints.tets[texelCoord];
        VertexData v1 = vertices.pos[vConst.v1];
        VertexData v2 = vertices.pos[vConst.v2];
        VertexData v3 = vertices.pos[vConst.v3];
        VertexData v4 = vertices.pos[vConst.v4];
        
        float v = dot(cross((v2.pos - v1.pos), (v3.pos - v1.pos)), (v4.pos - v1.pos))/6;
        
        float w1 = 1/v1.mass;
        float w2 = 1/v2.mass;
        float w3 = 1/v3.mass;
        float w4 = 1/v4.mass;
        
        vec3 grad1 = cross(v4.pos - v2.pos, v3.pos - v2.pos);
        vec3 grad2 = cross(v3.pos - v1.pos, v4.pos - v1.pos);
        vec3 grad3 = cross(v4.pos - v1.pos, v2.pos - v1.pos);
        vec3 grad4 = cross(v2.pos - v1.pos, v3.pos - v1.pos);
        
        float dP = -6*(v - vConst.volume) / (w1 * pow(length(grad1),2) +
                                             w2 * pow(length(grad2),2) +
                                             w3 * pow(length(grad3),2) +
                                             w4 * pow(length(grad4),2));
        
        corrections.data[vConst.v1].dP += dP * w1 * grad1;
        corrections.data[vConst.v1].pad += 1;
        corrections.data[vConst.v2].dP += dP * w2 * grad2;
        corrections.data[vConst.v2].pad += 1;
        corrections.data[vConst.v3].dP += dP * w3 * grad3;
        corrections.data[vConst.v3].pad += 1;
        corrections.data[vConst.v4].dP += dP * w4 * grad4;
        corrections.data[vConst.v4].pad += 1;
    }
}
