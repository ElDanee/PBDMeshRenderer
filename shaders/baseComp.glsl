#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require

//size of a workgroup for compute
layout (local_size_x = 32) in;

struct EdgeConstraint{
    float l;
    uint v1;
    uint v2;
    float elasticity; // 0 < e <= 1
};

layout(set = 0, binding = 0) readonly buffer Constraints{
    EdgeConstraint edges[];
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
    
    if (texelCoord < constraints.edges.length()){
        EdgeConstraint edConst = constraints.edges[texelCoord];
        VertexData v1 = vertices.pos[edConst.v1];
        VertexData v2 = vertices.pos[edConst.v2];
        
        vec3 d = v1.pos - v2.pos;
        float l = length(d);
        vec3 n = normalize(d);
        
        float w1 = 1/v1.mass;
        float w2 = 1/v2.mass;
        
        vec3 dP = n * (l - edConst.l) / (w1 + w2);
        
        corrections.data[edConst.v1].dP -= dP * w1;
        corrections.data[edConst.v1].pad += 1;
        corrections.data[edConst.v2].dP += dP * w2;
        corrections.data[edConst.v2].pad += 1;
        
        //vertices.pos[edConst.v1].pos += dP;
        //vertices.pos[edConst.v2].pos -= dP;
        
        /*
        atomicAdd(vertices.pos[edConst.v1].pos, dP);
        atomicAdd(corrections.data[edConst.v1].dP.y, dP.y);
        atomicAdd(corrections.data[edConst.v1].dP.z, dP.z);
        atomicAdd(corrections.data[edConst.v2].dP.x, dP.x);
        atomicAdd(corrections.data[edConst.v2].dP.y, dP.y);
        atomicAdd(corrections.data[edConst.v2].dP.z, dP.z);*/
    }
}
