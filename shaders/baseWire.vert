#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;

layout(set = 0, binding = 0) uniform SceneData{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection; //w for sun power
    vec4 sunlightColor;
    vec4 eyePosition;
    float time;
} sceneData;

struct VertexData{
    vec3 position;
    float pad;
};

struct InstanceData{
    uint v1;
    uint v2;
    uint v3;
    uint pad;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{
    VertexData vertices[];
};

layout(set = 1, binding = 0) buffer Vertices{
    VertexData pos[];
} positions;

layout(set = 1, binding = 1) buffer Instances{
    InstanceData data[];
} instances;

//push constants block
layout( push_constant ) uniform constants
{
    mat4 render_matrix;
    VertexBuffer vertexBuffer;
} PushConstants;

void main()
{
    
    VertexData v  = positions.pos[gl_VertexIndex];
    vec3 oPos = v.position;
    gl_PointSize = 5.f;
    
    gl_Position =  sceneData.viewproj * vec4(v.position, 1.0f);
    
    outColor = vec3(0.8f, 0.0, 0.6f);
}
