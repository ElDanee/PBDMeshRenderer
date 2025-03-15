#version 450

#extension GL_GOOGLE_include_directive : require

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

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

void main()
{
    vec3 ambient = vec3(0.1f);
    vec3 color = inColor * max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.5f);
    outFragColor = vec4(0.9*color + ambient, 1.0f);
}
