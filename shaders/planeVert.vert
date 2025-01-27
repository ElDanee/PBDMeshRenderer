#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

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

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;

void main()
{
    if(gl_VertexIndex == 0){
        gl_Position =  sceneData.viewproj * vec4(100.0, 10.f, 100.0, 1.0f);
    } else if(gl_VertexIndex == 1){
        gl_Position =  sceneData.viewproj * vec4(100.0, 10.f, -100.0, 1.0f);
    } else if(gl_VertexIndex == 2){
        gl_Position =  sceneData.viewproj * vec4(-100.0, 10.f, -100.0, 1.0f);
    } else if(gl_VertexIndex == 3){
        gl_Position =  sceneData.viewproj * vec4(-100.0, 10.f, 100.0, 1.0f);
    }
    
    outNormal = vec3(0, -1, 0);
    outColor = vec3(0.f, 0.6, 0.9);
}
