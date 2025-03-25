#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec4 position;
layout (location = 1) in vec2 uv;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

layout (location = 0) out vec2 frag_uv;
void main() {
    frag_uv = uv;
    gl_Position = mvp.proj * mvp.view * mvp.model * position;
}
