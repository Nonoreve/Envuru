#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 uv;

layout (binding = 1) uniform sampler2D samplerColor;

layout (location = 0) out vec4 uFragColor;

void main() {
    vec4 texture_color = texture(samplerColor, uv);
    uFragColor = texture_color;
//    uFragColor = vec4(uv, 0.0, 1.0);
}
