#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 uv;

layout (binding = 1) uniform sampler2D samplerColor[1];

layout (location = 0) out vec4 uFragColor;

layout(push_constant) uniform PER_OBJECT
{
    int imgIdx;
}pc;

void main() {
    vec4 texture_color = texture(samplerColor[pc.imgIdx], uv);
    uFragColor = texture_color;
//    uFragColor = vec4(float(pc.imgIdx), uv.x, uv.y, 1.0);
}
