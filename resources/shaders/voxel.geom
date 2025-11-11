#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (points) in;
layout (triangle_strip, max_vertices = 36) out;

layout (location = 0) in vec2 uvs[];
layout (location = 1) in mat4 out_mvp[];

layout (location = 0) out vec2 frag_uv;

void main() {
    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();
    EndPrimitive();


    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();
    EndPrimitive();


    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();
    EndPrimitive();


    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();
    EndPrimitive();


    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 1.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();
    EndPrimitive();


    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(1.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, -1.0, 1.0));
    frag_uv = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = out_mvp[0] * (gl_in[0].gl_Position + vec4(0.0, 0.0, 0.0, 1.0));
    frag_uv = vec2(0.0, 0.0);
    EmitVertex();
    EndPrimitive();
}
