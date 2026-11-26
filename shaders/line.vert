#version 450

layout(location = 0) in vec2 inPos;

layout(push_constant) uniform PC {
    vec2 scale;
    vec2 offset;
    vec4 color;
} pc;

void main() {
    gl_Position = vec4(inPos * pc.scale + pc.offset, 0.0, 1.0);
}
