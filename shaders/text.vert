#version 450

// One instance per character: a quad covering the 5x9 glyph cell
// (7 rows above the baseline + 2 descender rows).
layout(location = 0) in vec2 inPos;     // top-left corner, pixels
layout(location = 1) in uint inGlyph;
layout(location = 2) in vec4 inColor;   // R8G8B8A8_UNORM
layout(location = 3) in float inScale;  // integer pixel scale

layout(push_constant) uniform PC {
    vec2 pixelToNdc;  // (2/width, 2/height)
} pc;

layout(location = 0) out vec2 vCell;
layout(location = 1) flat out uint vGlyph;
layout(location = 2) flat out vec4 vColor;

const vec2 kCorners[6] = vec2[](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
                                vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

void main() {
    vec2 c = kCorners[gl_VertexIndex];
    vec2 glyph = vec2(5.0, 9.0);
    vec2 p = inPos + c * glyph * inScale;
    gl_Position = vec4(p * pc.pixelToNdc - 1.0, 0.0, 1.0);
    vCell = c * glyph;
    vGlyph = inGlyph;
    vColor = inColor;
}
