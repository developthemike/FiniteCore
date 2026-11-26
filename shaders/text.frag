#version 450

layout(location = 0) in vec2 vCell;
layout(location = 1) flat in uint vGlyph;
layout(location = 2) flat in vec4 vColor;

layout(location = 0) out vec4 outColor;

// 9 rows per glyph, low 5 bits of each row, bit 4 = leftmost pixel.
layout(std430, set = 0, binding = 0) readonly buffer Font { uint rows[]; };

void main() {
    int ix = clamp(int(vCell.x), 0, 4);
    int iy = clamp(int(vCell.y), 0, 8);
    uint row = rows[vGlyph * 9u + uint(iy)];
    if (((row >> uint(4 - ix)) & 1u) == 0u) discard;
    outColor = vColor;
}
