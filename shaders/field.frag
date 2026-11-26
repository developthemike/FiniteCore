#version 450
#extension GL_GOOGLE_include_directive : require
#include "colormaps.glsl"

layout(location = 0) in float vT;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    vec2 scale;
    vec2 offset;
    float tmin;
    float tmax;
    int colormap;
    int contours;
} pc;

void main() {
    float range = pc.tmax - pc.tmin;
    float s = (vT - pc.tmin) / (abs(range) > 1e-30 ? range : 1.0);
    vec3 c = colormap(pc.colormap, s);

    // Isotherms: anti-aliased lines at interior levels of the colour range.
    if (pc.contours > 0) {
        float v = s * float(pc.contours);
        float w = fwidth(v);
        float k = floor(v + 0.5);
        if (k >= 1.0 && k <= float(pc.contours) - 1.0 && w > 1e-6) {
            float d = abs(v - k) / w;   // ~distance to the level line in pixels
            float line = 1.0 - smoothstep(0.5, 1.5, d);
            c *= 1.0 - 0.36 * line;
        }
    }
    outColor = vec4(c, 1.0);
}
