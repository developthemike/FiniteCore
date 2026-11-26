#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"
#include "colormaps.glsl"

layout(location = 0) in vec3 vWorld;
layout(location = 1) in float vT;
layout(location = 0) out vec4 outColor;

void main() {
    // Derivatives first: they must be taken before any invocation discards.
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    float range = S.range.y - S.range.x;
    float s = (vT - S.range.x) / (abs(range) > 1e-30 ? range : 1.0);
    float v = s * float(S.style.y);
    float w = fwidth(v);

    if (draw.kind == 0) {
        if (cutAway(vWorld)) discard;
    } else if (draw.kind <= 3) {
        if (!sliceVisible(vWorld, draw.kind - 1)) discard;
    }

    vec3 c = colormap(S.style.x, s);
    if (S.style.y > 0) {
        float k = floor(v + 0.5);
        if (k >= 1.0 && k <= float(S.style.y) - 1.0 && w > 1e-6) {
            float line = 1.0 - smoothstep(0.5, 1.5, abs(v - k) / w);
            c *= 1.0 - 0.36 * line;
        }
    }

    // Two-sided Lambert lighting with a strong ambient term so that colours
    // stay readable against the colour bar, plus a faint light-dependent sheen
    // so that faces in the darkest part of a colormap still show their shape.
    if (dot(n, S.eye.xyz - vWorld) < 0.0) n = -n;
    float diffuse = max(dot(n, S.light.xyz), 0.0);
    c = c * (S.light.w + (1.0 - S.light.w) * diffuse) + vec3(0.07 * diffuse * diffuse);
    outColor = vec4(min(c, vec3(1.0)), 1.0);
}
