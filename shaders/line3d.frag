#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout(location = 0) in vec3 vWorld;
layout(location = 0) out vec4 outColor;

void main() {
    // kind 0: on the exterior (hidden where cut away); 1..3: outline of the cut
    // normal to x/y/z (shown where that cut is shown); 4: always.
    if (draw.kind == 0 && cutAway(vWorld)) discard;
    if (draw.kind >= 1 && draw.kind <= 3 && !sliceVisible(vWorld, draw.kind - 1)) discard;
    outColor = draw.color;
}
