#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

// Planar cross-section through the tetrahedral mesh: every vertex lies on a
// mesh edge (a, b) at parameter s, so both position and temperature are
// linear interpolations - exactly the P1 field restricted to the plane. Only
// the geometry is precomputed; the temperatures come live from the solver.
layout(location = 0) in uvec2 inEdge;
layout(location = 1) in float inS;

layout(location = 0) out vec3 vWorld;
layout(location = 1) out float vT;

void main() {
    vec3 p = mix(nodePosition(inEdge.x), nodePosition(inEdge.y), inS);
    vWorld = p;
    vT = mix(T[inEdge.x], T[inEdge.y], inS);
    gl_Position = S.viewProj * vec4(p, 1.0);
}
