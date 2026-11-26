#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec3 vWorld;

void main() {
    vWorld = inPos;
    gl_Position = S.viewProj * vec4(inPos, 1.0);
    // Pull lines slightly towards the camera so they win against the faces
    // they lie on.
    gl_Position.z -= 2e-4 * gl_Position.w;
}
