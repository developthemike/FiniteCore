#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

// Exterior surface: drawn with an index buffer of boundary triangles, so the
// vertex index is the mesh node index. No vertex buffers needed.
layout(location = 0) out vec3 vWorld;
layout(location = 1) out float vT;

void main() {
    uint node = uint(gl_VertexIndex);
    vec3 p = nodePosition(node);
    vWorld = p;
    vT = T[node];
    gl_Position = S.viewProj * vec4(p, 1.0);
}
