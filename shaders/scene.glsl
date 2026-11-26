// Shared declarations for the 3D scene shaders.

layout(std140, set = 0, binding = 0) uniform Scene {
    mat4 viewProj;
    vec4 eye;     // camera position (xyz)
    vec4 light;   // xyz: direction towards the light, w: ambient fraction
    vec4 cut;     // cut point (xyz)
    vec4 side;    // per axis +1/-1: the half that is cut away (faces the camera)
    ivec4 mode;   // x: 0 surface, 1 clip, 2 octant, 3 slices; y: clip axis
    vec4 range;   // x: tmin, y: tmax
    ivec4 style;  // x: colormap, y: isotherm count
} S;

// Node positions (xyz per node) and the solver's temperature buffer.
layout(std430, set = 0, binding = 1) readonly buffer Positions { float P[]; };
layout(std430, set = 0, binding = 2) readonly buffer Temperatures { float T[]; };

// kind: 0 = exterior surface, 1..3 = slice normal to x/y/z, 4 = always visible
layout(push_constant) uniform Draw {
    vec4 color;
    int kind;
} draw;

vec3 nodePosition(uint i) { return vec3(P[3u * i], P[3u * i + 1u], P[3u * i + 2u]); }

// Signed distances from the cut point, positive on the cut-away side.
vec3 cutDistance(vec3 p) { return (p - S.cut.xyz) * S.side.xyz; }

// Is this exterior point removed by the current cut?
bool cutAway(vec3 p) {
    vec3 d = cutDistance(p);
    if (S.mode.x == 1) return d[S.mode.y] > 0.0;
    if (S.mode.x == 2) return d.x > 0.0 && d.y > 0.0 && d.z > 0.0;
    return false;
}

// Which part of the slice normal to `axis` is shown? In octant mode only the
// faces of the removed corner are visible.
bool sliceVisible(vec3 p, int axis) {
    if (S.mode.x != 2) return true;
    vec3 d = cutDistance(p);
    if (axis == 0) return d.y > 0.0 && d.z > 0.0;
    if (axis == 1) return d.x > 0.0 && d.z > 0.0;
    return d.x > 0.0 && d.y > 0.0;
}
