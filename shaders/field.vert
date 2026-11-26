#version 450

// Nodal positions + nodal temperatures (the solver's T buffer bound directly
// as a vertex buffer, so no copy is needed between compute and graphics).
layout(location = 0) in vec2 inPos;
layout(location = 1) in float inT;

layout(push_constant) uniform PC {
    vec2 scale;
    vec2 offset;
    float tmin;
    float tmax;
    int colormap;
    int contours;
} pc;

layout(location = 0) out float vT;

void main() {
    gl_Position = vec4(inPos * pc.scale + pc.offset, 0.0, 1.0);
    // Linear interpolation of the nodal values across each triangle is exactly
    // the P1 finite element field.
    vT = inT;
}
