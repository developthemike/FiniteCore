// Shared declarations for the FEM compute kernels (included by every *.comp file).
//
// Every kernel walks the rows/nodes with a grid-stride loop, and the host never
// launches more than MAX_GROUPS workgroups. That keeps each per-workgroup
// partial-sum array no longer than one workgroup, so the *next* kernel can
// finish a global reduction by itself instead of paying for an extra dispatch.

#define WG_SIZE 256
#define MAX_GROUPS 256

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0)  readonly buffer RowPtrBuf  { uint  rowPtr[];  };
layout(std430, set = 0, binding = 1)  readonly buffer ColIdxBuf  { uint  colIdx[];  };
layout(std430, set = 0, binding = 2)  readonly buffer ValSBuf    { float valS[];    }; // S = M + theta*dt*A (Dirichlet rows/cols = identity)
layout(std430, set = 0, binding = 3)  readonly buffer ValABuf    { float valA[];    }; // dt*A, Dirichlet rows zeroed
layout(std430, set = 0, binding = 4)  readonly buffer RowSumBuf  { float rowSumA[]; }; // row sums of dt*A
layout(std430, set = 0, binding = 5)  readonly buffer InvDiagBuf { float invDiag[]; }; // Jacobi preconditioner 1/S_ii
layout(std430, set = 0, binding = 6)  readonly buffer LoadBuf    { float loadF[];   }; // dt * (theta F^{n+1} + (1-theta) F^n)
layout(std430, set = 0, binding = 7)  buffer TempBuf { float T[];  };  // nodal temperatures
layout(std430, set = 0, binding = 8)  buffer XBuf    { float vx[]; };  // unknown of the solve: temperature increment
layout(std430, set = 0, binding = 9)  buffer RBuf    { float vr[]; };
layout(std430, set = 0, binding = 10) buffer ZBuf    { float vz[]; };
layout(std430, set = 0, binding = 11) buffer PBuf    { float vp[]; };
layout(std430, set = 0, binding = 12) buffer QBuf    { float vq[]; };
layout(std430, set = 0, binding = 13) buffer PartBuf { float partials[]; }; // 4 slots x MAX_GROUPS
layout(std430, set = 0, binding = 14) buffer ScalarBuf {
    uint  converged;   // 0 = iterating, 1 = converged, 2 = breakdown
    uint  iterations;
    float rr0;         // |r0|^2 at the start of the current step
    float rr;          // latest |r|^2
} sc;

layout(push_constant) uniform PushConstants {
    uint  n;
    uint  numGroups;
    uint  parity;      // CG iteration parity: picks the ping-pong slot holding r.z
    uint  force;       // apply_delta: apply the increment even without convergence
    float tolSq;       // relative tolerance, squared
    float absTolSq;    // absolute floor on |r|^2
} pc;

#define SLOT_PQ  0u
#define SLOT_RZ0 1u
#define SLOT_RZ1 2u
#define SLOT_RR  3u

uint slot(uint s, uint g) { return s * uint(MAX_GROUPS) + g; }
uint gridStride() { return gl_NumWorkGroups.x * uint(WG_SIZE); }

shared vec4 sRed[WG_SIZE];
shared uint sFlag;

// Workgroup-wide sum. Must be reached in uniform control flow.
vec4 workgroupSum(vec4 v) {
    uint lid = gl_LocalInvocationIndex;
    sRed[lid] = v;
    barrier();
    for (uint s = uint(WG_SIZE) / 2u; s > 0u; s >>= 1u) {
        if (lid < s) sRed[lid] += sRed[lid + s];
        barrier();
    }
    vec4 total = sRed[0];
    barrier();
    return total;
}

// Reads the solver status once per workgroup so early exits stay uniform
// within the workgroup (required because the kernels use barrier()).
uint workgroupStatus() {
    if (gl_LocalInvocationIndex == 0u) sFlag = sc.converged;
    barrier();
    return sFlag;
}

// (dt*A * T)_i in difference form: sum_j a_ij (T_j - T_i) + T_i * sum_j a_ij.
// For smooth fields the differences are small and exact-ish in fp32, which
// avoids the catastrophic cancellation of the naive sum.
float applyDtA(uint i) {
    float Ti = T[i];
    float a = rowSumA[i] * Ti;
    uint end = rowPtr[i + 1u];
    for (uint k = rowPtr[i]; k < end; ++k) a += valA[k] * (T[colIdx[k]] - Ti);
    return a;
}
