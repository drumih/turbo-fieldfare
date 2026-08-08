#include <metal_stdlib>
using namespace metal;

// ============================================================================
// trunk_k3 — generic-shape BF16 / FP32 trunk GEMVs for the K3 forward runner.
//
// The affine int4/int8 trunk GEMVs are the house `dequant_int4_gemv_simd` /
// `dequant_int8_gemv_simd` (already shape-generic; `K3TrunkGEMV` reuses those
// dispatchers). The house has no unquantized GEMV, so this module fills the
// remaining two formats with the same execution shape: one SIMD group per
// output row, eight rows per threadgroup, FP32 accumulation, two elements
// per lane per 64-wide group (the int8 kernel's loop structure).
//
// Function constants 95/96/97 specialize M/N (90–94 are used by embed_k3).
// Only the fp32 kernel has a canonical consumer (the 896×7168 router); the
// bf16 kernel runs its generic path.
// ============================================================================

constant constexpr uint kK3TrunkGroup = 64;

constant uint FC_K3_TRUNK_M [[function_constant(95)]];
constant uint FC_K3_TRUNK_N [[function_constant(96)]];
constant bool FC_K3_TRUNK_USE_FC [[function_constant(97)]];

static inline uint k3_trunk_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_K3_TRUNK_USE_FC) &&
            FC_K3_TRUNK_USE_FC &&
            is_function_constant_defined(FC_K3_TRUNK_M)) ? FC_K3_TRUNK_M : M;
}

static inline uint k3_trunk_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_K3_TRUNK_USE_FC) &&
            FC_K3_TRUNK_USE_FC &&
            is_function_constant_defined(FC_K3_TRUNK_N)) ? FC_K3_TRUNK_N : N;
}

// y[m] = W[m] . x, W BF16 [M, N], x/y FP16. N must be a multiple of 64.
[[kernel, max_total_threads_per_threadgroup(256)]]
void k3_gemv_bf16(
    device const bfloat* W      [[buffer(0)]],   // [M, N]
    device const half*   x      [[buffer(1)]],   // [N]
    device       half*   y      [[buffer(2)]],   // [M]
    constant     uint&   M      [[buffer(3)]],
    constant     uint&   N      [[buffer(4)]],
    uint                 tg_idx [[threadgroup_position_in_grid]],
    uint                 sg_idx [[simdgroup_index_in_threadgroup]],
    uint                 lane   [[thread_index_in_simdgroup]]
) {
    const uint MM = k3_trunk_fc_m(M);
    const uint NN = k3_trunk_fc_n(N);
    const uint row = tg_idx * 8u + sg_idx;
    if (row >= MM) return;
    device const bfloat* W_row = W + uint(row) * NN;

    float acc = 0.0f;
    for (uint g = 0; g < NN / kK3TrunkGroup; ++g) {
        const uint i0 = g * kK3TrunkGroup + lane * 2u;
        const uint i1 = i0 + 1u;
        acc = fma(float(W_row[i0]), float(x[i0]), acc);
        acc = fma(float(W_row[i1]), float(x[i1]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// y[m] = W[m] . x, W FP32 [M, N], x/y FP32. N must be a multiple of 64.
[[kernel, max_total_threads_per_threadgroup(256)]]
void k3_gemv_f32(
    device const float* W      [[buffer(0)]],   // [M, N]
    device const float* x      [[buffer(1)]],   // [N]
    device       float* y      [[buffer(2)]],   // [M]
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  N      [[buffer(4)]],
    uint                tg_idx [[threadgroup_position_in_grid]],
    uint                sg_idx [[simdgroup_index_in_threadgroup]],
    uint                lane   [[thread_index_in_simdgroup]]
) {
    const uint MM = k3_trunk_fc_m(M);
    const uint NN = k3_trunk_fc_n(N);
    const uint row = tg_idx * 8u + sg_idx;
    if (row >= MM) return;
    device const float* W_row = W + uint(row) * NN;

    float acc = 0.0f;
    for (uint g = 0; g < NN / kK3TrunkGroup; ++g) {
        const uint i0 = g * kK3TrunkGroup + lane * 2u;
        const uint i1 = i0 + 1u;
        acc = fma(W_row[i0], x[i0], acc);
        acc = fma(W_row[i1], x[i1], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}
