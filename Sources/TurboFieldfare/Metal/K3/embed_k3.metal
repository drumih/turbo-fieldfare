#include <metal_stdlib>
using namespace metal;

// ============================================================================
// embed_k3 — Kimi K3 embedding gather + lm_head GEMV (int8 affine g64).
//
//   k3_embed_gather_int8   out[d] = dequant(table[tokenID][d])   (fp16 out)
//   k3_lmhead_gemv_int8    logits[m] = W[m] . x                  (fp32 out)
//
// Both read the MLX-style group-64 affine int8 layout shared with the v1
// runtime (dequant_int8.metal): one unsigned byte per weight, BF16 scale +
// bias per 64-weight group, w = q * s + b. The GEMV mirrors
// `dequant_int8_gemv_simd` numerics exactly (per-group fma(s, dot, acc) /
// fma(b, sum_x, acc), simd_sum reduction) but writes FP32 logits — K3 keeps
// the sampler input at FP32 (vocab 163,840; no softcap).
//
// Function constants (90+ block; 50–59 and 70–84 are used by other K3
// modules): 90/91 specialize the lm_head M/N, 92 gates their use; 93
// specializes the gather row width, 94 gates its use.
// ============================================================================

constant constexpr uint kK3Int8GroupSize = 64;

constant uint FC_K3_LMHEAD_M [[function_constant(90)]];
constant uint FC_K3_LMHEAD_N [[function_constant(91)]];
constant bool FC_K3_LMHEAD_USE_FC [[function_constant(92)]];
constant uint FC_K3_EMBED_D [[function_constant(93)]];
constant bool FC_K3_EMBED_USE_FC [[function_constant(94)]];

static inline uint k3_lmhead_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_K3_LMHEAD_USE_FC) &&
            FC_K3_LMHEAD_USE_FC &&
            is_function_constant_defined(FC_K3_LMHEAD_M)) ? FC_K3_LMHEAD_M : M;
}

static inline uint k3_lmhead_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_K3_LMHEAD_USE_FC) &&
            FC_K3_LMHEAD_USE_FC &&
            is_function_constant_defined(FC_K3_LMHEAD_N)) ? FC_K3_LMHEAD_N : N;
}

static inline uint k3_embed_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_K3_EMBED_USE_FC) &&
            FC_K3_EMBED_USE_FC &&
            is_function_constant_defined(FC_K3_EMBED_D)) ? FC_K3_EMBED_D : D;
}

// Gather + dequant one row of the int8 embedding table. One thread per
// element; the row is D <= a few thousand elements, so a handful of
// threadgroups. `table` is [V, D] packed bytes, scales/biases [V, D/64].
[[kernel, max_total_threads_per_threadgroup(256)]]
void k3_embed_gather_int8(
    device const uint8_t* table   [[buffer(0)]],   // [V, D] packed
    device const bfloat*  scales  [[buffer(1)]],   // [V, D/64]
    device const bfloat*  biases  [[buffer(2)]],   // [V, D/64]
    device       half*    out     [[buffer(3)]],   // [D] fp16
    constant     uint&    tokenID [[buffer(4)]],
    constant     uint&    D       [[buffer(5)]],
    uint                  idx     [[thread_position_in_grid]]
) {
    const uint DD = k3_embed_fc_d(D);
    if (idx >= DD) return;
    const uint n_groups = DD / kK3Int8GroupSize;
    device const uint8_t* row = table + uint(tokenID) * DD;
    device const bfloat*  s_row = scales + uint(tokenID) * n_groups;
    device const bfloat*  b_row = biases + uint(tokenID) * n_groups;
    const uint g = idx / kK3Int8GroupSize;
    const float w = float(uint(row[idx])) * float(s_row[g]) + float(b_row[g]);
    out[idx] = half(w);
}

// y[m] = W[m] . x with FP32 accumulation and FP32 output. One SIMD group per
// output row, eight rows per threadgroup — the `dequant_int8_gemv_simd`
// shape and per-group numerics, differing only in the output dtype.
[[kernel, max_total_threads_per_threadgroup(256)]]
void k3_lmhead_gemv_int8(
    device const uint8_t* W      [[buffer(0)]],   // [M, N] packed
    device const bfloat*  scales [[buffer(1)]],   // [M, N/64]
    device const bfloat*  biases [[buffer(2)]],   // [M, N/64]
    device const half*    x      [[buffer(3)]],   // [N] fp16
    device       float*   y      [[buffer(4)]],   // [M] fp32
    constant     uint&    M      [[buffer(5)]],
    constant     uint&    N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint MM = k3_lmhead_fc_m(M);
    const uint NN = k3_lmhead_fc_n(N);
    const uint row = tg_idx * 8u + sg_idx;
    if (row >= MM) return;
    const uint n_groups = NN / kK3Int8GroupSize;
    device const uint8_t* W_row = W      + uint(row) * NN;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint i0 = g * kK3Int8GroupSize + lane * 2u;
        const uint i1 = i0 + 1u;
        const float q0 = float(uint(W_row[i0]));
        const float q1 = float(uint(W_row[i1]));
        const float x0 = float(x[i0]);
        const float x1 = float(x[i1]);
        const float dot_qx = q0 * x0 + q1 * x1;
        const float sum_x  = x0 + x1;
        acc = fma(s, dot_qx, acc);
        acc = fma(b, sum_x,  acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}
