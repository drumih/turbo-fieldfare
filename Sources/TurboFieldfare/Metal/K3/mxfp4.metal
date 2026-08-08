#include <metal_stdlib>
using namespace metal;

// ============================================================================
// mxfp4 — Kimi K3 routed-expert MXFP4 dequant + simd-group GEMV.
//
// Format (QuantizationMXFP4.swift is the canonical contract; this file must
// match it bit-for-bit):
//   packed  : N/2 bytes per row, two E2M1 nibbles per byte, LOW nibble = even
//             element index.
//   scales  : N/32 bytes per row, one E8M0 byte per group of 32 elements.
//             value = 2^(byte - 127); byte 0 is the subnormal 2^-127; byte
//             255 (E8M0 NaN) decodes as 0 and zeroes its group.
//   value   : w[i] = e2m1(nibble[i]) * scale[i/32].
//
// Bit-exactness: Apple GPUs flush subnormals in FP32 arithmetic regardless
// of math mode, so a float multiply can NOT reproduce the reference's
// subnormal products (scale byte 0, or byte 1 with a 0.5 LUT entry). But
// every E2M1 value is (1 or 1.5) x 2^k and every scale is 2^e, so each
// product is exactly representable (down to 2^-128, well inside the
// subnormal range) and is CONSTRUCTED bit-by-bit below — no FP multiply on
// the dequant path. The K3 shader library is additionally compiled with
// safe math (see K3MetalLibrary.swift) so transcendentals are precise and
// nothing reassociates.
//
// Row-dot factoring: lut values and the E8M0 scale are exact powers-of-two
// products, so sum_k (lut_k * s) * x_k = s * sum_k(lut_k * x_k) is computed
// as one FMA per group on top of an unscaled 32-element dot, mirroring the
// affine factoring in dequant_int4.metal. (FTZ on a subnormal scale there
// drops a ~1e-38 contribution — invisible at GEMV tolerances.)
// ============================================================================

#ifndef K3_MXFP4_COMMON
#define K3_MXFP4_COMMON

constant constexpr uint kK3Mxfp4GroupSize = 32;
constant constexpr uint kK3Mxfp4GroupBytes = 16;

// E2M1 decode table indexed by the raw nibble (sign << 3 | exponent << 1 |
// mantissa). Index 8 is negative zero.
constant constexpr float kK3E2M1LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float k3_e2m1_value(uint nibble) {
    return kK3E2M1LUT[nibble & 0xFu];
}

// E8M0 scale: 2^(byte - 127) via the exponent-field bit trick (exact for
// bytes 1...254); byte 0 is the subnormal 2^-127; byte 255 zeroes the group.
static inline float k3_e8m0_decode(uint8_t b) {
    if (b == 255u) { return 0.0f; }
    if (b == 0u)   { return as_type<float>(0x00400000u); }
    return as_type<float>(uint(b) << 23);
}

// Exact e2m1(nibble) * e8m0(scale) via integer bit construction — the
// bit-exact counterpart of QuantizationMXFP4's `LUT[...] * decodeScale(...)`.
// E2M1: nibble = sign<<3 | E<<1 | M. E=0 is subnormal-style (magnitude
// M x 0.5, NO implicit leading 1); E>0 is (1 + M/2) x 2^(E-1). Every product
// with 2^e is exactly representable (down to 2^-128, well inside the
// subnormal range) and is CONSTRUCTED bit-by-bit — a float multiply would
// FTZ below 2^-126 on the GPU.
static inline float k3_e2m1_times_e8m0(uint nibble, uint8_t scale_byte) {
    if (scale_byte == 255u) {
        // Zeroed group: keep the reference's signed-zero product
        // (-0 LUT entry x +0 scale = -0).
        return kK3E2M1LUT[nibble & 0xFu] * 0.0f;
    }
    const uint mag = nibble & 0x7u;
    if (mag == 0u) {
        return kK3E2M1LUT[nibble & 0xFu];  // +/-0 times a positive scale
    }
    // The M bit contributes 0.5 to the significand only when E > 0.
    const uint m_bit = (mag > 1u) ? (mag & 1u) : 0u;
    const int e = (scale_byte == 0u) ? -127 : (int(scale_byte) - 127);
    const int k = int(mag >> 1) - 1;               // E - 1
    const int eTot = k + e;                        // 2^eTot scaling
    const uint sign = (nibble & 0x8u) << 28;       // bit 3 -> bit 31
    uint bits;
    if (eTot >= 128) {
        // Overflow: m x 2^eTot exceeds the fp32 range — the reference's
        // float multiply rounds this to infinity.
        bits = sign | 0x7F800000u;
    } else if (eTot >= -126) {
        // Normal: exponent field eTot+127, mantissa bit 22 = M.
        bits = sign | (uint(eTot + 127) << 23) | (m_bit << 22);
    } else {
        // Subnormal: value = mantissa_int x 2^-149. Only eTot -128/-127
        // occur (k >= -1, e >= -127), so no precision is ever lost.
        bits = sign | (m_bit != 0u ? (3u << uint(eTot + 148))
                                   : (2u << uint(eTot + 148)));
    }
    return as_type<float>(bits);
}

// SiTU-GLU halves (fp32; docs/K3_DATAFLOW.md "Primitives"):
//   gate half: 4 * tanh(g / 4) * sigmoid(g)   — sigmoid reads the UNCAPPED g
//   up half  : 25 * tanh(u / 25)
// The tanh argument is clamped like gelu_pytorch_tanh's: tanh saturates to
// exactly +/-1 in fp32 well below |20|, so the clamp is a NaN guard only.
static inline float k3_situ_gate(float g) {
    const float t = tanh(clamp(g / 4.0f, -20.0f, 20.0f));
    const float sg = 1.0f / (1.0f + exp(-g));
    return 4.0f * t * sg;
}

static inline float k3_situ_up(float u) {
    return 25.0f * tanh(clamp(u / 25.0f, -20.0f, 20.0f));
}

static inline float k3_situ_and_mul(float g, float u) {
    return k3_situ_gate(g) * k3_situ_up(u);
}

// Unscaled 32-element dot of one MXFP4 group against x (any float-castable
// element type read through a caller-supplied loader is overkill; the two
// variants below cover fp16 and fp32 x). Weight reads use ushort loads:
// packed rows need only be 2-byte aligned (enforced by the dispatchers).
static inline float k3_mxfp4_group_dot_f16x(
    device const uint8_t* w_group,
    device const half*    x_group
) {
    device const ushort* wp = (device const ushort*)w_group;
    float dot = 0.0f;
    for (uint b = 0; b < 8u; ++b) {
        const uint w2 = uint(wp[b]);
        dot = fma(k3_e2m1_value(w2 & 0xFu),         float(x_group[b * 4u]),      dot);
        dot = fma(k3_e2m1_value((w2 >> 4) & 0xFu),  float(x_group[b * 4u + 1u]), dot);
        dot = fma(k3_e2m1_value((w2 >> 8) & 0xFu),  float(x_group[b * 4u + 2u]), dot);
        dot = fma(k3_e2m1_value(w2 >> 12),          float(x_group[b * 4u + 3u]), dot);
    }
    return dot;
}

static inline float k3_mxfp4_group_dot_f32x(
    device const uint8_t* w_group,
    device const float*   x_group
) {
    device const ushort* wp = (device const ushort*)w_group;
    float dot = 0.0f;
    for (uint b = 0; b < 8u; ++b) {
        const uint w2 = uint(wp[b]);
        dot = fma(k3_e2m1_value(w2 & 0xFu),         x_group[b * 4u],      dot);
        dot = fma(k3_e2m1_value((w2 >> 4) & 0xFu),  x_group[b * 4u + 1u], dot);
        dot = fma(k3_e2m1_value((w2 >> 8) & 0xFu),  x_group[b * 4u + 2u], dot);
        dot = fma(k3_e2m1_value(w2 >> 12),          x_group[b * 4u + 3u], dot);
    }
    return dot;
}

// One SIMD lane striding over the groups of a single row; caller reduces
// with simd_sum. `w_row`/`s_row` point at the row start; n_groups = N/32.
static inline float k3_mxfp4_row_dot_f16x(
    device const uint8_t* w_row,
    device const uint8_t* s_row,
    device const half*    x,
    uint n_groups,
    uint lane
) {
    float acc = 0.0f;
    for (uint g = lane; g < n_groups; g += 32u) {
        const float s = k3_e8m0_decode(s_row[g]);
        const float dot = k3_mxfp4_group_dot_f16x(
            w_row + g * kK3Mxfp4GroupBytes, x + g * kK3Mxfp4GroupSize);
        acc = fma(s, dot, acc);
    }
    return acc;
}

static inline float k3_mxfp4_row_dot_f32x(
    device const uint8_t* w_row,
    device const uint8_t* s_row,
    device const float*   x,
    uint n_groups,
    uint lane
) {
    float acc = 0.0f;
    for (uint g = lane; g < n_groups; g += 32u) {
        const float s = k3_e8m0_decode(s_row[g]);
        const float dot = k3_mxfp4_group_dot_f32x(
            w_row + g * kK3Mxfp4GroupBytes, x + g * kK3Mxfp4GroupSize);
        acc = fma(s, dot, acc);
    }
    return acc;
}

// Fused w1/w3 row pair sharing the fp16 x reads (LatentMoE phase 1 shape:
// both matrices are F x D_lat). Returns (w1 dot, w3 dot), pre-simd_sum.
static inline float2 k3_mxfp4_dual_row_dot_f16x(
    device const uint8_t* w1_row,
    device const uint8_t* s1_row,
    device const uint8_t* w3_row,
    device const uint8_t* s3_row,
    device const half*    x,
    uint n_groups,
    uint lane
) {
    float g_acc = 0.0f;
    float u_acc = 0.0f;
    for (uint g = lane; g < n_groups; g += 32u) {
        const uint byte0 = g * kK3Mxfp4GroupBytes;
        const uint elem0 = g * kK3Mxfp4GroupSize;
        const float gs = k3_e8m0_decode(s1_row[g]);
        const float us = k3_e8m0_decode(s3_row[g]);
        const float g_dot = k3_mxfp4_group_dot_f16x(w1_row + byte0, x + elem0);
        const float u_dot = k3_mxfp4_group_dot_f16x(w3_row + byte0, x + elem0);
        g_acc = fma(gs, g_dot, g_acc);
        u_acc = fma(us, u_dot, u_acc);
    }
    return float2(g_acc, u_acc);
}

#endif // K3_MXFP4_COMMON

constant uint FC_K3_MXFP4_M [[function_constant(50)]];
constant uint FC_K3_MXFP4_N [[function_constant(51)]];
constant bool FC_K3_MXFP4_USE_FC [[function_constant(52)]];

static inline uint k3_mxfp4_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_K3_MXFP4_USE_FC) &&
            FC_K3_MXFP4_USE_FC &&
            is_function_constant_defined(FC_K3_MXFP4_M)) ? FC_K3_MXFP4_M : M;
}

static inline uint k3_mxfp4_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_K3_MXFP4_USE_FC) &&
            FC_K3_MXFP4_USE_FC &&
            is_function_constant_defined(FC_K3_MXFP4_N)) ? FC_K3_MXFP4_N : N;
}

// y[m] = sum_n dequant(W[m, n]) * x[n]. One SIMD per row, eight rows per
// threadgroup; lane 0 writes after simd_sum. Requires N % 32 == 0 (validated
// by the dispatcher). FP32 accumulation throughout.
static inline void k3_mxfp4_gemv_body(
    device const uint8_t* W,
    device const uint8_t* scales,
    device const half*    x,
    device half*          y,
    uint M,
    uint N,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kK3Mxfp4GroupSize;
    const uint row_bytes = N / 2u;
    float acc = k3_mxfp4_row_dot_f16x(
        W + uint(row) * row_bytes,
        scales + uint(row) * n_groups,
        x, n_groups, lane);
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

kernel void k3_mxfp4_gemv(
    device const uint8_t* W      [[buffer(0)]],
    device const uint8_t* scales [[buffer(1)]],
    device const half*    x      [[buffer(2)]],
    device half*          y      [[buffer(3)]],
    constant uint&        M      [[buffer(4)]],
    constant uint&        N      [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    k3_mxfp4_gemv_body(W, scales, x, y,
                       k3_mxfp4_fc_m(M), k3_mxfp4_fc_n(N),
                       rows_per_tg, tg_idx, sg_idx, lane);
}

// FP32-in/FP32-out sibling (phase-2 style activations, router-adjacent
// plumbing, and the f32 parity tests). Same dequant and accumulation.
kernel void k3_mxfp4_gemv_f32(
    device const uint8_t* W      [[buffer(0)]],
    device const uint8_t* scales [[buffer(1)]],
    device const float*   x      [[buffer(2)]],
    device float*         y      [[buffer(3)]],
    constant uint&        M      [[buffer(4)]],
    constant uint&        N      [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = k3_mxfp4_fc_m(M);
    const uint NN = k3_mxfp4_fc_n(N);
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= MM) return;
    const uint n_groups  = NN / kK3Mxfp4GroupSize;
    const uint row_bytes = NN / 2u;
    float acc = k3_mxfp4_row_dot_f32x(
        W + uint(row) * row_bytes,
        scales + uint(row) * n_groups,
        x, n_groups, lane);
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}

// Standalone dequant: out[r, c] = e2m1(nibble) * e8m0(scale). One thread per
// element. The product is built by integer bit construction (see
// k3_e2m1_times_e8m0), making this bit-exact against
// QuantizationMXFP4.dequantizeMxfp4Group — including byte-255 groups zeroing
// (with the reference's signed zero), the byte-0 subnormal scale, and
// overflow-to-infinity at the top of the E8M0 range.
kernel void k3_mxfp4_dequant(
    device const uint8_t* W      [[buffer(0)]],
    device const uint8_t* scales [[buffer(1)]],
    device float*         out    [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = k3_mxfp4_fc_m(M) * k3_mxfp4_fc_n(N);
    if (gid >= total) return;
    const uint NN = k3_mxfp4_fc_n(N);
    const uint row = gid / NN;
    const uint col = gid % NN;
    const uint8_t byte = W[uint(row) * (NN / 2u) + (col >> 1)];
    const uint nibble = (col & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
    const uint8_t scale_byte = scales[
        uint(row) * (NN / kK3Mxfp4GroupSize) + (col / kK3Mxfp4GroupSize)];
    out[gid] = k3_e2m1_times_e8m0(nibble, scale_byte);
}
