#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_k3 — small utility kernels for the K3 forward runner
// (docs/K3_DATAFLOW.md). Everything here is deliberately simple: one thread
// per element for the elementwise ops, one 256-thread threadgroup for the
// RMSNorms (the house two-stage block reduce). No function constants — these
// run once per model load (buffer dequant) or on vectors at most a few
// thousand elements wide (add / SiTU / norms), so specialization buys
// nothing.
//
//   k3_dequant_int4_buffer / k3_dequant_int8_buffer
//                   Affine g64 (BF16 scale+bias) matrix -> fp16, for the
//                   per-load MLA kv_b plane expansion (k3_mla_kvb_expand
//                   consumes fp16). Nibble order matches the repo's int4
//                   packing convention (low nibble = even element).
//   k3_situ_mul     SiTU-GLU on separate gate/up vectors (the runner issues
//                   two GEMVs instead of one fused cat(gate, up) GEMV):
//                   out = (b1*tanh(g/b1)*sigmoid(g)) . (b2*tanh(u/b2)),
//                   fp32 math on fp16 io, tanh argument clamped at +/-20
//                   exactly like k3_situ_and_mul in moe_k3.metal.
//   k3_add          out = a + b (fp32 math, fp16 io). In-place safe: each
//                   element is read before it is written by the same thread.
//                   Used for the AttnRes prefix updates and the MoE
//                   shared-expert add.
//   k3_rmsnorm_f32w KimiRMSNorm with an FP32 weight (MLA q_a_layernorm):
//                   fp16 in/out, fp32-internal per the reference.
//   k3_rmsnorm_f32x KimiRMSNorm over an FP32 input with a BF16 weight
//                   (routed_expert_norm over the fp32 y_lat), fp16 out for
//                   the up-projection trunk GEMV.
//   k3_cvt_f16_f32  fp16 -> fp32 widening (router input, KDA beta logits).
//   k3_add_bias_f32 out = float(x[i]) + bias[i] (KDA z = f_b(f_a(x)) +
//                   dt_bias), fp16 x, fp32 bias/out.
// ============================================================================

constant constexpr uint kK3UtilThreads = 256;
constant constexpr uint kK3UtilMaxSimdGroups = kK3UtilThreads / 32;
constant constexpr uint kK3AffineGroup = 64;

// y[r * N + i] = dequant(W[r][i]) for one thread per element, int4 variant.
kernel void k3_dequant_int4_buffer(
    device const uint8_t* W      [[buffer(0)]],   // [M, N/2] packed nibbles
    device const bfloat*  scales [[buffer(1)]],   // [M, N/64]
    device const bfloat*  biases [[buffer(2)]],   // [M, N/64]
    device       half*    y      [[buffer(3)]],   // [M, N] fp16
    constant     uint&    M      [[buffer(4)]],
    constant     uint&    N      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M * N) return;
    const uint row = gid / N;
    const uint col = gid % N;
    const uint packedByte = W[row * (N / 2u) + col / 2u];
    const uint q = (col & 1u) == 0u ? (packedByte & 0x0Fu) : (packedByte >> 4u);
    const uint g = col / kK3AffineGroup;
    const uint groups = N / kK3AffineGroup;
    const float w = float(q) * float(scales[row * groups + g])
        + float(biases[row * groups + g]);
    y[gid] = half(w);
}

// Same, int8 variant (one unsigned byte per weight).
kernel void k3_dequant_int8_buffer(
    device const uint8_t* W      [[buffer(0)]],   // [M, N]
    device const bfloat*  scales [[buffer(1)]],   // [M, N/64]
    device const bfloat*  biases [[buffer(2)]],   // [M, N/64]
    device       half*    y      [[buffer(3)]],   // [M, N] fp16
    constant     uint&    M      [[buffer(4)]],
    constant     uint&    N      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M * N) return;
    const uint row = gid / N;
    const uint col = gid % N;
    const uint g = col / kK3AffineGroup;
    const uint groups = N / kK3AffineGroup;
    const float w = float(uint(W[gid])) * float(scales[row * groups + g])
        + float(biases[row * groups + g]);
    y[gid] = half(w);
}

static inline float k3_util_sigmoid(float v) {
    return 1.0f / (1.0f + exp(-v));
}

// SiTU-GLU: numerics identical to k3_situ_and_mul (moe_k3.metal), including
// the +/-20 tanh-argument clamp and the uncapped-gate sigmoid.
kernel void k3_situ_mul(
    device const half*  gate  [[buffer(0)]],   // [n] fp16
    device const half*  up    [[buffer(1)]],   // [n] fp16
    device       half*  out   [[buffer(2)]],   // [n] fp16
    constant     uint&  n     [[buffer(3)]],
    constant     float& beta1 [[buffer(4)]],
    constant     float& beta2 [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const float g = float(gate[gid]);
    const float u = float(up[gid]);
    const float t1 = tanh(clamp(g / beta1, -20.0f, 20.0f));
    const float situGate = beta1 * t1 * k3_util_sigmoid(g);
    const float situUp = beta2 * tanh(clamp(u / beta2, -20.0f, 20.0f));
    out[gid] = half(situGate * situUp);
}

// Elementwise add, fp32 math on fp16 io. In-place safe (out may alias a or b).
kernel void k3_add(
    device const half* a   [[buffer(0)]],   // [n] fp16
    device const half* b   [[buffer(1)]],   // [n] fp16
    device       half* out [[buffer(2)]],   // [n] fp16
    constant     uint& n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = half(float(a[gid]) + float(b[gid]));
}

// House two-stage block reduction (per-SIMD simd_sum, scratch, one SIMD
// merges, broadcast slot doubles as the return value).
static inline float k3_util_block_reduce_sum(
    float v,
    uint simd_lane_id,
    uint simd_group_id,
    uint simdgroups,
    threadgroup float* scratch,
    threadgroup float* bcast
) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}

// KimiRMSNorm, fp16 in/out, FP32 weight (MLA q_a_layernorm). One
// threadgroup, 256 threads.
kernel void k3_rmsnorm_f32w(
    device const half*  x      [[buffer(0)]],   // [D] fp16
    device const float* weight [[buffer(1)]],   // [D] fp32
    device       half*  out    [[buffer(2)]],   // [D] fp16
    constant     uint&  D      [[buffer(3)]],
    constant     float& eps    [[buffer(4)]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]
) {
    threadgroup float scratch[kK3UtilMaxSimdGroups];
    threadgroup float bcast;
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = float(x[i]);
        ss = fma(v, v, ss);
    }
    const float total = k3_util_block_reduce_sum(
        ss, simd_lane_id, simd_group_id, simdgroups, scratch, &bcast);
    const float inv = rsqrt(total / float(D) + eps);
    for (uint i = lid; i < D; i += lsize) {
        out[i] = half(weight[i] * float(x[i]) * inv);
    }
}

// KimiRMSNorm over an FP32 input with a BF16 weight (routed_expert_norm on
// the fp32 y_lat), fp16 out. One threadgroup, 256 threads.
kernel void k3_rmsnorm_f32x(
    device const float*  x      [[buffer(0)]],   // [D] fp32
    device const bfloat* weight [[buffer(1)]],   // [D] bf16
    device       half*   out    [[buffer(2)]],   // [D] fp16
    constant     uint&   D      [[buffer(3)]],
    constant     float&  eps    [[buffer(4)]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]
) {
    threadgroup float scratch[kK3UtilMaxSimdGroups];
    threadgroup float bcast;
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = x[i];
        ss = fma(v, v, ss);
    }
    const float total = k3_util_block_reduce_sum(
        ss, simd_lane_id, simd_group_id, simdgroups, scratch, &bcast);
    const float inv = rsqrt(total / float(D) + eps);
    for (uint i = lid; i < D; i += lsize) {
        out[i] = half(float(weight[i]) * x[i] * inv);
    }
}

// fp16 -> fp32 widening, one thread per element.
kernel void k3_cvt_f16_f32(
    device const half* x   [[buffer(0)]],   // [n] fp16
    device       float* out [[buffer(1)]],  // [n] fp32
    constant     uint& n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = float(x[gid]);
}

// out[i] = float(x[i]) + bias[i]; fp16 x, fp32 bias/out. One thread per
// element.
kernel void k3_add_bias_f32(
    device const half*  x    [[buffer(0)]],   // [n] fp16
    device const float* bias [[buffer(1)]],   // [n] fp32
    device       float* out  [[buffer(2)]],   // [n] fp32
    constant     uint&  n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = float(x[gid]) + bias[gid];
}
