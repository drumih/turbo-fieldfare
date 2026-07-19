#include <metal_stdlib>
using namespace metal;

// ============================================================================
// rmsnorm — RMS-norm over hidden dim.
//
//   inv     = rsqrt(mean(x[i]^2) + eps)
//   y[i]    = x[i] * inv * weight[i]
//
// FP32 accumulator (numerical stability: D=2816 with FP16 inputs can overflow
// FP16 sum-of-squares once activations grow past ~1.2 in magnitude).
// FP16 storage in and out. Learned weights are BF16 where present.
//
// Dispatch: one threadgroup per row, 256 threads per group. Two-stage block
// reduce — SIMD-group simd_sum, then a single SIMD-group merges the partials.
// ============================================================================

// Threadgroup memory carries at most simdgroups_per_threadgroup = 256/32 = 8
// partial sums. Slot 0 is reused after the merge to broadcast the final inv.
constant constexpr uint kRmsMaxSimdGroups = 8;
constant uint FC_RMS_D [[function_constant(30)]];
constant bool FC_RMS_USE_FC [[function_constant(31)]];

static inline uint rms_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_RMS_USE_FC) &&
            FC_RMS_USE_FC &&
            is_function_constant_defined(FC_RMS_D)) ? FC_RMS_D : D;
}

// Common block reduction. Returns `inv = rsqrt(mean(x^2) + eps)` broadcast to
// every thread via threadgroup memory slot 0.
static inline float rms_block_inv(
    device const half* x,
    uint  D,
    float eps,
    uint  lid,
    uint  lsize,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

// Gemma 4 RMS norms ship as BF16 weight vectors (all 30 layers' input /
// post-attn / pre-FFN / post-FFN norms, plus q/k_norm). Math is identical to
// the no-scale form below, with a learned weight applied after normalization.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w(
    device const half*   x          [[buffer(0)]],   // [D] FP16
    device const bfloat* weight     [[buffer(1)]],   // [D] BF16
    device       half*   out        [[buffer(2)]],   // [D] FP16
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        float wv = float(weight[i]);
        out[i] = half(xv * inv * wv);
    }
}

// Gemma 4 applies q_norm/k_norm
// (BF16 weight, shared across heads) and v_norm (no-scale) to each attention
// head independently. These kernels process all heads in one dispatch, with
// one threadgroup per head, avoiding a chain of tiny serialized encoders.
// Math is identical to the single-row kernels applied per head.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_perhead(
    device const half*   x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device const bfloat* weight     [[buffer(1)]],   // [headDim] BF16, shared per head
    device       half*   out        [[buffer(2)]],   // [numHeads * headDim] FP16
    constant     uint&   headDim    [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(weight[i]);
        oh[i] = half(xv * inv * wv);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale_perhead(
    device const half*  x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device       half*  out        [[buffer(1)]],   // [numHeads * headDim] FP16
    constant     uint&  headDim    [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        oh[i] = half(float(xh[i]) * inv);
    }
}

// Gemma 4 v_norm and the MoE router's internal norm are no-scale RMSNorm:
// y[i] = x[i] * rsqrt(mean(x^2) + eps). There is no resident weight tensor.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale(
    device const half*  x          [[buffer(0)]],   // [D] FP16
    device       half*  out        [[buffer(1)]],   // [D] FP16
    constant     uint&  D          [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        out[i] = half(xv * inv);
    }
}
