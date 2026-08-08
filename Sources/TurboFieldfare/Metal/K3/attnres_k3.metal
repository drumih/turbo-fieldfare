#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attnres_k3 — Kimi K3 AttnRes (attention residuals), single token.
// docs/K3_DATAFLOW.md "Primitives" and modeling_kimi_linear.py
// _apply_attn_res:
//
//   v        = [blocks ; prefix]              (blocks first, prefix LAST)
//   k_i      = v_i_fp32 * rsqrt(mean(v_i_fp32^2) + eps)   (no norm weight)
//   score_i  = sum_j k_i[j] * score_vec[j]    where score_vec = norm.weight
//                                                     ⊙ proj.weight (fused,
//                                                     precomputed on the CPU)
//   probs    = softmax(scores)                (fp32, max-subtracted)
//   out      = sum_i probs_i * v_i_fp32       (cast back to fp16)
//
// The output REPLACES the stream — there is no residual add. With zero
// blocks the softmax is over the prefix alone, so out == prefix bit-exactly.
//
// One threadgroup (256 threads); at most 9 blocks + 1 prefix = 10 vectors.
// ============================================================================

constant constexpr uint kK3AttnResThreads = 256;
constant constexpr uint kK3AttnResMaxSimdGroups = kK3AttnResThreads / 32;
constant constexpr uint kK3AttnResMaxBlocks = 9;
constant constexpr uint kK3AttnResMaxVectors = kK3AttnResMaxBlocks + 1;

constant uint FC_K3_ATTNRES_HIDDEN [[function_constant(82)]];
constant bool FC_K3_ATTNRES_USE_FC [[function_constant(84)]];

static inline uint k3_attnres_fc_hidden(constant uint& H) {
    return (is_function_constant_defined(FC_K3_ATTNRES_USE_FC) &&
            FC_K3_ATTNRES_USE_FC &&
            is_function_constant_defined(FC_K3_ATTNRES_HIDDEN))
        ? FC_K3_ATTNRES_HIDDEN : H;
}

static inline float k3_attnres_block_reduce_sum(
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

kernel void k3_attnres(
    device const half*  blocks     [[buffer(0)]],   // [num_blocks][hidden] fp16
    device const half*  prefix     [[buffer(1)]],   // [hidden] fp16
    device const float* score_vec  [[buffer(2)]],   // [hidden] fp32 fused
    device half*        out        [[buffer(3)]],   // [hidden] fp16
    constant uint&      hidden     [[buffer(4)]],
    constant uint&      num_blocks [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]
) {
    threadgroup float scores[kK3AttnResMaxVectors];
    threadgroup float probs[kK3AttnResMaxVectors];
    threadgroup float reduce_scratch[kK3AttnResMaxSimdGroups];
    threadgroup float bcast;
    const uint H = k3_attnres_fc_hidden(hidden);
    const uint B = num_blocks;
    if (B > kK3AttnResMaxBlocks) return;
    const uint num_vectors = B + 1;

    // Per vector: k = v * rsqrt(mean(v^2) + eps), score = k . score_vec.
    for (uint vi = 0; vi < num_vectors; ++vi) {
        device const half* v = (vi < B) ? (blocks + uint(vi) * H) : prefix;
        float ss = 0.0f;
        for (uint j = lid; j < H; j += lsize) {
            const float x = float(v[j]);
            ss = fma(x, x, ss);
        }
        const float total = k3_attnres_block_reduce_sum(
            ss, simd_lane_id, simd_group_id, simdgroups,
            reduce_scratch, &bcast);
        const float inv = rsqrt(total / float(H) + eps);

        float sc = 0.0f;
        for (uint j = lid; j < H; j += lsize) {
            sc = fma(float(v[j]) * inv, score_vec[j], sc);
        }
        const float score = k3_attnres_block_reduce_sum(
            sc, simd_lane_id, simd_group_id, simdgroups,
            reduce_scratch, &bcast);
        if (lid == 0) { scores[vi] = score; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Softmax (max-subtracted) on one thread; num_vectors <= 10.
    if (lid == 0) {
        float m = scores[0];
        for (uint vi = 1; vi < num_vectors; ++vi) { m = max(m, scores[vi]); }
        float sum = 0.0f;
        for (uint vi = 0; vi < num_vectors; ++vi) {
            const float e = exp(scores[vi] - m);
            probs[vi] = e;
            sum += e;
        }
        const float inv_sum = 1.0f / sum;
        for (uint vi = 0; vi < num_vectors; ++vi) { probs[vi] *= inv_sum; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted sum of the RAW (unnormalized) vectors.
    for (uint j = lid; j < H; j += lsize) {
        float acc = 0.0f;
        for (uint vi = 0; vi < num_vectors; ++vi) {
            device const half* v = (vi < B) ? (blocks + uint(vi) * H) : prefix;
            acc = fma(probs[vi], float(v[j]), acc);
        }
        out[j] = half(acc);
    }
}
