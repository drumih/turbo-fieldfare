#include <metal_stdlib>
using namespace metal;

// Fused Walsh-Hadamard transform and K4/V4 Lloyd-Max cache packing.
// One threadgroup handles each flattened (token, head) pair.

constant constexpr float kTQCodebook4[16] = {
    -2.7326f, -2.0690f, -1.6181f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6181f,  2.0690f,  2.7326f
};

constant constexpr float kTQDecision4[15] = {
    (-2.7326f + -2.0690f) * 0.5f,
    (-2.0690f + -1.6181f) * 0.5f,
    (-1.6181f + -1.2562f) * 0.5f,
    (-1.2562f + -0.9424f) * 0.5f,
    (-0.9424f + -0.6568f) * 0.5f,
    (-0.6568f + -0.3881f) * 0.5f,
    (-0.3881f + -0.1284f) * 0.5f,
    (-0.1284f +  0.1284f) * 0.5f,
    ( 0.1284f +  0.3881f) * 0.5f,
    ( 0.3881f +  0.6568f) * 0.5f,
    ( 0.6568f +  0.9424f) * 0.5f,
    ( 0.9424f +  1.2562f) * 0.5f,
    ( 1.2562f +  1.6181f) * 0.5f,
    ( 1.6181f +  2.0690f) * 0.5f,
    ( 2.0690f +  2.7326f) * 0.5f
};

inline uint tq_nearest_index4(float v) {
    uint idx = 0;
    idx += (v >= kTQDecision4[ 0]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 1]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 2]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 3]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 4]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 5]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 6]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 7]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 8]) ? 1u : 0u;
    idx += (v >= kTQDecision4[ 9]) ? 1u : 0u;
    idx += (v >= kTQDecision4[10]) ? 1u : 0u;
    idx += (v >= kTQDecision4[11]) ? 1u : 0u;
    idx += (v >= kTQDecision4[12]) ? 1u : 0u;
    idx += (v >= kTQDecision4[13]) ? 1u : 0u;
    idx += (v >= kTQDecision4[14]) ? 1u : 0u;
    return idx;
}

inline float tq_centroid4(uint idx) {
    return kTQCodebook4[idx];
}

// =========================================================================
// Walsh-Hadamard transform
//
// Iterative butterfly, log2(D) stages. Each stage:
//   stride s = 1, 2, 4, ..., D/2
//   for each (i, j=i+s) with (i & s) == 0:
//       a = x[i]; b = x[j]
//       x[i] = (a + b) * INV_SQRT2
//       x[j] = (a - b) * INV_SQRT2
//
// One threadgroup per (token, head). D threads, one element per thread held in
// threadgroup memory. Cross-thread butterflies are handled by reading the
// partner element across the threadgroup with a barrier between stages. The
// largest supported head uses 2 KiB of shared float storage, keeping the fused
// cache-write path bounded.
// =========================================================================

constant constexpr float kInvSqrt2 = 0.7071067811865475f;

struct TurboQuantWHTParams {
    uint num_heads;
    uint layer;
    uint rotation_seed;
    uint apply_rotation;
};

inline ulong tq_mix64(ulong z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

inline uint tq_rotation_sign(uint layer, uint head, uint dim, uint seed) {
    ulong z = ulong(seed);
    z ^= ulong(layer) * 0x9E3779B97F4A7C15ul;
    z ^= ulong(head)  * 0xBF58476D1CE4E5B9ul;
    z ^= ulong(dim)   * 0x94D049BB133111EBul;
    return uint(tq_mix64(z) >> 63);
}

inline float tq_apply_rotation(float v,
                               constant TurboQuantWHTParams& params,
                               uint pair,
                               uint dim) {
    if (params.apply_rotation == 0u) {
        return v;
    }
    const uint head = pair % params.num_heads;
    return tq_rotation_sign(params.layer, head, dim, params.rotation_seed) != 0u ? -v : v;
}

inline float wht_body(float v,
                      threadgroup float* shared,
                      uint lid,
                      uint D,
                      uint simd_lane) {
    // FP32 throughout for stability; D=512 with FP16 amax ~1 produces a
    // post-WHT scale roughly sqrt(D)=22, well within FP16 range, but
    // intermediate sums overflow if accumulated in FP16.
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
        float other = simd_shuffle_xor(v, stride);
        v = ((simd_lane & stride) == 0u)
            ? (v + other) * kInvSqrt2
            : (other - v) * kInvSqrt2;
    }

    shared[lid] = v;
    for (uint stride = 32u; stride < D; stride <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((lid & stride) == 0u) {
            uint j = lid | stride;
            float a = shared[lid];
            float b = shared[j];
            shared[lid] = (a + b) * kInvSqrt2;
            shared[j]   = (a - b) * kInvSqrt2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[lid];
}

struct TurboQuantKVWriteParams {
    uint D;
    uint num_heads;
    uint bytes_per_head;
    uint packed_offset;
    uint scale_offset;
    uint token_base;
    uint bytes_per_token;
};

constant uint FC_TQ_D [[function_constant(100)]];
constant uint FC_TQ_NUM_HEADS [[function_constant(101)]];
constant bool FC_TQ_USE_FC [[function_constant(103)]];

static inline bool tq_use_fc() {
    return is_function_constant_defined(FC_TQ_USE_FC) && FC_TQ_USE_FC;
}

static inline uint tq_kv_fc_d(constant TurboQuantKVWriteParams& params) {
    return (tq_use_fc() && is_function_constant_defined(FC_TQ_D)) ? FC_TQ_D : params.D;
}

static inline uint tq_kv_fc_num_heads(constant TurboQuantKVWriteParams& params) {
    return (tq_use_fc() && is_function_constant_defined(FC_TQ_NUM_HEADS)) ? FC_TQ_NUM_HEADS : params.num_heads;
}

static inline uint tq_kv_write_head_base(constant TurboQuantKVWriteParams& params, uint pair) {
    const uint num_heads = tq_kv_fc_num_heads(params);
    const uint token_in_span = pair / num_heads;
    const uint head = pair % num_heads;
    return token_in_span * params.bytes_per_token + head * params.bytes_per_head;
}

[[kernel, max_total_threads_per_threadgroup(512)]]
void turboquant_quant_kv_write_wht(
    device const half*       x        [[buffer(0)]],   // [pairs, D] FP16, pre-WHT
    device       uint8_t*    cache    [[buffer(1)]],   // [heads, packed values + scale]
    constant TurboQuantKVWriteParams& params [[buffer(2)]],
    constant TurboQuantWHTParams& wht_params [[buffer(3)]],
    uint  lid          [[thread_position_in_threadgroup]],
    uint  lsize        [[threads_per_threadgroup]],
    uint  pair         [[threadgroup_position_in_grid]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]],
    uint  simdgroups   [[simdgroups_per_threadgroup]]
) {
    const uint D = tq_kv_fc_d(params);
    threadgroup float values[512];
    threadgroup float partial[16];

    const uint base = pair * D;
    float v = tq_apply_rotation(float(x[base + lid]), wht_params, pair, lid);
    v = wht_body(v, values, lid, D, simd_lane);
    values[lid] = float(half(v));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float w = values[i];
        acc = fma(w, w, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) {
        partial[simd_group] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float l2_inv_tg;
    threadgroup float l2_scale_tg;
    if (simd_group == 0) {
        float sum = (simd_lane < simdgroups) ? partial[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) {
            float s = sqrt(max(sum / float(D), 1e-12f));
            l2_scale_tg = s;
            l2_inv_tg = 1.0f / s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint head_base = tq_kv_write_head_base(params, pair);

    if (lid == 0) {
        device half* scale = reinterpret_cast<device half*>(cache + head_base + params.scale_offset);
        *scale = half(l2_scale_tg);
    }

    threadgroup uint8_t code4[1024];
    const float inv = l2_inv_tg;
    for (uint i = lid; i < D; i += lsize) {
        float qv = values[i] * inv;
        code4[i] = uint8_t(tq_nearest_index4(qv));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < D / 2u) {
        uint d0 = lid * 2u;
        uint packed = (uint(code4[d0 + 0]) & 0xFu)
                    | ((uint(code4[d0 + 1]) & 0xFu) << 4);
        cache[head_base + params.packed_offset + lid] = uint8_t(packed);
    }
}
