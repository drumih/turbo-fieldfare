#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention — split-KV tiled softmax attention for single-token decode.
//
// Decode path only: M_q = 1 (one query token), arbitrary seq_len history.
// The MPP prefill path handles M_q > 1 separately.
//
// Layout (caller-side contract):
//   Q   : [num_q_heads,  head_dim]                      FP16, contiguous.
//   K   : [seq_len, num_kv_heads, head_dim]             FP16, contiguous.
//   V   : [seq_len, num_kv_heads, head_dim]             FP16, same shape as K.
//         Full attention reuses the raw K projection for V, but its separate
//         normalization and RoPE paths make these buffers distinct here.
//   out : [num_q_heads,  head_dim]                      FP16.
//
// GQA: q_head -> kv_head = q_head / (num_q_heads / num_kv_heads).
//      Multiple Q heads share one KV head; the dispatch indexes Q heads.
//
// Online softmax recurrence (FP32 accumulators) — Milakov & Gimelshein 2018,
// also FlashAttention:
//   m_new   = max(m, s)
//   alpha   = exp(m - m_new)                 // rescale factor for past state
//   d       = d * alpha + exp(s - m_new)
//   o[i]    = o[i] * alpha + exp(s - m_new) * V[p, i]
//   m       = m_new
// Final normalization: out[i] = o[i] / d.
//
// ============================================================================

constant constexpr uint kAttnThreads      = 256;
// kAttnMaxSimdGroups must cover kAttnThreads / 32 = 8.
constant constexpr uint kAttnMaxSimdGroups = 8;
constant constexpr uint kAttnMaxQPerKV     = 2;
constant constexpr uint kAttnMaxFullQPerKV = 8;
constant constexpr uint kAttnFullQPerThreadgroup = 2;
// Largest head_dim we run with (full-attention layers). SWA uses 256 — the
// kernel still allocates the 512-slot scratch but only touches the live half.
constant constexpr uint kAttnMaxHeadDim   = 512;
constant uint FC_ATTN_HEAD_DIM [[function_constant(60)]];
constant uint FC_ATTN_NUM_Q_HEADS [[function_constant(61)]];
constant uint FC_ATTN_NUM_KV_HEADS [[function_constant(62)]];
constant bool FC_ATTN_USE_FC [[function_constant(63)]];
constant float FC_ATTN_SCALE [[function_constant(64)]];
constant uint FC_ATTN_NUM_CHUNKS [[function_constant(65)]];
constant uint FC_ATTN_RING_CAP [[function_constant(69)]];

static inline uint attn_fc_head_dim(constant uint& head_dim) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_HEAD_DIM))
        ? FC_ATTN_HEAD_DIM
        : head_dim;
}
static inline uint attn_fc_num_q_heads(constant uint& num_q_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_Q_HEADS))
        ? FC_ATTN_NUM_Q_HEADS
        : num_q_heads;
}

static inline uint attn_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_KV_HEADS))
        ? FC_ATTN_NUM_KV_HEADS
        : num_kv_heads;
}

static inline float attn_fc_scale(float scale) {
    return is_function_constant_defined(FC_ATTN_SCALE) ? FC_ATTN_SCALE : scale;
}

static inline uint attn_fc_num_chunks(constant uint& num_chunks) {
    return is_function_constant_defined(FC_ATTN_NUM_CHUNKS) ? FC_ATTN_NUM_CHUNKS : num_chunks;
}

static inline uint attn_ring_slot(uint p) {
    return (is_function_constant_defined(FC_ATTN_RING_CAP) &&
            FC_ATTN_RING_CAP != 0u)
        ? (p % FC_ATTN_RING_CAP)
        : p;
}

static inline float attn_softmax_exp(float x) {
    return fast::exp(x);
}

// Block reduce: per-SIMD-group simd_sum, write partial to scratch, lane 0 of
// SIMD-group 0 finishes the merge with a second simd_sum and broadcasts.
// `scratch` must hold at least `simdgroups` floats; `bcast` is one float used
// to publish the final reduced value to all threads.
inline float block_reduce_sum(float v,
                              uint simd_lane_id,
                              uint simd_group_id,
                              uint simdgroups,
                              threadgroup float* scratch,
                              threadgroup float* bcast) {
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

// ============================================================================
// TurboQuant packed-KV attention.
// ============================================================================

struct AttentionTurboQuantKVParams {
    uint head_dim;
    uint num_q_heads;
    uint num_kv_heads;
    uint seq_len;
    uint kv_start;
    float scale;
    uint layer;
    uint rotation_seed;
    uint key_bytes_per_head;
    uint key_packed_offset;
    uint key_scale_offset;
    uint value_bytes_per_head;
    uint value_packed_offset;
    uint value_scale_offset;
    uint q_pretransformed;
};

static inline uint attn_tq_key_bit_width(constant AttentionTurboQuantKVParams&) {
    return 4u;
}

static inline uint attn_tq_value_bit_width(constant AttentionTurboQuantKVParams&) {
    return 4u;
}

constant constexpr float kAttnTQCodebook4[16] = {
    -2.7326f, -2.0690f, -1.6181f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6181f,  2.0690f,  2.7326f
};

static inline float attn_tq_centroid(uint idx) {
    return kAttnTQCodebook4[idx];
}

static inline ulong attn_tq_mix64(ulong z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

static inline bool attn_tq_rotation_is_negative(uint layer, uint head, uint dim, uint seed) {
    ulong z = ulong(seed);
    z ^= ulong(layer) * 0x9E3779B97F4A7C15ul;
    z ^= ulong(head)  * 0xBF58476D1CE4E5B9ul;
    z ^= ulong(dim)   * 0x94D049BB133111EBul;
    return (attn_tq_mix64(z) >> 63) != 0ul;
}

static inline float attn_tq_apply_rotation(float v,
                                           uint layer,
                                           uint head,
                                           uint dim,
                                           uint seed) {
    return attn_tq_rotation_is_negative(layer, head, dim, seed) ? -v : v;
}

static inline void attn_tq_wht_inplace(threadgroup float* data,
                                       uint D,
                                       uint lid,
                                       uint lsize) {
    for (uint stride = 1u; stride < D; stride <<= 1u) {
        for (uint j = lid; j < D / 2u; j += lsize) {
            const uint block = j / stride;
            const uint lane = j - block * stride;
            const uint a_idx = block * stride * 2u + lane;
            const uint b_idx = a_idx + stride;
            const float a = data[a_idx];
            const float b = data[b_idx];
            data[a_idx] = (a + b) * 0.7071067811865475f;
            data[b_idx] = (a - b) * 0.7071067811865475f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

static inline void attn_tq_wht_inplace_d256(threadgroup float* data,
                                            uint lid,
                                            uint simd_lane) {
    float v = data[lid];
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
        float other = simd_shuffle_xor(v, stride);
        v = ((simd_lane & stride) == 0u)
            ? (v + other) * 0.7071067811865475f
            : (other - v) * 0.7071067811865475f;
    }

    data[lid] = v;
    for (uint stride = 32u; stride < 256u; stride <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((lid & stride) == 0u) {
            uint j = lid | stride;
            float a = data[lid];
            float b = data[j];
            data[lid] = (a + b) * 0.7071067811865475f;
            data[j]   = (a - b) * 0.7071067811865475f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void attn_tq_wht_inplace_d512(threadgroup float* data,
                                            uint lid,
                                            uint simd_lane) {
    float v0 = data[lid];
    float v1 = data[lid + 256u];
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
        float other0 = simd_shuffle_xor(v0, stride);
        float other1 = simd_shuffle_xor(v1, stride);
        const bool lo = (simd_lane & stride) == 0u;
        v0 = lo
            ? (v0 + other0) * 0.7071067811865475f
            : (other0 - v0) * 0.7071067811865475f;
        v1 = lo
            ? (v1 + other1) * 0.7071067811865475f
            : (other1 - v1) * 0.7071067811865475f;
    }

    data[lid] = v0;
    data[lid + 256u] = v1;
    for (uint stride = 32u; stride < 256u; stride <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((lid & stride) == 0u) {
            uint j = lid | stride;
            float a0 = data[lid];
            float b0 = data[j];
            data[lid] = (a0 + b0) * 0.7071067811865475f;
            data[j]   = (a0 - b0) * 0.7071067811865475f;

            float a1 = data[lid + 256u];
            float b1 = data[j + 256u];
            data[lid + 256u] = (a1 + b1) * 0.7071067811865475f;
            data[j + 256u]   = (a1 - b1) * 0.7071067811865475f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a = data[lid];
    float b = data[lid + 256u];
    data[lid] = (a + b) * 0.7071067811865475f;
    data[lid + 256u] = (a - b) * 0.7071067811865475f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void attn_tq_wht_inplace_fast(threadgroup float* data,
                                            uint D,
                                            uint lid,
                                            uint lsize,
                                            uint simd_lane) {
    if (D == 256u && lsize >= 256u) {
        attn_tq_wht_inplace_d256(data, lid, simd_lane);
    } else if (D == 512u && lsize >= 256u) {
        attn_tq_wht_inplace_d512(data, lid, simd_lane);
    } else {
        attn_tq_wht_inplace(data, D, lid, lsize);
    }
}

static inline float attn_tq_read_packed(device const uint8_t* cache,
                                        uint head_base,
                                        uint packed_bits,
                                        uint packed_offset,
                                        uint D,
                                        uint dim,
                                        float scale) {
    const uint packed_base = head_base + packed_offset;
    const uint byte = dim >> 1u;
    const uint shift = (dim & 1u) << 2u;
    const uint idx = (uint(cache[packed_base + byte]) >> shift) & 0xFu;
    return attn_tq_centroid(idx) * scale;
}

static inline uint attn_tq_read_index(device const uint8_t* cache,
                                      uint head_base,
                                      uint packed_bits,
                                      uint packed_offset,
                                      uint D,
                                      uint dim) {
    const uint packed_base = head_base + packed_offset;
    const uint byte = dim >> 1u;
    const uint shift = (dim & 1u) << 2u;
    return (uint(cache[packed_base + byte]) >> shift) & 0xFu;
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_turboquant_q_rht(
    device const half*    Q       [[buffer(0)]],
    device       half*    out     [[buffer(1)]],
    constant AttentionTurboQuantKVParams& params [[buffer(2)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    const uint HD = params.head_dim;
    const uint q_head = tg_id;
    const uint kv_head = q_head / (params.num_q_heads / params.num_kv_heads);
    device const half* Q_row = Q + q_head * HD;
    for (uint i = lid; i < HD; i += lsize) {
        const float q = float(Q_row[i]);
        q_smem[i] = attn_tq_apply_rotation(q,
                                            params.layer,
                                            kv_head,
                                            i,
                                            params.rotation_seed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_tq_wht_inplace_fast(q_smem, HD, lid, lsize, simd_lane_id);

    device half* out_row = out + q_head * HD;
    for (uint i = lid; i < HD; i += lsize) {
        out_row[i] = half(q_smem[i]);
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_turboquant_decode_partial(
    device const half*    Q       [[buffer(0)]],
    device const uint8_t* K_cache [[buffer(1)]],
    device const uint8_t* V_cache [[buffer(2)]],
    device       float*   m_out   [[buffer(3)]],
    device       float*   d_out   [[buffer(4)]],
    device       float*   o_out   [[buffer(5)]],
    constant AttentionTurboQuantKVParams& params [[buffer(6)]],
    constant uint& chunk_len      [[buffer(7)]],
    constant uint& num_chunks     [[buffer(8)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;

    const uint HD = attn_fc_head_dim(params.head_dim);
    const uint NQ = attn_fc_num_q_heads(params.num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(params.num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id / NC;
    const uint chunk = tg_id % NC;
    const uint kv_head = q_head / (NQ / NKV);
    const uint key_bytes_per_token = params.key_bytes_per_head * NKV;
    const uint value_bytes_per_token = params.value_bytes_per_head * NKV;
    const uint p_start = params.kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > params.seq_len) { p_end = params.seq_len; }

    device const half* Q_row = Q + q_head * HD;
    for (uint i = lid; i < HD; i += lsize) {
        if (params.q_pretransformed != 0u) {
            q_smem[i] = float(Q_row[i]);
        } else {
            const float q = float(Q_row[i]);
            q_smem[i] = attn_tq_apply_rotation(q,
                                                params.layer,
                                                kv_head,
                                                i,
                                                params.rotation_seed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (params.q_pretransformed == 0u) {
        attn_tq_wht_inplace_fast(q_smem, HD, lid, lsize, simd_lane_id);
    }

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint key_head_base = p * key_bytes_per_token
            + kv_head * params.key_bytes_per_head;
        const uint value_head_base = p * value_bytes_per_token
            + kv_head * params.value_bytes_per_head;
        const device half* key_scale_ptr = reinterpret_cast<const device half*>(
            K_cache + key_head_base + params.key_scale_offset);
        const device half* value_scale_ptr = reinterpret_cast<const device half*>(
            V_cache + value_head_base + params.value_scale_offset);
        const float key_scale = float(*key_scale_ptr);
        const float value_scale = float(*value_scale_ptr);

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            const float k_val = attn_tq_read_packed(K_cache,
                                                    key_head_base,
                                                    4u,
                                                    params.key_packed_offset,
                                                    HD,
                                                    i,
                                                    key_scale);
            partial = fma(q_smem[i], k_val, partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= params.scale;

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            float v_val = attn_tq_read_packed(V_cache,
                                              value_head_base,
                                              4u,
                                              params.value_packed_offset,
                                              HD,
                                              i,
                                              value_scale);
            o_local[slot] = o_local[slot] * alpha + p_exp * v_val;
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = q_head * NC + chunk;
    if (lid == 0) {
        m_out[base] = m_run;
        d_out[base] = d_run;
    }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_turboquant_decode_gqa_swa_partial(
    device const half*    Q       [[buffer(0)]],
    device const uint8_t* K_cache [[buffer(1)]],
    device const uint8_t* V_cache [[buffer(2)]],
    device       float*   m_out   [[buffer(3)]],
    device       float*   d_out   [[buffer(4)]],
    device       float*   o_out   [[buffer(5)]],
    constant AttentionTurboQuantKVParams& params [[buffer(6)]],
    constant uint& chunk_len      [[buffer(7)]],
    constant uint& num_chunks     [[buffer(8)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV];

    const uint HD = attn_fc_head_dim(params.head_dim);
    const uint NQ = attn_fc_num_q_heads(params.num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(params.num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk = tg_id % NC;
    const uint q_base = kv_head * q_per_kv;
    const uint key_bytes_per_token = params.key_bytes_per_head * NKV;
    const uint value_bytes_per_token = params.value_bytes_per_head * NKV;
    const uint p_start = params.kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > params.seq_len) { p_end = params.seq_len; }

    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            if (params.q_pretransformed != 0u) {
                Q_s[i] = float(Q_row[i]);
            } else {
                const float q = float(Q_row[i]);
                Q_s[i] = attn_tq_apply_rotation(q,
                                                 params.layer,
                                                 kv_head,
                                                 i,
                                                 params.rotation_seed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (params.q_pretransformed == 0u) {
        for (uint qg = 0; qg < q_per_kv; ++qg) {
            attn_tq_wht_inplace_fast(q_smem + qg * kAttnMaxHeadDim, HD, lid, lsize, simd_lane_id);
        }
    }

    const uint groups_per_q = max(1u, (lsize / 32u) / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint key_head_base = p * key_bytes_per_token
            + kv_head * params.key_bytes_per_head;
        const uint value_head_base = p * value_bytes_per_token
            + kv_head * params.value_bytes_per_head;
        const device half* key_scale_ptr = reinterpret_cast<const device half*>(
            K_cache + key_head_base + params.key_scale_offset);
        const device half* value_scale_ptr = reinterpret_cast<const device half*>(
            V_cache + value_head_base + params.value_scale_offset);
        const float key_scale = float(*key_scale_ptr);
        const float value_scale = float(*value_scale_ptr);

        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const float k_val = attn_tq_read_packed(K_cache,
                                                    key_head_base,
                                                    4u,
                                                    params.key_packed_offset,
                                                    HD,
                                                    i,
                                                    key_scale);
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
        }

        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            reduce_scratch[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_group == 0) {
            float t = (simd_lane_id < groups_per_q)
                ? reduce_scratch[active_q * kAttnMaxSimdGroups + simd_lane_id]
                : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[active_q] = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = bcast[active_q] * params.scale;

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
        m_run = m_new;

        uint slot = 0;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            float v_val = attn_tq_read_packed(V_cache,
                                              value_head_base,
                                              4u,
                                              params.value_packed_offset,
                                              HD,
                                              i,
                                              value_scale);
            o_local[slot] += p_exp * v_val;
            slot += 1;
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = q_head * NC + chunk;
    if (local_lid == 0) {
        m_out[base] = m_run;
        d_out[base] = d_run;
    }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_turboquant_decode_combine(
    device const float*   m_in    [[buffer(0)]],
    device const float*   d_in    [[buffer(1)]],
    device const float*   o_in    [[buffer(2)]],
    device       half*    out     [[buffer(3)]],
    constant AttentionTurboQuantKVParams& params [[buffer(4)]],
    constant uint& num_chunks     [[buffer(5)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]]
) {
    threadgroup float out_smem[kAttnMaxHeadDim];
    const uint HD = attn_fc_head_dim(params.head_dim);
    const uint NQ = attn_fc_num_q_heads(params.num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(params.num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id;
    const uint kv_head = q_head / (NQ / NKV);
    device const float* m_row = m_in + q_head * NC;
    device const float* d_row = d_in + q_head * NC;
    device const float* o_base = o_in + q_head * NC * HD;

    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) {
        m_glob = max(m_glob, m_row[c]);
    }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) {
        D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob);
    }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_smem[i] = acc * inv_d;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_tq_wht_inplace_fast(out_smem, HD, lid, lsize, simd_lane_id);

    device half* out_row = out + q_head * HD;
    for (uint i = lid; i < HD; i += lsize) {
        out_row[i] = half(attn_tq_apply_rotation(out_smem[i],
                                                 params.layer,
                                                 kv_head,
                                                 i,
                                                 params.rotation_seed));
    }
}

// ============================================================================
// Split-KV (Flash-Decoding) decode attention — the default path.
//
// The single-pass kernels above run one threadgroup per Q head, each serially
// streaming the whole K/V history. At decode (M_q=1) that is occupancy- and
// latency-bound: 16 Q heads -> 16 threadgroups (cannot fill the GPU's cores),
// and the per-head critical path is seq_len positions long. Splitting each
// head's [kv_start, seq_len) range across `num_chunks` threadgroups turns it
// into 16 * num_chunks threadgroups whose critical path is ~seq_len/num_chunks.
//
// This D512 specialization adapts MLX v0.32.0's one-pass vector geometry. MLX
// does not instantiate its vector kernel at D512. TurboFieldfare selects this path
// for full-attention decode.
kernel void attention_decode_mlx_geometry_full_v2(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint q_head [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    constexpr uint kHeadDim = 512;
    constexpr uint kNumKVHeads = 2;
    constexpr uint kQPerKV = 8;
    constexpr uint kSIMDGroups = 32;
    constexpr uint kComponentsPerLane = 16;

    thread float q[kComponentsPerLane];
    thread float key[kComponentsPerLane];
    thread float output[kComponentsPerLane];
    threadgroup float output_exchange[kSIMDGroups * 32];
    threadgroup float max_scores[kSIMDGroups];
    threadgroup float sum_exp_scores[kSIMDGroups];

    const uint kv_head = q_head / kQPerKV;
    const uint component = simd_lid * kComponentsPerLane;
    const uint q_base = q_head * kHeadDim + component;
    for (uint i = 0; i < kComponentsPerLane; ++i) {
        q[i] = scale * float(Q[q_base + i]);
        output[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    for (uint position = simd_gid; position < seq_len; position += kSIMDGroups) {
        const uint kv_base = (position * kNumKVHeads + kv_head) * kHeadDim + component;
        float score = 0.0f;
        for (uint i = 0; i < kComponentsPerLane; ++i) {
            key[i] = float(K[kv_base + i]);
            score += q[i] * key[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float old_factor = fast::exp(max_score - new_max);
        const float score_factor = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * old_factor + score_factor;
        for (uint i = 0; i < kComponentsPerLane; ++i) {
            output[i] = output[i] * old_factor + score_factor * float(V[kv_base + i]);
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    const float merged_max = simd_max(max_score);
    const float merge_factor = fast::exp(max_score - merged_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * merge_factor);

    for (uint i = 0; i < kComponentsPerLane; ++i) {
        output_exchange[simd_lid * kSIMDGroups + simd_gid] = output[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float merged = simd_sum(
            output_exchange[simd_gid * kSIMDGroups + simd_lid] * merge_factor);
        output[i] = sum_exp_score == 0.0f ? merged : merged / sum_exp_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        const uint out_base = q_head * kHeadDim + simd_gid * kComponentsPerLane;
        for (uint i = 0; i < kComponentsPerLane; ++i) {
            out[out_base + i] = half(output[i]);
        }
    }
}

// Pass 1 (attention_decode_partial): grid = num_q_heads * num_chunks. Each TG
//   runs the same online-softmax recurrence over its chunk [p_start, p_end) and
//   writes the UN-normalized partial state (m_chunk, d_chunk, o_chunk[head_dim])
//   to scratch — no division yet.
// Pass 2 (attention_decode_combine): grid = num_q_heads. Each TG merges its
//   head's num_chunks partials with the standard online-softmax rescale
//   (m_glob = max_c m_c; D = Σ d_c·e^{m_c−m_glob}; O = Σ o_c·e^{m_c−m_glob}) and
//   writes out[i] = O[i] / D in FP16.
//
// At num_chunks == 1 the chunk spans the whole [kv_start, seq_len) range and
// the partial is the exact single-pass accumulation; the combine's only chunk
// has m_glob == m_chunk so e^0 == 1 and out == o/d — byte-identical to the
// single-pass kernels above. num_chunks > 1 changes the FP rounding of the
// partial sums only (same position summation order), not the algorithm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // p_start can land past the end when num_chunks > range length (the tail
    // chunks are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(V_row[i]);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_gqa_swa_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_base = kv_head * q_per_kv;
    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            Q_s[i] = float(Q_row[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const float k_val = float(K_row[i]);
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
        }

        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            reduce_scratch[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_group == 0) {
            float t = (simd_lane_id < groups_per_q)
                ? reduce_scratch[active_q * kAttnMaxSimdGroups + simd_lane_id]
                : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[active_q] = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = bcast[active_q] * attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
        m_run = m_new;

        uint slot = 0;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            o_local[slot] += p_exp * float(V_row[i]);
            slot += 1;
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = uint(q_head) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * head_dim]
    device       half*  out          [[buffer(3)]],    // [num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * NC;
    device const float* d_row  = d_in + uint(q_head) * NC;
    device const float* o_base = o_in + uint(q_head) * NC * HD;

    // num_chunks is small (<= a few dozen); each thread recomputes the global
    // max and denominator rather than pay a threadgroup reduction + barriers.
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}
