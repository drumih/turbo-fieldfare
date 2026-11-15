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
//   K   : [seq_len, num_kv_heads, head_dim]             FP16 or affine INT8/4.
//   V   : [seq_len, num_kv_heads, head_dim]             Same format as K.
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

static inline float attn_load_kv(
    device const uchar* cache,
    uint physical_position,
    uint flat_element,
    uint elements_per_row,
    uint bits,
    uint row_stride,
    uint values_bytes,
    uint group_size
) {
    if (bits == 16u) {
        device const half* fp16 = reinterpret_cast<device const half*>(cache);
        return float(fp16[physical_position * elements_per_row + flat_element]);
    }
    device const uchar* row = cache + physical_position * row_stride;
    uint quantized;
    if (bits == 8u) {
        quantized = uint(row[flat_element]);
    } else {
        const uchar packed = row[flat_element / 2u];
        quantized = (flat_element & 1u) == 0u
            ? uint(packed & 0x0fu) : uint(packed >> 4u);
    }
    const uint groups = (elements_per_row + group_size - 1u) / group_size;
    device const half* scales = reinterpret_cast<device const half*>(row + values_bytes);
    device const half* biases = scales + groups;
    const uint group = flat_element / group_size;
    return float(quantized) * float(scales[group]) + float(biases[group]);
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
// Split-KV (Flash-Decoding) decode attention — the default path.
//

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
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
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
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    // Sparse-attention selection: keep[p] == 0 drops key p entirely. Bound
    // always (a one-byte dummy when unused) because Metal requires it; the
    // branch is uniform across the threadgroup, so the cost when off is a
    // predicted branch and the saving when on is the whole dot product for a
    // dropped key, not just its softmax term.
    device const uchar* keep          [[buffer(18)]],
    constant     uint&  use_keep      [[buffer(19)]],
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
        if (use_keep != 0u && keep[p] == 0u) { continue; }
        const uint phys_p = attn_ring_slot(p);
        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            const uint flat = kv_head * HD + i;
            const float kval = attn_load_kv(K, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            partial = fma(q_smem[i], kval, partial);
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
            const uint flat = kv_head * HD + i;
            const float vval = attn_load_kv(V, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            o_local[slot] = o_local[slot] * alpha + p_exp * vval;
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
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
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
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
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
        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const uint flat = kv_head * HD + i;
            const float k_val = attn_load_kv(K, phys_p, flat, NKV * HD,
                                              kv_bits, kv_stride, kv_value_bytes,
                                              kv_group_size);
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
            const uint flat = kv_head * HD + i;
            const float v_val = attn_load_kv(V, phys_p, flat, NKV * HD,
                                              kv_bits, kv_stride, kv_value_bytes,
                                              kv_group_size);
            o_local[slot] += p_exp * v_val;
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
    if (m_glob == -INFINITY) {
        // All chunks empty (e.g. seq_len == kv_start): zero the row rather
        // than producing NaN from exp(-inf - -inf).
        device half* out_row = out + uint(q_head) * HD;
        for (uint i = lid; i < HD; i += lsize) { out_row[i] = 0.0h; }
        return;
    }
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

// ============================================================================
// Split-KV pass 1, simdgroup-per-key layout (attention_decode_partial_simd).
//
// attention_decode_partial walks its chunk one key at a time with the whole
// threadgroup: a 128-wide dot spread over 256 threads, a two-barrier block
// reduction, then the V update -- latency-bound, ~10 us per key. At 3.7k keys
// that is 2.7 ms per full-attention layer, 32 ms/token over 12 layers, and it
// grows with the context.
//
// Here each simdgroup takes every `simdgroups`-th key of the chunk, a lane
// holds head_dim/32 contiguous dims of q, the dot is one simd_sum, and the
// online-softmax state is per simdgroup; the eight partial states merge once
// at the end through threadgroup memory. Same buffer contract and the same
// (m, d, o) partial layout, so attention_decode_combine is unchanged. The
// summation order differs from the serial kernel (keys interleave across
// simdgroups), so this is selected only when a sparse selection is in play --
// past the dense window -- and the short-prompt goldens keep the old kernel.
// Requires head_dim % 32 == 0 and head_dim <= 512.
// ============================================================================
// head_dim <= 256 for this kernel: the merge buffer is what bounds
// threadgroup memory (8 x 256 floats = 8 KB) and so threadgroups per core.
constant constexpr uint kAttnSimdMaxHeadDim = 256u;
constant constexpr uint kAttnSimdMaxDimsPerLane = kAttnSimdMaxHeadDim / 32u;

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial_simd(
    device const half*  Q             [[buffer(0)]],
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    device const uchar* keep          [[buffer(18)]],
    constant     uint&  use_keep      [[buffer(19)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint lane            [[thread_index_in_simdgroup]],
    uint sg              [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float m_sg[kAttnMaxSimdGroups];
    threadgroup float d_sg[kAttnMaxSimdGroups];
    threadgroup float o_sg[kAttnMaxSimdGroups * kAttnSimdMaxHeadDim];
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
    const uint dpl = HD / 32u;                 // dims per lane
    const uint d0 = lane * dpl;                // this lane's first dim
    float q_reg[kAttnSimdMaxDimsPerLane];
    float o_reg[kAttnSimdMaxDimsPerLane];
    for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
        q_reg[j] = (j < dpl) ? float(Q_row[d0 + j]) : 0.0f;
        o_reg[j] = 0.0f;
    }
    const uint row_elems = NKV * HD;
    const uint flat0 = kv_head * HD + d0;
    const float sc = attn_fc_scale(scale);
    float m_run = -INFINITY;
    float d_run = 0.0f;
    // int8 rows with this lane's dims inside one quant group: one scale and
    // one bias per lane per key, plain byte loads, no per-element division.
    // Anything else takes the generic per-element helper.
    const bool fast8 = (kv_bits == 8u) && (kv_group_size % dpl == 0u);
    const uint groups = (row_elems + kv_group_size - 1u) / kv_group_size;
    const uint grp = flat0 / kv_group_size;
    for (uint p = p_start + sg; p < p_end; p += simdgroups) {
        if (use_keep != 0u && keep[p] == 0u) { continue; }
        const uint phys_p = attn_ring_slot(p);
        float partial = 0.0f;
        if (fast8) {
            device const uchar* krow = K + phys_p * kv_stride;
            device const half* ks = reinterpret_cast<device const half*>(krow + kv_value_bytes);
            const float kscale = float(ks[grp]);
            const float kbias = float(ks[groups + grp]);
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float kval = float(krow[flat0 + j]) * kscale + kbias;
                    partial = fma(q_reg[j], kval, partial);
                }
            }
        } else {
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float kval = attn_load_kv(K, phys_p, flat0 + j, row_elems,
                                                    kv_bits, kv_stride, kv_value_bytes,
                                                    kv_group_size);
                    partial = fma(q_reg[j], kval, partial);
                }
            }
        }
        const float s = simd_sum(partial) * sc;
        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        if (fast8) {
            device const uchar* vrow = V + phys_p * kv_stride;
            device const half* vs = reinterpret_cast<device const half*>(vrow + kv_value_bytes);
            const float vscale = float(vs[grp]);
            const float vbias = float(vs[groups + grp]);
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float vval = float(vrow[flat0 + j]) * vscale + vbias;
                    o_reg[j] = o_reg[j] * alpha + p_exp * vval;
                }
            }
        } else {
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float vval = attn_load_kv(V, phys_p, flat0 + j, row_elems,
                                                    kv_bits, kv_stride, kv_value_bytes,
                                                    kv_group_size);
                    o_reg[j] = o_reg[j] * alpha + p_exp * vval;
                }
            }
        }
        m_run = m_new;
    }
    if (lane == 0) { m_sg[sg] = m_run; d_sg[sg] = d_run; }
    for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
        if (j < dpl) o_sg[sg * HD + d0 + j] = o_reg[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    // Merge the simdgroup states into this (head, chunk) partial.
    float m_g = -INFINITY;
    for (uint c = 0; c < simdgroups; ++c) m_g = max(m_g, m_sg[c]);
    const uint base = uint(q_head) * NC + chunk;
    device float* o_row = o_out + base * HD;
    if (m_g == -INFINITY) {
        if (lane == 0) { m_out[base] = -INFINITY; d_out[base] = 0.0f; }
        for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
            if (j < dpl) o_row[d0 + j] = 0.0f;
        }
        return;
    }
    float d_g = 0.0f;
    float w[kAttnMaxSimdGroups];
    for (uint c = 0; c < kAttnMaxSimdGroups; ++c) {
        w[c] = (c < simdgroups && m_sg[c] != -INFINITY) ? attn_softmax_exp(m_sg[c] - m_g) : 0.0f;
        if (c < simdgroups) d_g += d_sg[c] * w[c];
    }
    for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
        if (j < dpl) {
            float acc = 0.0f;
            for (uint c = 0; c < simdgroups; ++c) acc += o_sg[c * HD + d0 + j] * w[c];
            o_row[d0 + j] = acc;
        }
    }
    if (lane == 0) { m_out[base] = m_g; d_out[base] = d_g; }
}

// ============================================================================
// Windowed decode attention.
//
// The host-KV tier keeps the authoritative rows in a host arena and a window
// of `windowTokens` rows in VRAM. The selection is gathered into that window
// before the call, so these kernels iterate a compact list of selected keys
// rather than a logical range: `sel_index[i]` is the key's logical position
// (diagnostics only) and `sel_row[i]` is the window row that holds it.
//
// The chunking is over the selection, not over positions: chunk 0 reads the
// first `chunk_len` selected keys, chunk 1 the next, and so on. That keeps the
// per-chunk work uniform even when the selected keys are scattered across a
// million-position history, and it is why the cost is O(selection) instead of
// O(context).
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_window_partial(
    device const half*  Q             [[buffer(0)]],
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  sel_count     [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    device const uint*  sel_index     [[buffer(18)]],
    device const uint*  sel_row       [[buffer(19)]],
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
    const uint p_start = chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > sel_count) { p_end = sel_count; }

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

    for (uint s = p_start; s < p_end; ++s) {
        const uint phys_p = sel_row[s];
        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            const uint flat = kv_head * HD + i;
            const float kval = attn_load_kv(K, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            partial = fma(q_smem[i], kval, partial);
        }
        float s_score = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s_score *= attn_fc_scale(scale);

        const float m_new = max(m_run, s_score);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s_score - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            const uint flat = kv_head * HD + i;
            const float vval = attn_load_kv(V, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            o_local[slot] = o_local[slot] * alpha + p_exp * vval;
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
void attention_decode_window_partial_simd(
    device const half*  Q             [[buffer(0)]],
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  sel_count     [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    device const uint*  sel_index     [[buffer(18)]],
    device const uint*  sel_row       [[buffer(19)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint lane            [[thread_index_in_simdgroup]],
    uint sg              [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float m_sg[kAttnMaxSimdGroups];
    threadgroup float d_sg[kAttnMaxSimdGroups];
    threadgroup float o_sg[kAttnMaxSimdGroups * kAttnSimdMaxHeadDim];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > sel_count) { p_end = sel_count; }
    const uint kv_head = q_head / (NQ / NKV);
    device const half* Q_row = Q + uint(q_head) * HD;
    const uint dpl = HD / 32u;
    const uint d0 = lane * dpl;
    float q_reg[kAttnSimdMaxDimsPerLane];
    float o_reg[kAttnSimdMaxDimsPerLane];
    for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
        q_reg[j] = (j < dpl) ? float(Q_row[d0 + j]) : 0.0f;
        o_reg[j] = 0.0f;
    }
    const uint row_elems = NKV * HD;
    const uint flat0 = kv_head * HD + d0;
    const float sc = attn_fc_scale(scale);
    float m_run = -INFINITY;
    float d_run = 0.0f;
    const bool fast8 = (kv_bits == 8u) && (kv_group_size % dpl == 0u);
    const uint groups = (row_elems + kv_group_size - 1u) / kv_group_size;
    const uint grp = flat0 / kv_group_size;
    for (uint s = p_start + sg; s < p_end; s += simdgroups) {
        const uint phys_p = sel_row[s];
        float partial = 0.0f;
        if (fast8) {
            device const uchar* krow = K + phys_p * kv_stride;
            device const half* ks = reinterpret_cast<device const half*>(krow + kv_value_bytes);
            const float kscale = float(ks[grp]);
            const float kbias = float(ks[groups + grp]);
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float kval = float(krow[flat0 + j]) * kscale + kbias;
                    partial = fma(q_reg[j], kval, partial);
                }
            }
        } else {
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float kval = attn_load_kv(K, phys_p, flat0 + j, row_elems,
                                                    kv_bits, kv_stride, kv_value_bytes,
                                                    kv_group_size);
                    partial = fma(q_reg[j], kval, partial);
                }
            }
        }
        const float s_score = simd_sum(partial) * sc;
        const float m_new = max(m_run, s_score);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s_score - m_new);
        d_run = d_run * alpha + p_exp;
        if (fast8) {
            device const uchar* vrow = V + phys_p * kv_stride;
            device const half* vs = reinterpret_cast<device const half*>(vrow + kv_value_bytes);
            const float vscale = float(vs[grp]);
            const float vbias = float(vs[groups + grp]);
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float vval = float(vrow[flat0 + j]) * vscale + vbias;
                    o_reg[j] = o_reg[j] * alpha + p_exp * vval;
                }
            }
        } else {
            for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
                if (j < dpl) {
                    const float vval = attn_load_kv(V, phys_p, flat0 + j, row_elems,
                                                    kv_bits, kv_stride, kv_value_bytes,
                                                    kv_group_size);
                    o_reg[j] = o_reg[j] * alpha + p_exp * vval;
                }
            }
        }
        m_run = m_new;
    }
    if (lane == 0) { m_sg[sg] = m_run; d_sg[sg] = d_run; }
    for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
        if (j < dpl) o_sg[sg * HD + d0 + j] = o_reg[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float m_g = -INFINITY;
    for (uint c = 0; c < simdgroups; ++c) m_g = max(m_g, m_sg[c]);
    const uint base = uint(q_head) * NC + chunk;
    device float* o_row = o_out + base * HD;
    if (m_g == -INFINITY) {
        if (lane == 0) { m_out[base] = -INFINITY; d_out[base] = 0.0f; }
        for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
            if (j < dpl) o_row[d0 + j] = 0.0f;
        }
        return;
    }
    float d_g = 0.0f;
    float w[kAttnMaxSimdGroups];
    for (uint c = 0; c < kAttnMaxSimdGroups; ++c) {
        w[c] = (c < simdgroups && m_sg[c] != -INFINITY) ? attn_softmax_exp(m_sg[c] - m_g) : 0.0f;
        if (c < simdgroups) d_g += d_sg[c] * w[c];
    }
    for (uint j = 0; j < kAttnSimdMaxDimsPerLane; ++j) {
        if (j < dpl) {
            float acc = 0.0f;
            for (uint c = 0; c < simdgroups; ++c) acc += o_sg[c * HD + d0 + j] * w[c];
            o_row[d0 + j] = acc;
        }
    }
    if (lane == 0) { m_out[base] = m_g; d_out[base] = d_g; }
}
