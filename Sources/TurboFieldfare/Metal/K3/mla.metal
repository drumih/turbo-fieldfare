#include <metal_stdlib>
using namespace metal;

// ============================================================================
// mla — Kimi K3 MLA absorbed single-token decode attention over the latent
// cache (docs/K3_DATAFLOW.md "MLA layer"). The model is fully NoPE: the
// rope-part channels are cached and consumed raw, never rotated.
//
// Cache per token (fp16, L+R values): [ kv_a_layernorm(latent L) | rope R ].
// kv_b (the 512 -> H*(N+V) projection, N = qk_nope, V = v_head) is absorbed:
//   q~_h   = W_kvb_k,h^T q_nope_h              (L-dim, precomputed per token)
//   s_h(t) = (q~_h . latent_t + q_rope_h . rope_t) * scale
//   out_h  = W_kvb_v,h ( softmax_h . latents )  (per-head V-dim)
// kv_b is linear, so absorbing into q / folding the value side is exact
// eager attention — the CPU reference (naive expansion) is the ground truth.
//
// Weight layout assumption for k3_mla_kvb_expand: the input is the
// DEQUANTIZED kv_b projection as row-major fp16 [H*(N+V), L] (nn.Linear
// out-major), with per-head blocks of N k rows followed by V v rows. The
// quantized-source fused variant is a later stage.
// ============================================================================

constant constexpr uint kK3MlaThreads = 256;
constant constexpr uint kK3MlaMaxSimdGroups = kK3MlaThreads / 32;
constant constexpr uint kK3MlaMaxLatent = 512;
constant constexpr uint kK3MlaMaxCacheRow = 576;   // L + R
constant constexpr uint kK3MlaMaxHeadDim = 128;    // max(N, V)

constant uint FC_K3_MLA_NUM_HEADS [[function_constant(74)]];
constant uint FC_K3_MLA_LATENT [[function_constant(75)]];
constant uint FC_K3_MLA_ROPE [[function_constant(76)]];
constant uint FC_K3_MLA_NOPE [[function_constant(77)]];
constant uint FC_K3_MLA_VHEAD [[function_constant(78)]];
constant bool FC_K3_MLA_USE_FC [[function_constant(79)]];

static inline uint k3_mla_fc_num_heads(constant uint& H) {
    return (is_function_constant_defined(FC_K3_MLA_USE_FC) &&
            FC_K3_MLA_USE_FC &&
            is_function_constant_defined(FC_K3_MLA_NUM_HEADS))
        ? FC_K3_MLA_NUM_HEADS : H;
}

static inline uint k3_mla_fc_latent(constant uint& L) {
    return (is_function_constant_defined(FC_K3_MLA_USE_FC) &&
            FC_K3_MLA_USE_FC &&
            is_function_constant_defined(FC_K3_MLA_LATENT))
        ? FC_K3_MLA_LATENT : L;
}

static inline uint k3_mla_fc_rope(constant uint& R) {
    return (is_function_constant_defined(FC_K3_MLA_USE_FC) &&
            FC_K3_MLA_USE_FC &&
            is_function_constant_defined(FC_K3_MLA_ROPE))
        ? FC_K3_MLA_ROPE : R;
}

static inline uint k3_mla_fc_nope(constant uint& N) {
    return (is_function_constant_defined(FC_K3_MLA_USE_FC) &&
            FC_K3_MLA_USE_FC &&
            is_function_constant_defined(FC_K3_MLA_NOPE))
        ? FC_K3_MLA_NOPE : N;
}

static inline uint k3_mla_fc_vhead(constant uint& V) {
    return (is_function_constant_defined(FC_K3_MLA_USE_FC) &&
            FC_K3_MLA_USE_FC &&
            is_function_constant_defined(FC_K3_MLA_VHEAD))
        ? FC_K3_MLA_VHEAD : V;
}

static inline float k3_mla_sigmoid(float v) {
    return 1.0f / (1.0f + exp(-v));
}

// House two-stage block reduction (per-SIMD simd_sum, scratch, one SIMD
// merges, broadcast slot doubles as the return value).
static inline float k3_mla_block_reduce_sum(
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

// Per-head layout/convert of the dequantized kv_b weight:
//   kT[h][l][i] = kv_b[h*(N+V) + i][l]         (k_nope part, transposed)
//   v[h][i][l]  = kv_b[h*(N+V) + N + i][l]     (value part, row copy)
// One thread per output element; kT range first, then the v range.
kernel void k3_mla_kvb_expand(
    device const half* kv_b [[buffer(0)]],   // [H*(N+V)][L] fp16
    device half*       kT   [[buffer(1)]],   // [H][L][N] fp16
    device half*       vOut [[buffer(2)]],   // [H][V][L] fp16
    constant uint&     num_heads [[buffer(3)]],
    constant uint&     latent    [[buffer(4)]],
    constant uint&     nope      [[buffer(5)]],
    constant uint&     v_head    [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint H = k3_mla_fc_num_heads(num_heads);
    const uint L = k3_mla_fc_latent(latent);
    const uint N = k3_mla_fc_nope(nope);
    const uint V = k3_mla_fc_vhead(v_head);
    const uint k_total = H * L * N;
    if (gid < k_total) {
        const uint i = gid % N;
        const uint l = (gid / N) % L;
        const uint h = gid / (N * L);
        kT[gid] = kv_b[(h * (N + V) + i) * L + l];
        return;
    }
    const uint vg = gid - k_total;
    if (vg >= H * V * L) return;
    const uint l = vg % L;
    const uint i = (vg / L) % V;
    const uint h = vg / (L * V);
    vOut[vg] = kv_b[(h * (N + V) + N + i) * L + l];
}

// q~_h = W_kvb_k,h^T q_nope_h, with the rope part passed through:
//   q_abs[h] = [ q~_h (L) | q_rope_h (R) ]   fp32
// One SIMD per (h, l) dot over N; rope rows are written by lane 0.
kernel void k3_mla_absorb_q(
    device const half* kT     [[buffer(0)]],   // [H][L][N] fp16
    device const half* q      [[buffer(1)]],   // [H][N+R] fp16
    device float*      q_abs  [[buffer(2)]],   // [H][L+R] fp32
    constant uint&     num_heads [[buffer(3)]],
    constant uint&     latent [[buffer(4)]],
    constant uint&     rope   [[buffer(5)]],
    constant uint&     nope   [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint H = k3_mla_fc_num_heads(num_heads);
    const uint L = k3_mla_fc_latent(latent);
    const uint R = k3_mla_fc_rope(rope);
    const uint N = k3_mla_fc_nope(nope);
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= H * (L + R)) return;
    const uint h = row / (L + R);
    const uint l = row % (L + R);
    if (l >= L) {
        if (lane == 0) {
            q_abs[row] = float(q[h * (N + R) + N + (l - L)]);
        }
        return;
    }
    device const half* kT_row = kT + (uint(h) * L + l) * N;
    device const half* q_row = q + uint(h) * (N + R);
    float acc = 0.0f;
    for (uint i = lane; i < N; i += 32u) {
        acc = fma(float(kT_row[i]), float(q_row[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        q_abs[row] = acc;
    }
}

// Split-KV partial: grid = H * num_chunks, one 256-thread threadgroup per
// (head, chunk). Because both q_abs and cache rows are [latent | rope]
// contiguous, the score is one dot over L+R values, scaled afterwards.
// Online softmax state (m, d) and the unnormalized o accumulation are FP32;
// o accumulates the LATENT only.
kernel void k3_mla_attn_decode_partial(
    device const half*  cache      [[buffer(0)]],   // [cap][L+R] fp16
    device const float* q_abs      [[buffer(1)]],   // [H][L+R] fp32
    device float*       m_out      [[buffer(2)]],   // [H * num_chunks]
    device float*       d_out      [[buffer(3)]],   // [H * num_chunks]
    device float*       o_out      [[buffer(4)]],   // [H * num_chunks * L]
    constant uint&      num_heads  [[buffer(5)]],
    constant uint&      latent     [[buffer(6)]],
    constant uint&      rope       [[buffer(7)]],
    constant uint&      seq_len    [[buffer(8)]],
    constant uint&      chunk_len  [[buffer(9)]],
    constant uint&      num_chunks [[buffer(10)]],
    constant float&     scale      [[buffer(11)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kK3MlaMaxCacheRow];
    threadgroup float reduce_scratch[kK3MlaMaxSimdGroups];
    threadgroup float bcast;
    const uint H = k3_mla_fc_num_heads(num_heads);
    const uint L = k3_mla_fc_latent(latent);
    const uint R = k3_mla_fc_rope(rope);
    const uint row_len = L + R;

    const uint h = tg_id / num_chunks;
    const uint chunk = tg_id % num_chunks;
    const uint p_start = chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    device const float* q_row = q_abs + uint(h) * row_len;
    for (uint i = lid; i < row_len; i += lsize) {
        q_smem[i] = q_row[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread =
        (kK3MlaMaxLatent + kK3MlaThreads - 1) / kK3MlaThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // Tail chunks can be empty (p_start >= seq_len); they contribute the
    // (-inf, 0, 0) partial, which the combine weights to zero via e^-inf.
    for (uint p = p_start; p < p_end; ++p) {
        device const half* c_row = cache + uint(p) * row_len;
        float partial = 0.0f;
        for (uint i = lid; i < row_len; i += lsize) {
            partial = fma(q_smem[i], float(c_row[i]), partial);
        }
        float s = k3_mla_block_reduce_sum(
            partial, simd_lane_id, simd_group_id, simdgroups,
            reduce_scratch, &bcast);
        s *= scale;

        const float m_new = max(m_run, s);
        const float alpha = exp(m_run - m_new);
        const float p_exp = exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        uint slot = 0;
        for (uint i = lid; i < L; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(c_row[i]);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(h) * num_chunks + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * L;
    uint slot = 0;
    for (uint i = lid; i < L; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// Split-KV combine: grid = H. Merges chunk partials with the standard
// rescale and writes the fp32 out_lat (the input of k3_mla_out_project).
kernel void k3_mla_attn_decode_combine(
    device const float* m_in      [[buffer(0)]],   // [H * num_chunks]
    device const float* d_in      [[buffer(1)]],
    device const float* o_in      [[buffer(2)]],   // [H * num_chunks * L]
    device float*       out_lat   [[buffer(3)]],   // [H][L] fp32
    constant uint&      latent    [[buffer(4)]],
    constant uint&      num_heads [[buffer(5)]],
    constant uint&      num_chunks [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    const uint H = k3_mla_fc_num_heads(num_heads);
    const uint L = k3_mla_fc_latent(latent);
    const uint NC = num_chunks;
    const uint h = tg_id;
    if (h >= H) return;
    device const float* m_row = m_in + uint(h) * NC;
    device const float* d_row = d_in + uint(h) * NC;
    device const float* o_base = o_in + uint(h) * NC * L;

    // num_chunks is small; every thread recomputes the max/denominator
    // rather than paying a threadgroup reduction + barriers (house style).
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * exp(m_row[c] - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device float* out_row = out_lat + uint(h) * L;
    for (uint i = lid; i < L; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * L + i] * exp(m_row[c] - m_glob);
        }
        out_row[i] = acc * inv_d;
    }
}

// o_h = W_kvb_v,h . out_lat_h, flattened, then . sigmoid(gate). One SIMD
// per (h, i) dot over L. Output is fp16 for the external o_proj trunk GEMV.
kernel void k3_mla_out_project(
    device const half*  v_w     [[buffer(0)]],   // [H][V][L] fp16
    device const float* out_lat [[buffer(1)]],   // [H][L] fp32
    device const half*  gate    [[buffer(2)]],   // [H*V] fp16 (g_proj output)
    device half*        out     [[buffer(3)]],   // [H*V] fp16
    constant uint&      num_heads [[buffer(4)]],
    constant uint&      latent  [[buffer(5)]],
    constant uint&      v_head  [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint H = k3_mla_fc_num_heads(num_heads);
    const uint L = k3_mla_fc_latent(latent);
    const uint V = k3_mla_fc_vhead(v_head);
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= H * V) return;
    const uint h = row / V;

    device const half* w_row = v_w + uint(row) * L;
    device const float* x_row = out_lat + uint(h) * L;
    float acc = 0.0f;
    for (uint i = lane; i < L; i += 32u) {
        acc = fma(float(w_row[i]), x_row[i], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[row] = half(acc * k3_mla_sigmoid(float(gate[row])));
    }
}

// Cache append: writes [ kv_a_layernorm(latent) | rope ] as fp16 at
// `position`. Input is the raw kv_a projection (fp16, L+R); the RMSNorm over
// the first L channels (fp32 math, learned fp32 weight) runs in-kernel.
// One threadgroup, 256 threads.
kernel void k3_mla_cache_append(
    device const half*  kv_a        [[buffer(0)]],   // [L+R] fp16
    device const float* norm_weight [[buffer(1)]],   // [L] fp32
    device half*        cache       [[buffer(2)]],   // [cap][L+R] fp16
    constant uint&      position    [[buffer(3)]],
    constant uint&      latent      [[buffer(4)]],
    constant uint&      rope        [[buffer(5)]],
    constant float&     eps         [[buffer(6)]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[kK3MlaMaxSimdGroups];
    threadgroup float bcast;
    const uint L = k3_mla_fc_latent(latent);
    const uint R = k3_mla_fc_rope(rope);

    float ss = 0.0f;
    for (uint i = lid; i < L; i += lsize) {
        const float v = float(kv_a[i]);
        ss = fma(v, v, ss);
    }
    const float total = k3_mla_block_reduce_sum(
        ss, simd_lane_id, simd_group_id, simdgroups, reduce_scratch, &bcast);
    const float inv = rsqrt(total / float(L) + eps);

    device half* row = cache + uint(position) * (L + R);
    for (uint i = lid; i < L; i += lsize) {
        row[i] = half(norm_weight[i] * float(kv_a[i]) * inv);
    }
    for (uint i = L + lid; i < L + R; i += lsize) {
        row[i] = kv_a[i];   // rope part: cached raw, never rotated
    }
}
