#include <metal_stdlib>
using namespace metal;

// ============================================================================
// sampling_k3 — Kimi K3 output sampler (vocab 163,840, NO softcap).
//
//   k3_softmax_f32   p[i] = exp(z[i] - max) / sum exp(z - max)   (fp32 in/out)
//   k3_sample_f32    one token id from the probs                  (fp32 in)
//
// Logic mirrors the house `logit.metal` (`logit_softcap_softmax` + `sample`)
// exactly, minus the Gemma final-logit softcap, and reads/writes FP32 instead
// of FP16 (the K3 lm_head materializes fp32 logits; 163,840 floats = 640 KB,
// no need for fp16 storage). Sampler order is the house / mlx-lm order:
// repetition penalty (host) → softmax → top-p → top-k → temperature draw.
// Greedy (temperature == 0) is a pure GPU argmax. The PRNG streams
// (xorshift64* for the CDF draw, splitmix64-domain Gumbel for the plain
// temperature path) are bit-identical to the house kernels so seeded runs are
// reproducible; the K3 library compiles with safe math, so transcendentals
// are the precise variants.
// ============================================================================

constant constexpr uint kK3SampleMaxSimdGroups = 8;   // 256 threads / 32-lane SIMD
constant constexpr float kK3SampleTopMaxK = 256.0f;   // cap for top-k mask scan

// Thomas's "xorshift64*" variant — identical to the house sampler.
inline uint64_t k3_xorshift64(thread uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s * 2685821657736338717ULL;
}

inline float k3_uniform01(thread uint64_t& s) {
    uint32_t bits = uint32_t(k3_xorshift64(s) >> 40);
    return float(bits) * (1.0f / 16777216.0f);
}

// Counter-based SplitMix64, same constants as the house `lmhead_splitmix64`.
inline uint64_t k3_splitmix64(uint64_t x) {
    x = x + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline float k3_gumbel_for(uint64_t seed, uint position, uint row) {
    uint64_t key = seed ^ (uint64_t(position) * 0xD2B74407B1CE6E93ull) ^ uint64_t(row);
    uint64_t r   = k3_splitmix64(key);
    uint32_t bits = uint32_t(r >> 40);
    float u = (float(bits) + 0.5f) * (1.0f / 16777216.0f);
    return -log(-log(u));
}

// ----------------------------------------------------------------------------
// Online safe softmax over V (Milakov & Gimelshein), the house
// `logit_softcap_softmax` two-stage threadgroup reduction minus the softcap.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void k3_softmax_f32(
    device const float* logits   [[buffer(0)]],   // [V] fp32
    device       float* probs    [[buffer(1)]],   // [V] fp32
    constant     uint&  V        [[buffer(2)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_m[kK3SampleMaxSimdGroups];
    threadgroup float partial_d[kK3SampleMaxSimdGroups];
    threadgroup float final_m;
    threadgroup float final_inv_d;

    float m = -INFINITY;
    float d = 0.0f;

    for (uint i = lid; i < V; i += lsize) {
        float z  = logits[i];
        float mn = max(m, z);
        float scale = (m == -INFINITY) ? 0.0f : exp(m - mn);
        d = d * scale + exp(z - mn);
        m = mn;
    }

    float m_simd = simd_max(m);
    float d_simd = simd_sum((m == -INFINITY) ? 0.0f : d * exp(m - m_simd));

    if (simd_lane_id == 0) {
        partial_m[simd_group_id] = m_simd;
        partial_d[simd_group_id] = d_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float mp = (simd_lane_id < simdgroups) ? partial_m[simd_lane_id] : -INFINITY;
        float dp = (simd_lane_id < simdgroups) ? partial_d[simd_lane_id] : 0.0f;
        float m_all = simd_max(mp);
        float d_all = simd_sum((mp == -INFINITY) ? 0.0f : dp * exp(mp - m_all));
        if (simd_lane_id == 0) {
            final_m     = m_all;
            final_inv_d = 1.0f / d_all;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float m_final     = final_m;
    const float inv_d_final = final_inv_d;

    for (uint i = lid; i < V; i += lsize) {
        probs[i] = exp(logits[i] - m_final) * inv_d_final;
    }
}

// ----------------------------------------------------------------------------
// Sampler. Same three paths as the house `sample` kernel:
//   temperature == 0                          greedy argmax (lowest index ties)
//   temperature > 0, no top-k/top-p           Gumbel-argmax fast path
//   temperature > 0, top-k and/or top-p set   k-pass extraction, top-p cut on
//                                             descending mass, temperature
//                                             reweight, seeded CDF draw
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void k3_sample_f32(
    device const float*    probs        [[buffer(0)]],   // [V] fp32
    device       uint*     out_token    [[buffer(1)]],   // [1] UInt32
    constant     uint&     V            [[buffer(2)]],
    constant     float&    temperature  [[buffer(3)]],
    constant     uint&     top_k        [[buffer(4)]],
    constant     float&    top_p        [[buffer(5)]],
    constant     uint64_t& seed         [[buffer(6)]],
    constant     uint&     position     [[buffer(7)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_val[kK3SampleMaxSimdGroups];
    threadgroup uint  partial_idx[kK3SampleMaxSimdGroups];
    threadgroup uint  chosen;

    // -- Greedy path: argmax(probs); softmax is monotonic.
    if (temperature <= 0.0f) {
        float best_v = -INFINITY;
        uint  best_i = 0;

        for (uint i = lid; i < V; i += lsize) {
            float p = probs[i];
            if (p > best_v) {
                best_v = p;
                best_i = i;
            }
        }

        float m_simd = simd_max(best_v);
        uint  i_simd = (best_v == m_simd) ? best_i : 0xFFFFFFFFu;
        i_simd = simd_min(i_simd);

        if (simd_lane_id == 0) {
            partial_val[simd_group_id] = m_simd;
            partial_idx[simd_group_id] = i_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial_val[simd_lane_id] : -INFINITY;
            uint  i = (simd_lane_id < simdgroups) ? partial_idx[simd_lane_id] : 0xFFFFFFFFu;
            float m_all = simd_max(v);
            uint  i_all = (v == m_all) ? i : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (simd_lane_id == 0) {
                out_token[0] = i_all;
            }
        }
        return;
    }

    // -- Fast stochastic path: no top-k, no top-p. argmax_i(log p_i / T + g_i).
    if (top_k == 0u && (top_p <= 0.0f || top_p >= 1.0f)) {
        float inv_t = 1.0f / temperature;
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = lid; i < V; i += lsize) {
            float p = probs[i];
            if (!(p > 0.0f)) continue;
            float s = inv_t * log(p) + k3_gumbel_for(seed, position, i);
            if (s > best_v) {
                best_v = s;
                best_i = i;
            }
        }

        float m_simd = simd_max(best_v);
        uint  i_simd = (best_v == m_simd) ? best_i : 0xFFFFFFFFu;
        i_simd = simd_min(i_simd);

        if (simd_lane_id == 0) {
            partial_val[simd_group_id] = m_simd;
            partial_idx[simd_group_id] = i_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial_val[simd_lane_id] : -INFINITY;
            uint  i = (simd_lane_id < simdgroups) ? partial_idx[simd_lane_id] : 0xFFFFFFFFu;
            float m_all = simd_max(v);
            uint  i_all = (v == m_all) ? i : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (simd_lane_id == 0) {
                out_token[0] = (i_all == 0xFFFFFFFFu) ? 0u : i_all;
            }
        }
        return;
    }

    // -- Truncating stochastic path (top-k and/or top-p set).
    threadgroup float topk_val[256];
    threadgroup uint  topk_idx[256];
    threadgroup uint  topk_count;

    uint k_req = (top_k == 0u) ? V : top_k;
    uint k     = min(k_req, uint(kK3SampleTopMaxK));
    k          = min(k, V);
    float inv_temp = 1.0f / temperature;

    if (lid == 0) {
        topk_count = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < k; ++slot) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = lid; i < V; i += lsize) {
            float p = probs[i];
            bool claimed = false;
            for (uint c = 0; c < slot; ++c) {
                if (topk_idx[c] == i) { claimed = true; break; }
            }
            if (claimed) continue;
            if (p > best_v) {
                best_v = p;
                best_i = i;
            }
        }

        float m_simd = simd_max(best_v);
        uint  i_simd = (best_v == m_simd) ? best_i : 0xFFFFFFFFu;
        i_simd = simd_min(i_simd);

        if (simd_lane_id == 0) {
            partial_val[simd_group_id] = m_simd;
            partial_idx[simd_group_id] = i_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial_val[simd_lane_id] : -INFINITY;
            uint  i = (simd_lane_id < simdgroups) ? partial_idx[simd_lane_id] : 0xFFFFFFFFu;
            float m_all = simd_max(v);
            uint  i_all = (v == m_all) ? i : 0xFFFFFFFFu;
            i_all = simd_min(i_all);
            if (simd_lane_id == 0) {
                topk_val[slot] = m_all;
                topk_idx[slot] = i_all;
                topk_count     = slot + 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Top-p truncation and sample, single-threaded (k <= 256 elements).
    if (lid == 0) {
        uint kept = 0;
        while (kept < topk_count
               && topk_idx[kept] < V
               && isfinite(topk_val[kept])) {
            kept += 1;
        }
        // Top-P against the still-full-vocabulary-normalized masses, then the
        // top-k list is the limiting filter (house/mlx-lm order).
        if (top_p > 0.0f && top_p < 1.0f) {
            float cum = 0.0f;
            uint  cut = kept;
            for (uint i = 0; i < kept; ++i) {
                cum += topk_val[i];
                if (cum >= top_p) { cut = i + 1; break; }
            }
            kept = cut;
        }

        for (uint i = 0; i < kept; ++i) {
            if (temperature != 1.0f) {
                topk_val[i] = pow(topk_val[i], inv_temp);
            }
        }

        float surviving = 0.0f;
        for (uint i = 0; i < kept; ++i) surviving += topk_val[i];

        uint64_t rng = seed;
        // One xorshift step before sampling (same as the house kernel).
        (void)k3_xorshift64(rng);
        float u = k3_uniform01(rng) * surviving;

        float run = 0.0f;
        uint  picked = topk_idx[0];
        for (uint i = 0; i < kept; ++i) {
            run += topk_val[i];
            if (u <= run) { picked = topk_idx[i]; break; }
        }

        chosen = picked;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        out_token[0] = chosen;
    }
}
