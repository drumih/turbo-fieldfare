#include <metal_stdlib>
using namespace metal;

// ============================================================================
// logit.metal — output-head kernels (#12, #13 in the inventory).
//
//   logit_softcap_softmax  Apply 30 * tanh(z / 30) to raw logits and produce a
//                          numerically-stable softmax over V=262144.
//   sample                 Sample a single token id from softmaxed probs.
//                          Supports greedy (temperature=0), temperature, top-k,
//                          top-p with a seeded PRNG.
//
// Both kernels run in a single threadgroup of 256 threads. V is far too large
// to fit in threadgroup memory (262144 * 4 B = 1 MB), so reductions are done
// in two stages: each thread strides over V, then a SIMD-group + cross-SIMD
// merge collapses partials. Threadgroup memory stores only one float per
// SIMD-group (up to 8 lanes for 256-thread groups on Apple silicon).
// ============================================================================

constant constexpr uint kLogitMaxSimdGroups = 8;
constant constexpr float kSampleTopMaxK     = 256.0f;  // cap for top-k mask scan

// ----------------------------------------------------------------------------
// K8: logit_softcap_softmax
//
// Online safe softmax (Milakov & Gimelshein, arXiv:1805.02867). Each thread
// walks its stride of V and maintains a running (m, d) pair: m is the max
// softcap'd logit seen so far, d is sum(exp(logit_i - m)). On every new logit
// the rule is:
//
//   m_new = max(m, z')
//   d_new = d * exp(m - m_new) + exp(z' - m_new)
//
// Softcap is applied inline before the max update — going through DRAM with a
// separate softcap pass costs 1 MB of round-trip traffic.
//
// After the per-thread loop the (m, d) pairs are merged across the threadgroup
// using the same rule, broadcast, and a second pass writes normalized FP16
// probabilities.
// ----------------------------------------------------------------------------

inline float softcap_value(float z, float softcap) {
    // A NaN logit must not take the rest of the row with it. `max(x, NaN)` and
    // `exp(NaN)` both stay NaN, so one bad value leaves the running max and the
    // running sum NaN for the whole vocabulary and the normalized row NaN; the
    // sampler can only answer an in-range index, so it answered token 0 for
    // every remaining position and a broken model looked like a model that
    // likes token 0. Folding it to the -inf the sum is already built to
    // tolerate excludes that element and leaves the softmax over the *finite*
    // logits, which is the distribution a reader expects.
    //
    // This has to come before the softcap == 0 early return: with capping
    // disabled (`finalLogitSoftcap == 0`) there is no tanh to swallow it, and
    // the NaN would reach the reduction unchanged.
    //
    // Only NaN is folded. A +inf logit is a claim rather than a failure, and
    // `tanh` below saturates it to `softcap` -- the most likely token, which is
    // what an infinite logit means.
    if (isnan(z)) return -INFINITY;
    // softcap <= 0 disables capping (architectures without a final logit
    // softcap, e.g. Qwen 3.6).
    if (softcap <= 0.0f) return z;
    // tanh saturates well before |z/softcap|=10, so values like +1e3 collapse
    // cleanly to softcap=30 without exp overflow downstream.
    return softcap * precise::tanh(z / softcap);
}

inline float logit_softmax_exp(float x) {
    return fast::exp(x);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void logit_softcap_softmax(
    device const half*  logits   [[buffer(0)]],   // [V] FP16
    device       half*  probs    [[buffer(1)]],   // [V] FP16
    constant     uint&  V        [[buffer(2)]],
    constant     float& softcap  [[buffer(3)]],
    device       float* row_max  [[buffer(4)]],   // [1] the max this row used
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    // Per-SIMD-group partials for both the running max and the running sum.
    // Sized for the worst case of 8 SIMD-groups (256 threads / 32-lane SIMD).
    threadgroup float partial_m[kLogitMaxSimdGroups];
    threadgroup float partial_d[kLogitMaxSimdGroups];
    threadgroup float final_m;
    threadgroup float final_inv_d;
    // -log-of-zero sentinel: any real logit beats this on the first compare.
    float m = -INFINITY;
    float d = 0.0f;

    for (uint i = lid; i < V; i += lsize) {
        float z  = softcap_value(float(logits[i]), softcap);
        float mn = max(m, z);
        // Two guards, and the sum needs both. `scale` rescales the running sum
        // to the new max; when the first element is -inf the running max is
        // still -inf, so `exp(-inf - -inf)` would be NaN and `d * 0 + NaN`
        // poisons the thread's sum for the rest of its elements even though a
        // -inf logit contributes nothing. `contribution` is that term. This is
        // reachable whenever the softcap is disabled (`finalLogitSoftcap == 0`,
        // the Qwen 3.6 production setting) and a logit arrives as -inf: before
        // this the row came out empty and the sampler answered token 0.
        float scale = (m == -INFINITY) ? 0.0f : logit_softmax_exp(m - mn);
        float contribution = (mn == -INFINITY) ? 0.0f : logit_softmax_exp(z - mn);
        d = d * scale + contribution;
        m = mn;
    }

    // -- SIMD-group reduce.
    // simd_max gives the SIMD-wide max in one instruction; we then rescale our
    // own d to that common max so all 32 lanes share a comparable d term.
    float m_simd = simd_max(m);
    // Lanes that saw no element (m == -inf) must contribute 0, not NaN: when an
    // entire SIMD group is empty (V < threads_per_threadgroup, e.g. small-vocab
    // toy models) m_simd is also -inf, and exp(-inf - -inf) = exp(NaN) = NaN
    // would poison the sum. 0 * NaN is NaN, so guard before the multiply.
    float d_simd = simd_sum((m == -INFINITY) ? 0.0f : d * logit_softmax_exp(m - m_simd));

    if (simd_lane_id == 0) {
        partial_m[simd_group_id] = m_simd;
        partial_d[simd_group_id] = d_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Cross-SIMD merge in SIMD-group 0. Up to kLogitMaxSimdGroups partials.
    // The defaults are written by thread 0 only — the same thread that writes
    // the real values below, so no barrier is needed between them.
    //
    // Writing them from every thread (to silence the sometimes-uninitialized
    // diagnostic) is a data race: a thread in another SIMD group can store the
    // default *after* lane 0 of group 0 has stored the real value, leaving
    // final_m = -inf and final_inv_d = 0. The normalize loop below then
    // computes exp(z - -inf) * 0 = inf * 0 = NaN for every entry, so the whole
    // probability row becomes NaN. Downstream, `sample`'s top-k reduction finds
    // no finite mass, every slot collapses to the UINT_MAX sentinel, and the
    // draw returns an out-of-range token id.
    if (lid == 0) {
        final_m     = -INFINITY;
        final_inv_d = 0.0f;
    }
    if (simd_group_id == 0) {
        float mp = (simd_lane_id < simdgroups) ? partial_m[simd_lane_id] : -INFINITY;
        float dp = (simd_lane_id < simdgroups) ? partial_d[simd_lane_id] : 0.0f;

        float m_all = simd_max(mp);
        // Same rescale trick, same empty-lane guard as above: an empty partial
        // (mp == -inf) contributes 0, never NaN.
        float d_all = simd_sum((mp == -INFINITY) ? 0.0f : dp * logit_softmax_exp(mp - m_all));

        if (simd_lane_id == 0) {
            final_m     = m_all;
            // Reciprocal once so the normalize loop is a single multiply.
            final_inv_d = (d_all > 0.0f) ? (1.0f / d_all) : 0.0f;
            // Published so the host can tell an empty row from a peaked one
            // without a vocabulary-sized readback. Non-finite here means the
            // row carried no finite logit at all.
            row_max[0] = m_all;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float m_final     = final_m;
    const float inv_d_final = final_inv_d;
    // `m_final` is non-finite only when every logit was (the NaN fold above, or
    // an all -inf row). The row then has no mass; writing zeros keeps it a
    // *defined* empty distribution instead of exp(-inf - -inf) * 0 = NaN, so
    // the sampler's own in-range fallback is what answers and the logits stay
    // the only thing that was broken.
    const bool has_mass = isfinite(m_final);

    for (uint i = lid; i < V; i += lsize) {
        float z = softcap_value(float(logits[i]), softcap);
        probs[i] = has_mass
            ? half(logit_softmax_exp(z - m_final) * inv_d_final)
            : half(0.0f);
    }
}

// ----------------------------------------------------------------------------
// Tiled softcap+softmax. Same math as `logit_softcap_softmax`, but the two
// full-vocabulary passes are spread across the device instead of running in
// one 256-thread threadgroup. Stage 1 emits per-tile (max, sum-of-exp)
// partials; stage 2 merges them into a single (max, 1/sum); stage 3
// normalizes. At V = 248,320 the single-threadgroup form left the rest of the
// GPU idle for both passes, which is the same pathology the tiled Top-K
// reduction removed from the selection half.
constant constexpr uint kSoftmaxTile = 4096;

[[kernel, max_total_threads_per_threadgroup(256)]]
void logit_softcap_softmax_tiled_stage1(
    device const half*  logits   [[buffer(0)]],
    device float*       out_max  [[buffer(1)]],
    device float*       out_sum  [[buffer(2)]],
    constant uint&      V        [[buffer(3)]],
    constant float&     softcap  [[buffer(4)]],
    uint  group            [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_m[kLogitMaxSimdGroups];
    threadgroup float partial_d[kLogitMaxSimdGroups];
    const uint base = group * kSoftmaxTile;
    const uint end  = min(base + kSoftmaxTile, V);

    float m = -INFINITY;
    float d = 0.0f;
    for (uint i = base + lid; i < end; i += lsize) {
        float z  = softcap_value(float(logits[i]), softcap);
        float mn = max(m, z);
        // Same two guards as the single-threadgroup form, same reason.
        float scale = (m == -INFINITY) ? 0.0f : logit_softmax_exp(m - mn);
        float contribution = (mn == -INFINITY) ? 0.0f : logit_softmax_exp(z - mn);
        d = d * scale + contribution;
        m = mn;
    }
    float m_simd = simd_max(m);
    float d_simd = simd_sum((m == -INFINITY) ? 0.0f : d * logit_softmax_exp(m - m_simd));
    if (simd_lane_id == 0) {
        partial_m[simd_group_id] = m_simd;
        partial_d[simd_group_id] = d_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float mp = (simd_lane_id < simdgroups) ? partial_m[simd_lane_id] : -INFINITY;
        float dp = (simd_lane_id < simdgroups) ? partial_d[simd_lane_id] : 0.0f;
        float m_all = simd_max(mp);
        float d_all = simd_sum((mp == -INFINITY) ? 0.0f : dp * logit_softmax_exp(mp - m_all));
        if (simd_lane_id == 0) {
            out_max[group] = m_all;
            out_sum[group] = d_all;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void logit_softcap_softmax_tiled_merge(
    device const float* in_max   [[buffer(0)]],
    device const float* in_sum   [[buffer(1)]],
    device float*       out_pair [[buffer(2)]],   // [max, 1/sum]
    constant uint&      tiles    [[buffer(3)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_m[kLogitMaxSimdGroups];
    threadgroup float partial_d[kLogitMaxSimdGroups];
    float m = -INFINITY;
    float d = 0.0f;
    for (uint i = lid; i < tiles; i += lsize) {
        float tm = in_max[i];
        float mn = max(m, tm);
        float scale = (m == -INFINITY) ? 0.0f : logit_softmax_exp(m - mn);
        float contribution = (tm == -INFINITY)
            ? 0.0f : in_sum[i] * logit_softmax_exp(tm - mn);
        d = d * scale + contribution;
        m = mn;
    }
    float m_simd = simd_max(m);
    float d_simd = simd_sum((m == -INFINITY) ? 0.0f : d * logit_softmax_exp(m - m_simd));
    if (simd_lane_id == 0) {
        partial_m[simd_group_id] = m_simd;
        partial_d[simd_group_id] = d_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float mp = (simd_lane_id < simdgroups) ? partial_m[simd_lane_id] : -INFINITY;
        float dp = (simd_lane_id < simdgroups) ? partial_d[simd_lane_id] : 0.0f;
        float m_all = simd_max(mp);
        float d_all = simd_sum((mp == -INFINITY) ? 0.0f : dp * logit_softmax_exp(mp - m_all));
        if (simd_lane_id == 0) {
            out_pair[0] = m_all;
            out_pair[1] = (d_all > 0.0f) ? (1.0f / d_all) : 0.0f;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void logit_softcap_softmax_tiled_normalize(
    device const half*  logits   [[buffer(0)]],
    device half*        probs    [[buffer(1)]],
    device const float* pair     [[buffer(2)]],
    constant uint&      V        [[buffer(3)]],
    constant float&     softcap  [[buffer(4)]],
    uint  gid [[thread_position_in_grid]]
) {
    if (gid >= V) return;
    const float z = softcap_value(float(logits[gid]), softcap);
    // Same no-mass rule as the single-threadgroup form; `pair[0]` is the row
    // max the merge pass settled on and is non-finite only for a row with no
    // finite logit at all.
    probs[gid] = isfinite(pair[0])
        ? half(logit_softmax_exp(z - pair[0]) * pair[1])
        : half(0.0f);
}

// ----------------------------------------------------------------------------
// K9: sample
//
// Reads softmaxed probabilities and writes one token id. Behaviour:
//
//   temperature == 0   greedy argmax (the top-k / top-p / rng inputs are
//                      ignored; the argmax of probs is also the argmax of
//                      logits because softmax is monotonic).
//   temperature  > 0   top-p filters against the full normalized probability
//                      distribution, top-k caps the surviving set, then
//                      temperature reweights only that final categorical
//                      draw. This matches mlx-lm's sampler-chain order.
//
// xorshift64 is chosen for two reasons: it has a single 64-bit state (cheap
// to seed and to broadcast through threadgroup memory) and the output is bit-
// reproducible across hardware and compilers, which is what the tests assert.
// ----------------------------------------------------------------------------

// Thomas's "xorshift64*" variant — period 2^64-1, bias is negligible for
// sampling and the multiplier scrambles low-bit correlation. We only call it
// from one thread per dispatch, so contention is a non-issue.
inline uint64_t xorshift64(thread uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s * 2685821657736338717ULL;
}

inline float uniform01(thread uint64_t& s) {
    // Use the top 24 bits as a [0,1) float — matches IEEE-754 mantissa width
    // so the result is exactly representable.
    uint32_t bits = uint32_t(xorshift64(s) >> 40);
    return float(bits) * (1.0f / 16777216.0f);
}

// Counter-based SplitMix64 keeps Gumbel draws deterministic across replays at
// a fixed seed without a precomputed random buffer.
inline uint64_t lmhead_splitmix64(uint64_t x) {
    x = x + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline float lmhead_gumbel_for(uint64_t seed, uint position, uint row) {
    // Domain separation: mix (seed, position, row) by xors before the
    // splitmix scramble so adjacent rows yield independent draws.
    uint64_t key = seed ^ (uint64_t(position) * 0xD2B74407B1CE6E93ull) ^ uint64_t(row);
    uint64_t r   = lmhead_splitmix64(key);
    // [0, 1) with 24 mantissa bits; clamp away from zero/one so the double-log
    // can never overflow. 1/16777216 maps to ~5.96e-8, well above FP32 ULP.
    uint32_t bits = uint32_t(r >> 40);
    float u = (float(bits) + 0.5f) * (1.0f / 16777216.0f);
    return -log(-log(u));
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample(
    device const half*    probs        [[buffer(0)]],   // [V] FP16
    device       uint*    out_token    [[buffer(1)]],   // [1] UInt32
    constant     uint&    V            [[buffer(2)]],
    constant     float&   temperature  [[buffer(3)]],
    constant     uint&    top_k        [[buffer(4)]],
    constant     float&   top_p        [[buffer(5)]],
    constant     uint64_t& seed        [[buffer(6)]],
    constant     uint&    position     [[buffer(7)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_val[kLogitMaxSimdGroups];
    threadgroup uint  partial_idx[kLogitMaxSimdGroups];
    threadgroup uint  chosen;

    // ------------------------------------------------------------------
    // Greedy path (temperature == 0). Pure argmax(probs).
    // Softmax is monotonic, so we can argmax over probs directly without
    // looking back at the raw logits.
    // ------------------------------------------------------------------
    if (temperature <= 0.0f) {
        float best_v = -INFINITY;
        uint  best_i = 0;

        for (uint i = lid; i < V; i += lsize) {
            float p = float(probs[i]);
            if (p > best_v) {
                best_v = p;
                best_i = i;
            }
        }

        // SIMD-group argmax: simd_max gives the value; we need the matching
        // index, which simd_max can't return. Roll it manually: every lane
        // broadcasts (val, idx) via simd_shuffle and keeps the best pair.
        // 32-lane fan-in is one log2(32)=5-step tree but the lane-by-lane
        // scan is equally fast at this size and simpler to read.
        float m_simd = simd_max(best_v);
        uint  i_simd = best_i;
        // Tie-break: lowest index wins. Lanes whose value isn't the SIMD max
        // bow out with idx = UINT_MAX so simd_min picks the surviving smallest.
        i_simd = (best_v == m_simd) ? best_i : 0xFFFFFFFFu;
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

    // ------------------------------------------------------------------
    // Fast stochastic path: no top-k, no top-p. Sampling from p^(1/T)/Z is
    // exactly argmax_i(log(p_i)/T + g_i) with iid Gumbel noise g_i — one
    // parallel V-pass, same shape as the greedy argmax above. The top-k
    // extraction loop below costs k full V-passes (k clamps to 256 when
    // top_k is "disabled"), which measured at ~2.6 s/token at V=262144;
    // this path replaces it for the common plain-temperature case.
    // ------------------------------------------------------------------
    if (top_k == 0u && (top_p <= 0.0f || top_p >= 1.0f)) {
        float inv_t = 1.0f / temperature;
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = lid; i < V; i += lsize) {
            float p = float(probs[i]);
            if (!(p > 0.0f)) continue;
            float s = inv_t * log(p) + lmhead_gumbel_for(seed, position, i);
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
                // All-zero probs cannot occur post-softmax, but a fully
                // skipped scan would leave UINT_MAX; fall back to index 0.
                out_token[0] = (i_all == 0xFFFFFFFFu) ? 0u : i_all;
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // Truncating stochastic path (top-k and/or top-p set).
    //
    // 1. Find the top-k threshold by repeatedly extracting the max and
    //    storing it in a small threadgroup buffer (k <= kSampleTopMaxK).
    // 2. Top-p truncates against the full-vocabulary probability mass.
    // 3. Temperature reweights the survivors as p^(1/T).
    // 4. CDF inverse-transform sample with the seeded PRNG.
    //
    // We use a *single* thread for the top-k / top-p / sample logic. Each
    // top-k slot costs one full V-pass, so explicit small top_k stays
    // tolerable; the k=256 worst case is exactly what the fast path above
    // removes from the default configuration.
    // ------------------------------------------------------------------

    // -- Threadgroup buffers for the sampler state. 256 floats + 256 indices
    // = 3 KB, comfortably inside the 32 KB threadgroup limit on Apple9+.
    threadgroup float topk_val[256];
    threadgroup uint  topk_idx[256];
    threadgroup uint  topk_count;

    // Effective k. top_k == 0 disables top-k entirely; we represent that as
    // k = V (subject to the kSampleTopMaxK working-buffer cap).
    uint k_req = (top_k == 0u) ? V : top_k;
    uint k     = min(k_req, uint(kSampleTopMaxK));
    k          = min(k, V);
    float inv_temp = 1.0f / temperature;

    if (lid == 0) {
        topk_count = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Extract top-k via k repeated argmax passes.
    // For each slot: every thread scans its V-stride for the argmax that
    // hasn't already been claimed, then we reduce across the threadgroup.
    // Claimed slots are tracked by index, not by zeroing the input — leaves
    // the probs buffer untouched (it's `device const`).
    for (uint slot = 0; slot < k; ++slot) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = lid; i < V; i += lsize) {
            float p = float(probs[i]);
            // Skip already-claimed indices.
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
        // Drop trailing invalid slots. When the working set k exceeds the
        // number of rankable entries (high temperature flattens ties; the tail
        // can exhaust before k slots fill), a slot's reduction yields a -INF
        // value / UINT_MAX index. Those sort last (descending by mass), so the
        // first invalid slot bounds the real count. Slot 0 is the global argmax
        // and is always valid, so kept >= 1 holds whenever V > 0.
        uint kept = 0;
        while (kept < topk_count
               && topk_idx[kept] < V
               && isfinite(topk_val[kept])) {
            kept += 1;
        }
        // mlx-lm applies Top-P before Top-K. The probabilities still carry
        // their full-vocabulary normalization here, so comparing the raw
        // descending cumulative mass to top_p produces the same intersection
        // without sorting the entire vocabulary. If Top-64 holds less than
        // top_p mass, all 64 survive and Top-K is the limiting filter.
        if (top_p > 0.0f && top_p < 1.0f) {
            float cum = 0.0f;
            uint  cut = kept;
            for (uint i = 0; i < kept; ++i) {
                cum += topk_val[i];
                if (cum >= top_p) { cut = i + 1; break; }
            }
            kept = cut;
        }

        // MLX applies temperature in categorical_sampling after both filters.
        for (uint i = 0; i < kept; ++i) {
            if (temperature != 1.0f) {
                topk_val[i] = pow(topk_val[i], inv_temp);
            }
        }

        // Sum the reweighted surviving mass for inverse-CDF sampling.
        float surviving = 0.0f;
        for (uint i = 0; i < kept; ++i) surviving += topk_val[i];

        uint64_t rng = seed;
        // One xorshift step before sampling so identical seeds across calls
        // don't produce the same first draw when the seed is small (a few
        // bits short of full entropy on xorshift's first output).
        (void)xorshift64(rng);
        float u = uniform01(rng) * surviving;

        // Pick by walking the CDF until we cross u. With kept <= 256 this is
        // a tight loop and stays inside L1.
        // Default to slot 0 (the global argmax — always a valid index) so a
        // CDF that never crosses u (FP rounding leaves run slightly below the
        // surviving mass) still returns a real token, never a tail sentinel.
        // `kept == 0` means slot 0 itself failed the `< V` / isfinite test
        // above — every entry reduced to the UINT_MAX sentinel, which happens
        // when the probability row carries no finite mass. Seeding `picked`
        // from topk_idx[0] unconditionally would then emit that sentinel as a
        // token id. Fall back to a real index instead: an arbitrary token beats
        // an out-of-range one that the generation loop would use to index the
        // vocabulary.
        float run = 0.0f;
        uint  picked = (kept > 0) ? topk_idx[0] : 0u;
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

// Specialized Top-64 selection. Each stage reduces sorted
// 1,024-entry tiles to 64 pairs, so the full 262,144-entry vocabulary reaches
// one final tile in three dispatches without a full-vocabulary FP32 copy.
inline bool sample_topk64_better(float lhs_value, uint lhs_index,
                                 float rhs_value, uint rhs_index) {
    return lhs_value > rhs_value
        || (lhs_value == rhs_value && lhs_index < rhs_index);
}

inline void sample_topk64_sort_tile(threadgroup float* values,
                                    threadgroup uint* indices,
                                    uint lid) {
    for (uint width = 2; width <= 1024; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid; i < 1024; i += 256) {
                uint partner = i ^ stride;
                if (partner <= i) continue;

                float lhs_value = values[i];
                uint lhs_index = indices[i];
                float rhs_value = values[partner];
                uint rhs_index = indices[partner];
                bool descending = (i & width) == 0;
                bool swap = descending
                    ? sample_topk64_better(rhs_value, rhs_index,
                                           lhs_value, lhs_index)
                    : sample_topk64_better(lhs_value, lhs_index,
                                           rhs_value, rhs_index);
                if (swap) {
                    values[i] = rhs_value;
                    indices[i] = rhs_index;
                    values[partner] = lhs_value;
                    indices[partner] = lhs_index;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample_topk64_stage1(
    device const half* probs [[buffer(0)]],
    device float* out_values [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    constant uint& V [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    threadgroup uint indices[1024];
    uint base = group * 1024;

    for (uint i = lid; i < 1024; i += 256) {
        uint source = base + i;
        bool valid = source < V;
        values[i] = valid ? float(probs[source]) : -INFINITY;
        indices[i] = valid ? source : 0xFFFFFFFFu;
    }
    sample_topk64_sort_tile(values, indices, lid);

    if (lid < 64) {
        uint destination = group * 64 + lid;
        out_values[destination] = values[lid];
        out_indices[destination] = indices[lid];
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample_topk64_reduce(
    device const float* in_values [[buffer(0)]],
    device const uint* in_indices [[buffer(1)]],
    device float* out_values [[buffer(2)]],
    device uint* out_indices [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    threadgroup uint indices[1024];
    uint base = group * 1024;

    for (uint i = lid; i < 1024; i += 256) {
        uint source = base + i;
        bool valid = source < count;
        values[i] = valid ? in_values[source] : -INFINITY;
        indices[i] = valid ? in_indices[source] : 0xFFFFFFFFu;
    }
    sample_topk64_sort_tile(values, indices, lid);

    if (lid < 64) {
        uint destination = group * 64 + lid;
        out_values[destination] = values[lid];
        out_indices[destination] = indices[lid];
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sample_topk64_final(
    device const float* in_values [[buffer(0)]],
    device const uint* in_indices [[buffer(1)]],
    device uint* out_token [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    constant float& top_p [[buffer(5)]],
    constant uint64_t& seed [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    threadgroup uint indices[1024];

    for (uint i = lid; i < 1024; i += 256) {
        bool valid = i < count;
        values[i] = valid ? in_values[i] : -INFINITY;
        indices[i] = valid ? in_indices[i] : 0xFFFFFFFFu;
    }
    sample_topk64_sort_tile(values, indices, lid);

    if (lid == 0) {
        uint kept = 0;
        while (kept < 64
               && indices[kept] != 0xFFFFFFFFu
               && isfinite(values[kept])) {
            kept += 1;
        }

        // Top-K caps the working set *before* the Top-P scan, which is what
        // makes this path bit-identical to the generic `sample` kernel for
        // k <= 64. There, only k slots are ever extracted, so the cumulative
        // mass Top-P compares against is the mass of the top k — not of the
        // top 64. Cutting after Top-P instead would let a distribution whose
        // 0.95 mass is reached at rank 30 keep 30 candidates at k=20, where
        // the generic kernel keeps 20. Order matters; this is not a clamp
        // that can be moved for tidiness.
        kept = min(kept, max(1u, top_k));

        if (top_p > 0.0f && top_p < 1.0f) {
            float cumulative = 0.0f;
            uint cut = kept;
            for (uint i = 0; i < kept; ++i) {
                cumulative += values[i];
                if (cumulative >= top_p) {
                    cut = i + 1;
                    break;
                }
            }
            kept = cut;
        }

        float inv_temperature = 1.0f / temperature;
        for (uint i = 0; i < kept; ++i) {
            if (temperature != 1.0f) {
                values[i] = pow(values[i], inv_temperature);
            }
        }

        float surviving = 0.0f;
        for (uint i = 0; i < kept; ++i) surviving += values[i];
        uint64_t rng = seed;
        (void)xorshift64(rng);
        float u = uniform01(rng) * surviving;
        float running = 0.0f;
        // Same fallback as the generic `sample` kernel, for the same reason:
        // `kept == 0` means every slot failed the isfinite test, so `indices[0]`
        // is still the UINT_MAX sentinel. Emitting it hands the generation loop
        // an out-of-range id to index the vocabulary with. The default path for
        // topK <= 64 is this kernel, not the generic one.
        uint picked = (kept > 0) ? indices[0] : 0u;
        for (uint i = 0; i < kept; ++i) {
            running += values[i];
            if (u <= running) {
                picked = indices[i];
                break;
            }
        }
        out_token[0] = picked;
    }
}


// Fused greedy lm-head path. Eight SIMD groups each evaluate one INT4 row;
// a second dispatch reduces the per-threadgroup argmax summaries.
constant constexpr uint kLMHeadRowsPerTG = 8;
constant constexpr uint kLMHeadGroupSize = 64;
constant constexpr uint kLMHeadRowSummaryStride = 2;
constant uint FC_HEAD_D [[function_constant(10)]];
constant uint FC_HEAD_V [[function_constant(11)]];
constant bool FC_HEAD_USE_FC [[function_constant(13)]];

static inline uint lmhead_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_HEAD_USE_FC) &&
            FC_HEAD_USE_FC &&
            is_function_constant_defined(FC_HEAD_D)) ? FC_HEAD_D : D;
}

static inline uint lmhead_fc_v(constant uint& V) {
    return (is_function_constant_defined(FC_HEAD_USE_FC) &&
            FC_HEAD_USE_FC &&
            is_function_constant_defined(FC_HEAD_V)) ? FC_HEAD_V : V;
}

inline float lmhead_int4_gemv_row_simd_dev(device const uint8_t*    W,
                                           device const bfloat*     scales,
                                           device const bfloat*     biases,
                                           device const half*       x,
                                           uint row,
                                           uint D,
                                           uint lane) {
    const uint n_groups  = D / kLMHeadGroupSize;
    const uint row_bytes = D / 2u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kLMHeadGroupSize / 2) + lane];
        const float x0 = float(x[g * kLMHeadGroupSize + lane * 2u]);
        const float x1 = float(x[g * kLMHeadGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    return simd_sum(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void lm_head_greedy_int4_rows_chunk_raw(
    device const half*    x_normed     [[buffer(0)]],
    device const uint8_t* W            [[buffer(1)]],
    device const bfloat*  scales       [[buffer(2)]],
    device const bfloat*  biases       [[buffer(3)]],
    device       float*   summaries    [[buffer(4)]],
    constant     uint&    D            [[buffer(5)]],
    constant     uint&    V            [[buffer(6)]],
    uint  tg_idx         [[threadgroup_position_in_grid]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_v[kLogitMaxSimdGroups];
    threadgroup uint partial_i[kLogitMaxSimdGroups];
    const uint DD = lmhead_fc_d(D);
    const uint VV = lmhead_fc_v(V);

    const uint row = tg_idx * kLMHeadRowsPerTG + simd_group_id;
    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;

    if (row < VV) {
        float z = lmhead_int4_gemv_row_simd_dev(W, scales, biases,
                                                x_normed, row, DD, simd_lane_id);
        if (simd_lane_id == 0 && isfinite(z)) {
            best_v = z;
            best_i = row;
        }
    }

    if (simd_lane_id == 0) {
        partial_v[simd_group_id] = best_v;
        partial_i[simd_group_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const bool active = simd_lane_id < simdgroups;
        const float v = active ? partial_v[simd_lane_id] : -INFINITY;
        const uint idx = active ? partial_i[simd_lane_id] : 0xFFFFFFFFu;
        const float v_all = simd_max(v);
        uint i_all = (v == v_all) ? idx : 0xFFFFFFFFu;
        i_all = simd_min(i_all);
        if (simd_lane_id == 0) {
            device float* slot = summaries + tg_idx * kLMHeadRowSummaryStride;
            slot[0] = v_all;
            slot[1] = as_type<float>(i_all);
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void lm_head_greedy_int4_rows_reduce(
    device const float*   summaries    [[buffer(0)]],
    device       uint*    out_token    [[buffer(1)]],
    constant     uint&    row_groups   [[buffer(2)]],
    uint  lid            [[thread_position_in_threadgroup]],
    uint  lsize          [[threads_per_threadgroup]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_v[kLogitMaxSimdGroups];
    threadgroup uint  partial_i[kLogitMaxSimdGroups];

    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;
    for (uint i = lid; i < row_groups; i += lsize) {
        device const float* slot = summaries + i * kLMHeadRowSummaryStride;
        const float v = slot[0];
        const uint idx = as_type<uint>(slot[1]);
        if (v > best_v || (v == best_v && idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }

    float v_simd = simd_max(best_v);
    uint i_simd = (best_v == v_simd) ? best_i : 0xFFFFFFFFu;
    i_simd = simd_min(i_simd);

    if (simd_lane_id == 0) {
        partial_v[simd_group_id] = v_simd;
        partial_i[simd_group_id] = i_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const bool active = simd_lane_id < simdgroups;
        const float v = active ? partial_v[simd_lane_id] : -INFINITY;
        const uint idx = active ? partial_i[simd_lane_id] : 0xFFFFFFFFu;
        const float v_all = simd_max(v);
        uint i_all = (v == v_all) ? idx : 0xFFFFFFFFu;
        i_all = simd_min(i_all);
        if (simd_lane_id == 0) {
            out_token[0] = (i_all == 0xFFFFFFFFu) ? 0u : i_all;
        }
    }
}
