#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMoEGroupSize = 64;
// Upper bound on routed experts per token, sizing the argument buffer's blob
// pointer array. Metal array extents must be compile-time, so this cannot be a
// function constant -- it is one bound for every family. 16 covers top-8
// (Qwen 3.6, Ornith) and top-10 (Qwen3.8-Flash-Next) at a cost of eight unused
// pointers, 64 bytes per argument buffer. Kernels read only [0, top_k), so a
// family using fewer slots is unaffected byte-for-byte.
constant constexpr uint kMaxStreamedExperts = 16;
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;
// Max hidden size for the phase-1 threadgroup-staged activation. Covers
// qwen36_35B_A3B's 2048, qwen38flash's 2560 and Ornith-1.5-122B's 3072; the
// cooperative load is guarded for smaller D.
constant constexpr uint kMoEXMaxD = 3072;

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant uint FC_ROUTER_TOP_K [[function_constant(42)]];
constant bool FC_ROUTER_USE_FC [[function_constant(43)]];

constant uint FC_MOE_D [[function_constant(0)]];
constant uint FC_MOE_F [[function_constant(1)]];
constant uint FC_MOE_TOP_K [[function_constant(2)]];
constant bool FC_MOE_USE_FC [[function_constant(3)]];
// Hidden activation for expert FFNs: unset/false = gelu_pytorch_tanh,
// true = silu (Qwen 3.6 SwiGLU). Applies to routed decode, routed prefill,
// and the fused INT8 shared-expert kernel.
constant bool FC_MOE_ACT_SILU [[function_constant(4)]];
constant uint FC_MOE_WEIGHT_BITS [[function_constant(5)]];
constant bool FC_MOE_EVENT_GATED [[function_constant(6)]];

/// Whether an event-gated dispatch may dereference its expert slots.
///
/// Ordering is already guaranteed by `encodeWaitForEvent` on this command
/// buffer: the coordinator writes the status word and only then advances the
/// shared event, so a kernel that runs has necessarily observed a published
/// token. The gate therefore exists for exactly one case -- a read that
/// *failed* (status 2), whose slots hold incomplete bytes that must not be
/// dereferenced.
///
/// Requiring status == 1 instead made "not yet observed" indistinguishable
/// from "failed" and silently skipped the whole dispatch, so every expert
/// contributed zero and the model emitted fluent nonsense at full speed. The
/// 3 arm was dead: nothing has ever written that value.
static inline bool moe_io_ready(device const uint* io_status) {
    return !(is_function_constant_defined(FC_MOE_EVENT_GATED) && FC_MOE_EVENT_GATED)
        || io_status[0] != 2u;
}

static inline uint moe_weight_bits() {
    return is_function_constant_defined(FC_MOE_WEIGHT_BITS)
        ? FC_MOE_WEIGHT_BITS : 4u;
}

static inline uint moe_affine_value(device const uint8_t* packed,
                                    uint element, uint bits) {
    const uint bit = element * bits;
    const uint byte = bit >> 3;
    const uint shift = bit & 7u;
    uint word = uint(packed[byte]);
    if (shift + bits > 8u) word |= uint(packed[byte + 1u]) << 8;
    return (word >> shift) & ((1u << bits) - 1u);
}

static inline uint router_fc_num_experts(constant uint& num_experts) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_NUM_EXPERTS))
        ? FC_ROUTER_NUM_EXPERTS
        : num_experts;
}

static inline uint router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_D))
        ? FC_ROUTER_D
        : D;
}

static inline uint moe_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_D)) ? FC_MOE_D : D;
}

static inline uint moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_F)) ? FC_MOE_F : F;
}

static inline uint moe_fc_top_k(constant uint& top_k) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_TOP_K)) ? FC_MOE_TOP_K : top_k;
}

static inline float gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    // Clamping avoids Metal tanh producing NaN at large magnitudes while being
    // equivalent to the saturated result at FP32 precision.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

struct ExpertResidencyGPU {
    uint slot;
    uint state;
    ulong generation;
};

/// Classifies the router's exact top-k result against the CPU-published cache
/// map. Only eight positions are touched; one lane avoids atomics and preserves
/// router order in both compact lists.
kernel void moe_classify_expert_residency(
    device const uint* topk_indices [[buffer(0)]],
    device const ExpertResidencyGPU* residency [[buffer(1)]],
    device uint* hit_count [[buffer(2)]],
    device uint* hit_positions [[buffer(3)]],
    device uint* miss_count [[buffer(4)]],
    device uint* miss_positions [[buffer(5)]],
    device uint* miss_experts [[buffer(6)]],
    device uint* resolved_slots [[buffer(7)]],
    device ulong* resolved_generations [[buffer(8)]],
    constant uint& top_k [[buffer(9)]],
    constant uint& num_experts [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (lane != 0) return;
    uint hits = 0, misses = 0;
    for (uint position = 0; position < top_k; ++position) {
        const uint expert = min(topk_indices[position], num_experts - 1u);
        const ExpertResidencyGPU entry = residency[expert];
        if (entry.state == 2u && entry.slot != 0xffffffffu) {
            hit_positions[hits++] = position;
            resolved_slots[position] = entry.slot;
            resolved_generations[position] = entry.generation;
        } else {
            miss_positions[misses] = position;
            miss_experts[misses] = expert;
            ++misses;
            resolved_slots[position] = 0xffffffffu;
            resolved_generations[position] = entry.generation;
        }
    }
    hit_count[0] = hits;
    miss_count[0] = misses;
}

static inline float moe_hidden_activation(float x) {
    if (is_function_constant_defined(FC_MOE_ACT_SILU) && FC_MOE_ACT_SILU) {
        return x / (1.0f + exp(-x));
    }
    return gelu_pytorch_tanh(x);
}

struct ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct RoutedBlobs {
    device const uint8_t* blob[kMaxStreamedExperts];
};

constant uint FC_ROUTER_BITS [[function_constant(44)]];

static inline uint router_affine_bits() {
    return is_function_constant_defined(FC_ROUTER_BITS) ? FC_ROUTER_BITS : 8u;
}

static inline uint router_affine_value(device const uint8_t* packed,
                                       uint element,
                                       uint bits) {
    const uint bit_offset = element * bits;
    const uint byte_offset = bit_offset >> 3;
    const uint shift = bit_offset & 7u;
    uint word = uint(packed[byte_offset]);
    if (shift + bits > 8u) word |= uint(packed[byte_offset + 1u]) << 8u;
    return (word >> shift) & ((1u << bits) - 1u);
}

static inline void router_gemv_body(
    device const uint8_t* W,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    const uint bits = router_affine_bits();
    const uint n_groups = DD / kMoEGroupSize;
    const uint row_bytes = DD * bits / 8u;
    device const uint8_t* W_row = W + uint(e) * row_bytes;
    device const bfloat* s_row = scales + uint(e) * n_groups;
    device const bfloat* b_row = biases + uint(e) * n_groups;

    float acc = 0.0f;
    // bits == 16 means the router was promoted to the checkpoint's own bf16:
    // no packing, no per-group scale or bias, one multiply per weight. The
    // row stride above already lands in the right place -- DD * 16 / 8 is the
    // byte length of a bf16 row -- so only the inner read changes.
    //
    // The router produces a *ranking*, and a ranking either matches or it does
    // not. That is why this one is worth promoting where a bulk projection is
    // not: at 8 bits its top-10 selection agrees with bf16 on 98.8% of tokens,
    // and the disagreements pick different experts.
    if (bits == 16u) {
        device const bfloat* W_bf = (device const bfloat*)W_row;
        for (uint g = 0; g < n_groups; ++g) {
            const uint idx = g * kMoEGroupSize + lane * 2u;
            const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
            const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
            acc = fma(float(W_bf[idx]), x0, acc);
            acc = fma(float(W_bf[idx + 1u]), x1, acc);
        }
    } else {
        for (uint g = 0; g < n_groups; ++g) {
            const float s = float(s_row[g]);
            const float b = float(b_row[g]);
            const uint idx = g * kMoEGroupSize + lane * 2u;
            const float q0 = float(router_affine_value(W_row, idx, bits));
            const float q1 = float(router_affine_value(W_row, idx + 1u, bits));
            const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
            const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
            acc = fma(s, q0 * x0 + q1 * x1, acc);
            acc = fma(b, x0 + x1, acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

kernel void router_gemv_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_body(W, scales, biases, hidden, effective_scale,
                            out_logits, num_experts, D, 4, tg_idx, sg_idx, lane);
}

kernel void router_topk_select_k8(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[8];
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        // Equal scores at the boundary are still considered so the inner
        // loop's lower-index tie-break (matching the reference) applies.
        if (s < top_score[7]) continue;
        uint pos = 8u;
        for (uint i = 0; i < 8; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 8u) continue;
        for (uint i = 7; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// Same selection as router_topk_select_kn on one simdgroup instead of one
// thread. The serial kernel scans NE logits with an insertion sort on a
// single lane, which at 512 experts is thousands of dependent steps per layer.
// Here each lane keeps NE/32 logits in registers and the top-k is K rounds
// of (max score, then lowest index among equals) simd reductions, which is
// the serial kernel's order exactly: score descending, lower index on ties,
// NaN never selected. Lane 0 then runs the identical exp/sum sequence, so the
// weights are bit-identical too.
constant constexpr uint kRouterSimdMaxPerLane = 32u;   // NE <= 1024

kernel void router_topk_select_kn_simd(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint K = min(top_k, kMaxStreamedExperts);
    const uint per_lane = (NE + 31u) / 32u;
    float v[kRouterSimdMaxPerLane];
    for (uint j = 0; j < kRouterSimdMaxPerLane; ++j) {
        const uint e = lane + 32u * j;
        v[j] = (j < per_lane && e < NE) ? logits[e] : -INFINITY;
    }
    uint top_idx[kMaxStreamedExperts];
    float top_score[kMaxStreamedExperts];
    for (uint r = 0; r < K; ++r) {
        float best = -INFINITY;
        uint bidx = 0xFFFFFFFFu;
        for (uint j = 0; j < kRouterSimdMaxPerLane; ++j) {
            if (j < per_lane && v[j] > best) {   // strict: first (lowest e) wins
                best = v[j];
                bidx = lane + 32u * j;
            }
        }
        const float gmax = simd_max(best);
        const uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
        const uint gidx = simd_min(cand);
        // Nothing selectable (all -inf or NaN): the serial kernel leaves the
        // slot at index 0 with score -inf.
        top_score[r] = gmax;
        top_idx[r] = (gidx == 0xFFFFFFFFu) ? 0u : gidx;
        if (gidx != 0xFFFFFFFFu && (gidx & 31u) == lane) {
            for (uint j = 0; j < kRouterSimdMaxPerLane; ++j) {
                if (lane + 32u * j == gidx) v[j] = -INFINITY;
            }
        }
    }
    if (lane != 0) return;
    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[kMaxStreamedExperts];
    for (uint i = 0; i < K; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < K; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// Top-k router select for k != 8. The k8 kernel above is left exactly as it
// was: it is on the golden path for every shipping model, and a shared
// implementation would put a k-dependent loop bound in front of output that is
// pinned byte-for-byte. This variant is identical in method -- same insertion
// sort, same lower-index tie-break, same softmax over the SELECTED scores
// (which is what norm_topk_prob=true means: renormalizing over the top-k is
// the same as softmax-over-all then renormalize).
kernel void router_topk_select_kn(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    const uint K = min(top_k, kMaxStreamedExperts);
    uint top_idx[kMaxStreamedExperts];
    float top_score[kMaxStreamedExperts];
    for (uint i = 0; i < K; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        if (s < top_score[K - 1u]) continue;
        uint pos = K;
        for (uint i = 0; i < K; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= K) continue;
        for (uint i = K - 1u; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[kMaxStreamedExperts];
    for (uint i = 0; i < K; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < K; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// Each SIMD computes one affine INT4 row. Four adjacent groups are loaded as
// two 16-bit weight halves (sub-tensor offsets are only guaranteed 2-byte
// aligned, so a 4-byte `uint*` load would be misaligned — undefined); the
// remaining groups use one byte per lane.
static inline float moe_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Read the 4-byte weight chunk as two ushorts: row stride N/2,
        // sub-tensor weightsOffset, and byte_base are all even, so a
        // `ushort*` load is always aligned even when the row is only
        // 2-byte aligned overall.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
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
        const uint8_t byte = W_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

static inline float moe_affine_gemv_row_simd(
    device const uint8_t* W, device const bfloat* S,
    device const bfloat* B, device const half* x,
    uint row, uint N, uint lane
) {
    const uint bits = moe_weight_bits();
    const uint groups = N / kMoEGroupSize;
    const uint row_bytes = N * bits / 8u;
    device const uint8_t* wr = W + row * row_bytes;
    device const bfloat* sr = S + row * groups;
    device const bfloat* br = B + row * groups;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const uint e = g * kMoEGroupSize + lane * 2u;
        const float x0 = float(x[e]);
        const float x1 = float(x[e + 1u]);
        const float q0 = float(moe_affine_value(wr, e, bits));
        const float q1 = float(moe_affine_value(wr, e + 1u, bits));
        acc = fma(float(sr[g]), q0 * x0 + q1 * x1, acc);
        acc = fma(float(br[g]), x0 + x1, acc);
    }
    return simd_sum(acc);
}

static inline float2 moe_affine_gate_up_rows_simd(
    device const uint8_t* gateW, device const bfloat* gateS,
    device const bfloat* gateB, device const uint8_t* upW,
    device const bfloat* upS, device const bfloat* upB,
    device const half* x, uint row, uint N, uint lane
) {
    return float2(moe_affine_gemv_row_simd(gateW, gateS, gateB, x, row, N, lane),
                  moe_affine_gemv_row_simd(upW, upS, upB, x, row, N, lane));
}

// Gate and up rows share activation loads. Two 16-bit loads assemble each
// 4-byte weight chunk because packed sub-tensor offsets need only be 2-byte aligned.
static inline float2 moe_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

// Threadgroup-staged activation for phase-1: all rows of a threadgroup share
// the same x vector, but each SIMD row loop re-loads the full x from L2.
// Staging it in threadgroup memory once per threadgroup (cooperatively, then
// a barrier) removes those re-reads — measured 38% -> 56% of peak bandwidth
// on the M3 with 16 rows/threadgroup (TinyTitanBench moe_phase1_xsh16).
static inline float2 moe_int4_gate_up_rows_simd_tgmem_u16load(
    threadgroup const half* x,
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const float e0 = float(x[elem]), e1 = float(x[elem + 1u]);
        const float e2 = float(x[elem + 2u]), e3 = float(x[elem + 3u]);
        const float e4 = float(x[elem + 4u]), e5 = float(x[elem + 5u]);
        const float e6 = float(x[elem + 6u]), e7 = float(x[elem + 7u]);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

static inline void moe_phase1_gate_up_act_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

static inline void moe_phase1_gate_up_act_subset_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    device const uint* active_slots,
    uint active_count,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint active_idx = rowg / F;
    const uint slot = active_slots[active_idx];
    if (slot >= top_k) return;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    // 16 rows per threadgroup with the activation staged in threadgroup
    // memory. Each simdgroup redundantly loads the full x (the shared input),
    // then a barrier makes it visible to the row loops.
    constexpr uint rows_per_tg = 16;
    threadgroup half xt[kMoEXMaxD];
    const uint DD = moe_fc_d(D);
    for (uint i = lane; i < DD; i += 32u) {
        xt[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * moe_fc_f(F)) return;
    const uint slot = rowg / moe_fc_f(F);
    const uint f = rowg % moe_fc_f(F);

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int4_gate_up_rows_simd_tgmem_u16load(
        xt, base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        f, DD, lane);
    if (lane == 0) acts[slot * moe_fc_f(F) + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_subset_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    device const uint* io_status [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    threadgroup half xt[kMoEXMaxD];
    const uint DD = moe_fc_d(D);
    for (uint i = lane; i < DD; i += 32u) {
        xt[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * moe_fc_f(F)) return;
    const uint active_idx = rowg / moe_fc_f(F);
    const uint slot = active_slots[active_idx];
    if (slot >= moe_fc_top_k(top_k)) return;
    const uint f = rowg % moe_fc_f(F);

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int4_gate_up_rows_simd_tgmem_u16load(
        xt, base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        f, DD, lane);
    if (lane == 0) acts[slot * moe_fc_f(F) + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load_r16(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void moe_phase1_gate_up_act_u16load_r8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void moe_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    device const uint* io_status [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;
    if (!moe_io_ready(io_status)) {
        if (sg_idx == 0 && lane == 0) y[d] = residual[d];
        return;
    }

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}

kernel void moe_affine_phase1_gate_up_act(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& re [[buffer(1)]],
    device const half* x [[buffer(2)]], device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], device const uint* io_status [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F), TK = moe_fc_top_k(top_k);
    const uint rowg = tg * rows_per_tg + sg;
    if (rowg >= TK * FF) return;
    const uint slot = rowg / FF, row = rowg % FF;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_affine_gate_up_rows_simd(
        base + re.gate_W_off, (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off), base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off), x, row, DD, lane);
    if (lane == 0) acts[slot * FF + row] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_affine_phase1_gate_up_act_subset(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& re [[buffer(1)]],
    device const half* x [[buffer(2)]], device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]], device const uint* io_status [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F), TK = moe_fc_top_k(top_k);
    const uint rowg = tg * rows_per_tg + sg;
    if (rowg >= active_count * FF) return;
    const uint slot = active_slots[rowg / FF];
    if (slot >= TK) return;
    const uint row = rowg % FF;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_affine_gate_up_rows_simd(
        base + re.gate_W_off, (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off), base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off), x, row, DD, lane);
    if (lane == 0) acts[slot * FF + row] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_affine_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& re [[buffer(1)]],
    device const half* acts [[buffer(2)]], device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]], device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]], constant uint& F [[buffer(7)]],
    device const uint* io_status [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    if (d >= DD) return;
    if (!moe_io_ready(io_status)) {
        if (sg == 0 && lane == 0) y[d] = residual[d];
        return;
    }
    device const uint8_t* base = routed.blob[sg];
    const float value = moe_affine_gemv_row_simd(
        base + re.down_W_off, (device const bfloat*)(base + re.down_s_off),
        (device const bfloat*)(base + re.down_b_off), acts + sg * FF,
        d, FF, lane);
    if (lane == 0) partial[sg] = float(routing_w[sg]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float acc = float(residual[d]);
        for (uint i = 0; i < 8u; ++i) acc += partial[i];
        y[d] = half(acc);
    }
}

// Phase-2 reduce for k != 8. The k8 kernels stay untouched on the golden
// path; here the partial array is sized to the argument buffer's bound and
// simdgroups past k return early rather than reading an unset blob pointer.
kernel void moe_phase2_down_reduce_kn(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    device const uint* io_status [[buffer(8)]],
    constant uint& top_k [[buffer(9)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kMaxStreamedExperts];
    const uint K = min(top_k, kMaxStreamedExperts);
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;
    if (!moe_io_ready(io_status)) {
        if (sg_idx == 0 && lane == 0) y[d] = residual[d];
        return;
    }

    // No sg_idx >= K guard: the dispatch launches exactly K simdgroups, so
    // every simdgroup owns a real slot. An early return here would skip the
    // threadgroup_barrier below, which every thread in the group must reach.
    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        // Summed in slot order so the result is bit-identical to the k8
        // kernel's unrolled chain when k == 8.
        for (uint i = 0; i < K; ++i) acc += partial[i];
        y[d] = half(acc);
    }
}

// Phase-2 reduce for k != 8. The k8 kernels stay untouched on the golden
// path; here the partial array is sized to the argument buffer's bound and
// simdgroups past k return early rather than reading an unset blob pointer.
kernel void moe_affine_phase2_down_reduce_kn(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& re [[buffer(1)]],
    device const half* acts [[buffer(2)]], device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]], device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]], constant uint& F [[buffer(7)]],
    device const uint* io_status [[buffer(8)]],
    constant uint& top_k [[buffer(9)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[kMaxStreamedExperts];
    const uint K = min(top_k, kMaxStreamedExperts);
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    if (d >= DD) return;
    if (!moe_io_ready(io_status)) {
        if (sg == 0 && lane == 0) y[d] = residual[d];
        return;
    }
    device const uint8_t* base = routed.blob[sg];
    const float value = moe_affine_gemv_row_simd(
        base + re.down_W_off, (device const bfloat*)(base + re.down_s_off),
        (device const bfloat*)(base + re.down_b_off), acts + sg * FF,
        d, FF, lane);
    if (lane == 0) partial[sg] = float(routing_w[sg]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float acc = float(residual[d]);
        // K, not 8: this variant exists precisely because k differs.
        for (uint i = 0; i < K; ++i) acc += partial[i];
        y[d] = half(acc);
    }
}

// ---------------------------------------------------------------------------
// Phase 1, second layout (TinyTitanBench moe_phase1_v2; wired only if it wins).
// Same per-row arithmetic as moe_int4_gate_up_rows_simd_tgmem_u16load -- the
// fma order, the s*dot + b*sum factoring and the half rounding are untouched,
// so rows are bit-identical -- with three changes to the load pattern, which
// is what the 38 GB/s was: the activation is staged once by the whole
// threadgroup instead of once per simdgroup (16x fewer staging loads), it is
// read back as half4 (two loads per block instead of eight), and each
// simdgroup carries two rows so one activation read feeds four dots.
// ---------------------------------------------------------------------------
static inline void moe_int4_gate_up_row_pointers(
    device const RoutedBlobs& routed, constant ExpertOffsets& re,
    uint rowg, uint F, uint row_bytes, uint n_groups,
    thread device const uint8_t*& gW, thread device const bfloat*& gS,
    thread device const bfloat*& gB, thread device const uint8_t*& uW,
    thread device const bfloat*& uS, thread device const bfloat*& uB,
    thread uint& slot, thread uint& f)
{
    slot = rowg / F;
    f = rowg - slot * F;
    device const uint8_t* base = routed.blob[slot];
    gW = base + re.gate_W_off + f * row_bytes;
    gS = (device const bfloat*)(base + re.gate_s_off) + f * n_groups;
    gB = (device const bfloat*)(base + re.gate_b_off) + f * n_groups;
    uW = base + re.up_W_off + f * row_bytes;
    uS = (device const bfloat*)(base + re.up_s_off) + f * n_groups;
    uB = (device const bfloat*)(base + re.up_b_off) + f * n_groups;
}

#define MOE_GU_BLOCK(gW, gS, gB, uW, uS, uB, g_acc, u_acc)                       \
    {                                                                            \
        device const ushort* gp = (device const ushort*)(gW + byte_base);        \
        device const ushort* up = (device const ushort*)(uW + byte_base);        \
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);                      \
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);                      \
        const float gs = float(gS[g]);                                           \
        const float gb = float(gB[g]);                                           \
        const float us = float(uS[g]);                                           \
        const float ub = float(uB[g]);                                           \
        const uint gb0 = gw4 & 0xFFu, gb1 = (gw4 >> 8) & 0xFFu;                  \
        const uint gb2 = (gw4 >> 16) & 0xFFu, gb3 = (gw4 >> 24) & 0xFFu;         \
        float g_dot = 0.0f;                                                      \
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot); \
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot); \
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot); \
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot); \
        const uint ub0 = uw4 & 0xFFu, ub1 = (uw4 >> 8) & 0xFFu;                  \
        const uint ub2 = (uw4 >> 16) & 0xFFu, ub3 = (uw4 >> 24) & 0xFFu;         \
        float u_dot = 0.0f;                                                      \
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot); \
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot); \
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot); \
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot); \
        g_acc = fma(gs, g_dot, g_acc);                                           \
        g_acc = fma(gb, sum, g_acc);                                             \
        u_acc = fma(us, u_dot, u_acc);                                           \
        u_acc = fma(ub, sum, u_acc);                                             \
    }

#define MOE_GU_TAIL(gW, gS, gB, uW, uS, uB, g_acc, u_acc)                        \
    {                                                                            \
        const float gs = float(gS[g]);                                           \
        const float gb = float(gB[g]);                                           \
        const float us = float(uS[g]);                                           \
        const float ub = float(uB[g]);                                           \
        const uint8_t gbv = gW[g * (kMoEGroupSize / 2) + lane];                  \
        const uint8_t ubv = uW[g * (kMoEGroupSize / 2) + lane];                  \
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);                   \
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);                           \
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);                   \
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);                           \
        g_acc = fma(gs, g_dot, g_acc);                                           \
        g_acc = fma(gb, sum, g_acc);                                             \
        u_acc = fma(us, u_dot, u_acc);                                           \
        u_acc = fma(ub, sum, u_acc);                                             \
    }

kernel void moe_phase1_gate_up_act_u16load_v2(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_sg = 2;
    threadgroup half xt[kMoEXMaxD];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint total = moe_fc_top_k(top_k) * FF;
    for (uint i = lid; i < DD; i += lsize) xt[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sgs = lsize / 32u;
    const uint row0 = (tg_idx * sgs + sg_idx) * rows_per_sg;
    if (row0 >= total) return;
    const bool two = row0 + 1u < total;
    const uint n_groups = DD / kMoEGroupSize;
    const uint row_bytes = DD / 2;
    device const uint8_t* gW0; device const bfloat* gS0; device const bfloat* gB0;
    device const uint8_t* uW0; device const bfloat* uS0; device const bfloat* uB0;
    device const uint8_t* gW1; device const bfloat* gS1; device const bfloat* gB1;
    device const uint8_t* uW1; device const bfloat* uS1; device const bfloat* uB1;
    uint slot0, f0, slot1 = 0u, f1 = 0u;
    moe_int4_gate_up_row_pointers(routed, routed_offsets, row0, FF, row_bytes, n_groups,
                                  gW0, gS0, gB0, uW0, uS0, uB0, slot0, f0);
    if (two) {
        moe_int4_gate_up_row_pointers(routed, routed_offsets, row0 + 1u, FF, row_bytes, n_groups,
                                      gW1, gS1, gB1, uW1, uS1, uB1, slot1, f1);
    } else {
        gW1 = gW0; gS1 = gS0; gB1 = gB0; uW1 = uW0; uS1 = uS0; uB1 = uB0;
    }

    float g_acc0 = 0.0f, u_acc0 = 0.0f, g_acc1 = 0.0f, u_acc1 = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint g = blk * 4u + (lane >> 3);
        const uint elem = byte_base * 2u;
        const half4 xa = *((threadgroup const half4*)(xt + elem));
        const half4 xb = *((threadgroup const half4*)(xt + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        MOE_GU_BLOCK(gW0, gS0, gB0, uW0, uS0, uB0, g_acc0, u_acc0)
        if (two) MOE_GU_BLOCK(gW1, gS1, gB1, uW1, uS1, uB1, g_acc1, u_acc1)
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float x0 = float(xt[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(xt[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        MOE_GU_TAIL(gW0, gS0, gB0, uW0, uS0, uB0, g_acc0, u_acc0)
        if (two) MOE_GU_TAIL(gW1, gS1, gB1, uW1, uS1, uB1, g_acc1, u_acc1)
    }
    const float g0 = simd_sum(g_acc0), u0 = simd_sum(u_acc0);
    const float g1 = simd_sum(g_acc1), u1 = simd_sum(u_acc1);
    if (lane == 0) {
        acts[slot0 * FF + f0] = half(moe_hidden_activation(g0) * u0);
        if (two) acts[slot1 * FF + f1] = half(moe_hidden_activation(g1) * u1);
    }
}

// K2-Horizon's router selection.
//
// This is not the Qwen softmax router. Scores are `sigmoid(logits)`; the bias
// tensor is added ONLY to the ranking (it must not change the mixture weights);
// the selected weights are the sigmoid scores renormalized over the top-k, then
// multiplied by the family's `router_scaling_factor`. K2 uses the same
// selection for its 100-expert MoE (top-8) and its 64-expert MoVA value router
// (top-4), so the expert count and k are runtime arguments.
//
// The order is the CPU reference's and `router_topk_select_kn`'s: descending
// selection score, lower expert index on ties. One thread does the scan, which
// is what the reference does and what a 100-expert router can afford; the
// dispatch is one 32-thread threadgroup like the Qwen selector.
kernel void k2_router_topk_select_kn(
    device const float*  logits      [[buffer(0)]],
    device const bfloat* bias        [[buffer(1)]],
    device uint*         out_indices [[buffer(2)]],
    device half*         out_weights [[buffer(3)]],
    constant uint&       num_experts [[buffer(4)]],
    constant uint&       top_k       [[buffer(5)]],
    constant float&      scaling     [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    const uint K = min(top_k, kMaxStreamedExperts);
    if (K == 0) return;
    uint top_idx[kMaxStreamedExperts];
    float top_score[kMaxStreamedExperts];   // sigmoid score: the mixture weight
    float top_rank[kMaxStreamedExperts];    // score + bias: the selection key
    for (uint i = 0; i < K; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
        top_rank[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = 1.0f / (1.0f + exp(-logits[e]));
        const float rank = s + float(bias[e]);
        if (rank < top_rank[K - 1]) continue;
        uint pos = K;
        for (uint i = 0; i < K; ++i) {
            if (rank > top_rank[i] || (rank == top_rank[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= K) continue;
        for (uint i = K - 1; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
            top_rank[i] = top_rank[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
        top_rank[pos] = rank;
    }

    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) sum += top_score[i];
    const float norm = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint i = 0; i < K; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(top_score[i] * norm * scaling);
    }
}

// The MoVA value epilogue.
//
// K2's sparse layers replace `v_proj` with four selected value experts, each
// `hidden -> kvHeads * headDim`, activated with SiLU before weighting. The
// four GEMVs are separate dispatches (they read different resident weights);
// this kernel is the part that has to see all four at once: it applies SiLU
// and the router weights and sums them into the value states.
kernel void mova_combine_fp16(
    device const half* experts [[buffer(0)]],   // [4][count]
    device const half* weights [[buffer(1)]],   // [4]
    device half*       out     [[buffer(2)]],   // [count]
    constant uint&     count   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    float total = 0.0f;
    for (uint slot = 0; slot < 4u; ++slot) {
        const float v = float(experts[slot * count + tid]);
        const float w = float(weights[slot]);
        total += w * (v / (1.0f + exp(-v)));
    }
    out[tid] = half(total);
}

// K2's routed-expert sum: out += Σ_slot weight[slot] * expertOutput[slot][i].
//
// The expert GEMVs are separate dispatches (they read different resident
// blobs), so each writes its own slot and this pass applies the router weight.
// It adds into `out`, which the shared expert's down projection has already
// written, matching the checkpoint's "shared + routed" sum.
kernel void k2_moe_accumulate_fp16(
    device const half* experts [[buffer(0)]],   // [slots][count]
    device const half* weights [[buffer(1)]],   // [slots]
    device half*       out     [[buffer(2)]],   // [count], shared already written
    constant uint&     count   [[buffer(3)]],
    constant uint&     slots   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    float total = 0.0f;
    for (uint slot = 0; slot < slots; ++slot) {
        total += float(weights[slot]) * float(experts[slot * count + tid]);
    }
    out[tid] = half(float(out[tid]) + total);
}

// K2's fused routed-expert pass, written for its 768x2560 experts.
//
// The Qwen fused kernels stage 16 rows of a 512-wide expert per threadgroup,
// a shape K2 does not share: its experts are wider (2560) and taller (768), and
// that staging loses more than it gains there. These two kernels keep one
// simdgroup per row and read the selected experts through argument buffers, so a
// whole top-8 set costs two dispatches per layer instead of four per expert.
constant uint K2_FUSED_BITS [[function_constant(120)]];

static inline uint k2_fused_bits() {
    return is_function_constant_defined(K2_FUSED_BITS) ? K2_FUSED_BITS : 4u;
}

struct K2ExpertBuffers {
    device const uint*   w[kMaxStreamedExperts];
    device const bfloat* s[kMaxStreamedExperts];
    device const bfloat* b[kMaxStreamedExperts];
};

// acts[slot * F + row] = silu(gate_row . x) * (up_row . x)
kernel void k2_moe_gate_up_act(
    device const K2ExpertBuffers& gate [[buffer(0)]],
    device const K2ExpertBuffers& up   [[buffer(1)]],
    device const half* x               [[buffer(2)]],
    device half* acts                  [[buffer(3)]],
    constant uint& D                   [[buffer(4)]],
    constant uint& F                   [[buffer(5)]],
    constant uint& topK                [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tg * 8u + sg;
    const uint slot = row / F;
    if (slot >= topK) return;
    const uint local = row - slot * F;
    const uint bits = k2_fused_bits();
    const uint groups = D / kMoEGroupSize;
    const uint words = (D * bits) / 32u;
    device const uint* gw = gate.w[slot] + local * words;
    device const bfloat* gs = gate.s[slot] + local * groups;
    device const bfloat* gb = gate.b[slot] + local * groups;
    device const uint* uw = up.w[slot] + local * words;
    device const bfloat* us = up.s[slot] + local * groups;
    device const bfloat* ub = up.b[slot] + local * groups;
    // One word per lane, one group per lane stride: 8 (4-bit) or 4 (8-bit)
    // consecutive weights share a word and a group, so a lane's scale and bias
    // are single scalars and the simdgroup's word reads are one coalesced
    // line. The first form read two elements per lane per group, which issued
    // twice the transactions for the same bytes.
    const uint elemsPerWord = 32u / bits;
    const uint lanesPerGroup = kMoEGroupSize / elemsPerWord;
    const uint groupsPerIteration = 32u / lanesPerGroup;
    const uint wordsPerGroup = kMoEGroupSize / elemsPerWord;
    const uint mask = (1u << bits) - 1u;
    float gateAcc = 0.0f;
    float upAcc = 0.0f;
    for (uint base = 0; base < groups; base += groupsPerIteration) {
        const uint group = base + lane / lanesPerGroup;
        if (group >= groups) continue;
        const uint localWord = lane % lanesPerGroup;
        const uint word = group * wordsPerGroup + localWord;
        const uint elementBase = group * kMoEGroupSize + localWord * elemsPerWord;
        const uint gWord = gw[word];
        const uint uWord = uw[word];
        float gateDot = 0.0f;
        float upDot = 0.0f;
        float sumX = 0.0f;
        for (uint i = 0; i < elemsPerWord; ++i) {
            const uint shift = i * bits;
            const float xv = float(x[elementBase + i]);
            gateDot = fma(float((gWord >> shift) & mask), xv, gateDot);
            upDot = fma(float((uWord >> shift) & mask), xv, upDot);
            sumX += xv;
        }
        const float gScale = float(gs[group]);
        const float gBias = float(gb[group]);
        gateAcc = fma(gScale, gateDot, gateAcc);
        gateAcc = fma(gBias, sumX, gateAcc);
        upAcc = fma(float(us[group]), upDot, upAcc);
        upAcc = fma(float(ub[group]), sumX, upAcc);
    }
    gateAcc = simd_sum(gateAcc);
    upAcc = simd_sum(upAcc);
    if (lane == 0u) {
        acts[row] = half((gateAcc / (1.0f + exp(-gateAcc))) * upAcc);
    }
}

// out[d] = residual[d] + Σ_slot weight[slot] * down_slot[d] . acts_slot
kernel void k2_moe_down(
    device const K2ExpertBuffers& down [[buffer(0)]],
    device const half* acts            [[buffer(1)]],
    device const half* weights         [[buffer(2)]],
    device const half* residual        [[buffer(3)]],
    device half* out                   [[buffer(4)]],
    constant uint& D                   [[buffer(5)]],
    constant uint& F                   [[buffer(6)]],
    constant uint& topK                [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tg * 8u + sg;
    if (row >= D) return;
    const uint bits = k2_fused_bits();
    const uint groups = F / kMoEGroupSize;
    const uint words = (F * bits) / 32u;
    float total = 0.0f;
    for (uint slot = 0; slot < topK; ++slot) {
        device const uint* dw = down.w[slot] + row * words;
        device const bfloat* ds = down.s[slot] + row * groups;
        device const bfloat* db = down.b[slot] + row * groups;
        device const half* a = acts + slot * F;
        const uint elemsPerWord = 32u / bits;
        const uint lanesPerGroup = kMoEGroupSize / elemsPerWord;
        const uint groupsPerIteration = 32u / lanesPerGroup;
        const uint wordsPerGroup = kMoEGroupSize / elemsPerWord;
        const uint mask = (1u << bits) - 1u;
        float acc = 0.0f;
        for (uint base = 0; base < groups; base += groupsPerIteration) {
            const uint group = base + lane / lanesPerGroup;
            if (group >= groups) continue;
            const uint localWord = lane % lanesPerGroup;
            const uint word = group * wordsPerGroup + localWord;
            const uint elementBase = group * kMoEGroupSize + localWord * elemsPerWord;
            const uint dWord = dw[word];
            float dot = 0.0f;
            float sumA = 0.0f;
            for (uint i = 0; i < elemsPerWord; ++i) {
                const uint shift = i * bits;
                const float av = float(a[elementBase + i]);
                dot = fma(float((dWord >> shift) & mask), av, dot);
                sumA += av;
            }
            acc = fma(float(ds[group]), dot, acc);
            acc = fma(float(db[group]), sumA, acc);
        }
        total += float(weights[slot]) * simd_sum(acc);
    }
    if (lane == 0u) {
        out[row] = half(float(residual[row]) + total);
    }
}

// K2's MoVA values with the router's choice resolved on the GPU. The four
// value experts are resident, so instead of reading the selected ids back to
// the CPU (which forces the value pass into its own command buffer) the kernel
// indexes three per-expert offset tables into the shared resident buffer. The
// GPU dependency is the selection kernel earlier in the same command buffer.
kernel void k2_mova_values_routed(
    device const uchar* resident          [[buffer(0)]],
    device const ulong* weightOffsets     [[buffer(1)]],
    device const ulong* scaleOffsets      [[buffer(2)]],
    device const ulong* biasOffsets       [[buffer(3)]],
    device const uint* selected           [[buffer(4)]],
    device const half* x                  [[buffer(5)]],
    device half* out                      [[buffer(6)]],
    constant uint& D                      [[buffer(7)]],
    constant uint& R                      [[buffer(8)]],
    constant uint& topK                   [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tg * 8u + sg;
    const uint slot = row / R;
    if (slot >= topK) return;
    const uint local = row - slot * R;
    const uint id = selected[slot];
    const uint bits = k2_fused_bits();
    const uint groups = D / kMoEGroupSize;
    const uint words = (D * bits) / 32u;
    device const uint* vw =
        (device const uint*)(resident + (size_t)weightOffsets[id]) + local * words;
    device const bfloat* vs =
        (device const bfloat*)(resident + (size_t)scaleOffsets[id]) + local * groups;
    device const bfloat* vb =
        (device const bfloat*)(resident + (size_t)biasOffsets[id]) + local * groups;
    const uint elemsPerWord = 32u / bits;
    const uint lanesPerGroup = kMoEGroupSize / elemsPerWord;
    const uint groupsPerIteration = 32u / lanesPerGroup;
    const uint wordsPerGroup = kMoEGroupSize / elemsPerWord;
    const uint mask = (1u << bits) - 1u;
    float acc = 0.0f;
    for (uint base = 0; base < groups; base += groupsPerIteration) {
        const uint group = base + lane / lanesPerGroup;
        if (group >= groups) continue;
        const uint localWord = lane % lanesPerGroup;
        const uint word = group * wordsPerGroup + localWord;
        const uint elementBase = group * kMoEGroupSize + localWord * elemsPerWord;
        const uint vWord = vw[word];
        float dot = 0.0f;
        float sumX = 0.0f;
        for (uint i = 0; i < elemsPerWord; ++i) {
            const uint shift = i * bits;
            const float xv = float(x[elementBase + i]);
            dot = fma(float((vWord >> shift) & mask), xv, dot);
            sumX += xv;
        }
        acc = fma(float(vs[group]), dot, acc);
        acc = fma(float(vb[group]), sumX, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out[row] = half(acc);
    }
}
