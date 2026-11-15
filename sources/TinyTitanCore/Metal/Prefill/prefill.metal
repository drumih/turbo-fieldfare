#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif

constant constexpr uint kPrefillGroupSize = 64;
constant constexpr uint kPrefillRmsMaxSimdGroups = 8;
// 512, for Qwen3.8-Flash-Next. This sizes a threadgroup score array AND
// clamps the scored range (`NE = min(num_experts, ...)`), so a model with more
// experts than this does not fail -- it silently scores only the first N and
// routes as if the rest did not exist. 512 floats is 2 KiB of threadgroup
// memory, well inside the limit.
constant constexpr uint kPrefillRouterMaxExperts = 512;
constant constexpr uint kPrefillRouterMaxTopK = 64;
constant constexpr uint kPrefillAttentionMaxSimdGroups = 16;
constant constexpr uint kPrefillMaxTileExperts = 16;
constant constexpr float kPrefillGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kPrefillGeluCubicCoeff = 0.044715f;
constant uint FC_PREFILL_KV_RING_CAP [[function_constant(76)]];
// Unset/false = gelu_pytorch_tanh, true = silu (Qwen 3.6 SwiGLU).
constant bool FC_PREFILL_ACT_SILU [[function_constant(77)]];
// Packed affine weight width for chunked embedding, projections, and routed
// experts. Leaving it unset preserves the original INT4 specialization.
constant uint FC_PREFILL_AFFINE_BITS [[function_constant(78)]];
constant uint FC_PREFILL_ROUTER_BITS [[function_constant(79)]];

static inline uint prefill_affine_bits() {
    return is_function_constant_defined(FC_PREFILL_AFFINE_BITS)
        ? FC_PREFILL_AFFINE_BITS : 4u;
}

static inline uint prefill_affine_value(
    device const uint8_t* packed,
    uint element,
    uint bits
) {
    const uint bit_offset = element * bits;
    const uint byte_offset = bit_offset >> 3;
    const uint shift = bit_offset & 7u;
    uint word = uint(packed[byte_offset]);
    if (shift + bits > 8u) {
        word |= uint(packed[byte_offset + 1u]) << 8u;
    }
    return (word >> shift) & ((1u << bits) - 1u);
}

static inline float prefill_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kPrefillGeluSqrt2OverPi * (x + kPrefillGeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

static inline float prefill_hidden_activation(float x) {
    if (is_function_constant_defined(FC_PREFILL_ACT_SILU) &&
        FC_PREFILL_ACT_SILU) {
        return x / (1.0f + exp(-x));
    }
    return prefill_gelu_pytorch_tanh(x);
}
kernel void prefill_embed_lookup_affine_block(
    device const uint8_t* table     [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const uint*    tokens    [[buffer(3)]],
    device half*          out       [[buffer(4)]],
    constant uint&        T         [[buffer(5)]],
    constant uint&        D         [[buffer(6)]],
    constant float&       out_scale [[buffer(7)]],
    constant uint&        vocab     [[buffer(8)]],
    uint2                 gid       [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    const uint token = tokens[t];
    // OOB token guard: the CPU clamps routed token ids, but a clamped id can
    // still land at the table edge; never index past the vocab rows. Zero
    // embeddings keep the prefill pipeline well-defined for OOB tokens.
    if (token >= vocab) {
        out[t * D + d] = half(0.0f);
        return;
    }
    const uint groups_per_row = D / kPrefillGroupSize;
    const uint bits = prefill_affine_bits();
    const uint row_bytes = D * bits / 8u;
    device const uint8_t* row_q = table + token * row_bytes;
    device const bfloat*  row_s = scales + token * groups_per_row;
    device const bfloat*  row_b = biases + token * groups_per_row;

    const uint q = prefill_affine_value(row_q, d, bits);
    const float s = float(row_s[d / kPrefillGroupSize]);
    const float b = float(row_b[d / kPrefillGroupSize]);
    out[t * D + d] = half((float(q) * s + b) * out_scale);
}

static inline float prefill_rms_block_inv(
    device const half* x,
    uint D,
    float eps,
    uint lid,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id,
    uint simdgroups,
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
            partial[0] = rsqrt(v / float(D) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_block(
    device const half*   x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device half*         out     [[buffer(2)]],
    constant uint&       T       [[buffer(3)]],
    constant uint&       D       [[buffer(4)]],
    constant float&      eps     [[buffer(5)]],
    uint                 row     [[threadgroup_position_in_grid]],
    uint                 lid     [[thread_position_in_threadgroup]],
    uint                 lsize   [[threads_per_threadgroup]],
    uint                 lane    [[thread_index_in_simdgroup]],
    uint                 sg      [[simdgroup_index_in_threadgroup]],
    uint                 sgs     [[simdgroups_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xr = x + row * D;
    device half* yr = out + row * D;
    const float inv = prefill_rms_block_inv(xr, D, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < D; i += lsize) {
        yr[i] = half(float(xr[i]) * inv * float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device const bfloat* weight              [[buffer(1)]],
    device half*         out                 [[buffer(2)]],
    constant uint&       T                   [[buffer(3)]],
    constant uint&       head_dim            [[buffer(4)]],
    constant uint&       num_heads           [[buffer(5)]],
    constant uint&       token_stride_elems  [[buffer(6)]],
    constant float&      eps                 [[buffer(7)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv * float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_no_scale_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device half*         out                 [[buffer(1)]],
    constant uint&       T                   [[buffer(2)]],
    constant uint&       head_dim            [[buffer(3)]],
    constant uint&       num_heads           [[buffer(4)]],
    constant uint&       token_stride_elems  [[buffer(5)]],
    constant float&      eps                 [[buffer(6)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv);
    }
}

struct PrefillTokenExpertPairMSL {
    uint token;
    uint expert;
    uint rank;
    uint weight_bits_and_reserved;
};

struct PrefillStreamedRoutedBlobsMSL {
    device const uint8_t* blob[kPrefillMaxTileExperts];
};

struct PrefillGroupedRoutedMoEStreamedParamsMSL {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride_elements;
    uint live_expert_count;
    uint local_expert_0;
    uint local_expert_1;
    uint local_expert_2;
    uint local_expert_3;
    uint local_expert_4;
    uint local_expert_5;
    uint local_expert_6;
    uint local_expert_7;
    uint local_expert_8;
    uint local_expert_9;
    uint local_expert_10;
    uint local_expert_11;
    uint local_expert_12;
    uint local_expert_13;
    uint local_expert_14;
    uint local_expert_15;
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

static inline uint prefill_streamed_local_expert_id(
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p,
    uint slot
) {
    switch (slot) {
        case 0: return p.local_expert_0;
        case 1: return p.local_expert_1;
        case 2: return p.local_expert_2;
        case 3: return p.local_expert_3;
        case 4: return p.local_expert_4;
        case 5: return p.local_expert_5;
        case 6: return p.local_expert_6;
        case 7: return p.local_expert_7;
        case 8: return p.local_expert_8;
        case 9: return p.local_expert_9;
        case 10: return p.local_expert_10;
        case 11: return p.local_expert_11;
        case 12: return p.local_expert_12;
        case 13: return p.local_expert_13;
        case 14: return p.local_expert_14;
        default: return p.local_expert_15;
    }
}

static inline float prefill_moe_affine_gemv_row_dev(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    const uint bits = prefill_affine_bits();
    const uint row_bytes = N * bits / 8u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const half* xg = x + g * kPrefillGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        if (bits == 4u) {
            device const uint8_t* Wg = W_row + g * (kPrefillGroupSize / 2u);
            for (uint k = 0; k < kPrefillGroupSize / 2u; ++k) {
                const uint8_t packed = Wg[k];
                const float x0 = float(xg[2u * k]);
                const float x1 = float(xg[2u * k + 1u]);
                dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
                dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
                sum_x += x0 + x1;
            }
        } else {
            const uint group_base = g * kPrefillGroupSize;
            for (uint k = 0; k < kPrefillGroupSize; ++k) {
                const float xv = float(xg[k]);
                const uint q = prefill_affine_value(W_row, group_base + k, bits);
                dot_qx = fma(float(q), xv, dot_qx);
                sum_x += xv;
            }
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

kernel void prefill_router_block(
    device const uint8_t* W                [[buffer(0)]],
    device const bfloat*  scales           [[buffer(1)]],
    device const bfloat*  biases           [[buffer(2)]],
    device const half*    hidden           [[buffer(3)]],
    device const bfloat*  effective_scale  [[buffer(4)]],
    device const bfloat*  per_expert_scale [[buffer(5)]],
    device uint*          out_indices      [[buffer(6)]],
    device half*          out_weights      [[buffer(7)]],
    constant uint&        T                [[buffer(8)]],
    constant uint&        num_experts      [[buffer(9)]],
    constant uint&        D                [[buffer(10)]],
    constant uint&        top_k            [[buffer(11)]],
    constant uint&        hidden_stride    [[buffer(12)]],
    uint                  row              [[threadgroup_position_in_grid]],
    uint                  tid              [[thread_position_in_threadgroup]],
    uint                  tg_size          [[threads_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float scores[kPrefillRouterMaxExperts];
    const uint NE = min(num_experts, kPrefillRouterMaxExperts);
    const uint KK = min(top_k, kPrefillRouterMaxTopK);
    device const half* row_hidden = hidden + row * hidden_stride;

    for (uint e = tid; e < NE; e += tg_size) {
        const uint n_groups = D / kPrefillGroupSize;
        const uint bits = is_function_constant_defined(FC_PREFILL_ROUTER_BITS)
            ? FC_PREFILL_ROUTER_BITS : 8u;
        const uint row_bytes = D * bits / 8u;
        device const uint8_t* W_row = W + e * row_bytes;
        device const bfloat* s_row = scales + e * n_groups;
        device const bfloat* b_row = biases + e * n_groups;

        float acc = 0.0f;
        // bits == 16: the router was promoted to the checkpoint's own bf16.
        // `row_bytes` above is already D * 16 / 8, the byte length of a bf16
        // row, so only the read changes. Kept in step with the decode path in
        // moe.metal -- prefill and decode must agree on routing or a prompt
        // and its continuation would select different experts.
        if (bits == 16u) {
            device const bfloat* W_bf = (device const bfloat*)W_row;
            for (uint g = 0; g < n_groups; ++g) {
                device const half* xg = row_hidden + g * kPrefillGroupSize;
                device const bfloat* eg = effective_scale + g * kPrefillGroupSize;
                for (uint k = 0; k < kPrefillGroupSize; ++k) {
                    const uint idx = g * kPrefillGroupSize + k;
                    acc = fma(float(W_bf[idx]),
                              float(xg[k]) * float(eg[k]), acc);
                }
            }
        } else {
        for (uint g = 0; g < n_groups; ++g) {
            float s = float(s_row[g]);
            float b = float(b_row[g]);
            device const half* xg = row_hidden + g * kPrefillGroupSize;
            device const bfloat* eg = effective_scale + g * kPrefillGroupSize;
            float dot_qx = 0.0f;
            float sum_x = 0.0f;
            for (uint k = 0; k < kPrefillGroupSize; ++k) {
                float q = float(prefill_affine_value(
                    W_row, g * kPrefillGroupSize + k, bits));
                float xv = float(xg[k]) * float(eg[k]);
                dot_qx = fma(q, xv, dot_qx);
                sum_x += xv;
            }
            acc = fma(s, dot_qx, acc);
            acc = fma(b, sum_x, acc);
        }
        }
        scores[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint top_idx[kPrefillRouterMaxTopK];
        float top_score[kPrefillRouterMaxTopK];
        for (uint i = 0; i < kPrefillRouterMaxTopK; ++i) {
            top_idx[i] = 0u;
            top_score[i] = -INFINITY;
        }

        for (uint e = 0; e < NE; ++e) {
            float s = scores[e];
            // Strictly less than, mirroring the decode router. `<=` skipped an
            // expert on an exact tie with the k-th score, while decode admitted
            // it and let the lower-index tie-break decide -- so a prompt and its
            // continuation could select different experts for identical logits,
            // which this kernel's own nearby comment forbids.
            if (KK > 0 && s < top_score[KK - 1]) continue;
            uint pos = KK;
            for (uint i = 0; i < KK; ++i) {
                if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                    pos = i;
                    break;
                }
            }
            if (pos >= KK) continue;
            for (uint i = KK - 1; i > pos; --i) {
                top_idx[i] = top_idx[i - 1];
                top_score[i] = top_score[i - 1];
            }
            top_idx[pos] = e;
            top_score[pos] = s;
        }

        float max_s = top_score[0];
        float sum_exp = 0.0f;
        float exps[kPrefillRouterMaxTopK];
        for (uint i = 0; i < KK; ++i) {
            float e = fast::exp(top_score[i] - max_s);
            exps[i] = e;
            sum_exp += e;
        }
        for (uint i = 0; i < KK; ++i) {
            const uint expert_idx = top_idx[i];
            const float w = exps[i] / sum_exp;
            const float gain = float(per_expert_scale[expert_idx]);
            out_indices[row * top_k + i] = expert_idx;
            out_weights[row * top_k + i] = half(w * gain);
        }
    }
}

kernel void prefill_moe_reduce_token_major(
    device const half* route_partials [[buffer(0)]],
    device const half* route_weights  [[buffer(1)]],
    device half*       h2             [[buffer(2)]],
    constant uint&     T              [[buffer(3)]],
    constant uint&     top_k          [[buffer(4)]],
    constant uint&     D              [[buffer(5)]],
    uint2              gid            [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    float acc = 0.0f;
    for (uint r = 0; r < top_k; ++r) {
        const uint partial_index = (t * top_k + r) * D + d;
        acc = fma(float(route_weights[t * top_k + r]),
                  float(route_partials[partial_index]),
                  acc);
    }
    h2[t * D + d] = half(acc);
}

kernel void prefill_grouped_routed_moe_batched_phase1(
    device const half*                                   hidden               [[buffer(0)]],
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         gate_up_act_scratch  [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint f = gid.x;
    const uint pair_local = gid.y;
    if (f >= p.F || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const half* x = hidden + pair.token * p.hidden_stride_elements;
    device const uint8_t* gate_W = expert + p.gate_W_off;
    device const bfloat* gate_s = reinterpret_cast<device const bfloat*>(expert + p.gate_s_off);
    device const bfloat* gate_b = reinterpret_cast<device const bfloat*>(expert + p.gate_b_off);
    device const uint8_t* up_W = expert + p.up_W_off;
    device const bfloat* up_s = reinterpret_cast<device const bfloat*>(expert + p.up_s_off);
    device const bfloat* up_b = reinterpret_cast<device const bfloat*>(expert + p.up_b_off);

    const float gate = prefill_moe_affine_gemv_row_dev(gate_W, gate_s, gate_b, x, f, p.D);
    const float up = prefill_moe_affine_gemv_row_dev(up_W, up_s, up_b, x, f, p.D);
    const uint row_elements = p.pair_count * p.F;
    const uint index = pair_local * p.F + f;
    // Only the activated third region is ever read: batched_down consumes
    // `gate_up_act_scratch + 2 * pair_count * F`. The raw gate/up halves were
    // dead traffic — skip those stores (the layout offset stays so the
    // reader's math is unchanged).
    gate_up_act_scratch[2u * row_elements + index] =
        half(prefill_hidden_activation(gate) * up);
}

kernel void prefill_grouped_routed_moe_batched_down(
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         route_partials       [[buffer(5)]],
    device const half*                                   gate_up_act_scratch  [[buffer(7)]],
    device half*                                         down_scratch         [[buffer(8)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint pair_local = gid.y;
    if (d >= p.D || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const uint8_t* down_W = expert + p.down_W_off;
    device const bfloat* down_s = reinterpret_cast<device const bfloat*>(expert + p.down_s_off);
    device const bfloat* down_b = reinterpret_cast<device const bfloat*>(expert + p.down_b_off);
    device const half* act = gate_up_act_scratch + 2u * p.pair_count * p.F + pair_local * p.F;
    const half value = half(prefill_moe_affine_gemv_row_dev(down_W, down_s, down_b, act, d, p.F));
    down_scratch[pair_local * p.D + d] = value;
    route_partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}

kernel void prefill_dequant_affine_qmm_f16_block(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        T      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        K      [[buffer(7)]],
    uint2                 tid    [[thread_position_in_threadgroup]],
    uint2                 tgid   [[threadgroup_position_in_grid]]
) {
    const uint n = tgid.x * 8u + tid.x;
    const uint t = tgid.y * 8u + tid.y;
    if (t >= T || n >= N) return;

    const uint groups = K / kPrefillGroupSize;
    const uint bits = prefill_affine_bits();
    const uint row_bytes = K * bits / 8u;
    device const uint8_t* w_row = W + n * row_bytes;
    device const bfloat* s_row = scales + n * groups;
    device const bfloat* b_row = biases + n * groups;
    device const half* x_row = X + t * K;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        const uint group_base = g * kPrefillGroupSize;
        for (uint kk = 0; kk < kPrefillGroupSize; ++kk) {
            const uint k = group_base + kk;
            const uint q = prefill_affine_value(w_row, k, bits);
            const float w = fma(float(q), scale, bias);
            acc = fma(w, float(x_row[k]), acc);
        }
    }
    Y[t * N + n] = half(acc);
}

static inline void prefill_rope_apply_neox_pair(
    device half* head_ptr,
    uint i,
    uint half_dim,
    uint freq_divisor,
    float position,
    float theta_base
) {
    const float exponent = -float(2u * i) / float(freq_divisor);
    const float freq = pow(theta_base, exponent);
    const float angle = position * freq;
    const float c = cos(angle);
    const float s = sin(angle);

    const uint i0 = i;
    const uint i1 = half_dim + i;
    const float x0 = float(head_ptr[i0]);
    const float x1 = float(head_ptr[i1]);
    head_ptr[i0] = half(x0 * c - x1 * s);
    head_ptr[i1] = half(x0 * s + x1 * c);
}

kernel void prefill_rope_default_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    const uint half_dim = head_dim / 2u;
    if (i >= half_dim) return;
    if (h >= num_heads) return;

    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base);
}

kernel void prefill_rope_proportional_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    constant uint& rotated_pairs       [[buffer(6)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (i >= rotated_pairs) return;
    if (h >= num_heads) return;

    const uint half_dim = head_dim / 2u;
    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base);
}

// Qwen-style partial RoPE: rotation confined to the first `rotary_dim`
// elements per head, pairing (i, rotary_dim/2 + i), frequency divisor =
// rotary_dim; the remaining elements pass through untouched.
kernel void prefill_rope_neox_subdim_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    constant uint& rotary_dim          [[buffer(6)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    const uint half_rotary = rotary_dim / 2u;
    if (i >= half_rotary) return;
    if (h >= num_heads) return;

    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_rotary, rotary_dim,
                                 float(start_position + t), theta_base);
}

kernel void prefill_rope_yarn_neox_subdim_block(
    device half* data [[buffer(0)]],
    device const float* inverse_frequencies [[buffer(1)]],
    constant uint& start_position [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& token_stride_elems [[buffer(5)]],
    constant uint& rotary_dim [[buffer(6)]],
    constant float& attention_factor [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head = gid.y;
    const uint token = gid.z;
    const uint half_rotary = rotary_dim / 2u;
    if (pair >= half_rotary || head >= num_heads) return;
    device half* row = data + token * token_stride_elems + head * head_dim;
    const float angle = float(start_position + token) * inverse_frequencies[pair];
    const float cosine = cos(angle) * attention_factor;
    const float sine = sin(angle) * attention_factor;
    const float x0 = float(row[pair]);
    const float x1 = float(row[half_rotary + pair]);
    row[pair] = half(x0 * cosine - x1 * sine);
    row[half_rotary + pair] = half(x0 * sine + x1 * cosine);
}

struct PrefillAttentionParams {
    uint startPosition;
    uint queryCount;
    uint headDim;
    uint numQHeads;
    uint numKVHeads;
    uint kvValidCount;
    uint slidingWindow;
    uint kvTokenStrideElements;
    uint qTokenStrideElements;
    uint oTokenStrideElements;
    float scale;
    uint kvBits;
    uint kvTokenStrideBytes;
    uint kvValueBytes;
    uint kvGroupSize;
};

static inline float prefill_load_kv(
    device const uchar* cache,
    uint physical_position,
    uint flat_element,
    constant PrefillAttentionParams& p
) {
    const uint elements_per_row = p.numKVHeads * p.headDim;
    if (p.kvBits == 16u) {
        device const half* fp16 = reinterpret_cast<device const half*>(cache);
        return float(fp16[physical_position * p.kvTokenStrideElements + flat_element]);
    }
    device const uchar* row = cache + physical_position * p.kvTokenStrideBytes;
    uint quantized;
    if (p.kvBits == 8u) {
        quantized = uint(row[flat_element]);
    } else {
        const uchar packed = row[flat_element / 2u];
        quantized = (flat_element & 1u) == 0u
            ? uint(packed & 0x0fu) : uint(packed >> 4u);
    }
    const uint groups = (elements_per_row + p.kvGroupSize - 1u) / p.kvGroupSize;
    device const half* scales = reinterpret_cast<device const half*>(row + p.kvValueBytes);
    device const half* biases = scales + groups;
    const uint group = flat_element / p.kvGroupSize;
    return float(quantized) * float(scales[group]) + float(biases[group]);
}

static inline uint prefill_kv_slot(uint logical) {
    return (is_function_constant_defined(FC_PREFILL_KV_RING_CAP) &&
            FC_PREFILL_KV_RING_CAP != 0u)
        ? (logical % FC_PREFILL_KV_RING_CAP)
        : logical;
}

static inline float prefill_attention_tg_sum(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    float s = simd_sum(value);
    if (lane == 0u) {
        partial[simd_group] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        float v = lane < simdgroups ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

/// One key's Q.K with the affine-group work hoisted out of the element loop.
///
/// `prefill_load_kv` is correct but costs two integer divisions and three loads
/// per element, and the scale/bias pair only changes every `kvGroupSize`
/// elements -- two distinct groups for a 128-wide head. Per group this instead
/// computes
///
///     sum_e q_e * (quant_e * s + b)  ==  s * sum_e q_e*quant_e + b * sum_e q_e
///
/// so the scale and bias are applied once per group rather than once per
/// element, and `sum_e q_e` does not depend on the key at all -- the caller
/// precomputes it once per query and passes it in.
static inline float prefill_qsa_dot(
    device const uchar* cache,
    uint phys,
    uint base,
    uint HD,
    threadgroup const half* q_smem,
    threadgroup const float* q_group_sum,
    constant PrefillAttentionParams& p
) {
    if (p.kvBits == 16u) {
        device const half* fp16 = reinterpret_cast<device const half*>(cache);
        device const half* row = fp16 + phys * p.kvTokenStrideElements + base;
        float dot = 0.0f;
        for (uint e = 0u; e < HD; ++e) { dot = fma(float(q_smem[e]), float(row[e]), dot); }
        return dot;
    }
    device const uchar* row = cache + phys * p.kvTokenStrideBytes;
    const uint elements_per_row = p.numKVHeads * p.headDim;
    const uint groups = (elements_per_row + p.kvGroupSize - 1u) / p.kvGroupSize;
    device const half* scales = reinterpret_cast<device const half*>(row + p.kvValueBytes);
    device const half* biases = scales + groups;

    float dot = 0.0f;
    uint e = 0u;
    // Span index, not e / kvGroupSize: the head's base need not sit on a group
    // boundary, in which case the first span is short and every later index
    // would be off by one against the producer.
    uint gi = 0u;
    while (e < HD) {
        const uint flat = base + e;
        const uint group = flat / p.kvGroupSize;
        // How far to the end of this affine group, clamped to the head.
        const uint span = min(HD - e, p.kvGroupSize - (flat % p.kvGroupSize));
        float sq = 0.0f;
        if (p.kvBits == 8u) {
            for (uint i = 0u; i < span; ++i) {
                sq = fma(float(q_smem[e + i]), float(row[flat + i]), sq);
            }
        } else {
            for (uint i = 0u; i < span; ++i) {
                const uint fe = flat + i;
                const uchar packed = row[fe / 2u];
                const uint q = (fe & 1u) == 0u ? uint(packed & 0x0fu) : uint(packed >> 4u);
                sq = fma(float(q_smem[e + i]), float(q), sq);
            }
        }
        // q_group_sum holds sum(q) for the same span, so the bias term is one
        // multiply instead of `span` adds.
        dot = fma(float(scales[group]), sq,
                  fma(float(biases[group]), q_group_sum[gi], dot));
        e += span;
        gi += 1u;
    }
    return dot;
}

constant constexpr uint kPrefillQSATile = 128u;
constant constexpr uint kPrefillQSAMaxHeadDim = 512u;

/// Same result as `attention_prefill_causal_tiled`, arranged so the threadgroup
/// synchronises once per *tile* of keys instead of once per key.
///
/// The original owns one head-dim element per thread and reduces across the
/// threadgroup for every key's dot product: ~2,048 reductions per query per
/// head at the shipped budget, which is what left QSA layers at 6.8x a GDN
/// layer even after the selection was compacted. Here a thread computes one
/// key's dot product end to end -- no reduction at all -- the tile's scores go
/// to threadgroup memory, and the running softmax advances once per tile.
///
/// Traffic is unchanged: phase A reads TILE x headDim of K, phase C reads
/// TILE x headDim of V, exactly as before. Only the barrier count moves, from
/// O(keys) to O(keys / TILE).
kernel void attention_prefill_causal_qsa_tiled(
    device const half* Q [[buffer(0)]],
    device const uchar* K [[buffer(1)]],
    device const uchar* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    device const uchar* keep [[buffer(5)]],
    constant uint& useKeep [[buffer(6)]],
    constant uint& keepStride [[buffer(7)]],
    device const uint* keepIdx [[buffer(8)]],
    device const uint* keepCount [[buffer(9)]],
    constant uint& keepIdxStride [[buffer(10)]],
    // useKeep == 3: host-KV window. `keepIdx` holds the logical key and this
    // parallel list holds the window row that actually stores it.
    device const uint* keepRows [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]
) {
    const uint t = tg.x;
    const uint qh = tg.y;
    if (t >= p.queryCount || qh >= p.numQHeads) return;

    threadgroup float s_score[kPrefillQSATile];
    threadgroup uint  s_key[kPrefillQSATile];
    threadgroup half  q_smem[kPrefillQSAMaxHeadDim];
    threadgroup float q_gsum[kPrefillQSAMaxHeadDim / 16u];

    const uint threads = threads3.x;
    const uint d = tid.x;
    const uint HD = p.headDim;
    const bool owns = d < HD;
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh / q_per_kv;
    const uint abs_q = p.startPosition + t;
    uint first = 0u;
    if (p.slidingWindow != 0u && abs_q + 1u > p.slidingWindow) {
        first = abs_q + 1u - p.slidingWindow;
    }
    const uint last_exclusive = min(p.kvValidCount, abs_q + 1u);

    device const half* q_row = Q + t * p.qTokenStrideElements + qh * HD;
    for (uint i = d; i < HD; i += threads) { q_smem[i] = q_row[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // sum(q) per affine group: the bias half of every key's dot product, and
    // the same for all of them.
    const uint qBase = kvh * HD;   // K and V are indexed from the kv head
    if (d == 0u) {
        uint e = 0u;
        uint gi = 0u;
        while (e < HD) {
            const uint flat = qBase + e;
            const uint span = min(HD - e, p.kvGroupSize - (flat % p.kvGroupSize));
            float acc_q = 0.0f;
            for (uint i = 0u; i < span; ++i) { acc_q += float(q_smem[e + i]); }
            q_gsum[gi] = acc_q;
            e += span;
            gi += 1u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool compacted = (useKeep >= 2u);
    const bool windowed = (useKeep == 3u);
    const uint iterations = compacted ? keepCount[t] : (last_exclusive - first);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (uint base = 0u; base < iterations; base += kPrefillQSATile) {
        const uint n = min(kPrefillQSATile, iterations - base);

        // Phase A: one key per thread, whole dot product, no reduction.
        for (uint i = d; i < n; i += threads) {
            const uint step = base + i;
            uint key;
            bool valid;
            uint phys = 0u;
            if (compacted) {
                key = keepIdx[t * keepIdxStride + step];
                valid = (key >= first && key < last_exclusive);
                if (valid) {
                    phys = windowed ? keepRows[t * keepIdxStride + step]
                                    : prefill_kv_slot(key);
                }
            } else {
                key = first + step;
                valid = (useKeep == 0u) || (keep[t * keepStride + key] != 0u);
                if (valid) { phys = prefill_kv_slot(key); }
            }
            float dot = 0.0f;
            if (valid) {
                dot = prefill_qsa_dot(K, phys, kvh * HD, HD,
                                      q_smem, q_gsum, p);
            }
            s_key[i] = phys;
            // A dropped key must contribute nothing through the softmax, and
            // exp(-inf - m) is exactly zero for any finite running maximum.
            s_score[i] = valid ? dot * p.scale : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase B: every thread reads the whole tile from threadgroup memory
        // rather than reducing across threads -- n scratch loads beat a
        // barrier, and it keeps the running softmax identical in every thread.
        float tile_max = -INFINITY;
        for (uint i = 0u; i < n; ++i) { tile_max = max(tile_max, s_score[i]); }
        const float new_max = max(row_max, tile_max);
        // A tile in which every key was dropped leaves new_max at -inf, and
        // exp(-inf - -inf) is NaN rather than 0. Nothing in such a tile can
        // contribute, so carry the running state through untouched. The test
        // is uniform across the threadgroup, so the barrier below is still
        // reached by every thread.
        const bool tileContributes = isfinite(new_max);
        const float old_scale = (row_sum > 0.0f && tileContributes)
            ? fast::exp(row_max - new_max) : (row_sum > 0.0f ? 1.0f : 0.0f);

        // Phase C: accumulate this thread's output element over the tile.
        float tile_sum = 0.0f;
        float a = acc * old_scale;
        for (uint i = 0u; tileContributes && i < n; ++i) {
            const float w = isfinite(s_score[i])
                ? fast::exp(s_score[i] - new_max) : 0.0f;
            tile_sum += w;
            if (owns && w > 0.0f) {
                a = fma(w, prefill_load_kv(V, s_key[i], kvh * HD + d, p), a);
            }
        }
        acc = a;
        row_sum = row_sum * old_scale + tile_sum;
        if (tileContributes) { row_max = new_max; }
        // s_score/s_key are rewritten next iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (owns) {
        device half* out_row = O + t * p.oTokenStrideElements + qh * HD;
        out_row[d] = row_sum > 0.0f ? half(acc / row_sum) : half(0.0f);
    }
}

[[kernel, max_total_threads_per_threadgroup(512)]]
kernel void attention_prefill_causal_tiled(
    device const half* Q [[buffer(0)]],
    device const uchar* K [[buffer(1)]],
    device const uchar* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    // Sparse-attention selection, one row of `keepStride` bytes per query.
    // Unlike decode's single query, a chunk's rows each pick their own keys,
    // so this is a matrix and not a vector.
    device const uchar* keep [[buffer(5)]],
    constant uint& useKeep [[buffer(6)]],
    constant uint& keepStride [[buffer(7)]],
    // useKeep == 2: the selection arrived compacted as ascending key ids,
    // `keepCounts[query]` of them. The mask form made this loop O(context) --
    // one byte-load per visible key per query per head -- to discover an
    // answer the indexer already knew was O(budget). At 10k context that was
    // 4.53s per layer-chunk against 1.42s at 2.4k, where the two coincide.
    device const uint* keepIdx [[buffer(8)]],
    device const uint* keepCount [[buffer(9)]],
    constant uint& keepIdxStride [[buffer(10)]],
    // useKeep == 3: host-KV window, same logical/row split as the QSA kernel.
    device const uint* keepRows [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint t = tg.x;
    const uint qh = tg.y;
    if (t >= p.queryCount || qh >= p.numQHeads) return;

    threadgroup float partial[2u * kPrefillAttentionMaxSimdGroups];

    const uint d = tid.x;
    const bool owns = d < p.headDim;
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh / q_per_kv;
    const uint abs_q = p.startPosition + t;
    uint first = 0u;
    if (p.slidingWindow != 0u && abs_q + 1u > p.slidingWindow) {
        first = abs_q + 1u - p.slidingWindow;
    }
    const uint last_exclusive = min(p.kvValidCount, abs_q + 1u);

    device const half* q_row = Q + t * p.qTokenStrideElements + qh * p.headDim;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc = 0.0f;

    // Compacted selections iterate the budget; everything else iterates the
    // visible range. Both bounds are uniform across the threadgroup, so the
    // reduction below is reached by every thread the same number of times.
    const bool compacted = (useKeep >= 2u);
    const bool windowed = (useKeep == 3u);
    const uint iterations = compacted ? keepCount[t] : (last_exclusive - first);
    for (uint step = 0u; step < iterations; ++step) {
        uint key;
        uint phys_key;
        if (compacted) {
            key = keepIdx[t * keepIdxStride + step];
            // The host selects only visible keys, but a stale or truncated
            // list must not read outside the cache rather than merely give a
            // wrong answer.
            if (key < first || key >= last_exclusive) { continue; }
            phys_key = windowed ? keepRows[t * keepIdxStride + step]
                                : prefill_kv_slot(key);
        } else {
            key = first + step;
            // Uniform across the threadgroup -- `t` and `key` are the same for
            // every thread in it -- so skipping here cannot strand anyone at the
            // threadgroup reduction below.
            if (useKeep != 0u && keep[t * keepStride + key] == 0u) { continue; }
            phys_key = prefill_kv_slot(key);
        }
        const float qv = owns ? float(q_row[d]) : 0.0f;
        const uint flat = kvh * p.headDim + d;
        const float kv = owns ? prefill_load_kv(K, phys_key, flat, p) : 0.0f;
        const uint bank = key & 1u;
        const float score = prefill_attention_tg_sum(
            qv * kv,
            lane,
            simd_group,
            simdgroups,
            partial + bank * kPrefillAttentionMaxSimdGroups) * p.scale;

        const float new_max = max(row_max, score);
        const float old_scale = row_sum > 0.0f ? fast::exp(row_max - new_max) : 0.0f;
        const float new_scale = fast::exp(score - new_max);
        if (owns) {
            const float vv = prefill_load_kv(V, phys_key, flat, p);
            acc = fma(new_scale, vv, acc * old_scale);
        }
        row_sum = row_sum * old_scale + new_scale;
        row_max = new_max;
    }

    if (owns) {
        device half* out_row = O + t * p.oTokenStrideElements + qh * p.headDim;
        out_row[d] = row_sum > 0.0f ? half(acc / row_sum) : half(0.0f);
    }
}

/// Causal prefill attention, eight queries per threadgroup.
///
/// The scalar tiled kernel launches one threadgroup per (query, head) and
/// re-reads every visible K/V row for each of them, which is ~100 GB of cache
/// traffic for a 2.5K-token chunk. Here one simdgroup owns one query and the
/// eight simdgroups of a threadgroup cover eight consecutive queries, so the
/// same K/V row is read once per threadgroup instead of once per query. The
/// reduction is a `simd_sum`, so the kernel takes no threadgroup barrier in
/// the key loop at all.
constant constexpr uint kPrefillAttnQBlock = 8;
constant constexpr uint kPrefillAttnMaxDimsPerLane = 8;

kernel void attention_prefill_causal_q8(
    device const half* Q [[buffer(0)]],
    device const uchar* K [[buffer(1)]],
    device const uchar* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    device const uchar* keep [[buffer(5)]],
    constant uint& useKeep [[buffer(6)]],
    constant uint& keepStride [[buffer(7)]],
    device const uint* keepIdx [[buffer(8)]],
    device const uint* keepCount [[buffer(9)]],
    constant uint& keepIdxStride [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    const uint q0 = tg.x * kPrefillAttnQBlock;
    const uint qh = tg.y;
    if (q0 >= p.queryCount || qh >= p.numQHeads) return;
    const uint q = q0 + sg;
    if (q >= p.queryCount) return;
    // Gated on the host to headDim % 32 == 0 and headDim / 32 <= 8.
    const uint dims_per_lane = p.headDim / 32u;
    const uint d0 = lane * dims_per_lane;
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh / q_per_kv;
    const uint abs_q = p.startPosition + q;
    uint first = 0u;
    if (p.slidingWindow != 0u && abs_q + 1u > p.slidingWindow) {
        first = abs_q + 1u - p.slidingWindow;
    }
    const uint last_exclusive = min(p.kvValidCount, abs_q + 1u);
    device const half* q_row = Q + q * p.qTokenStrideElements + qh * p.headDim;
    const uint flat = kvh * p.headDim + d0;
    // The per-element helper recomputed the row pointer, the group count and
    // the group index for every element. All three are invariant across keys
    // except the row, so they are hoisted here: one division per lane instead
    // of two per element.
    const uint elements_per_row = p.numKVHeads * p.headDim;
    const uint kv_groups =
        (elements_per_row + p.kvGroupSize - 1u) / p.kvGroupSize;
    const uint g0 = flat / p.kvGroupSize;
    uint group_step[kPrefillAttnMaxDimsPerLane];
    {
        uint g = g0;
        uint group_end = (g0 + 1u) * p.kvGroupSize;
        for (uint i = 0; i < dims_per_lane; ++i) {
            if (flat + i >= group_end) {
                g += 1u;
                group_end += p.kvGroupSize;
            }
            group_step[i] = g - g0;
        }
    }
    const bool kv16 = (p.kvBits == 16u);
    const bool kv4 = (p.kvBits == 4u);

    float qreg[kPrefillAttnMaxDimsPerLane];
    float acc[kPrefillAttnMaxDimsPerLane];
    for (uint i = 0; i < dims_per_lane; ++i) {
        qreg[i] = float(q_row[d0 + i]);
        acc[i] = 0.0f;
    }
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (uint key = first; key < last_exclusive; ++key) {
        const uint phys_key = prefill_kv_slot(key);
        float partial = 0.0f;
        if (kv16) {
            device const half* kbase =
                reinterpret_cast<device const half*>(K)
                + phys_key * p.kvTokenStrideElements + flat;
            for (uint i = 0; i < dims_per_lane; ++i) {
                partial = fma(qreg[i], float(kbase[i]), partial);
            }
        } else {
            device const uchar* krow = K + phys_key * p.kvTokenStrideBytes;
            device const half* ksc =
                reinterpret_cast<device const half*>(krow + p.kvValueBytes);
            device const half* kbi = ksc + kv_groups;
            for (uint i = 0; i < dims_per_lane; ++i) {
                const uint element = flat + i;
                float quantized;
                if (kv4) {
                    const uchar packed = krow[element / 2u];
                    quantized = (element & 1u) == 0u
                        ? float(uint(packed & 0x0fu))
                        : float(uint(packed >> 4u));
                } else {
                    quantized = float(uint(krow[element]));
                }
                const uint g = g0 + group_step[i];
                partial = fma(
                    qreg[i],
                    quantized * float(ksc[g]) + float(kbi[g]),
                    partial);
            }
        }
        const float score = simd_sum(partial) * p.scale;
        const float new_max = max(row_max, score);
        const float old_scale = row_sum > 0.0f ? fast::exp(row_max - new_max) : 0.0f;
        const float new_scale = fast::exp(score - new_max);
        if (kv16) {
            device const half* vbase =
                reinterpret_cast<device const half*>(V)
                + phys_key * p.kvTokenStrideElements + flat;
            for (uint i = 0; i < dims_per_lane; ++i) {
                acc[i] = fma(new_scale, float(vbase[i]), acc[i] * old_scale);
            }
        } else {
            device const uchar* vrow = V + phys_key * p.kvTokenStrideBytes;
            device const half* vsc =
                reinterpret_cast<device const half*>(vrow + p.kvValueBytes);
            device const half* vbi = vsc + kv_groups;
            for (uint i = 0; i < dims_per_lane; ++i) {
                const uint element = flat + i;
                float quantized;
                if (kv4) {
                    const uchar packed = vrow[element / 2u];
                    quantized = (element & 1u) == 0u
                        ? float(uint(packed & 0x0fu))
                        : float(uint(packed >> 4u));
                } else {
                    quantized = float(uint(vrow[element]));
                }
                const uint g = g0 + group_step[i];
                acc[i] = fma(
                    new_scale,
                    quantized * float(vsc[g]) + float(vbi[g]),
                    acc[i] * old_scale);
            }
        }
        row_sum = row_sum * old_scale + new_scale;
        row_max = new_max;
    }

    device half* out_row = O + q * p.oTokenStrideElements + qh * p.headDim;
    for (uint i = 0; i < dims_per_lane; ++i) {
        out_row[d0 + i] = row_sum > 0.0f ? half(acc[i] / row_sum) : half(0.0f);
    }
}

#if defined(__HAVE_TENSOR__)

constant constexpr int kPrefillTensorOpsOutputs = 8;
constant constexpr int kPrefillTensorOpsKeys = 64;
constant constexpr int kPrefillTensorOpsHeadDim = 512;

static inline void attention_prefill_full_tensorops_2d_validity_v2_impl(
    device const half* Q,
    device half* K,
    device half* V,
    device half* O,
    constant PrefillAttentionParams& p,
    uint3 tg,
    uint lid,
    uint threads,
    threadgroup half* query_tile,
    threadgroup float* score_tile,
    threadgroup float* weight_tile,
    threadgroup float* row_max,
    threadgroup float* row_sum,
    threadgroup float* row_old_scale
) {
    constexpr auto qk_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsKeys,
        kPrefillTensorOpsHeadDim,
        false, true, false);
    constexpr auto pv_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsHeadDim,
        kPrefillTensorOpsKeys,
        false, false, false);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<4>> pv_op;

    using device_half_tensor =
        tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor =
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_float_tensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    const uint query_start = tg.x;
    const uint qh_start = tg.y * uint(kPrefillTensorOpsOutputs);
    const uint valid_query_rows =
        min(1u, p.queryCount - min(query_start, p.queryCount));
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh_start / q_per_kv;

    for (uint linear = lid;
         linear < uint(kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim);
         linear += threads) {
        const uint output_row =
            linear / uint(kPrefillTensorOpsHeadDim);
        const uint d = linear % uint(kPrefillTensorOpsHeadDim);
        if (valid_query_rows != 0u) {
            query_tile[linear] = Q[
                query_start * p.qTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d];
        } else {
            query_tile[linear] = half(0.0f);
        }
    }
    if (lid < uint(kPrefillTensorOpsOutputs)) {
        row_max[lid] = -INFINITY;
        row_sum[lid] = 0.0f;
        row_old_scale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup_half_tensor query_tensor(
        query_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsHeadDim,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsHeadDim}));
    threadgroup_float_tensor weight_tensor(
        weight_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsKeys,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsKeys}));
    device_half_tensor key_tensor(
        K + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));
    device_half_tensor value_tensor(
        V + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));

    auto query_slice = query_tensor.slice(0, 0);
    auto first_value_slice = value_tensor.slice(0, 0);
    auto output_accumulator =
        pv_op.get_destination_cooperative_tensor<
            decltype(weight_tensor), decltype(first_value_slice), float>();
    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (output_accumulator.is_valid_element(element)) {
            output_accumulator[element] = 0.0f;
        }
    }

    // This full-attention kernel starts at key zero and ignores slidingWindow.
    // The Swift selector must dispatch it only when every prior key is visible.
    const uint last =
        min(p.kvValidCount, p.startPosition + query_start + valid_query_rows);
    for (uint key_start = 0u;
         key_start < last;
         key_start += uint(kPrefillTensorOpsKeys)) {
        auto key_slice = key_tensor.slice(0, int32_t(key_start));
        auto score_product =
            qk_op.get_destination_cooperative_tensor<
                decltype(query_slice), decltype(key_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (score_product.is_valid_element(element)) {
                score_product[element] = 0.0f;
            }
        }
        qk_op.run(query_slice, key_slice, score_product);

        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (!score_product.is_valid_element(element)) continue;
            const auto position =
                score_product.get_multidimensional_index(element);
            const uint key_column = uint(position[0]);
            const uint output_row = uint(position[1]);
            score_tile[
                output_row * uint(kPrefillTensorOpsKeys) + key_column] =
                score_product[element] * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < uint(kPrefillTensorOpsOutputs)) {
            const uint output_row = lid;
            const uint causal_last = valid_query_rows != 0u
                ? min(p.kvValidCount, p.startPosition + query_start + 1u)
                : 0u;
            const uint visible =
                causal_last > key_start
                    ? min(uint(kPrefillTensorOpsKeys), causal_last - key_start)
                    : 0u;

            float tile_max = -INFINITY;
            for (uint key = 0u; key < visible; ++key) {
                tile_max = max(
                    tile_max,
                    score_tile[
                        output_row * uint(kPrefillTensorOpsKeys) + key]);
            }
            const float next_max = max(row_max[output_row], tile_max);
            const float old_scale = row_sum[output_row] > 0.0f
                ? fast::exp(row_max[output_row] - next_max)
                : 0.0f;
            float tile_sum = 0.0f;
            for (uint key = 0u;
                 key < uint(kPrefillTensorOpsKeys);
                 ++key) {
                const float weight = key < visible
                    ? fast::exp(
                        score_tile[
                            output_row * uint(kPrefillTensorOpsKeys) + key]
                        - next_max)
                    : 0.0f;
                weight_tile[
                    output_row * uint(kPrefillTensorOpsKeys) + key] = weight;
                tile_sum += weight;
            }
            row_old_scale[output_row] = old_scale;
            row_sum[output_row] =
                row_sum[output_row] * old_scale + tile_sum;
            row_max[output_row] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto value_slice = value_tensor.slice(0, int32_t(key_start));
        auto output_product =
            pv_op.get_destination_cooperative_tensor<
                decltype(weight_tensor), decltype(value_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_product.get_capacity();
             ++element) {
            if (output_product.is_valid_element(element)) {
                output_product[element] = 0.0f;
            }
        }
        pv_op.run(weight_tensor, value_slice, output_product);
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_accumulator.get_capacity();
             ++element) {
            if (!output_accumulator.is_valid_element(element)
                || !output_product.is_valid_element(element)) {
                continue;
            }
            const auto position =
                output_accumulator.get_multidimensional_index(element);
            const uint output_row = uint(position[1]);
            output_accumulator[element] =
                fma(
                    1.0f,
                    output_product[element],
                    output_accumulator[element] * row_old_scale[output_row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (!output_accumulator.is_valid_element(element)) continue;
        const auto position =
            output_accumulator.get_multidimensional_index(element);
        const uint d = uint(position[0]);
        const uint output_row = uint(position[1]);
        if (valid_query_rows != 0u) {
            const float denominator = row_sum[output_row];
            O[
                query_start * p.oTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d] = denominator > 0.0f
                    ? half(output_accumulator[element] / denominator)
                    : half(0.0f);
        }
    }
}

kernel void attention_prefill_full_tensorops_2d_validity_v2(
    device const half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]
) {
    threadgroup half query_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim];
    threadgroup float score_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float weight_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float row_max[kPrefillTensorOpsOutputs];
    threadgroup float row_sum[kPrefillTensorOpsOutputs];
    threadgroup float row_old_scale[kPrefillTensorOpsOutputs];
    attention_prefill_full_tensorops_2d_validity_v2_impl(
        Q, K, V, O, p, tg, lid, threads3.x,
        query_tile, score_tile, weight_tile,
        row_max, row_sum, row_old_scale);
}

#endif

/// Elementwise shared-expert activation: act = hidden_activation(gate) * up.
/// Used by the batched prefill path, where gate and up are full T x N GEMM
/// outputs rather than one row. The activation is the same helper the fused
/// per-token kernel uses, so the two paths cannot drift.
kernel void prefill_shared_act_block(
    device const half* gate  [[buffer(0)]],
    device const half* up    [[buffer(1)]],
    device half*       act   [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const float g = float(gate[gid]);
    const float u = float(up[gid]);
    act[gid] = half(prefill_hidden_activation(g) * u);
}
