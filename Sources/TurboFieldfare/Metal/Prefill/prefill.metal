#include <metal_stdlib>
using namespace metal;

constant constexpr uint kPrefillGroupSize = 64;
constant constexpr uint kPrefillRmsMaxSimdGroups = 8;
constant constexpr uint kPrefillPostMaxD = 4096;
constant constexpr uint kPrefillRouterMaxExperts = 256;
constant constexpr uint kPrefillRouterMaxTopK = 64;
constant constexpr uint kPrefillAttentionMaxSimdGroups = 16;
constant constexpr uint kPrefillMaxTileExperts = 16;
constant constexpr float kPrefillGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kPrefillGeluCubicCoeff = 0.044715f;
constant uint FC_PREFILL_KV_RING_CAP [[function_constant(76)]];

static inline float prefill_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kPrefillGeluSqrt2OverPi * (x + kPrefillGeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

kernel void prefill_embed_lookup_int4_block(
    device const uint8_t* table     [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const uint*    tokens    [[buffer(3)]],
    device half*          out       [[buffer(4)]],
    constant uint&        T         [[buffer(5)]],
    constant uint&        D         [[buffer(6)]],
    constant float&       out_scale [[buffer(7)]],
    uint2                 gid       [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    const uint token = tokens[t];
    const uint groups_per_row = D / kPrefillGroupSize;
    device const uint8_t* row_q = table  + token * (D / 2u);
    device const bfloat*  row_s = scales + token * groups_per_row;
    device const bfloat*  row_b = biases + token * groups_per_row;

    const uint8_t byte = row_q[d >> 1];
    const uint q = (d & 1u) == 0u ? uint(byte & 0x0Fu) : uint(byte >> 4);
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

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_post_attn_setup_block(
    device       half*   hidden                [[buffer(0)]],
    device const half*   attn                  [[buffer(1)]],
    device       half*   dense_x               [[buffer(2)]],
    device       half*   routed_x              [[buffer(3)]],
    device       half*   router_x              [[buffer(4)]],
    device const bfloat* w_post_attn           [[buffer(5)]],
    device const bfloat* w_pre_ffn             [[buffer(6)]],
    device const bfloat* w_pre_ffn2            [[buffer(7)]],
    constant uint&       T                     [[buffer(8)]],
    constant uint&       D                     [[buffer(9)]],
    constant uint&       hidden_stride_elems   [[buffer(10)]],
    constant uint&       attn_stride_elems     [[buffer(11)]],
    constant uint&       dense_stride_elems    [[buffer(12)]],
    constant uint&       routed_stride_elems   [[buffer(13)]],
    constant uint&       router_stride_elems   [[buffer(14)]],
    constant float&      rms_eps               [[buffer(15)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half attn_norm_tg[kPrefillPostMaxD];
    threadgroup half hidden_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device half* hidden_row = hidden + row * hidden_stride_elems;
    device const half* attn_row = attn + row * attn_stride_elems;
    device half* dense_row = dense_x + row * dense_stride_elems;
    device half* routed_row = routed_x + row * routed_stride_elems;
    device half* router_row = router_x + row * router_stride_elems;

    const float attn_inv = prefill_rms_block_inv(attn_row, D, rms_eps,
                                                 lid, lsize, lane, sg, sgs,
                                                 partial);
    for (uint i = lid; i < D; i += lsize) {
        attn_norm_tg[i] = half(float(attn_row[i]) * attn_inv * float(w_post_attn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        half h = half(float(hidden_row[i]) + float(attn_norm_tg[i]));
        hidden_tg[i] = h;
        hidden_row[i] = h;
        float hf = float(h);
        acc = fma(hf, hf, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float sum = (lane < sgs) ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) {
            partial[0] = rsqrt(sum / float(D) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float hidden_inv = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        const float h = float(hidden_tg[i]) * hidden_inv;
        dense_row[i] = half(h * float(w_pre_ffn[i]));
        routed_row[i] = half(h * float(w_pre_ffn2[i]));
        router_row[i] = half(h);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_layer_tail_block(
    device const half*   h2                    [[buffer(0)]],
    device const half*   h1                    [[buffer(1)]],
    device       half*   hidden                [[buffer(2)]],
    device const bfloat* w_postffn2            [[buffer(3)]],
    device const bfloat* w_postffn             [[buffer(4)]],
    constant uint&       T                     [[buffer(5)]],
    constant uint&       D                     [[buffer(6)]],
    constant uint&       h2_stride_elems       [[buffer(7)]],
    constant uint&       h1_stride_elems       [[buffer(8)]],
    constant uint&       hidden_stride_elems   [[buffer(9)]],
    constant float&      rms_eps               [[buffer(10)]],
    constant float&      layer_scalar          [[buffer(11)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half tmp_tg[kPrefillPostMaxD];
    threadgroup half h12_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device const half* h2_row = h2 + row * h2_stride_elems;
    device const half* h1_row = h1 + row * h1_stride_elems;
    device half* hidden_row = hidden + row * hidden_stride_elems;

    const float inv_h2 = prefill_rms_block_inv(h2_row, D, rms_eps,
                                               lid, lsize, lane, sg, sgs,
                                               partial);
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = half(float(h2_row[i]) * inv_h2 * float(w_postffn2[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        h12_tg[i] = h1_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(h12_tg[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float sum = (lane < sgs) ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) {
            partial[0] = rsqrt(sum / float(D) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_h12 = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = half(float(h12_tg[i]) * inv_h12 * float(w_postffn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const half h_scale = half(layer_scalar);
    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] * h_scale;
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

static inline float prefill_moe_int4_gemv_row_dev(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kPrefillGroupSize / 2u);
        device const half* xg = x + g * kPrefillGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kPrefillGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

static inline float prefill_moe_int4_gemv_row_tg(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    threadgroup const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kPrefillGroupSize / 2u);
        threadgroup const half* xg = x + g * kPrefillGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kPrefillGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

kernel void prefill_router_gemma4_block(
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
        device const uint8_t* W_row = W + e * D;
        device const bfloat* s_row = scales + e * n_groups;
        device const bfloat* b_row = biases + e * n_groups;

        float acc = 0.0f;
        for (uint g = 0; g < n_groups; ++g) {
            float s = float(s_row[g]);
            float b = float(b_row[g]);
            device const uint8_t* Wg = W_row + g * kPrefillGroupSize;
            device const half* xg = row_hidden + g * kPrefillGroupSize;
            device const bfloat* eg = effective_scale + g * kPrefillGroupSize;
            float dot_qx = 0.0f;
            float sum_x = 0.0f;
            for (uint k = 0; k < kPrefillGroupSize; ++k) {
                float q = float(uint(Wg[k]));
                float xv = float(xg[k]) * float(eg[k]);
                dot_qx = fma(q, xv, dot_qx);
                sum_x += xv;
            }
            acc = fma(s, dot_qx, acc);
            acc = fma(b, sum_x, acc);
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
            if (KK > 0 && s <= top_score[KK - 1]) continue;
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

    const float gate = prefill_moe_int4_gemv_row_dev(gate_W, gate_s, gate_b, x, f, p.D);
    const float up = prefill_moe_int4_gemv_row_dev(up_W, up_s, up_b, x, f, p.D);
    const uint row_elements = p.pair_count * p.F;
    const uint index = pair_local * p.F + f;
    gate_up_act_scratch[index] = half(gate);
    gate_up_act_scratch[row_elements + index] = half(up);
    gate_up_act_scratch[2u * row_elements + index] =
        half(prefill_gelu_pytorch_tanh(gate) * up);
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
    const half value = half(prefill_moe_int4_gemv_row_dev(down_W, down_s, down_b, act, d, p.F));
    down_scratch[pair_local * p.D + d] = value;
    route_partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}

kernel void prefill_dequant_int4_qmm_f16_block(
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
    const uint row_bytes = K / 2u;
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
            const uint8_t packed = w_row[k >> 1];
            const uint q = (k & 1u) == 0u ? uint(packed & 0x0Fu) : uint(packed >> 4);
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
};

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

static inline float prefill_attention_tg_sum_single_bank(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    const float result = prefill_attention_tg_sum(
        value, lane, simd_group, simdgroups, partial);
    // A single scratch bank needs an explicit reader-to-next-writer edge.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

[[kernel, max_total_threads_per_threadgroup(512)]]
kernel void attention_prefill_causal_tiled(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
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

    for (uint key = first; key < last_exclusive; ++key) {
        const uint phys_key = prefill_kv_slot(key);
        device const half* k_row = K + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
        const float qv = owns ? float(q_row[d]) : 0.0f;
        const float kv = owns ? float(k_row[d]) : 0.0f;
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
            device const half* v_row = V + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
            acc = fma(new_scale, float(v_row[d]), acc * old_scale);
        }
        row_sum = row_sum * old_scale + new_scale;
        row_max = new_max;
    }

    if (owns) {
        device half* out_row = O + t * p.oTokenStrideElements + qh * p.headDim;
        out_row[d] = row_sum > 0.0f ? half(acc / row_sum) : half(0.0f);
    }
}

struct PrefillTurboQuantAttentionParams {
    uint startPosition;
    uint queryCount;
    uint headDim;
    uint numQHeads;
    uint numKVHeads;
    uint kvValidCount;
    uint slidingWindow;
    uint qTokenStrideElements;
    uint oTokenStrideElements;
    float scale;
    uint layer;
    uint rotationSeed;
    uint keyBytesPerHead;
    uint keyPackedOffset;
    uint keyScaleOffset;
    uint valueBytesPerHead;
    uint valuePackedOffset;
    uint valueScaleOffset;
};

constant constexpr float kPrefillTQCodebook4[16] = {
    -2.7326f, -2.0690f, -1.6181f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6181f,  2.0690f,  2.7326f
};

static inline float prefill_tq_centroid(uint idx) {
    return kPrefillTQCodebook4[idx];
}

static inline ulong prefill_tq_mix64(ulong z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

static inline bool prefill_tq_rotation_is_negative(uint layer,
                                                   uint head,
                                                   uint dim,
                                                   uint seed) {
    ulong z = ulong(seed);
    z ^= ulong(layer) * 0x9E3779B97F4A7C15ul;
    z ^= ulong(head)  * 0xBF58476D1CE4E5B9ul;
    z ^= ulong(dim)   * 0x94D049BB133111EBul;
    return (prefill_tq_mix64(z) >> 63) != 0ul;
}

static inline float prefill_tq_apply_rotation(float v,
                                              uint layer,
                                              uint head,
                                              uint dim,
                                              uint seed) {
    return prefill_tq_rotation_is_negative(layer, head, dim, seed) ? -v : v;
}

static inline void prefill_tq_wht_inplace(threadgroup float* data,
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

static inline uint prefill_tq_read_index(device const uint8_t* cache,
                                         uint head_base,
                                         uint packed_offset,
                                         uint dim) {
    const uint packed_base = head_base + packed_offset;
    const uint byte = dim >> 1u;
    const uint shift = (dim & 1u) << 2u;
    return (uint(cache[packed_base + byte]) >> shift) & 0xFu;
}

static inline float prefill_tq_read_packed(device const uint8_t* cache,
                                           uint head_base,
                                           uint packed_offset,
                                           uint dim,
                                           float scale) {
    return prefill_tq_centroid(
        prefill_tq_read_index(cache, head_base, packed_offset, dim)) * scale;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void attention_prefill_turboquant_causal_tiled(
    device const half* Q [[buffer(0)]],
    device const uint8_t* K_cache [[buffer(1)]],
    device const uint8_t* V_cache [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillTurboQuantAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 lsize3 [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint t = tg.x;
    const uint qh = tg.y;
    if (t >= p.queryCount || qh >= p.numQHeads) return;

    threadgroup float q_smem[512];
    threadgroup float out_smem[512];
    threadgroup float partial[kPrefillAttentionMaxSimdGroups];

    const uint D = p.headDim;
    const uint tid = tid3.x;
    const uint lsize = lsize3.x;
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh / q_per_kv;
    const uint key_bytes_per_token = p.keyBytesPerHead * p.numKVHeads;
    const uint value_bytes_per_token = p.valueBytesPerHead * p.numKVHeads;
    const uint abs_q = p.startPosition + t;

    uint first = 0u;
    if (p.slidingWindow != 0u && abs_q + 1u > p.slidingWindow) {
        first = abs_q + 1u - p.slidingWindow;
    }
    const uint last_exclusive = min(p.kvValidCount, abs_q + 1u);

    device const half* q_row = Q + t * p.qTokenStrideElements + qh * D;
    for (uint i = tid; i < D; i += lsize) {
        const float qv = float(q_row[i]);
        q_smem[i] = prefill_tq_apply_rotation(qv, p.layer, kvh, i, p.rotationSeed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefill_tq_wht_inplace(q_smem, D, tid, lsize);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint d0 = tid;
    const uint d1 = tid + lsize;
    const bool owns0 = d0 < D;
    const bool owns1 = d1 < D;

    for (uint key = first; key < last_exclusive; ++key) {
        const uint key_head_base = key * key_bytes_per_token + kvh * p.keyBytesPerHead;
        const uint value_head_base = key * value_bytes_per_token + kvh * p.valueBytesPerHead;
        const device half* key_scale_ptr = reinterpret_cast<const device half*>(
            K_cache + key_head_base + p.keyScaleOffset);
        const device half* value_scale_ptr = reinterpret_cast<const device half*>(
            V_cache + value_head_base + p.valueScaleOffset);
        const float key_scale = float(*key_scale_ptr);
        const float value_scale = float(*value_scale_ptr);

        float partial_dot = 0.0f;
        for (uint i = tid; i < D; i += lsize) {
            const float k_val = prefill_tq_read_packed(K_cache,
                                                       key_head_base,
                                                       p.keyPackedOffset,
                                                       i,
                                                       key_scale);
            partial_dot = fma(q_smem[i], k_val, partial_dot);
        }

        float score = prefill_attention_tg_sum_single_bank(
            partial_dot, lane, simd_group, simdgroups, partial);
        score *= p.scale;

        const float new_max = max(row_max, score);
        const float old_scale = row_sum > 0.0f ? fast::exp(row_max - new_max) : 0.0f;
        const float new_scale = fast::exp(score - new_max);
        if (owns0) {
            const float v0 = prefill_tq_read_packed(V_cache,
                                                    value_head_base,
                                                    p.valuePackedOffset,
                                                    d0,
                                                    value_scale);
            acc0 = fma(new_scale, v0, acc0 * old_scale);
        }
        if (owns1) {
            const float v1 = prefill_tq_read_packed(V_cache,
                                                    value_head_base,
                                                    p.valuePackedOffset,
                                                    d1,
                                                    value_scale);
            acc1 = fma(new_scale, v1, acc1 * old_scale);
        }
        row_sum = row_sum * old_scale + new_scale;
        row_max = new_max;
    }

    if (owns0) {
        out_smem[d0] = row_sum > 0.0f ? acc0 / row_sum : 0.0f;
    }
    if (owns1) {
        out_smem[d1] = row_sum > 0.0f ? acc1 / row_sum : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    prefill_tq_wht_inplace(out_smem, D, tid, lsize);

    if (owns0) {
        device half* out_row = O + t * p.oTokenStrideElements + qh * D;
        const float value0 = prefill_tq_apply_rotation(
            out_smem[d0], p.layer, kvh, d0, p.rotationSeed);
        out_row[d0] = half(value0);
        if (owns1) {
            const float value1 = prefill_tq_apply_rotation(
                out_smem[d1], p.layer, kvh, d1, p.rotationSeed);
            out_row[d1] = half(value1);
        }
    }
}
