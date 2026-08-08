#include <metal_stdlib>
using namespace metal;

// ============================================================================
// moe_k3 — fused Kimi K3 LatentMoE decode for the top-16 routed experts.
//
// Per token (docs/K3_DATAFLOW.md "MoE block"), with x_lat already computed:
//   phase 1: per selected expert k and intermediate row f:
//              g = w1_k[f] . x_lat, u = w3_k[f] . x_lat   (MXFP4, fp32 accum)
//              h[k*F + f] = SiTU(g, u)                    (fp32, stored fp32)
//   phase 2: per latent row d:
//              e_k[d] = w2_k[d] . h_k
//              y_lat[d] = sum_k routing_w[k] * e_k[d]     (fp32 accumulate)
// There is no residual add in the routed latent path (unlike the Gemma MoE);
// y_lat feeds routed_expert_norm + W_up elsewhere.
//
// All selected expert blobs live in ONE device buffer; `slot_offsets` holds
// the k byte offsets (UInt64 — a full K3 layer's expert region is ~15.7 GB).
// Keeping every expert in one binding means one command buffer covers all k
// expert reads, so the SSD fetches overlap instead of serializing.
//
// Subtensor layout inside one expert blob (canonical K3, row-major):
//   w1_packed [F x D_lat/2] | w1_scales [F x D_lat/32]
//   w2_packed [D_lat x F/2] | w2_scales [D_lat x F/32]
//   w3_packed [F x D_lat/2] | w3_scales [F x D_lat/32]
// The six offsets ride along as K3ExpertOffsets (computed on the Swift side
// by K3ExpertSubtensorOffsets.canonical).
//
// Compiled together with mxfp4.metal (K3MetalLibrary); the shared MXFP4
// helpers come from there. The include-guarded copy below keeps this file
// self-sufficient when compiled on its own.
// ============================================================================

#ifndef K3_MXFP4_COMMON
#define K3_MXFP4_COMMON

constant constexpr uint kK3Mxfp4GroupSize = 32;
constant constexpr uint kK3Mxfp4GroupBytes = 16;

constant constexpr float kK3E2M1LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float k3_e2m1_value(uint nibble) {
    return kK3E2M1LUT[nibble & 0xFu];
}

static inline float k3_e8m0_decode(uint8_t b) {
    if (b == 255u) { return 0.0f; }
    if (b == 0u)   { return as_type<float>(0x00400000u); }
    return as_type<float>(uint(b) << 23);
}

// Exact e2m1(nibble) * e8m0(scale) via integer bit construction (see
// mxfp4.metal for the full rationale; kept in sync with the guarded copy
// there). Not used by the fused MoE kernels — they take the factored
// per-group scale path — but part of the shared helper block.
static inline float k3_e2m1_times_e8m0(uint nibble, uint8_t scale_byte) {
    if (scale_byte == 255u) {
        return kK3E2M1LUT[nibble & 0xFu] * 0.0f;
    }
    const uint mag = nibble & 0x7u;
    if (mag == 0u) {
        return kK3E2M1LUT[nibble & 0xFu];
    }
    const uint m_bit = (mag > 1u) ? (mag & 1u) : 0u;
    const int e = (scale_byte == 0u) ? -127 : (int(scale_byte) - 127);
    const int k = int(mag >> 1) - 1;
    const int eTot = k + e;
    const uint sign = (nibble & 0x8u) << 28;
    uint bits;
    if (eTot >= 128) {
        bits = sign | 0x7F800000u;
    } else if (eTot >= -126) {
        bits = sign | (uint(eTot + 127) << 23) | (m_bit << 22);
    } else {
        bits = sign | (m_bit != 0u ? (3u << uint(eTot + 148))
                                   : (2u << uint(eTot + 148)));
    }
    return as_type<float>(bits);
}

static inline float k3_situ_gate(float g) {
    const float t = tanh(clamp(g / 4.0f, -20.0f, 20.0f));
    const float sg = 1.0f / (1.0f + exp(-g));
    return 4.0f * t * sg;
}

static inline float k3_situ_up(float u) {
    return 25.0f * tanh(clamp(u / 25.0f, -20.0f, 20.0f));
}

static inline float k3_situ_and_mul(float g, float u) {
    return k3_situ_gate(g) * k3_situ_up(u);
}

static inline float k3_mxfp4_group_dot_f16x(
    device const uint8_t* w_group,
    device const half*    x_group
) {
    device const ushort* wp = (device const ushort*)w_group;
    float dot = 0.0f;
    for (uint b = 0; b < 8u; ++b) {
        const uint w2 = uint(wp[b]);
        dot = fma(k3_e2m1_value(w2 & 0xFu),         float(x_group[b * 4u]),      dot);
        dot = fma(k3_e2m1_value((w2 >> 4) & 0xFu),  float(x_group[b * 4u + 1u]), dot);
        dot = fma(k3_e2m1_value((w2 >> 8) & 0xFu),  float(x_group[b * 4u + 2u]), dot);
        dot = fma(k3_e2m1_value(w2 >> 12),          float(x_group[b * 4u + 3u]), dot);
    }
    return dot;
}

static inline float k3_mxfp4_group_dot_f32x(
    device const uint8_t* w_group,
    device const float*   x_group
) {
    device const ushort* wp = (device const ushort*)w_group;
    float dot = 0.0f;
    for (uint b = 0; b < 8u; ++b) {
        const uint w2 = uint(wp[b]);
        dot = fma(k3_e2m1_value(w2 & 0xFu),         x_group[b * 4u],      dot);
        dot = fma(k3_e2m1_value((w2 >> 4) & 0xFu),  x_group[b * 4u + 1u], dot);
        dot = fma(k3_e2m1_value((w2 >> 8) & 0xFu),  x_group[b * 4u + 2u], dot);
        dot = fma(k3_e2m1_value(w2 >> 12),          x_group[b * 4u + 3u], dot);
    }
    return dot;
}

static inline float k3_mxfp4_row_dot_f32x(
    device const uint8_t* w_row,
    device const uint8_t* s_row,
    device const float*   x,
    uint n_groups,
    uint lane
) {
    float acc = 0.0f;
    for (uint g = lane; g < n_groups; g += 32u) {
        const float s = k3_e8m0_decode(s_row[g]);
        const float dot = k3_mxfp4_group_dot_f32x(
            w_row + g * kK3Mxfp4GroupBytes, x + g * kK3Mxfp4GroupSize);
        acc = fma(s, dot, acc);
    }
    return acc;
}

static inline float2 k3_mxfp4_dual_row_dot_f16x(
    device const uint8_t* w1_row,
    device const uint8_t* s1_row,
    device const uint8_t* w3_row,
    device const uint8_t* s3_row,
    device const half*    x,
    uint n_groups,
    uint lane
) {
    float g_acc = 0.0f;
    float u_acc = 0.0f;
    for (uint g = lane; g < n_groups; g += 32u) {
        const uint byte0 = g * kK3Mxfp4GroupBytes;
        const uint elem0 = g * kK3Mxfp4GroupSize;
        const float gs = k3_e8m0_decode(s1_row[g]);
        const float us = k3_e8m0_decode(s3_row[g]);
        const float g_dot = k3_mxfp4_group_dot_f16x(w1_row + byte0, x + elem0);
        const float u_dot = k3_mxfp4_group_dot_f16x(w3_row + byte0, x + elem0);
        g_acc = fma(gs, g_dot, g_acc);
        u_acc = fma(us, u_dot, u_acc);
    }
    return float2(g_acc, u_acc);
}

#endif // K3_MXFP4_COMMON

constant constexpr uint kK3MoEMaxTopK = 16;

struct K3ExpertOffsets {
    uint w1_packed_off;
    uint w1_scales_off;
    uint w2_packed_off;
    uint w2_scales_off;
    uint w3_packed_off;
    uint w3_scales_off;
};

constant uint FC_K3_MOE_DLAT [[function_constant(53)]];
constant uint FC_K3_MOE_F [[function_constant(54)]];
constant uint FC_K3_MOE_TOPK [[function_constant(55)]];
constant bool FC_K3_MOE_USE_FC [[function_constant(56)]];

static inline uint k3_moe_fc_dlat(constant uint& D_lat) {
    return (is_function_constant_defined(FC_K3_MOE_USE_FC) &&
            FC_K3_MOE_USE_FC &&
            is_function_constant_defined(FC_K3_MOE_DLAT)) ? FC_K3_MOE_DLAT : D_lat;
}

static inline uint k3_moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_K3_MOE_USE_FC) &&
            FC_K3_MOE_USE_FC &&
            is_function_constant_defined(FC_K3_MOE_F)) ? FC_K3_MOE_F : F;
}

static inline uint k3_moe_fc_topk(constant uint& top_k) {
    return (is_function_constant_defined(FC_K3_MOE_USE_FC) &&
            FC_K3_MOE_USE_FC &&
            is_function_constant_defined(FC_K3_MOE_TOPK)) ? FC_K3_MOE_TOPK : top_k;
}

// Phase 1: fused w1/w3 GEMV + SiTU. Row space is top_k * F, one SIMD per
// (slot, f) pair, eight per threadgroup — the Gemma persistent-phase1
// shape, minus the argument buffer (offsets table instead).
kernel void k3_moe_phase1_situ(
    device const uint8_t*  experts      [[buffer(0)]],
    device const ulong*    slot_offsets [[buffer(1)]],
    device const half*     x_lat        [[buffer(2)]],
    device float*          h            [[buffer(3)]],
    constant K3ExpertOffsets& subtensor [[buffer(4)]],
    constant uint&         D_lat        [[buffer(5)]],
    constant uint&         F            [[buffer(6)]],
    constant uint&         top_k        [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = k3_moe_fc_dlat(D_lat);
    const uint FF = k3_moe_fc_f(F);
    const uint TK = k3_moe_fc_topk(top_k);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= TK * FF) return;
    const uint slot = rowg / FF;
    const uint f = rowg % FF;

    device const uint8_t* base = experts + slot_offsets[slot];
    device const uint8_t* w1 = base + subtensor.w1_packed_off;
    device const uint8_t* s1 = base + subtensor.w1_scales_off;
    device const uint8_t* w3 = base + subtensor.w3_packed_off;
    device const uint8_t* s3 = base + subtensor.w3_scales_off;

    const uint n_groups  = DD / kK3Mxfp4GroupSize;
    const uint row_bytes = DD / 2u;
    float2 gu = k3_mxfp4_dual_row_dot_f16x(
        w1 + uint(f) * row_bytes, s1 + uint(f) * n_groups,
        w3 + uint(f) * row_bytes, s3 + uint(f) * n_groups,
        x_lat, n_groups, lane);
    const float g = simd_sum(gu.x);
    const float u = simd_sum(gu.y);
    if (lane == 0) {
        h[slot * FF + f] = k3_situ_and_mul(g, u);
    }
}

// Phase 2: w2 GEMV + weighted fp32 reduce over the k experts. One
// threadgroup per y_lat row, one SIMD per selected expert (top_k simds per
// group, dispatched exactly). Mirrors moe_phase2_down_reduce_k8's
// partial-array reduction, without the residual add.
kernel void k3_moe_phase2_reduce(
    device const uint8_t*  experts      [[buffer(0)]],
    device const ulong*    slot_offsets [[buffer(1)]],
    device const float*    h            [[buffer(2)]],
    device const float*    routing_w    [[buffer(3)]],
    device float*          y_lat        [[buffer(4)]],
    constant K3ExpertOffsets& subtensor [[buffer(5)]],
    constant uint&         D_lat        [[buffer(6)]],
    constant uint&         F            [[buffer(7)]],
    constant uint&         top_k        [[buffer(8)]],
    uint d      [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kK3MoEMaxTopK];
    const uint DD = k3_moe_fc_dlat(D_lat);
    const uint FF = k3_moe_fc_f(F);
    const uint TK = k3_moe_fc_topk(top_k);
    if (d >= DD) return;

    const uint slot = sg_idx;
    if (slot < TK) {
        device const uint8_t* base = experts + slot_offsets[slot];
        device const uint8_t* w2 = base + subtensor.w2_packed_off;
        device const uint8_t* s2 = base + subtensor.w2_scales_off;

        const uint n_groups  = FF / kK3Mxfp4GroupSize;
        const uint row_bytes = FF / 2u;
        float acc = k3_mxfp4_row_dot_f32x(
            w2 + uint(d) * row_bytes, s2 + uint(d) * n_groups,
            h + slot * FF, n_groups, lane);
        acc = simd_sum(acc);
        if (lane == 0) {
            partial[slot] = routing_w[slot] * acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = partial[0];
        for (uint k = 1; k < TK; ++k) {
            acc += partial[k];
        }
        y_lat[d] = acc;
    }
}
