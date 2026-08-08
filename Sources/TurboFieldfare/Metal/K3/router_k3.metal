#include <metal_stdlib>
using namespace metal;

// ============================================================================
// router_k3 — Kimi K3 FP32 sigmoid router GEMV.
//
//   logits[e] = W_gate[e] . x        (W_gate FP32, x already upcast FP32)
//   scores[e] = sigmoid(logits[e])
//   scores_biased[e] = scores[e] + e_score_correction_bias[e]  (optional)
//
// Both score vectors land in small shared readback buffers (896 floats =
// 3.5 KB); top-16 selection and the (sum + 1e-20) renormalize run on the CPU
// in K3Router.swift, exactly as KimiMoEGate does — no GPU top-k here.
//
// One SIMD per expert row, four rows per threadgroup (the Gemma router's
// shape). Lane-strided scalar loads: lane l reads elements l, l+32, ... —
// fully coalesced 128-byte transactions per iteration for FP32.
// ============================================================================

constant uint FC_K3_ROUTER_NE [[function_constant(57)]];
constant uint FC_K3_ROUTER_D [[function_constant(58)]];
constant bool FC_K3_ROUTER_USE_FC [[function_constant(59)]];

static inline uint k3_router_fc_ne(constant uint& num_experts) {
    return (is_function_constant_defined(FC_K3_ROUTER_USE_FC) &&
            FC_K3_ROUTER_USE_FC &&
            is_function_constant_defined(FC_K3_ROUTER_NE)) ? FC_K3_ROUTER_NE : num_experts;
}

static inline uint k3_router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_K3_ROUTER_USE_FC) &&
            FC_K3_ROUTER_USE_FC &&
            is_function_constant_defined(FC_K3_ROUTER_D)) ? FC_K3_ROUTER_D : D;
}

static inline float k3_sigmoid(float v) {
    return 1.0f / (1.0f + exp(-v));
}

kernel void k3_router_gemv_f32(
    device const float* W_gate        [[buffer(0)]],
    device const float* x             [[buffer(1)]],
    device float*       scores        [[buffer(2)]],
    device const float* bias          [[buffer(3)]],
    device float*       scores_biased [[buffer(4)]],
    constant uint&      num_experts   [[buffer(5)]],
    constant uint&      D             [[buffer(6)]],
    constant uint&      write_biased  [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 4;
    const uint NE = k3_router_fc_ne(num_experts);
    const uint DD = k3_router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    device const float* w_row = W_gate + uint(e) * DD;
    float acc = 0.0f;
    for (uint i = lane; i < DD; i += 32u) {
        acc = fma(w_row[i], x[i], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        const float s = k3_sigmoid(acc);
        scores[e] = s;
        if (write_biased != 0u) {
            scores_biased[e] = s + bias[e];
        }
    }
}
