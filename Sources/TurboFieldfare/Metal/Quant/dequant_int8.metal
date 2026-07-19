#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int8 — MLX `affine` 8-bit dequant.
//
// Layout per weight row of length N:
//   q       : N unsigned bytes, range 0..255.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[i] = float(q[i]) * scale[i/64] + bias[i/64].
//
// Fused dequant + GEMV for the router and shared expert projections.
// ============================================================================

constant constexpr uint kInt8GroupSize = 64;
constant constexpr uint kRowsPerTGInt8 = 8;
constant uint FC_INT8_M [[function_constant(70)]];
constant uint FC_INT8_N [[function_constant(71)]];
constant bool FC_INT8_USE_FC [[function_constant(72)]];
constant uint FC_SHARED_INT8_ROWS_PER_TG [[function_constant(73)]];
constant constexpr float kInt8GeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kInt8GeluCubicCoeff  = 0.044715f;

static inline uint int8_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT8_USE_FC) &&
            FC_INT8_USE_FC &&
            is_function_constant_defined(FC_INT8_M)) ? FC_INT8_M : M;
}

static inline uint int8_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT8_USE_FC) &&
            FC_INT8_USE_FC &&
            is_function_constant_defined(FC_INT8_N)) ? FC_INT8_N : N;
}
static inline uint shared_int8_rows_per_tg() {
    return is_function_constant_defined(FC_SHARED_INT8_ROWS_PER_TG)
        ? FC_SHARED_INT8_ROWS_PER_TG
        : kRowsPerTGInt8;
}

inline float int8_gelu_pytorch_tanh(float x) {
    float x3 = x * x * x;
    float inner = kInt8GeluSqrt2OverPi * (x + kInt8GeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

// y[m] = sum_{n} W[m, n] * x[n]. One-SIMD-per-row variant: 32 threads
// cooperate on a single output row, each handling 2 elements per group of 64.
// Mirror of `dequant_int4_gemv_simd` but with one byte per element (no nibble
// unpack). Dispatch: threadgroupsPerGrid=(M,1,1), threadsPerThreadgroup=(32,1,1).
// See `dequant_int4_gemv_simd` for the multi-row-per-TG rationale; same trick
// applied here so the M=2112/2816 shared MLP GEMVs and the M=262144 lm_head
// stay in the GPU's preferred 256-thread/TG zone.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_int8_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint MM = int8_fc_m(M);
    const uint NN = int8_fc_n(N);
    const uint row = tg_idx * kRowsPerTGInt8 + sg_idx;
    if (row >= MM) return;
    const uint n_groups = NN / kInt8GroupSize;
    device const uint8_t* W_row = W      + uint(row) * NN;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        float s = float(s_row[g]);
        float b = float(b_row[g]);
        uint i0 = g * kInt8GroupSize + lane * 2;
        uint i1 = i0 + 1;
        float q0 = float(uint(W_row[i0]));
        float q1 = float(uint(W_row[i1]));
        float x0 = float(x[i0]);
        float x1 = float(x[i1]);
        float dot_qx = q0 * x0 + q1 * x1;
        float sum_x  = x0 + x1;
        acc = fma(s, dot_qx, acc);
        acc = fma(b, sum_x,  acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void shared_int8_gate_up_act_simd(
    device const uint8_t* gateW      [[buffer(0)]],
    device const bfloat*  gateScales [[buffer(1)]],
    device const bfloat*  gateBiases [[buffer(2)]],
    device const uint8_t* upW        [[buffer(3)]],
    device const bfloat*  upScales   [[buffer(4)]],
    device const bfloat*  upBiases   [[buffer(5)]],
    device const half*    x          [[buffer(6)]],
    device half*          act        [[buffer(7)]],
    constant uint&        M          [[buffer(8)]],
    constant uint&        N          [[buffer(9)]],
    uint                  tg_idx     [[threadgroup_position_in_grid]],
    uint                  sg_idx     [[simdgroup_index_in_threadgroup]],
    uint                  lane       [[thread_index_in_simdgroup]]
) {
    const uint MM = int8_fc_m(M);
    const uint NN = int8_fc_n(N);
    const uint row = tg_idx * shared_int8_rows_per_tg() + sg_idx;
    if (row >= MM) return;
    const uint n_groups = NN / kInt8GroupSize;
    device const uint8_t* gate_row = gateW + row * NN;
    device const uint8_t* up_row = upW + row * NN;
    device const bfloat* gate_s_row = gateScales + row * n_groups;
    device const bfloat* gate_b_row = gateBiases + row * n_groups;
    device const bfloat* up_s_row = upScales + row * n_groups;
    device const bfloat* up_b_row = upBiases + row * n_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        uint i0 = g * kInt8GroupSize + lane * 2u;
        uint i1 = i0 + 1u;
        float x0 = float(x[i0]);
        float x1 = float(x[i1]);
        float sum_x = x0 + x1;

        float gate_dot = float(uint(gate_row[i0])) * x0
                       + float(uint(gate_row[i1])) * x1;
        gate_acc = fma(float(gate_s_row[g]), gate_dot, gate_acc);
        gate_acc = fma(float(gate_b_row[g]), sum_x, gate_acc);

        float up_dot = float(uint(up_row[i0])) * x0
                     + float(uint(up_row[i1])) * x1;
        up_acc = fma(float(up_s_row[g]), up_dot, up_acc);
        up_acc = fma(float(up_b_row[g]), sum_x, up_acc);
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);
    if (lane == 0) {
        float gate_half = float(half(gate_acc));
        float up_half = float(half(up_acc));
        act[row] = half(int8_gelu_pytorch_tanh(gate_half) * up_half);
    }
}
