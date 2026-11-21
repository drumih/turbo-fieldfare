#include <metal_stdlib>
using namespace metal;

constant float e2m1_values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static inline float e4m3_value(uchar bits) {
    uint exponent = (bits >> 3) & 0x0F;
    uint mantissa = bits & 0x07;
    if ((bits & 0x7F) == 0x7F) return 0.0f;
    float magnitude;
    if (exponent == 0) {
        magnitude = float(mantissa) * 0.001953125f;
    } else {
        magnitude = (1.0f + float(mantissa) * 0.125f) * exp2(float(exponent) - 7.0f);
    }
    return (bits & 0x80) ? -magnitude : magnitude;
}

static inline float e2m1_value(uchar nibble) {
    float magnitude = e2m1_values[nibble & 0x07];
    return (nibble & 0x08) ? -magnitude : magnitude;
}

// Eight rows per threadgroup, one simdgroup per row: the same launch geometry
// as the bf16 GEMV. The earlier one-threadgroup-per-row version created eight
// times as many threadgroups and ran the dequant loop with 256 threads on one
// row, which measured slower than the memory it read.
kernel void nvfp4_gemv(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* outer [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg * 8u + sg;
    if (row >= rows) return;
    const uint blocks = columns / 16u;
    const float scale0 = outer[0];
    float sum = 0.0f;
    for (uint base = 0; base < columns; base += 64u) {
        const uint i = base + lane * 2u;
        const uint block = i / 16u;
        const float scale = e4m3_value(scales[row * blocks + block]) * scale0;
        const uint byteBase = row * (columns / 2u);
        const uchar b0 = packed[byteBase + i / 2u];
        const uchar b1 = packed[byteBase + (i + 1u) / 2u];
        const uchar n0 = (i & 1u) ? (b0 >> 4) : (b0 & 0x0Fu);
        const uchar n1 = ((i + 1u) & 1u) ? (b1 >> 4) : (b1 & 0x0Fu);
        sum = fma(e2m1_value(n0) * scale, x[i], sum);
        sum = fma(e2m1_value(n1) * scale, x[i + 1u], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}

kernel void mxfp8_gemv(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tg * 8u + sg;
    if (row >= rows) return;
    const uint blocks = columns / 32u;
    float sum = 0.0f;
    for (uint base = 0; base < columns; base += 64u) {
        const uint i = base + lane * 2u;
        const uint block = i / 32u;
        const float scale = exp2(float(scales[row * blocks + block]) - 127.0f);
        const uint rowBase = row * columns;
        sum = fma(e4m3_value(weights[rowBase + i]) * scale, x[i], sum);
        sum = fma(e4m3_value(weights[rowBase + i + 1u]) * scale, x[i + 1u], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}

// Softcapped SwiGLU for one expert: input is interleaved [gate, up] rows.
kernel void softcapped_swiglu(
    device const float* gate_up [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& gate_cap [[buffer(2)]],
    constant float& up_cap [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= intermediate) return;
    const float gate = gate_up[index * 2u];
    const float up = gate_up[index * 2u + 1u];
    const float g = tanh(gate / gate_cap) * gate_cap * (1.0f / (1.0f + exp(-gate)));
    const float u = tanh(up / up_cap) * up_cap;
    out[index] = g * u;
}
