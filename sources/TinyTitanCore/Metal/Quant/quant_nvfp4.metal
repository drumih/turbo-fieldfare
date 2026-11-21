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

// One threadgroup per output row: coalesced column reads, then a threadgroup
// reduction. The earlier one-thread-per-row version read a strided row per
// thread and ran at memory-latency speed.
kernel void nvfp4_gemv(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* outer [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float partial[256];
    const uint blocks = columns / 16u;
    const float scale0 = outer[0];
    float sum = 0.0f;
    for (uint i = tid; i < columns; i += tsize) {
        const uint block = i / 16u;
        const float scale = e4m3_value(scales[row * blocks + block]) * scale0;
        const uchar byte = packed[row * (columns / 2u) + i / 2u];
        const uchar nibble = (i & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        sum += e2m1_value(nibble) * scale * x[i];
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[row] = partial[0];
}

kernel void mxfp8_gemv(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float partial[256];
    const uint blocks = columns / 32u;
    float sum = 0.0f;
    for (uint i = tid; i < columns; i += tsize) {
        const uint block = i / 32u;
        const float scale = exp2(float(scales[row * blocks + block]) - 127.0f);
        const uchar bits = weights[row * columns + i];
        sum += e4m3_value(bits) * scale * x[i];
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsize / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[row] = partial[0];
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
