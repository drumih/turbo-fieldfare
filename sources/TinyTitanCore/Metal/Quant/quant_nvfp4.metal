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

// One thread per output row: dequantize the packed row and dot it with x.
kernel void nvfp4_gemv(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* outer [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const uint blocks = columns / 16u;
    const float scale0 = outer[0];
    float sum = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        const float scale = e4m3_value(scales[row * blocks + block]) * scale0;
        const uint byteBase = row * (columns / 2u) + block * 8u;
        for (uint j = 0; j < 16u; ++j) {
            const uchar byte = packed[byteBase + j / 2u];
            const uchar nibble = (j & 1u) ? (byte >> 4) : (byte & 0x0Fu);
            sum += e2m1_value(nibble) * scale * x[block * 16u + j];
        }
    }
    y[row] = sum;
}

kernel void mxfp8_gemv(
    device const uchar* weights [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const uint blocks = columns / 32u;
    float sum = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        const uchar scaleByte = scales[row * blocks + block];
        const float scale = exp2(float(scaleByte) - 127.0f);
        for (uint j = 0; j < 32u; ++j) {
            const uchar bits = weights[row * columns + block * 32u + j];
            sum += e4m3_value(bits) * scale * x[block * 32u + j];
        }
    }
    y[row] = sum;
}
