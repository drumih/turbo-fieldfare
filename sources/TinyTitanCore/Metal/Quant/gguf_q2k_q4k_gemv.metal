#include <metal_stdlib>
using namespace metal;

// Fused Q2_K / Q4_K GEMV for the Inkling2 GGUF path. One SIMD group per output
// row, eight rows per threadgroup, matching the shape of the bf16 and NVFP4
// GEMVs. Each lane handles eight consecutive elements of a 256-value block, so
// the 84/144-byte blocks are read once and no full-precision expert is
// materialised.

kernel void q2k_gemv(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = groupIndex * 8u + sg;
    if (row >= rows) return;
    const uint blockCount = columns / 256u;
    device const uchar* rowWeights = weights + row * blockCount * 84u;
    const uint e0 = lane * 8u;
    const uint sub = e0 / 16u;
    const uint halfBlock = e0 / 128u;
    const uint e2 = e0 % 128u;
    const uint j = e2 / 32u;
    const uint within = e2 % 32u;
    const uint qBase = halfBlock * 32u + (within / 16u) * 16u;
    const uint shift = 2u * j;
    const uint scIndex = halfBlock * 8u + j * 2u + within / 16u;
    const uint l0 = within % 16u;
    float sum = 0.0f;
    for (uint blockIndex = 0u; blockIndex < blockCount; ++blockIndex) {
        device const uchar* b = rowWeights + blockIndex * 84u;
        const float d = float(as_type<half>(ushort(ushort(b[80]) | (ushort(b[81]) << 8u))));
        const float dmin = float(as_type<half>(ushort(ushort(b[82]) | (ushort(b[83]) << 8u))));
        const uchar scale = b[scIndex];
        const float dl = d * float(scale & 0x0Fu);
        const float ml = dmin * float(scale >> 4u);
        device const uchar* qs = b + 16u + qBase;
        for (uint i = 0u; i < 8u; ++i) {
            const uint quant = (uint(qs[l0 + i]) >> shift) & 3u;
            sum = fma(dl * float(quant) - ml, x[blockIndex * 256u + e0 + i], sum);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}

kernel void q4k_gemv(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = groupIndex * 8u + sg;
    if (row >= rows) return;
    const uint blockCount = columns / 256u;
    device const uchar* rowWeights = weights + row * blockCount * 144u;
    const uint e0 = lane * 8u;
    const uint g = e0 / 32u;
    const uint within = e0 % 32u;
    float sum = 0.0f;
    for (uint blockIndex = 0u; blockIndex < blockCount; ++blockIndex) {
        device const uchar* b = rowWeights + blockIndex * 144u;
        const float d = float(as_type<half>(ushort(ushort(b[0]) | (ushort(b[1]) << 8u))));
        const float dmin = float(as_type<half>(ushort(ushort(b[2]) | (ushort(b[3]) << 8u))));
        const device uchar* scales = b + 4u;
        uint s;
        uint m;
        if (g < 4u) {
            s = uint(scales[g]) & 0x3Fu;
            m = uint(scales[g + 4u]) & 0x3Fu;
        } else {
            s = (uint(scales[g + 4u]) & 0x0Fu)
                | ((uint(scales[g - 4u]) >> 6u) << 4u);
            m = (uint(scales[g + 4u]) >> 4u)
                | ((uint(scales[g]) >> 6u) << 4u);
        }
        const float sc = float(s);
        const float mm = float(m);
        const float ds = d * sc;
        const float dm = dmin * mm;
        const bool low = (g & 1u) == 0u;
        device const uchar* qs = b + 16u + (g / 2u) * 32u;
        for (uint i = 0u; i < 8u; ++i) {
            const uint packed = uint(qs[within + i]);
            const uint nibble = low ? (packed & 0x0Fu) : (packed >> 4u);
            sum = fma(ds * float(nibble) - dm, x[blockIndex * 256u + e0 + i], sum);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}
