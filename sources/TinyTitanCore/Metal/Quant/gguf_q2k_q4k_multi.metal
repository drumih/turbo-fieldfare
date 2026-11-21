#include <metal_stdlib>
using namespace metal;

// Four output rows per SIMD group: eight lanes per row, 32 consecutive values
// per lane. The x chunk is loaded once per lane and reused for the row's four
// 8-value groups; the per-row reduction is three shuffle-xor steps inside the
// eight-lane subgroup rather than a full 32-lane sum.

static inline float reduce8(float sum, uint laneInRow) {
    sum += simd_shuffle_xor(sum, 1u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 4u);
    return sum;
}

kernel void q2k_gemv_multi(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint laneInRow = lane % 8u;
    const uint row = groupIndex * 32u + sg * 4u + lane / 8u;
    if (row >= rows) return;
    const uint blockCount = columns / 256u;
    device const uchar* rowWeights = weights + row * blockCount * 84u;
    const uint halfBlock = laneInRow / 4u;
    const uint j = laneInRow % 4u;
    const uint start = halfBlock * 128u + j * 32u;
    const uint qBase = halfBlock * 32u;
    const uint shift = 2u * j;
    const uint scBase = halfBlock * 8u + j * 2u;
    float sum = 0.0f;
    for (uint blockIndex = 0u; blockIndex < blockCount; ++blockIndex) {
        device const uchar* b = rowWeights + blockIndex * 84u;
        const float d = float(as_type<half>(ushort(ushort(b[80]) | (ushort(b[81]) << 8u))));
        const float dmin = float(as_type<half>(ushort(ushort(b[82]) | (ushort(b[83]) << 8u))));
        const uint xBase = blockIndex * 256u + start;
        // 32 values: two 16-value scale groups, each two 8-value chunks.
        for (uint sub = 0u; sub < 2u; ++sub) {
            const uchar scale = b[scBase + sub];
            const float dl = d * float(scale & 0x0Fu);
            const float ml = dmin * float(scale >> 4u);
            const ulong q0 = *(device const ulong*)(b + 16u + qBase + sub * 16u);
            const ulong q1 = *(device const ulong*)(b + 16u + qBase + sub * 16u + 8u);
            float qx = 0.0f;
            float sx = 0.0f;
            for (uint i = 0u; i < 8u; ++i) {
                const float xv = x[xBase + sub * 16u + i];
                const uint quant = (uint((q0 >> (8u * i)) & 0xFFu) >> shift) & 3u;
                qx = fma(float(quant), xv, qx);
                sx += xv;
            }
            for (uint i = 0u; i < 8u; ++i) {
                const float xv = x[xBase + sub * 16u + 8u + i];
                const uint quant = (uint((q1 >> (8u * i)) & 0xFFu) >> shift) & 3u;
                qx = fma(float(quant), xv, qx);
                sx += xv;
            }
            sum += dl * qx - ml * sx;
        }
    }
    sum = reduce8(sum, laneInRow);
    if (laneInRow == 0u) y[row] = sum;
}

kernel void q4k_gemv_multi(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint laneInRow = lane % 8u;
    const uint row = groupIndex * 32u + sg * 4u + lane / 8u;
    if (row >= rows) return;
    const uint blockCount = columns / 256u;
    device const uchar* rowWeights = weights + row * blockCount * 144u;
    const uint g = laneInRow;
    const uint start = g * 32u;
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
        const float ds = d * float(s);
        const float dm = dmin * float(m);
        const bool low = (g & 1u) == 0u;
        const device uchar* q0 = b + 16u + (g / 2u) * 32u;
        const ulong w0 = *(device const ulong*)q0;
        const ulong w1 = *(device const ulong*)(q0 + 8u);
        const ulong w2 = *(device const ulong*)(q0 + 16u);
        const ulong w3 = *(device const ulong*)(q0 + 24u);
        const uint xBase = blockIndex * 256u + start;
        float qx = 0.0f;
        float sx = 0.0f;
        for (uint chunk = 0u; chunk < 4u; ++chunk) {
            const ulong word = chunk == 0u ? w0 : chunk == 1u ? w1 : chunk == 2u ? w2 : w3;
            for (uint i = 0u; i < 8u; ++i) {
                const float xv = x[xBase + chunk * 8u + i];
                const uint byte = uint((word >> (8u * i)) & 0xFFu);
                const uint nibble = low ? (byte & 0x0Fu) : (byte >> 4u);
                qx = fma(float(nibble), xv, qx);
                sx += xv;
            }
        }
        sum += ds * qx - dm * sx;
    }
    sum = reduce8(sum, laneInRow);
    if (laneInRow == 0u) y[row] = sum;
}

static inline float reduce16(float sum) {
    sum += simd_shuffle_xor(sum, 1u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 4u);
    sum += simd_shuffle_xor(sum, 8u);
    return sum;
}

kernel void q2k_gemv_multi2(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint laneInRow = lane % 16u;
    const uint row = groupIndex * 16u + sg * 2u + lane / 16u;
    if (row >= rows) return;
    const uint blockCount = columns / 256u;
    device const uchar* rowWeights = weights + row * blockCount * 84u;
    const uint halfBlock = laneInRow / 8u;
    const uint within = laneInRow % 8u;
    const uint j = within / 2u;
    const uint sub = within % 2u;
    const uint qBase = halfBlock * 32u + sub * 16u;
    const uint shift = 2u * j;
    const uint scIndex = halfBlock * 8u + j * 2u + sub;
    float sum = 0.0f;
    for (uint blockIndex = 0u; blockIndex < blockCount; ++blockIndex) {
        device const uchar* b = rowWeights + blockIndex * 84u;
        const float d = float(as_type<half>(ushort(ushort(b[80]) | (ushort(b[81]) << 8u))));
        const float dmin = float(as_type<half>(ushort(ushort(b[82]) | (ushort(b[83]) << 8u))));
        const uchar scale = b[scIndex];
        const float dl = d * float(scale & 0x0Fu);
        const float ml = dmin * float(scale >> 4u);
        const ulong w0 = *(device const ulong*)(b + 16u + qBase);
        const ulong w1 = *(device const ulong*)(b + 16u + qBase + 8u);
        const uint xBase = blockIndex * 256u + laneInRow * 16u;
        float qx = 0.0f;
        float sx = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            const float xv = x[xBase + i];
            qx = fma(float((uint(w0 >> (8u * i)) & 0xFFu) >> shift & 3u), xv, qx);
            sx += xv;
        }
        for (uint i = 0u; i < 8u; ++i) {
            const float xv = x[xBase + 8u + i];
            qx = fma(float((uint(w1 >> (8u * i)) & 0xFFu) >> shift & 3u), xv, qx);
            sx += xv;
        }
        sum += dl * qx - ml * sx;
    }
    sum = reduce16(sum);
    if (laneInRow == 0u) y[row] = sum;
}

kernel void q4k_gemv_multi2(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint laneInRow = lane % 16u;
    const uint row = groupIndex * 16u + sg * 2u + lane / 16u;
    if (row >= rows) return;
    const uint blockCount = columns / 256u;
    device const uchar* rowWeights = weights + row * blockCount * 144u;
    const uint g = laneInRow / 2u;
    const uint sub = laneInRow % 2u;
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
        const float ds = d * float(s);
        const float dm = dmin * float(m);
        const bool low = (g & 1u) == 0u;
        const device uchar* q0 = b + 16u + (g / 2u) * 32u + sub * 16u;
        const ulong w0 = *(device const ulong*)q0;
        const ulong w1 = *(device const ulong*)(q0 + 8u);
        const uint xBase = blockIndex * 256u + laneInRow * 16u;
        float qx = 0.0f;
        float sx = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            const float xv = x[xBase + i];
            const uint byte = uint((w0 >> (8u * i)) & 0xFFu);
            const uint nibble = low ? (byte & 0x0Fu) : (byte >> 4u);
            qx = fma(float(nibble), xv, qx);
            sx += xv;
        }
        for (uint i = 0u; i < 8u; ++i) {
            const float xv = x[xBase + 8u + i];
            const uint byte = uint((w1 >> (8u * i)) & 0xFFu);
            const uint nibble = low ? (byte & 0x0Fu) : (byte >> 4u);
            qx = fma(float(nibble), xv, qx);
            sx += xv;
        }
        sum += ds * qx - dm * sx;
    }
    sum = reduce16(sum);
    if (laneInRow == 0u) y[row] = sum;
}
