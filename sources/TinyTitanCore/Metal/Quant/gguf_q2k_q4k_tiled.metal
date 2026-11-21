#include <metal_stdlib>
using namespace metal;

// Row-tiled Q2_K / Q4_K GEMV, the geometry of llama.cpp's
// kernel_mul_mv_q2_K_f32 / kernel_mul_mv_q4_K_f32 (ggml-metal mul_mv.metal).
//
// One SIMD group computes KQ_TILE_ROWS_* consecutive output rows with all 32
// lanes. The lanes split into four groups of eight (`blockLane`), each walking
// every fourth 256-value block, so four blocks of a row are in flight at once.
// A lane loads its 32 activations of a block once and
// reuses them for every row of the tile; that reuse is what the one-row kernel
// in gguf_q2k_q4k_gemv.metal cannot do. Quants are read as aligned 16-bit
// words and masked in place, with the shifted masks folded into constant
// per-plane factors (1, 1/4, 1/16, 1/64 and 1/256) instead of shifting each
// value. Rows past the end are computed on a clamped row and never stored.
//
// Block layouts (bytes):
//   Q2_K  84: scales[0..15], qs[16..79], d @80, dmin @82
//   Q4_K 144: d @0, dmin @2, scales[4..15], qs[16..143]

constant uint KQ_TILE_SIMDGROUPS = 2u;
constant uint KQ_TILE_ROWS_Q2K = 4u;
constant uint KQ_TILE_ROWS_Q4K = 2u;

static inline float kq_tile_half(device const uchar* p) {
    return float(as_type<half>(*(device const ushort*)p));
}

kernel void q2k_gemv_tiled(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint firstRow = (groupIndex * KQ_TILE_SIMDGROUPS + sg) * KQ_TILE_ROWS_Q2K;
    if (firstRow >= rows) return;
    const uint blockCount = columns / 256u;
    const uint rowBytes = blockCount * 84u;

    const uint blockLane = lane / 8u;      // 0...3: which block of four
    const uint laneInBlock = lane % 8u;
    const uint halfIndex = laneInBlock / 4u; // 0/1: first or second 128 values
    const uint quarter = laneInBlock % 4u;   // 0...3: 8-value slice of 32
    const uint scaleOffset = 8u * halfIndex + (8u * quarter) / 16u;
    const uint quantWord = 16u * halfIndex + 4u * quarter;

    device const uchar* rowBase[KQ_TILE_ROWS_Q2K];
    for (uint r = 0u; r < KQ_TILE_ROWS_Q2K; ++r) {
        rowBase[r] = weights + min(firstRow + r, rows - 1u) * rowBytes;
    }

    float yl[32];
    float sums[KQ_TILE_ROWS_Q2K] = {0.0f};
    device const float* xs = x + blockLane * 256u + 128u * halfIndex + 8u * quarter;

    for (uint blockIndex = blockLane; blockIndex < blockCount; blockIndex += 4u) {
        float4 sumx = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            yl[i + 0u] = xs[i + 0u];   sumx[0] += yl[i + 0u];
            yl[i + 8u] = xs[i + 32u];  sumx[1] += yl[i + 8u];
            yl[i + 16u] = xs[i + 64u]; sumx[2] += yl[i + 16u];
            yl[i + 24u] = xs[i + 96u]; sumx[3] += yl[i + 24u];
        }
        for (uint r = 0u; r < KQ_TILE_ROWS_Q2K; ++r) {
            device const uchar* b = rowBase[r] + blockIndex * 84u;
            device const uchar* sc = b + scaleOffset;
            device const ushort* qs = (device const ushort*)(b + 16u) + quantWord;
            float4 acc1 = 0.0f;
            float4 acc2 = 0.0f;
            for (uint i = 0u; i < 8u; i += 2u) {
                const ushort q = qs[i / 2u];
                acc1[0] += yl[i + 0u] * float(q & 0x0003u);
                acc2[0] += yl[i + 1u] * float(q & 0x0300u);
                acc1[1] += yl[i + 8u] * float(q & 0x000Cu);
                acc2[1] += yl[i + 9u] * float(q & 0x0C00u);
                acc1[2] += yl[i + 16u] * float(q & 0x0030u);
                acc2[2] += yl[i + 17u] * float(q & 0x3000u);
                acc1[3] += yl[i + 24u] * float(q & 0x00C0u);
                acc2[3] += yl[i + 25u] * float(q & 0xC000u);
            }
            const float d = kq_tile_half(b + 80u);
            const float dmin = kq_tile_half(b + 82u) * (1.0f / 16.0f);
            sums[r] += d * ((acc1[0] + (1.0f / 256.0f) * acc2[0]) * float(sc[0] & 0x0Fu)
                            + (acc1[1] + (1.0f / 256.0f) * acc2[1]) * float(sc[2] & 0x0Fu) * (1.0f / 4.0f)
                            + (acc1[2] + (1.0f / 256.0f) * acc2[2]) * float(sc[4] & 0x0Fu) * (1.0f / 16.0f)
                            + (acc1[3] + (1.0f / 256.0f) * acc2[3]) * float(sc[6] & 0x0Fu) * (1.0f / 64.0f))
                - dmin * (sumx[0] * float(sc[0] & 0xF0u) + sumx[1] * float(sc[2] & 0xF0u)
                          + sumx[2] * float(sc[4] & 0xF0u) + sumx[3] * float(sc[6] & 0xF0u));
        }
        xs += 4u * 256u;
    }

    for (uint r = 0u; r < KQ_TILE_ROWS_Q2K; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0u && firstRow + r < rows) y[firstRow + r] = total;
    }
}

kernel void q4k_gemv_tiled(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint firstRow = (groupIndex * KQ_TILE_SIMDGROUPS + sg) * KQ_TILE_ROWS_Q4K;
    if (firstRow >= rows) return;
    const uint blockCount = columns / 256u;
    const uint rowBytes = blockCount * 144u;

    const uint blockLane = lane / 8u;
    const uint laneInBlock = lane % 8u;
    const uint halfIndex = laneInBlock / 4u;
    const uint quarter = laneInBlock % 4u;

    device const uchar* rowBase[KQ_TILE_ROWS_Q4K];
    for (uint r = 0u; r < KQ_TILE_ROWS_Q4K; ++r) {
        rowBase[r] = weights + min(firstRow + r, rows - 1u) * rowBytes;
    }

    float yl[16];
    float yh[16];
    float sums[KQ_TILE_ROWS_Q4K] = {0.0f};
    device const float* xs = x + blockLane * 256u + 64u * halfIndex + 8u * quarter;

    for (uint blockIndex = blockLane; blockIndex < blockCount; blockIndex += 4u) {
        float4 sumx = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            yl[i + 0u] = xs[i + 0u];   sumx[0] += yl[i + 0u];
            yl[i + 8u] = xs[i + 32u];  sumx[1] += yl[i + 8u];
            yh[i + 0u] = xs[i + 128u]; sumx[2] += yh[i + 0u];
            yh[i + 8u] = xs[i + 160u]; sumx[3] += yh[i + 8u];
        }
        for (uint r = 0u; r < KQ_TILE_ROWS_Q4K; ++r) {
            device const uchar* b = rowBase[r] + blockIndex * 144u;
            device const ushort* sc = (device const ushort*)(b + 4u) + halfIndex;
            // Unpack the two 6-bit scale/min pairs this lane needs (sub-blocks
            // 2*halfIndex, 2*halfIndex+1 and the matching +4 pair).
            const ushort s0 = sc[0] & 0x3F3Fu;
            const ushort s1 = sc[2] & 0x3F3Fu;
            const ushort s2 = ((sc[4] >> 0u) & 0x0F0Fu) | ((sc[0] & 0xC0C0u) >> 2u);
            const ushort s3 = ((sc[4] >> 4u) & 0x0F0Fu) | ((sc[2] & 0xC0C0u) >> 2u);
            device const ushort* q1 = (device const ushort*)(b + 16u) + 16u * halfIndex + 4u * quarter;
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = 0.0f;
            float4 acc2 = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                const ushort lo = q1[i];
                const ushort hi = q2[i];
                acc1[0] += yl[2u * i + 0u] * float(lo & 0x000Fu);
                acc1[1] += yl[2u * i + 1u] * float(lo & 0x0F00u);
                acc1[2] += yl[2u * i + 8u] * float(lo & 0x00F0u);
                acc1[3] += yl[2u * i + 9u] * float(lo & 0xF000u);
                acc2[0] += yh[2u * i + 0u] * float(hi & 0x000Fu);
                acc2[1] += yh[2u * i + 1u] * float(hi & 0x0F00u);
                acc2[2] += yh[2u * i + 8u] * float(hi & 0x00F0u);
                acc2[3] += yh[2u * i + 9u] * float(hi & 0xF000u);
            }
            const float d = kq_tile_half(b + 0u);
            const float dmin = kq_tile_half(b + 2u);
            sums[r] += d * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(s0 & 0xFFu)
                            + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(s0 >> 8u) * (1.0f / 16.0f)
                            + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(s2 & 0xFFu)
                            + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(s2 >> 8u) * (1.0f / 16.0f))
                - dmin * (sumx[0] * float(s1 & 0xFFu) + sumx[1] * float(s1 >> 8u)
                          + sumx[2] * float(s3 & 0xFFu) + sumx[3] * float(s3 >> 8u));
        }
        xs += 4u * 256u;
    }

    for (uint r = 0u; r < KQ_TILE_ROWS_Q4K; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0u && firstRow + r < rows) y[firstRow + r] = total;
    }
}
