#include <metal_stdlib>
using namespace metal;

// IQ2_S GEMM for prefill: one weight row and up to 128 prompt tokens per
// threadgroup, K in 256-element blocks. Lanes 0-31 decode one block into
// threadgroup memory (8 values each, the same mapping as the dequant kernel),
// then every thread dots its own token against it. The row is decoded once per
// 128-token tile, not once per token, which is what makes a prompt cheaper than
// decoding it token by token.
kernel void iq2s_gemm_bf16(
    device const uchar*  W    [[buffer(0)]],
    device const ulong*  grid [[buffer(1)]],
    device const half*   A    [[buffer(2)]],
    device bfloat*       Y    [[buffer(3)]],
    constant uint&       M    [[buffer(4)]],
    constant uint&       K    [[buffer(5)]],
    constant uint&       T    [[buffer(6)]],
    uint tgFlat [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    // A flat grid of (row, token tile) pairs: the two thread-index attributes
    // must share one vector width, so the 2D position is decoded here.
    const uint row = tgFlat % M;
    const uint tokenBlock = tgFlat / M;
    const uint blocks = K / 256u;
    threadgroup float wTile[256];
    device const uchar* rowBase = W + row * blocks * 82u;
    const uint tokenBase = tokenBlock * 128u;
    float acc = 0.0f;
    for (uint b = 0u; b < blocks; ++b) {
        if (tid < 32u) {
            device const uchar* block = rowBase + b * 82u;
            const ushort dBits = ushort(block[0]) | (ushort(block[1]) << 8u);
            const float d = float(as_type<half>(dBits));
            const uint sub = tid / 4u;
            const uint l = tid % 4u;
            const uchar sc = block[74u + sub];
            const float db0 = d * (0.5f + float(sc & 0x0Fu)) * 0.25f;
            const float db1 = d * (0.5f + float(sc >> 4u)) * 0.25f;
            const float dl = l < 2u ? db0 : db1;
            const uint index = uint(block[2u + 4u * sub + l])
                | ((uint(block[66u + sub]) << (8u - 2u * l)) & 0x300u);
            const uchar signs = block[2u + 32u + 4u * sub + l];
            const ulong entry = grid[index];
            for (uint j = 0u; j < 8u; ++j) {
                const float v = float((entry >> (8u * j)) & 0xFFu);
                wTile[tid * 8u + j] = dl * v * ((signs & (1u << j)) ? -1.0f : 1.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint token = tokenBase + tid;
        if (token < T) {
            device const half* a = A + token * K + b * 256u;
            for (uint j = 0u; j < 256u; ++j) {
                acc += float(a[j]) * wTile[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint token = tokenBase + tid;
    if (token < T) {
        Y[token * M + row] = bfloat(acc);
    }
}
