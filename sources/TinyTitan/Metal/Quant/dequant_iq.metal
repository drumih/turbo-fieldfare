#include <metal_stdlib>
using namespace metal;

// IQ4_NL dequantisation to bf16: the GGUF block is one fp16 scale followed by
// 16 bytes of nibbles (32 values). The table is `kvalues_iq4nl` from
// ggml-common.h, the same values the CPU decoder in GGUFDequant uses, and low
// nibbles are elements 0...15, high nibbles 16...31.
//
// One thread per block: the scale is read once and the 32 outputs are written
// in order, which keeps this out of the integer div/mod path that made the
// element-per-thread version 14x slower than the hardware allows.
constant constexpr int8_t kIQ4NLTable[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

kernel void dequant_iq4nl_bf16(
    device const uchar*  blocks     [[buffer(0)]],
    device bfloat*       out        [[buffer(1)]],
    constant uint&       blockCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    const uint base = gid * 18u;
    const ushort dBits = ushort(blocks[base]) | (ushort(blocks[base + 1u]) << 8u);
    const float d = float(as_type<half>(dBits));
    device bfloat* dst = out + gid * 32u;
    for (uint j = 0u; j < 16u; ++j) {
        const uchar packed = blocks[base + 2u + j];
        dst[j] = bfloat(d * float(kIQ4NLTable[packed & 0x0Fu]));
        dst[j + 16u] = bfloat(d * float(kIQ4NLTable[packed >> 4u]));
    }
}

// IQ3_XXS: fp16 d + 64 grid-index bytes + 32 scale/sign bytes (4 words per
// 32-element sub-block). The grid (256 x uint32) and the 128-byte sign table
// come from ggml-common.h and are bound as buffers, so the shader carries no
// copy of them.
kernel void dequant_iq3xxs_bf16(
    device const uchar*  blocks     [[buffer(0)]],
    device const uint*   grid       [[buffer(1)]],
    device const uchar*  signsTable [[buffer(2)]],
    device bfloat*       out        [[buffer(3)]],
    constant uint&       blockCount [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    const uint base = gid * 98u;
    const ushort dBits = ushort(blocks[base]) | (ushort(blocks[base + 1u]) << 8u);
    const float d = float(as_type<half>(dBits));
    device const uchar* qs = blocks + base + 2u;
    device bfloat* dst = out + gid * 256u;
    for (uint sub = 0u; sub < 8u; ++sub) {
        const uint aux = uint(qs[64u + 4u * sub])
            | (uint(qs[65u + 4u * sub]) << 8u)
            | (uint(qs[66u + 4u * sub]) << 16u)
            | (uint(qs[67u + 4u * sub]) << 24u);
        const float db = d * (0.5f + float(aux >> 28u)) * 0.5f;
        for (uint l = 0u; l < 4u; ++l) {
            const uchar signs = signsTable[(aux >> (7u * l)) & 127u];
            const uint g1 = grid[qs[8u * sub + 2u * l]];
            const uint g2 = grid[qs[8u * sub + 2u * l + 1u]];
            device bfloat* group = dst + sub * 32u + l * 8u;
            for (uint j = 0u; j < 4u; ++j) {
                const float v1 = float((g1 >> (8u * j)) & 0xFFu);
                const float v2 = float((g2 >> (8u * j)) & 0xFFu);
                group[j] = bfloat(db * v1 * ((signs & (1u << j)) ? -1.0f : 1.0f));
                group[j + 4u] = bfloat(db * v2 * ((signs & (1u << (j + 4u))) ? -1.0f : 1.0f));
            }
        }
    }
}

// IQ2_S: fp16 d + 32 grid indices + 32 sign bytes + 8 high-bit bytes + 8 nibble
// scales. The 1024-entry 8-byte grid comes from ggml-common.h and is bound as a
// buffer. Per 32-element sub-block: two scales (low/high nibble) and four
// 8-value grid entries whose top 2 index bits come from `qh`.
kernel void dequant_iq2s_bf16(
    device const uchar*  blocks     [[buffer(0)]],
    device const ulong*  grid       [[buffer(1)]],
    device bfloat*       out        [[buffer(2)]],
    constant uint&       blockCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    const uint base = gid * 82u;
    const ushort dBits = ushort(blocks[base]) | (ushort(blocks[base + 1u]) << 8u);
    const float d = float(as_type<half>(dBits));
    device const uchar* qs = blocks + base + 2u;       // 64 bytes: indices, signs
    device const uchar* qh = blocks + base + 66u;      // 8 bytes
    device const uchar* scales = blocks + base + 74u;  // 8 bytes
    device bfloat* dst = out + gid * 256u;
    for (uint sub = 0u; sub < 8u; ++sub) {
        const uchar sc = scales[sub];
        const float db0 = d * (0.5f + float(sc & 0x0Fu)) * 0.25f;
        const float db1 = d * (0.5f + float(sc >> 4u)) * 0.25f;
        for (uint l = 0u; l < 4u; ++l) {
            const float dl = l < 2u ? db0 : db1;
            const uint index = uint(qs[4u * sub + l])
                | ((uint(qh[sub]) << (8u - 2u * l)) & 0x300u);
            const uchar signs = qs[32u + 4u * sub + l];
            const ulong entry = grid[index];
            device bfloat* group = dst + sub * 32u + l * 8u;
            for (uint j = 0u; j < 8u; ++j) {
                const float v = float((entry >> (8u * j)) & 0xFFu);
                group[j] = bfloat(dl * v * ((signs & (1u << j)) ? -1.0f : 1.0f));
            }
        }
    }
}
