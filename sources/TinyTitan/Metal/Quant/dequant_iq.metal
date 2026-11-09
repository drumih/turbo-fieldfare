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
