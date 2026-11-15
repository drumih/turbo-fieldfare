#include <metal_stdlib>
using namespace metal;

// PLE gather over the checkpoint's IQ4_NL table: one thread per requested row,
// 160 values wide = five 18-byte blocks. The table stays in its blocks on disk
// and in memory; only the rows a token asks for are decoded.
//
// `kIQ4NLTable` comes from dequant_iq.metal: MetalContext compiles all modules
// into one library, so the table has a single definition.
kernel void ple_gather_iq4nl_bf16(
    device const uchar*  table  [[buffer(0)]],
    device const uint*   rows   [[buffer(1)]],
    device bfloat*       out    [[buffer(2)]],
    constant uint&       count  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const uint row = rows[gid];
    const uint rowBytes = (160u / 32u) * 18u;
    device const uchar* base = table + uint(row) * rowBytes;
    device bfloat* dst = out + gid * 160u;
    for (uint b = 0u; b < 5u; ++b) {
        device const uchar* bp = base + b * 18u;
        const ushort dBits = ushort(bp[0]) | (ushort(bp[1]) << 8u);
        const float d = float(as_type<half>(dBits));
        device const uchar* qs = bp + 2u;
        device bfloat* block = dst + b * 32u;
        for (uint j = 0u; j < 16u; ++j) {
            const uchar packed = qs[j];
            block[j] = bfloat(d * float(kIQ4NLTable[packed & 0x0Fu]));
            block[j + 16u] = bfloat(d * float(kIQ4NLTable[packed >> 4u]));
        }
    }
}
