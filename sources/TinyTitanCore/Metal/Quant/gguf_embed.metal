#include <metal_stdlib>
using namespace metal;

// Token-embedding gather over Q6_K rows: one thread per requested row, 2560
// values = ten 210-byte blocks. The table stays in its blocks; only the rows a
// token asks for are decoded.
kernel void gguf_embed_gather_q6k_bf16(
    device const uchar* table [[buffer(0)]],
    device const uint*  rows  [[buffer(1)]],
    device bfloat*      out   [[buffer(2)]],
    constant uint&      count [[buffer(3)]],
    constant uint&      width [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const uint row = rows[gid];
    const uint blocks = width / 256u;
    device const uchar* base = table + uint(row) * blocks * 210u;
    device bfloat* dst = out + gid * width;
    for (uint b = 0u; b < blocks; ++b) {
        device const uchar* bp = base + b * 210u;
        const ushort dBits = ushort(bp[208]) | (ushort(bp[209]) << 8u);
        const float d = float(as_type<half>(dBits));
        device bfloat* block = dst + b * 256u;
        for (uint halfIndex = 0u; halfIndex < 2u; ++halfIndex) {
            device const uchar* ql = bp + halfIndex * 64u;
            device const uchar* qh = bp + 128u + halfIndex * 32u;
            device const uchar* sc = bp + 192u + halfIndex * 8u;
            device bfloat* halfOut = block + halfIndex * 128u;
            for (uint l = 0u; l < 32u; ++l) {
                const uint is = l / 16u;
                const int q1 = int((ql[l] & 0x0Fu) | ((qh[l] & 0x03u) << 4u)) - 32;
                const int q2 = int((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2u) & 0x03u) << 4u)) - 32;
                const int q3 = int((ql[l] >> 4u) | (((qh[l] >> 4u) & 0x03u) << 4u)) - 32;
                const int q4 = int((ql[l + 32u] >> 4u) | (((qh[l] >> 6u) & 0x03u) << 4u)) - 32;
                halfOut[l] = bfloat(d * float(as_type<char>(sc[is])) * float(q1));
                halfOut[l + 32u] = bfloat(d * float(as_type<char>(sc[is + 2u])) * float(q2));
                halfOut[l + 64u] = bfloat(d * float(as_type<char>(sc[is + 4u])) * float(q3));
                halfOut[l + 96u] = bfloat(d * float(as_type<char>(sc[is + 6u])) * float(q4));
            }
        }
    }
}
