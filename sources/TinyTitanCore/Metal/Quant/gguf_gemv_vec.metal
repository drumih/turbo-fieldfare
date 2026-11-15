#include <metal_stdlib>
using namespace metal;

// Vectorised GEMVs for the two GGUF formats the as-is decoder's resident
// matrices use: Q6_K (GDN and attention projections, the head) and Q8_0
// (hyper-connection projections, shared experts). The per-element kernels in
// dequant_iq.metal read one byte at a time behind a four-way branch and were
// measured latency-bound (~1.5 GB/s of weight reads); these read 4-byte packed
// vectors and have no data-dependent branch. Blocks are 210 and 34 bytes, so a
// block start is only 2-byte aligned: every weight load is a packed type.
//
// Both keep the layout of y = W x with W row-major in GGUF blocks, x and y in
// half, one SIMD group per row and 8 rows per threadgroup (256 threads), so
// they drop into the existing wrappers' dispatch shape.

// Q6_K: 256 values per block as two halves of 128. Within a half, position l
// (0...31) carries four values 32 apart:
//   v0 = ql[l]      & 0xF | (qh[l] >> 0 & 3) << 4   (group 0)
//   v1 = ql[l + 32] & 0xF | (qh[l] >> 2 & 3) << 4   (group 1)
//   v2 = ql[l]      >> 4  | (qh[l] >> 4 & 3) << 4   (group 2)
//   v3 = ql[l + 32] >> 4  | (qh[l] >> 6 & 3) << 4   (group 3)
// each minus 32, scaled by d * scales[l / 16 + 2 * group] (the 16-value
// sub-block of that group). A work item is 4 consecutive l of one half: three
// 4-byte loads give 16 weights. 16 items cover a block, so a SIMD group of 32
// covers two blocks per step.
kernel void q6k_gemv_vec(
    device const uchar* W  [[buffer(0)]],
    device const half*  x  [[buffer(1)]],
    device half*        y  [[buffer(2)]],
    constant uint&      M  [[buffer(3)]],
    constant uint&      N  [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * 8u + sg_idx;
    if (row >= M) return;
    const uint blocks = N / 256u;
    device const uchar* W_row = W + row * blocks * 210u;
    const uint item = lane % 16u;
    const uint halfIndex = item / 8u;
    const uint l0 = (item % 8u) * 4u;
    const uint scaleBase = halfIndex * 8u + l0 / 16u;
    float acc = 0.0f;
    for (uint b = lane / 16u; b < blocks; b += 2u) {
        device const uchar* bp = W_row + b * 210u;
        const float d = float(as_type<half>(ushort(ushort(bp[208]) | (ushort(bp[209]) << 8u))));
        device const uchar* ql = bp + halfIndex * 64u + l0;
        device const uchar* qh = bp + 128u + halfIndex * 32u + l0;
        const uint4 lo = uint4(uchar4(*(device const packed_uchar4*)ql));
        const uint4 hi = uint4(uchar4(*(device const packed_uchar4*)(ql + 32u)));
        const uint4 hb = uint4(uchar4(*(device const packed_uchar4*)qh));
        device const char* sc = (device const char*)(bp + 192u);
        device const half* xp = x + b * 256u + halfIndex * 128u + l0;
        const float4 x0 = float4(*(device const half4*)(xp));
        const float4 x1 = float4(*(device const half4*)(xp + 32u));
        const float4 x2 = float4(*(device const half4*)(xp + 64u));
        const float4 x3 = float4(*(device const half4*)(xp + 96u));
        const float4 q0 = float4(int4(lo & 0x0Fu) | (int4((hb >> 0u) & 3u) << 4)) - 32.0f;
        const float4 q1 = float4(int4(hi & 0x0Fu) | (int4((hb >> 2u) & 3u) << 4)) - 32.0f;
        const float4 q2 = float4(int4(lo >> 4u) | (int4((hb >> 4u) & 3u) << 4)) - 32.0f;
        const float4 q3 = float4(int4(hi >> 4u) | (int4((hb >> 6u) & 3u) << 4)) - 32.0f;
        acc += d * (float(sc[scaleBase + 0u]) * dot(q0, x0)
                  + float(sc[scaleBase + 2u]) * dot(q1, x1)
                  + float(sc[scaleBase + 4u]) * dot(q2, x2)
                  + float(sc[scaleBase + 6u]) * dot(q3, x3));
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}

// Q8_0: 32 values per block (fp16 d, 32 int8). A work item is 8 consecutive
// values of one block, two 4-byte loads; 4 items cover a block, so a SIMD group
// covers 8 blocks (256 values) per step.
kernel void q8_0_gemv_vec(
    device const uchar* W  [[buffer(0)]],
    device const half*  x  [[buffer(1)]],
    device half*        y  [[buffer(2)]],
    constant uint&      M  [[buffer(3)]],
    constant uint&      N  [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * 8u + sg_idx;
    if (row >= M) return;
    const uint blocks = N / 32u;
    device const uchar* W_row = W + row * blocks * 34u;
    const uint part = lane % 4u;
    float acc = 0.0f;
    for (uint b = lane / 4u; b < blocks; b += 8u) {
        device const uchar* bp = W_row + b * 34u;
        const float d = float(as_type<half>(ushort(ushort(bp[0]) | (ushort(bp[1]) << 8u))));
        device const char* qs = (device const char*)(bp + 2u + part * 8u);
        const float4 q0 = float4(char4(*(device const packed_char4*)qs));
        const float4 q1 = float4(char4(*(device const packed_char4*)(qs + 4)));
        device const half* xp = x + b * 32u + part * 8u;
        const float4 x0 = float4(*(device const half4*)(xp));
        const float4 x1 = float4(*(device const half4*)(xp + 4u));
        acc += d * (dot(q0, x0) + dot(q1, x1));
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}
