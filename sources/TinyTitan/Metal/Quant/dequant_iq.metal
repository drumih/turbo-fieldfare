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

// IQ4_XS: fp16 d, 16-bit high scales, 4 low-scale bytes, 128 nibble bytes.
// Per 32-element sub-block: a 6-bit scale (ls - 32) and 16 bytes whose low
// nibbles are elements 0...15 and high nibbles 16...31.
kernel void dequant_iq4xs_bf16(
    device const uchar* blocks     [[buffer(0)]],
    device bfloat*      out        [[buffer(1)]],
    constant uint&      blockCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    const uint base = gid * 136u;
    const ushort dBits = ushort(blocks[base]) | (ushort(blocks[base + 1u]) << 8u);
    const float d = float(as_type<half>(dBits));
    const ushort scalesH = ushort(blocks[base + 2u]) | (ushort(blocks[base + 3u]) << 8u);
    device const uchar* scalesL = blocks + base + 4u;
    device const uchar* qs = blocks + base + 8u;
    device bfloat* dst = out + gid * 256u;
    for (uint sub = 0u; sub < 8u; ++sub) {
        const uint low = (uint(scalesL[sub / 2u]) >> (4u * (sub % 2u))) & 0x0Fu;
        const uint high = (uint(scalesH) >> (2u * sub)) & 0x03u;
        const float dl = d * float(int(low | (high << 4u)) - 32);
        for (uint j = 0u; j < 16u; ++j) {
            const uchar packed = qs[sub * 16u + j];
            dst[sub * 32u + j] = bfloat(dl * float(kIQ4NLTable[packed & 0x0Fu]));
            dst[sub * 32u + 16u + j] = bfloat(dl * float(kIQ4NLTable[packed >> 4u]));
        }
    }
}

// Q2_0: fp16 d + 16 bytes of 2-bit quants, 64 values. The code is offset by
// one (00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2), as block_q2_0 defines it.
kernel void dequant_q2_0_bf16(
    device const uchar* blocks     [[buffer(0)]],
    device bfloat*      out        [[buffer(1)]],
    constant uint&      blockCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    const uint base = gid * 18u;
    const ushort dBits = ushort(blocks[base]) | (ushort(blocks[base + 1u]) << 8u);
    const float d = float(as_type<half>(dBits));
    device bfloat* dst = out + gid * 64u;
    for (uint j = 0u; j < 64u; ++j) {
        const uchar packed = blocks[base + 2u + (j / 4u)];
        const uint q = (uint(packed) >> (2u * (j % 4u))) & 0x03u;
        dst[j] = bfloat(float(int(q) - 1) * d);
    }
}

// Q6_K: 128 low-nibble bytes, 64 high-bit bytes, 16 int8 sub-block scales, and
// an fp16 super-scale. The four values one loop step produces land 32 apart.
kernel void dequant_q6k_bf16(
    device const uchar* blocks     [[buffer(0)]],
    device bfloat*      out        [[buffer(1)]],
    constant uint&      blockCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    const uint base = gid * 210u;
    const ushort dBits = ushort(blocks[base + 208u]) | (ushort(blocks[base + 209u]) << 8u);
    const float d = float(as_type<half>(dBits));
    device const uchar* ql = blocks + base;
    device const uchar* qh = blocks + base + 128u;
    device const uchar* sc = blocks + base + 192u;
    device bfloat* dst = out + gid * 256u;
    for (uint halfIndex = 0u; halfIndex < 2u; ++halfIndex) {
        device const uchar* qlH = ql + halfIndex * 64u;
        device const uchar* qhH = qh + halfIndex * 32u;
        device const uchar* scH = sc + halfIndex * 8u;
        device bfloat* halfOut = dst + halfIndex * 128u;
        for (uint l = 0u; l < 32u; ++l) {
            const uint is = l / 16u;
            const int q1 = int((qlH[l] & 0x0Fu) | ((qhH[l] & 0x03u) << 4u)) - 32;
            const int q2 = int((qlH[l + 32u] & 0x0Fu) | (((qhH[l] >> 2u) & 0x03u) << 4u)) - 32;
            const int q3 = int((qlH[l] >> 4u) | (((qhH[l] >> 4u) & 0x03u) << 4u)) - 32;
            const int q4 = int((qlH[l + 32u] >> 4u) | (((qhH[l] >> 6u) & 0x03u) << 4u)) - 32;
            const float s0 = float(as_type<char>(scH[is]));
            const float s1 = float(as_type<char>(scH[is + 2u]));
            const float s2 = float(as_type<char>(scH[is + 4u]));
            const float s3 = float(as_type<char>(scH[is + 6u]));
            halfOut[l] = bfloat(d * s0 * float(q1));
            halfOut[l + 32u] = bfloat(d * s1 * float(q2));
            halfOut[l + 64u] = bfloat(d * s2 * float(q3));
            halfOut[l + 96u] = bfloat(d * s3 * float(q4));
        }
    }
}

// Q8_0: fp16 d + 32 signed bytes.
kernel void dequant_q8_0_bf16(
    device const uchar* blocks     [[buffer(0)]],
    device bfloat*      out        [[buffer(1)]],
    constant uint&      blockCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= blockCount) return;
    const uint base = gid * 34u;
    const ushort dBits = ushort(blocks[base]) | (ushort(blocks[base + 1u]) << 8u);
    const float d = float(as_type<half>(dBits));
    device bfloat* dst = out + gid * 32u;
    for (uint j = 0u; j < 32u; ++j) {
        dst[j] = bfloat(float(as_type<char>(blocks[base + 2u + j])) * d);
    }
}

// IQ2_S GEMV: y[m] = sum_n W[m,n] * x[n], with W in IQ2_S blocks and both
// vectors in half. One SIMD per row; lane L owns elements L*8...L*8+7 of every
// 256-value block, which is exactly one 8-value grid entry (sub = L/4, entry
// l = L%4), so no cross-lane weight exchange is needed - only the final
// simd_sum.
kernel void iq2s_gemv_simd(
    device const uchar*  W     [[buffer(0)]],
    device const ulong*  grid  [[buffer(1)]],
    device const half*   x     [[buffer(2)]],
    device half*         y     [[buffer(3)]],
    constant uint&       M     [[buffer(4)]],
    constant uint&       N     [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8u;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint blocks = N / 256u;
    const uint rowBytes = blocks * 82u;
    device const uchar* W_row = W + uint(row) * rowBytes;
    const uint sub = lane / 4u;
    const uint l = lane % 4u;
    const uint elementBase = lane * 8u;
    float acc = 0.0f;
    for (uint b = 0u; b < blocks; ++b) {
        device const uchar* block = W_row + b * 82u;
        const ushort dBits = ushort(block[0]) | (ushort(block[1]) << 8u);
        const float d = float(as_type<half>(dBits));
        const uchar sc = block[74u + sub];
        const float dl = d * (0.5f + float(l < 2u ? (sc & 0x0Fu) : (sc >> 4u))) * 0.25f;
        const uint index = uint(block[2u + 4u * sub + l])
            | ((uint(block[66u + sub]) << (8u - 2u * l)) & 0x300u);
        const uchar signs = block[2u + 32u + 4u * sub + l];
        const ulong entry = grid[index];
        device const half* xp = x + b * 256u + elementBase;
        for (uint j = 0u; j < 8u; ++j) {
            const float v = float((entry >> (8u * j)) & 0xFFu);
            const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
            acc = fma(dl * v * sign, float(xp[j]), acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}

// IQ4_NL GEMV: y[m] = sum_n W[m,n] * x[n], W in IQ4_NL blocks (32 values, 18
// bytes). One SIMD per row; lane L owns value L of every block (byte L/2, low
// nibble for even lanes, high for odd), so the 32 lanes cover one block.
kernel void iq4nl_gemv_simd(
    device const uchar* W     [[buffer(0)]],
    device const half*  x     [[buffer(1)]],
    device half*        y     [[buffer(2)]],
    constant uint&      M     [[buffer(3)]],
    constant uint&      N     [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8u;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint blocks = N / 32u;
    device const uchar* W_row = W + uint(row) * blocks * 18u;
    float acc = 0.0f;
    for (uint b = 0u; b < blocks; ++b) {
        device const uchar* block = W_row + b * 18u;
        const ushort dBits = ushort(block[0]) | (ushort(block[1]) << 8u);
        const float d = float(as_type<half>(dBits));
        // ggml stores low nibbles as values 0...15 and high nibbles as 16...31.
        const uchar packed = block[2u + (lane & 15u)];
        const uint q = lane < 16u ? uint(packed & 0x0Fu) : uint(packed >> 4u);
        acc = fma(d * float(kIQ4NLTable[q]), float(x[b * 32u + lane]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}

// IQ3_XXS GEMV: lane L owns the 8 values of one (sub, l) pair per 256-value
// block (sub = L/4, l = L%4): 4 from grid1 then 4 from grid2.
kernel void iq3xxs_gemv_simd(
    device const uchar*  W     [[buffer(0)]],
    device const uint*   grid  [[buffer(1)]],
    device const uchar*  signs [[buffer(2)]],
    device const half*   x     [[buffer(3)]],
    device half*         y     [[buffer(4)]],
    constant uint&       M     [[buffer(5)]],
    constant uint&       N     [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8u;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint blocks = N / 256u;
    device const uchar* W_row = W + uint(row) * blocks * 98u;
    const uint sub = lane / 4u;
    const uint l = lane % 4u;
    float acc = 0.0f;
    for (uint b = 0u; b < blocks; ++b) {
        device const uchar* block = W_row + b * 98u;
        const ushort dBits = ushort(block[0]) | (ushort(block[1]) << 8u);
        const float d = float(as_type<half>(dBits));
        device const uchar* qs = block + 2u;
        const uint aux = uint(qs[64u + 4u * sub]) | (uint(qs[65u + 4u * sub]) << 8u)
            | (uint(qs[66u + 4u * sub]) << 16u) | (uint(qs[67u + 4u * sub]) << 24u);
        const float db = d * (0.5f + float(aux >> 28u)) * 0.5f;
        const uchar signByte = signs[(aux >> (7u * l)) & 127u];
        const uint g1 = grid[qs[8u * sub + 2u * l]];
        const uint g2 = grid[qs[8u * sub + 2u * l + 1u]];
        device const half* xp = x + b * 256u + sub * 32u + l * 8u;
        for (uint j = 0u; j < 4u; ++j) {
            const float s1 = (signByte & (1u << j)) ? -1.0f : 1.0f;
            const float s2 = (signByte & (1u << (j + 4u))) ? -1.0f : 1.0f;
            const float v1 = float((g1 >> (8u * j)) & 0xFFu);
            const float v2 = float((g2 >> (8u * j)) & 0xFFu);
            acc = fma(db * v1 * s1, float(xp[j]), acc);
            acc = fma(db * v2 * s2, float(xp[j + 4u]), acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}

// IQ4_XS GEMV: lane L owns 8 values of one 32-value sub-block (sub = L/4,
// l = L%4): l 0/1 are the low-nibble halves, l 2/3 the high-nibble halves.
kernel void iq4xs_gemv_simd(
    device const uchar*  W     [[buffer(0)]],
    device const half*   x     [[buffer(1)]],
    device half*         y     [[buffer(2)]],
    constant uint&       M     [[buffer(3)]],
    constant uint&       N     [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8u;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint blocks = N / 256u;
    device const uchar* W_row = W + uint(row) * blocks * 136u;
    const uint sub = lane / 4u;
    const uint l = lane % 4u;
    const bool highHalf = l >= 2u;
    const uint byteBase = (l & 1u) * 8u;
    float acc = 0.0f;
    for (uint b = 0u; b < blocks; ++b) {
        device const uchar* block = W_row + b * 136u;
        const ushort dBits = ushort(block[0]) | (ushort(block[1]) << 8u);
        const float d = float(as_type<half>(dBits));
        const ushort scalesH = ushort(block[2]) | (ushort(block[3]) << 8u);
        const uint low = (uint(block[4u + sub / 2u]) >> (4u * (sub % 2u))) & 0x0Fu;
        const uint high = (uint(scalesH) >> (2u * sub)) & 0x03u;
        const float dl = d * float(int(low | (high << 4u)) - 32);
        device const uchar* q = block + 8u + sub * 16u + byteBase;
        device const half* xp = x + b * 256u + sub * 32u + l * 8u;
        for (uint j = 0u; j < 8u; ++j) {
            const uchar packed = q[j];
            const uint nib = highHalf ? uint(packed >> 4u) : uint(packed & 0x0Fu);
            acc = fma(dl * float(kIQ4NLTable[nib]), float(xp[j]), acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}

// Q2_0 GEMV: lane L owns values 2L and 2L+1 of every 64-value block; both are
// in byte L/2, at bit offsets (L%2)*4 and (L%2)*4+2.
kernel void q2_0_gemv_simd(
    device const uchar*  W     [[buffer(0)]],
    device const half*   x     [[buffer(1)]],
    device half*         y     [[buffer(2)]],
    constant uint&       M     [[buffer(3)]],
    constant uint&       N     [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8u;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint blocks = N / 64u;
    device const uchar* W_row = W + uint(row) * blocks * 18u;
    const uint shift = (lane & 1u) * 4u;
    float acc = 0.0f;
    for (uint b = 0u; b < blocks; ++b) {
        device const uchar* block = W_row + b * 18u;
        const ushort dBits = ushort(block[0]) | (ushort(block[1]) << 8u);
        const float d = float(as_type<half>(dBits));
        const uchar packed = block[2u + lane / 2u];
        const int q0 = int((uint(packed) >> shift) & 0x03u) - 1;
        const int q1 = int((uint(packed) >> (shift + 2u)) & 0x03u) - 1;
        device const half* xp = x + b * 64u + lane * 2u;
        acc = fma(float(q0) * d, float(xp[0]), acc);
        acc = fma(float(q1) * d, float(xp[1]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}
