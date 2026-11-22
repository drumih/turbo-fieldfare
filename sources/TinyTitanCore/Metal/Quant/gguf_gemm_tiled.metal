#include <metal_stdlib>
using namespace metal;

// Tiled GEMM over GGUF weights for prefill: Y[p][r] = scale * sum_c W[r][c] * X[x(p)][c].
//
// Geometry follows llama.cpp's kernel_mul_mm: a threadgroup computes 64 weight
// rows x 32 tokens with four SIMD groups (each 32 rows x 16 tokens, eight 8x8
// fp32 accumulators); every 64-wide K step dequantises the weight tile and the
// activations into half in threadgroup memory (vector stores) and multiplies
// them with simdgroup matrices. K must be a multiple of 64. The weight block is decoded once per 32-token tile, not
// once per token.
//
// Two modes share the kernel:
//   dense    grid.z = 1; tile tokens are rows tokenTile*32 + j of X and Y.
//   experts  grid.z = expert; the tile's tokens are entries of that expert's
//            pair list (`lists[e * listStride + i]`, `counts[e]` long); a pair p
//            reads X row p / xDivisor and writes Y row p, and the output is
//            multiplied by scales[e] when useScale is set.

struct GGUFGemmArgs {
    uint rows;          // M: weight rows (output width)
    uint columns;       // K
    uint tokens;        // dense: token count
    ulong expertStride; // experts: bytes between expert matrices
    uint listStride;    // experts: capacity of each expert's pair list
    uint xDivisor;      // experts: pair -> X row divisor (slots per token, or 1)
    uint indirect;      // 1: experts mode
    uint useScale;
};

constant uint GEMM_BM = 64u;
constant uint GEMM_BN = 32u;
constant uint GEMM_BK = 64u;

// Sixteen consecutive weights of row `row` starting at column `c` (c % 16 == 0).
struct GemmQ8_0 {
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread half* out) {
        device const uchar* b = w + row * rowBytes + (c / 32u) * 34u;
        const float d = float(as_type<half>(*(device const ushort*)b));
        device const char* qs = (device const char*)(b + 2u) + (c % 32u);
        for (uint j = 0u; j < 16u; ++j) out[j] = half(d * float(qs[j]));
    }
    static uint rowBytes(uint columns) { return columns / 32u * 34u; }
};

// E2M1 code (4 bits) as an exact fp16: magnitudes 2..7 are their bits shifted
// into the fp16 exponent/mantissa plus the exponent bias difference (14);
// 0 and 1 (0 and 0.5) are selected; bit 3 is the sign.
static inline half gemm_e2m1(uint code) {
    const uint magnitude = code & 7u;
    const uint bits = magnitude >= 2u ? (magnitude << 9u) + (14u << 10u)
        : (magnitude == 1u ? 0x3800u : 0u);
    return as_type<half>(ushort(bits | ((code & 8u) << 12u)));
}

struct GemmNVFP4 {
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread half* out) {
        // One 16-value sub-block: value j in the low nibble of byte j (j < 8)
        // and value j in the high nibble of byte j - 8. Blocks are 36 bytes,
        // so the two quant words are 4-byte aligned.
        device const uchar* b = w + row * rowBytes + (c / 64u) * 36u;
        const uint sub = (c % 64u) / 16u;
        const uchar code = b[sub];
        const half d = code == 0x7Fu
            ? 0.0h
            : as_type<half>(ushort(ushort(code & 0x7Fu) << 7u)) * 256.0h;
        device const uint* q = (device const uint*)(b + 4u + sub * 8u);
        const uint words[2] = {q[0], q[1]};
        for (uint i = 0u; i < 2u; ++i) {
            for (uint j = 0u; j < 4u; ++j) {
                const uint byte = (words[i] >> (8u * j)) & 0xFFu;
                out[4u * i + j] = gemm_e2m1(byte & 0x0Fu) * d;
                out[8u + 4u * i + j] = gemm_e2m1(byte >> 4u) * d;
            }
        }
    }
    static uint rowBytes(uint columns) { return columns / 64u * 36u; }
};

struct GemmF32 {
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread half* out) {
        device const float* r = (device const float*)(w + row * rowBytes) + c;
        for (uint j = 0u; j < 16u; ++j) out[j] = half(r[j]);
    }
    static uint rowBytes(uint columns) { return columns * 4u; }
};

struct GemmBF16 {
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread half* out) {
        device const ushort* r = (device const ushort*)(w + row * rowBytes) + c;
        for (uint j = 0u; j < 16u; ++j) out[j] = half(as_type<float>(uint(r[j]) << 16u));
    }
    static uint rowBytes(uint columns) { return columns * 2u; }
};

template <typename Decoder>
static void gguf_gemm_tiled_impl(
    device const uchar* weights, device const float* x, device float* y,
    device const uint* lists, device const uint* counts, device const float* scales,
    constant GGUFGemmArgs& args, uint3 tile, uint sg, uint tid,
    threadgroup half* sa, threadgroup half* sb, threadgroup float* sc)
{
    const uint rowBase = tile.x * GEMM_BM;
    const uint tokenBase = tile.y * GEMM_BN;
    const uint expert = args.indirect != 0u ? tile.z : 0u;
    const uint tokenCount = args.indirect != 0u ? counts[expert] : args.tokens;
    if (rowBase >= args.rows || tokenBase >= tokenCount) return;
    device const uchar* matrix = weights + ulong(expert) * args.expertStride;
    const uint rowBytes = Decoder::rowBytes(args.columns);

    // This thread's loads per K step: weight row (tid / 2), columns
    // (tid % 2) * 16 and that + 32; activation token (tid / 4), 16 columns at
    // (tid % 4) * 16.
    const uint loadRow = min(rowBase + tid / 2u, args.rows - 1u);
    const uint loadToken = tokenBase + tid / 4u;
    const bool tokenValid = loadToken < tokenCount;
    uint xRow = 0u;
    if (tokenValid) {
        xRow = args.indirect != 0u
            ? lists[expert * args.listStride + loadToken] / args.xDivisor
            : loadToken;
    }
    device const float4* xs = (device const float4*)(x + ulong(xRow) * args.columns + (tid % 4u) * 16u);
    threadgroup half4* sa4 = (threadgroup half4*)(sa + (tid / 2u) * GEMM_BK + (tid % 2u) * 16u);
    threadgroup half4* sb4 = (threadgroup half4*)(sb + (tid / 4u) * GEMM_BK + (tid % 4u) * 16u);

    simdgroup_float8x8 acc[8];
    for (uint i = 0u; i < 8u; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint sgRow = (sg % 2u) * 32u;
    const uint sgToken = (sg / 2u) * 16u;

    for (uint k = 0u; k < args.columns; k += GEMM_BK) {
        for (uint part = 0u; part < 2u; ++part) {
            half wv[16];
            Decoder::decode(matrix, rowBytes, loadRow, k + (tid % 2u) * 16u + part * 32u, wv);
            for (uint j = 0u; j < 4u; ++j) {
                sa4[part * 8u + j] = half4(wv[4u * j], wv[4u * j + 1u], wv[4u * j + 2u], wv[4u * j + 3u]);
            }
        }
        for (uint j = 0u; j < 4u; ++j) {
            sb4[j] = tokenValid ? half4(xs[k / 4u + j]) : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < GEMM_BK; kk += 8u) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            for (uint i = 0u; i < 4u; ++i) {
                simdgroup_load(ma[i], sa + (sgRow + 8u * i) * GEMM_BK + kk, GEMM_BK);
            }
            for (uint i = 0u; i < 2u; ++i) {
                // B is X^T: element (kk + a, token b) = sb[(token) * BK + kk + a].
                simdgroup_load(mb[i], sb + (sgToken + 8u * i) * GEMM_BK + kk, GEMM_BK, ulong2(0, 0), true);
            }
            for (uint i = 0u; i < 4u; ++i) {
                for (uint j = 0u; j < 2u; ++j) {
                    simdgroup_multiply_accumulate(acc[i * 2u + j], ma[i], mb[j], acc[i * 2u + j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage the 64 x 32 tile (row-major rows x tokens), then write valid cells.
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(
                acc[i * 2u + j], sc + (sgRow + 8u * i) * GEMM_BN + sgToken + 8u * j, GEMM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = args.useScale != 0u ? scales[expert] : 1.0f;
    for (uint cell = tid; cell < GEMM_BM * GEMM_BN; cell += 128u) {
        const uint r = cell / GEMM_BN;
        const uint t = cell % GEMM_BN;
        const uint row = rowBase + r;
        const uint token = tokenBase + t;
        if (row >= args.rows || token >= tokenCount) continue;
        const uint yRow = args.indirect != 0u ? lists[expert * args.listStride + token] : token;
        y[ulong(yRow) * args.rows + row] = sc[r * GEMM_BN + t] * scale;
    }
}

#define GGUF_GEMM_TILED_KERNEL(name, decoder)                                        \
kernel void name(                                                                    \
    device const uchar* weights [[buffer(0)]],                                       \
    device const float* x [[buffer(1)]],                                             \
    device float* y [[buffer(2)]],                                                   \
    device const uint* lists [[buffer(3)]],                                          \
    device const uint* counts [[buffer(4)]],                                         \
    device const float* scales [[buffer(5)]],                                        \
    constant GGUFGemmArgs& args [[buffer(6)]],                                       \
    uint3 tile [[threadgroup_position_in_grid]],                                     \
    uint sg [[simdgroup_index_in_threadgroup]],                                      \
    uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                    \
    threadgroup half sa[GEMM_BM * GEMM_BK];                                          \
    threadgroup half sb[GEMM_BN * GEMM_BK];                                          \
    threadgroup float sc[GEMM_BM * GEMM_BN];                                         \
    gguf_gemm_tiled_impl<decoder>(                                                   \
        weights, x, y, lists, counts, scales, args, tile, sg, tid, sa, sb, sc);      \
}

GGUF_GEMM_TILED_KERNEL(gemm_tiled_q8_0, GemmQ8_0)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_nvfp4, GemmNVFP4)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_f32, GemmF32)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_bf16, GemmBF16)

// Groups (token, slot) pairs by expert: lists[e][i] = token * slots + slot for
// every pair routed to e, counts[e] how many. One thread per pair; the order
// inside a list is not deterministic, which does not change any sum.
kernel void moe_group_pairs(
    device const int* ids [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    device uint* lists [[buffer(2)]],
    constant uint& pairs [[buffer(3)]],
    constant uint& listStride [[buffer(4)]],
    uint p [[thread_position_in_grid]])
{
    if (p >= pairs) return;
    const uint e = uint(ids[p]);
    const uint slot = atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    lists[e * listStride + slot] = p;
}

// counts[i] = 0 for i < n (the expert pair counts, before moe_group_pairs).
kernel void clear_counts(
    device uint* counts [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) counts[i] = 0u;
}

// Indirect dispatch sizes for the expert-grouped GEMMs: the token-tile count is
// the largest expert's pair count over 32, so no threadgroup is launched for a
// tile no expert has. Writes (rowTiles[i], tiles, experts) for each of `shapes`.
kernel void moe_dispatch_sizes(
    device const uint* counts [[buffer(0)]],
    device uint* sizes [[buffer(1)]],
    constant uint& experts [[buffer(2)]],
    constant uint4& rowTiles [[buffer(3)]],  // up to four GEMM shapes
    constant uint& shapes [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint largest = 0u;
    for (uint e = lane; e < experts; e += 32u) largest = max(largest, counts[e]);
    largest = simd_max(largest);
    if (lane != 0u) return;
    const uint tiles = max(1u, (largest + 31u) / 32u);
    for (uint i = 0u; i < shapes; ++i) {
        sizes[3u * i + 0u] = rowTiles[i];
        sizes[3u * i + 1u] = tiles;
        sizes[3u * i + 2u] = experts;
    }
}
