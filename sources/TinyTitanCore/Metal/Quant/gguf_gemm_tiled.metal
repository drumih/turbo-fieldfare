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
// Row tile of the 32-token kernels (GGUFGemmTiled.wideTileRows).
constant constexpr uint GEMM_WIDE_ROWS = 128u;
// Row tile of the 8-token kernels (GGUFGemmTiled.narrowTileRows): 64 is the
// measured optimum at T=64 (32 rows 11.2-11.7 ms, 128 rows 13.7, 64 rows
// 10.3-11.1 per routed layer).
constant constexpr uint GEMM_NARROW_ROWS = 64u;
// K step of the 8-token kernels (GGUFGemmTiled.narrowTileDepth). 128 measured
// slower (11.1 -> 13.4-15.3 ms per routed layer at T=64): the larger
// threadgroup footprint costs more occupancy than the halved barriers save.
constant constexpr uint GEMM_NARROW_BK = 64u;

// Sixteen consecutive weights of row `row` starting at column `c` (c % 16 == 0).
struct GemmQ8_0 {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 32u) * 34u;
        const float d = float(as_type<half>(*(device const ushort*)b));
        device const char* qs = (device const char*)(b + 2u) + (c % 32u);
        for (uint j = 0u; j < 16u; ++j) out[j] = T(d * float(qs[j]));
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
    // E2M1 x UE4M3 products are exact in half, so T = float loses nothing.
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
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
                out[4u * i + j] = T(gemm_e2m1(byte & 0x0Fu) * d);
                out[8u + 4u * i + j] = T(gemm_e2m1(byte >> 4u) * d);
            }
        }
    }
    static uint rowBytes(uint columns) { return columns / 64u * 36u; }
};

struct GemmF32 {
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread half* out,
                       device const uchar* tables) {
        device const float* r = (device const float*)(w + row * rowBytes) + c;
        for (uint j = 0u; j < 16u; ++j) out[j] = half(r[j]);
    }
    static uint rowBytes(uint columns) { return columns * 4u; }
};

struct GemmBF16 {
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread half* out,
                       device const uchar* tables) {
        device const ushort* r = (device const ushort*)(w + row * rowBytes) + c;
        for (uint j = 0u; j < 16u; ++j) out[j] = half(as_type<float>(uint(r[j]) << 16u));
    }
    static uint rowBytes(uint columns) { return columns * 2u; }
};

// BN is the token tile: 32 for dense chunks and well-filled experts, 8 when
// experts get only a few tokens each (the 32-wide tile would spend 3/4 of its
// multiplies on padding). BM is the row tile. Four SIMD groups:
//   BM 128, BN 32: each 32 rows x 32 tokens (4 x 4 accumulators, 16 MMAs per
//                  8 threadgroup loads);
//   BM 64,  BN 32: each 32 rows x 16 tokens (4 x 2);
//   BM 64,  BN 8:  each 16 rows x 8 tokens (2 x 1). (128 rows x 8 tokens
//                  measured slower: 11.1 -> 13.7 ms per routed layer at T=64.)
// The output tile is staged in sa's memory once the K loop is done (BM * BN
// floats fit in BM * BK halves because BN <= BK / 2).
template <typename Decoder, uint BN, uint BM, uint BK>
static void gguf_gemm_tiled_impl(
    device const uchar* weights, device const float* x, device float* y,
    device const uint* lists, device const uint* counts, device const float* scales,
    constant GGUFGemmArgs& args, device const uchar* tables, uint3 tile, uint sg, uint tid,
    threadgroup half* sa, threadgroup half* sb)
{
    const uint rowBase = tile.x * BM;
    const uint tokenBase = tile.y * BN;
    const uint expert = args.indirect != 0u ? tile.z : 0u;
    const uint tokenCount = args.indirect != 0u ? counts[expert] : args.tokens;
    if (rowBase >= args.rows || tokenBase >= tokenCount) return;
    device const uchar* matrix = weights + ulong(expert) * args.expertStride;
    const uint rowBytes = Decoder::rowBytes(args.columns);

    // This thread's loads per K step: weight row tid / (128 / BM), 16-value
    // windows (tid % THREADS_PER_ROW) * 16 + part * 16 * THREADS_PER_ROW;
    // activation token tid / (128 / BN), BN / 2 columns (BN / 8 float4s).
    constexpr uint THREADS_PER_ROW = 128u / BM;
    constexpr uint PARTS = BK / 16u / THREADS_PER_ROW;
    constexpr uint THREADS_PER_TOKEN = 128u / BN;
    constexpr uint X4_PER_THREAD = BK * BN / 512u;
    constexpr uint ROW_TILES = (BM == 128u || BN == 32u) ? 4u : 2u;
    constexpr uint TOKEN_TILES = BM == 128u ? BN / 8u : (BN == 32u ? 2u : 1u);
    static_assert(BN * 2u <= BK, "the output tile is staged in sa");
    const uint loadRow = min(rowBase + tid / THREADS_PER_ROW, args.rows - 1u);
    const uint loadToken = tokenBase + tid / THREADS_PER_TOKEN;
    const bool tokenValid = loadToken < tokenCount;
    uint xRow = 0u;
    if (tokenValid) {
        xRow = args.indirect != 0u
            ? lists[expert * args.listStride + loadToken] / args.xDivisor
            : loadToken;
    }
    device const float4* xs = (device const float4*)(
        x + ulong(xRow) * args.columns + (tid % THREADS_PER_TOKEN) * X4_PER_THREAD * 4u);
    threadgroup half4* sa4 = (threadgroup half4*)(
        sa + (tid / THREADS_PER_ROW) * BK + (tid % THREADS_PER_ROW) * 16u);
    threadgroup half4* sb4 = (threadgroup half4*)(
        sb + (tid / THREADS_PER_TOKEN) * BK + (tid % THREADS_PER_TOKEN) * X4_PER_THREAD * 4u);

    simdgroup_float8x8 acc[ROW_TILES * TOKEN_TILES];
    for (uint i = 0u; i < ROW_TILES * TOKEN_TILES; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint sgRow = BM == 128u ? sg * 32u : (BN == 32u ? (sg % 2u) * 32u : sg * 16u);
    const uint sgToken = BM == 128u ? 0u : (BN == 32u ? (sg / 2u) * 16u : 0u);

    // One decode-store-barrier-MMA step per BK. (Prefetching step k + BK into
    // registers during step k's MMAs measured slower on both row tiles: a
    // routed layer at T=512 29.2 -> 32.1 ms on the 128-row tile.)
    for (uint k = 0u; k < args.columns; k += BK) {
        for (uint part = 0u; part < PARTS; ++part) {
            half wv[16];
            Decoder::decode(
                matrix, rowBytes, loadRow,
                k + (tid % THREADS_PER_ROW) * 16u + part * 16u * THREADS_PER_ROW, wv, tables);
            for (uint j = 0u; j < 4u; ++j) {
                sa4[part * 4u * THREADS_PER_ROW + j] =
                    half4(wv[4u * j], wv[4u * j + 1u], wv[4u * j + 2u], wv[4u * j + 3u]);
            }
        }
        for (uint j = 0u; j < X4_PER_THREAD; ++j) {
            sb4[j] = tokenValid ? half4(xs[k / 4u + j]) : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_half8x8 ma[ROW_TILES];
            simdgroup_half8x8 mb[TOKEN_TILES];
            for (uint i = 0u; i < ROW_TILES; ++i) {
                simdgroup_load(ma[i], sa + (sgRow + 8u * i) * BK + kk, BK);
            }
            for (uint i = 0u; i < TOKEN_TILES; ++i) {
                // B is X^T: element (kk + a, token b) = sb[(token) * BK + kk + a].
                simdgroup_load(mb[i], sb + (sgToken + 8u * i) * BK + kk, BK, ulong2(0, 0), true);
            }
            for (uint i = 0u; i < ROW_TILES; ++i) {
                for (uint j = 0u; j < TOKEN_TILES; ++j) {
                    simdgroup_multiply_accumulate(
                        acc[i * TOKEN_TILES + j], ma[i], mb[j], acc[i * TOKEN_TILES + j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage the BM x BN tile (row-major rows x tokens) in sa, then write valid cells.
    threadgroup float* sc = (threadgroup float*)sa;
    for (uint i = 0u; i < ROW_TILES; ++i) {
        for (uint j = 0u; j < TOKEN_TILES; ++j) {
            simdgroup_store(
                acc[i * TOKEN_TILES + j], sc + (sgRow + 8u * i) * BN + sgToken + 8u * j, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = args.useScale != 0u ? scales[expert] : 1.0f;
    for (uint cell = tid; cell < BM * BN; cell += 128u) {
        const uint r = cell / BN;
        const uint t = cell % BN;
        const uint row = rowBase + r;
        const uint token = tokenBase + t;
        if (row >= args.rows || token >= tokenCount) continue;
        const uint yRow = args.indirect != 0u ? lists[expert * args.listStride + token] : token;
        y[ulong(yRow) * args.rows + row] = sc[r * BN + t] * scale;
    }
}

#define GGUF_GEMM_TILED_KERNEL_BM(name, decoder, bn, bm, bk)                         \
kernel void name(                                                                    \
    device const uchar* weights [[buffer(0)]],                                       \
    device const float* x [[buffer(1)]],                                             \
    device float* y [[buffer(2)]],                                                   \
    device const uint* lists [[buffer(3)]],                                          \
    device const uint* counts [[buffer(4)]],                                         \
    device const float* scales [[buffer(5)]],                                        \
    constant GGUFGemmArgs& args [[buffer(6)]],                                       \
    device const uchar* tables [[buffer(7)]],                                        \
    uint3 tile [[threadgroup_position_in_grid]],                                     \
    uint sg [[simdgroup_index_in_threadgroup]],                                      \
    uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                    \
    threadgroup half sa[bm * bk];                                                    \
    threadgroup half sb[bn * bk];                                                    \
    gguf_gemm_tiled_impl<decoder, bn, bm, bk>(                                       \
        weights, x, y, lists, counts, scales, args, tables, tile, sg, tid, sa, sb);  \
}

// 32-token kernels take 128-row tiles; 8-token kernels 64-row tiles.
#define GGUF_GEMM_TILED_KERNEL(name, decoder, bn) \
    GGUF_GEMM_TILED_KERNEL_BM(                    \
        name, decoder, bn, (bn == 32u ? GEMM_WIDE_ROWS : GEMM_NARROW_ROWS), \
        (bn == 32u ? GEMM_BK : GEMM_NARROW_BK))

GGUF_GEMM_TILED_KERNEL(gemm_tiled_q8_0, GemmQ8_0, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_nvfp4, GemmNVFP4, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_f32, GemmF32, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_bf16, GemmBF16, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_q8_0_n8, GemmQ8_0, 8u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_nvfp4_n8, GemmNVFP4, 8u)
// The block decoders of gguf_block_decoders.metal (Inkling2 UD-Q2_K_XL types):
// projections in 32-token tiles, routed experts in both widths.
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq3_xxs, GemmIQ3XXS, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_q4_k, GemmQ4K, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_q5_k, GemmQ5K, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_q6_k, GemmQ6K, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq1_s, GemmIQ1S, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq2_xxs, GemmIQ2XXS, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq2_xs, GemmIQ2XS, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq2_s, GemmIQ2S, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_q2_0, GemmQ2_0, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_mxfp4, GemmMXFP4, 32u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq1_s_n8, GemmIQ1S, 8u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq2_xxs_n8, GemmIQ2XXS, 8u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq2_xs_n8, GemmIQ2XS, 8u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_iq2_s_n8, GemmIQ2S, 8u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_q2_0_n8, GemmQ2_0, 8u)
GGUF_GEMM_TILED_KERNEL(gemm_tiled_mxfp4_n8, GemmMXFP4, 8u)

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
