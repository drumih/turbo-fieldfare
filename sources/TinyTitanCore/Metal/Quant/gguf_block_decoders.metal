#include <metal_stdlib>
using namespace metal;

// GGUF block decoders: sixteen consecutive weights of row `row` from column `c`
// (c % 16 == 0), straight from the block bytes, as half for the tiled GEMM
// or float for the GEMV, for the tiled prefill GEMM and
// the expert-indexed decode GEMV. Each ports ggml's dequantize_row_* for its
// type (ggml-quants.c); the 16-value window always lies inside one scale group.
//
// IQ grids and value tables are one device buffer (`tables`), laid out as
// below and filled from GGUFIQTables (ggml-common.h):
constant uint GGUF_TABLE_IQ2XXS = 0u;      // uint64 x 256
constant uint GGUF_TABLE_IQ2XS = 2048u;    // uint64 x 512
constant uint GGUF_TABLE_IQ2S = 6144u;     // uint64 x 1024
constant uint GGUF_TABLE_IQ1S = 14336u;    // uint64 x 2048 (int8 entries)
constant uint GGUF_TABLE_IQ3XXS = 30720u;  // uint32 x 256
constant uint GGUF_TABLE_KSIGNS = 31744u;  // uint8 x 128
constant uint GGUF_TABLE_IQ4NL = 31872u;   // int8 x 16
constant uint GGUF_TABLE_BYTES = 31888u;

static inline float gguf_half_at(device const uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8u))));
}

static inline uint gguf_u16_at(device const uchar* p) {
    return uint(p[0]) | (uint(p[1]) << 8u);
}

static inline uint gguf_u32_at(device const uchar* p) {
    return uint(p[0]) | (uint(p[1]) << 8u) | (uint(p[2]) << 16u) | (uint(p[3]) << 24u);
}

// One 8-value grid entry as a single 64-bit load (table offsets are multiples
// of 8): eight byte loads per entry measured ~45 GB/s in the decode GEMV.
static inline ulong gguf_grid64(device const uchar* tables, uint table, uint entry) {
    return ((device const ulong*)(tables + table))[entry];
}

// IQ2_XXS (66 bytes / 256): d, then per 32 values 4 grid indices and a word of
// 4 x 7 sign indices with a 4-bit scale on top.
struct GemmIQ2XXS {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 66u;
        const uint v = c % 256u;
        device const uchar* group = b + 2u + (v / 32u) * 8u;
        const uint aux = gguf_u32_at(group + 4u);
        const float db = gguf_half_at(b) * (0.5f + float(aux >> 28u)) * 0.25f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint signs = tables[GGUF_TABLE_KSIGNS + ((aux >> (7u * l)) & 127u)];
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ2XXS, group[l]);
            for (uint j = 0u; j < 8u; ++j) {
                const float q = float(uint(grid >> (8u * j)) & 0xFFu);
                out[8u * h + j] = T(db * q * ((signs >> j) & 1u ? -1.0f : 1.0f));
            }
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 66u; }
};

// IQ2_XS (74 / 256): d, 32 uint16 (9-bit grid index, 7-bit sign index), then
// one byte of two 4-bit scales per 32 values (first 16 / second 16).
struct GemmIQ2XS {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 74u;
        const uint v = c % 256u;
        const uint ib = v / 32u;
        const uint scale = (v % 32u) < 16u ? (b[66u + ib] & 0x0Fu) : (b[66u + ib] >> 4u);
        const float db = gguf_half_at(b) * (0.5f + float(scale)) * 0.25f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint q = gguf_u16_at(b + 2u + (4u * ib + l) * 2u);
            const uint signs = tables[GGUF_TABLE_KSIGNS + (q >> 9u)];
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ2XS, q & 511u);
            for (uint j = 0u; j < 8u; ++j) {
                const float g = float(uint(grid >> (8u * j)) & 0xFFu);
                out[8u * h + j] = T(db * g * ((signs >> j) & 1u ? -1.0f : 1.0f));
            }
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 74u; }
};

// IQ2_S (82 / 256): d, 32 grid-index low bytes, 32 sign bytes, 8 qh bytes
// (2 high index bits per 8 values), 8 scale bytes.
struct GemmIQ2S {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 82u;
        const uint v = c % 256u;
        const uint sub = v / 32u;
        const float d = gguf_half_at(b);
        const uint packed = b[74u + sub];
        const float db = d * (0.5f + float((v % 32u) < 16u ? (packed & 0x0Fu) : (packed >> 4u))) * 0.25f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint index = uint(b[2u + sub * 4u + l]) | ((uint(b[66u + sub]) << (8u - 2u * l)) & 0x300u);
            const uint signs = b[2u + 32u + sub * 4u + l];
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ2S, index);
            for (uint j = 0u; j < 8u; ++j) {
                const float g = float(uint(grid >> (8u * j)) & 0xFFu);
                out[8u * h + j] = T(db * g * ((signs >> j) & 1u ? -1.0f : 1.0f));
            }
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 82u; }
};

// IQ1_S (50 / 256): d, 32 grid-index low bytes, 8 uint16 qh: per 32 values
// three high index bits per 8, a 3-bit scale and the delta sign.
struct GemmIQ1S {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 50u;
        const uint v = c % 256u;
        const uint ib = v / 32u;
        const uint qh = gguf_u16_at(b + 34u + ib * 2u);
        const float dl = gguf_half_at(b) * float(2u * ((qh >> 12u) & 7u) + 1u);
        const float delta = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint index = uint(b[2u + ib * 4u + l]) | (((qh >> (3u * l)) & 7u) << 8u);
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ1S, index);
            for (uint j = 0u; j < 8u; ++j) {
                const float g = float(char(uchar(grid >> (8u * j))));
                out[8u * h + j] = T(dl * (g + delta));
            }
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 50u; }
};

// IQ3_XXS (98 / 256): d, 64 grid-index bytes (4-value entries), then a uint32
// per 32 values of 4 x 7 sign indices with a 4-bit scale on top.
struct GemmIQ3XXS {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 98u;
        const uint v = c % 256u;
        const uint ib = v / 32u;
        const uint aux = gguf_u32_at(b + 2u + 64u + ib * 4u);
        const float db = gguf_half_at(b) * (0.5f + float(aux >> 28u)) * 0.5f;
        device const uchar* qs = b + 2u + ib * 8u;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint signs = tables[GGUF_TABLE_KSIGNS + ((aux >> (7u * l)) & 127u)];
            device const uint* grid3 = (device const uint*)(tables + GGUF_TABLE_IQ3XXS);
            const uint grids[2] = {grid3[qs[2u * l]], grid3[qs[2u * l + 1u]]};
            for (uint j = 0u; j < 8u; ++j) {
                const float g = float((grids[j / 4u] >> (8u * (j % 4u))) & 0xFFu);
                out[8u * h + j] = T(db * g * ((signs >> j) & 1u ? -1.0f : 1.0f));
            }
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 98u; }
};

// Q2_0 (18 / 64): d, then four 2-bit values per byte (00 = -1 ... 11 = +2).
struct GemmQ2_0 {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 64u) * 18u;
        const float d = gguf_half_at(b);
        const uint v = c % 64u;
        for (uint i = 0u; i < 16u; ++i) {
            const uint q = (uint(b[2u + (v + i) / 4u]) >> (2u * ((v + i) % 4u))) & 3u;
            out[i] = T(d * (float(q) - 1.0f));
        }
    }
    static uint rowBytes(uint columns) { return columns / 64u * 18u; }
};

// MXFP4 (17 / 32): E8M0 scale, then value j in the low and j + 16 in the high
// nibble of byte 1 + j; E2M1 values doubled, so the scale carries one half.
constant float GGUF_FP4_DOUBLED[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

struct GemmMXFP4 {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 32u) * 17u;
        const uint e = b[0];
        const float d = e < 2u ? as_type<float>(0x00200000u << e) : as_type<float>((e - 1u) << 23u);
        const bool high = (c % 32u) >= 16u;
        for (uint j = 0u; j < 16u; ++j) {
            const uint q = high ? (b[1u + j] >> 4u) : (b[1u + j] & 0x0Fu);
            out[j] = T(GGUF_FP4_DOUBLED[q] * d);
        }
    }
    static uint rowBytes(uint columns) { return columns / 32u * 17u; }
};

// K-quant 6-bit scale and minimum of 32-value group g from the 12 packed bytes.
static inline void gguf_scale_min_k4(uint g, device const uchar* s, thread float& sc, thread float& m) {
    if (g < 4u) {
        sc = float(s[g] & 63u);
        m = float(s[g + 4u] & 63u);
    } else {
        sc = float((s[g + 4u] & 0x0Fu) | ((s[g - 4u] >> 6u) << 4u));
        m = float((s[g + 4u] >> 4u) | ((s[g] >> 6u) << 4u));
    }
}

// Q4_K (144 / 256): d, dmin, 12 scale bytes, 128 nibble bytes; 64-value chunk
// j holds group 2j in the low and 2j + 1 in the high nibbles of 32 bytes.
struct GemmQ4K {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 144u;
        const uint v = c % 256u;
        const uint g = v / 32u;
        float sc, m;
        gguf_scale_min_k4(g, b + 4u, sc, m);
        const float d1 = gguf_half_at(b) * sc;
        const float m1 = gguf_half_at(b + 2u) * m;
        device const uchar* q = b + 16u + (g / 2u) * 32u + (v % 32u);
        const bool high = (g % 2u) == 1u;
        for (uint i = 0u; i < 16u; ++i) {
            const uint n = high ? (q[i] >> 4u) : (q[i] & 0x0Fu);
            out[i] = T(d1 * float(n) - m1);
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 144u; }
};

// Q5_K (176 / 256): Q4_K plus a 32-byte high-bit plane; bit g of qh[l] is the
// fifth bit of value l of group g.
struct GemmQ5K {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 176u;
        const uint v = c % 256u;
        const uint g = v / 32u;
        float sc, m;
        gguf_scale_min_k4(g, b + 4u, sc, m);
        const float d1 = gguf_half_at(b) * sc;
        const float m1 = gguf_half_at(b + 2u) * m;
        device const uchar* qh = b + 16u;
        device const uchar* q = b + 48u + (g / 2u) * 32u;
        const bool high = (g % 2u) == 1u;
        for (uint i = 0u; i < 16u; ++i) {
            const uint l = (v % 32u) + i;
            const uint n = (high ? (q[l] >> 4u) : (q[l] & 0x0Fu)) + (((qh[l] >> g) & 1u) << 4u);
            out[i] = T(d1 * float(n) - m1);
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 176u; }
};

// Q6_K (210 / 256): 128 low-nibble bytes, 64 high-2-bit bytes, 16 int8 scales,
// d at 208. Each 128-value half is four 32-value quadrants.
struct GemmQ6K {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 210u;
        const uint v = c % 256u;
        const uint n = v / 128u;
        const uint quadrant = (v % 128u) / 32u;
        const uint l0 = v % 32u;
        // Quadrants 1 and 3 read the second 32 low bytes; 2 and 3 the high nibbles.
        device const uchar* ql = b + 64u * n + 32u * (quadrant & 1u) + l0;
        device const uchar* qh = b + 128u + 32u * n + l0;
        const uint lowShift = 4u * (quadrant >> 1u);
        const uint highShift = 2u * quadrant;
        const float d = gguf_half_at(b + 208u)
            * float(char(b[192u + 8u * n + l0 / 16u + 2u * quadrant]));
        for (uint i = 0u; i < 16u; ++i) {
            const uint low = (uint(ql[i]) >> lowShift) & 0x0Fu;
            const int q = int(low | (((uint(qh[i]) >> highShift) & 3u) << 4u)) - 32;
            out[i] = T(d * float(q));
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 210u; }
};

// IQ4_XS (136 / 256): d, uint16 high scale bits, 4 low-scale bytes, 128 nibble
// bytes; group ib's 6-bit scale minus 32 times the IQ4_NL value table.
struct GemmIQ4XS {
    template <typename T>
    static void decode(device const uchar* w, uint rowBytes, uint row, uint c, thread T* out,
                       device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 136u;
        const uint v = c % 256u;
        const uint ib = v / 32u;
        const uint scalesHigh = gguf_u16_at(b + 2u);
        const uint ls = ((b[4u + ib / 2u] >> (4u * (ib % 2u))) & 0x0Fu) | (((scalesHigh >> (2u * ib)) & 3u) << 4u);
        const float dl = gguf_half_at(b) * (float(ls) - 32.0f);
        device const uchar* q = b + 8u + ib * 16u;
        const bool high = (v % 32u) >= 16u;
        for (uint j = 0u; j < 16u; ++j) {
            const uint n = high ? (q[j] >> 4u) : (q[j] & 0x0Fu);
            out[j] = T(dl * float(char(tables[GGUF_TABLE_IQ4NL + n])));
        }
    }
    static uint rowBytes(uint columns) { return columns / 256u * 136u; }
};

// The decode GEMV's 16-value dot product. By default the window is decoded to
// fp32 and dotted; the hot low-bit types specialise it to dot the raw grid or
// quant values and apply the group scale once (and flip signs on x rather
// than multiplying), ggml's mul_mv arrangement. Both are fp32 throughout.
template <typename Decoder>
struct GGUFDot16 {
    static float dot(device const uchar* w, uint rowBytes, uint row, uint c, thread const float* x,
                     device const uchar* tables) {
        float wv[16];
        Decoder::decode(w, rowBytes, row, c, wv, tables);
        float acc = 0.0f;
        for (uint i = 0u; i < 16u; ++i) acc = fma(wv[i], x[i], acc);
        return acc;
    }
};

static inline float gguf_signed(float x, uint signs, uint j) {
    return ((signs >> j) & 1u) != 0u ? -x : x;
}

template <>
struct GGUFDot16<GemmIQ2XXS> {
    static float dot(device const uchar* w, uint rowBytes, uint row, uint c, thread const float* x,
                     device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 66u;
        const uint v = c % 256u;
        device const uchar* group = b + 2u + (v / 32u) * 8u;
        const uint aux = gguf_u32_at(group + 4u);
        float acc = 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint signs = tables[GGUF_TABLE_KSIGNS + ((aux >> (7u * l)) & 127u)];
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ2XXS, group[l]);
            for (uint j = 0u; j < 8u; ++j) {
                acc = fma(gguf_signed(x[8u * h + j], signs, j), float(uint(grid >> (8u * j)) & 0xFFu), acc);
            }
        }
        return gguf_half_at(b) * (0.5f + float(aux >> 28u)) * 0.25f * acc;
    }
};

template <>
struct GGUFDot16<GemmIQ2XS> {
    static float dot(device const uchar* w, uint rowBytes, uint row, uint c, thread const float* x,
                     device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 74u;
        const uint v = c % 256u;
        const uint ib = v / 32u;
        const uint scale = (v % 32u) < 16u ? (b[66u + ib] & 0x0Fu) : (b[66u + ib] >> 4u);
        float acc = 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint q = gguf_u16_at(b + 2u + (4u * ib + l) * 2u);
            const uint signs = tables[GGUF_TABLE_KSIGNS + (q >> 9u)];
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ2XS, q & 511u);
            for (uint j = 0u; j < 8u; ++j) {
                acc = fma(gguf_signed(x[8u * h + j], signs, j), float(uint(grid >> (8u * j)) & 0xFFu), acc);
            }
        }
        return gguf_half_at(b) * (0.5f + float(scale)) * 0.25f * acc;
    }
};

template <>
struct GGUFDot16<GemmIQ2S> {
    static float dot(device const uchar* w, uint rowBytes, uint row, uint c, thread const float* x,
                     device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 82u;
        const uint v = c % 256u;
        const uint sub = v / 32u;
        const uint packed = b[74u + sub];
        const uint scale = (v % 32u) < 16u ? (packed & 0x0Fu) : (packed >> 4u);
        float acc = 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint index = uint(b[2u + sub * 4u + l]) | ((uint(b[66u + sub]) << (8u - 2u * l)) & 0x300u);
            const uint signs = b[2u + 32u + sub * 4u + l];
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ2S, index);
            for (uint j = 0u; j < 8u; ++j) {
                acc = fma(gguf_signed(x[8u * h + j], signs, j), float(uint(grid >> (8u * j)) & 0xFFu), acc);
            }
        }
        return gguf_half_at(b) * (0.5f + float(scale)) * 0.25f * acc;
    }
};

template <>
struct GGUFDot16<GemmIQ1S> {
    static float dot(device const uchar* w, uint rowBytes, uint row, uint c, thread const float* x,
                     device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 50u;
        const uint v = c % 256u;
        const uint ib = v / 32u;
        const uint qh = gguf_u16_at(b + 34u + ib * 2u);
        float acc = 0.0f;
        float sum = 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint index = uint(b[2u + ib * 4u + l]) | (((qh >> (3u * l)) & 7u) << 8u);
            const ulong grid = gguf_grid64(tables, GGUF_TABLE_IQ1S, index);
            for (uint j = 0u; j < 8u; ++j) {
                acc = fma(x[8u * h + j], float(char(uchar(grid >> (8u * j)))), acc);
                sum += x[8u * h + j];
            }
        }
        const float delta = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
        return gguf_half_at(b) * float(2u * ((qh >> 12u) & 7u) + 1u) * fma(delta, sum, acc);
    }
};

template <>
struct GGUFDot16<GemmIQ3XXS> {
    static float dot(device const uchar* w, uint rowBytes, uint row, uint c, thread const float* x,
                     device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 256u) * 98u;
        const uint v = c % 256u;
        const uint ib = v / 32u;
        const uint aux = gguf_u32_at(b + 2u + 64u + ib * 4u);
        device const uchar* qs = b + 2u + ib * 8u;
        device const uint* grid3 = (device const uint*)(tables + GGUF_TABLE_IQ3XXS);
        float acc = 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
            const uint l = ((v % 32u) / 8u) + h;
            const uint signs = tables[GGUF_TABLE_KSIGNS + ((aux >> (7u * l)) & 127u)];
            const uint grids[2] = {grid3[qs[2u * l]], grid3[qs[2u * l + 1u]]};
            for (uint j = 0u; j < 8u; ++j) {
                acc = fma(gguf_signed(x[8u * h + j], signs, j), float((grids[j / 4u] >> (8u * (j % 4u))) & 0xFFu), acc);
            }
        }
        return gguf_half_at(b) * (0.5f + float(aux >> 28u)) * 0.5f * acc;
    }
};

template <>
struct GGUFDot16<GemmQ2_0> {
    static float dot(device const uchar* w, uint rowBytes, uint row, uint c, thread const float* x,
                     device const uchar* tables) {
        device const uchar* b = w + row * rowBytes + (c / 64u) * 18u;
        device const uchar* q = b + 2u + (c % 64u) / 4u;
        float acc = 0.0f;
        float sum = 0.0f;
        for (uint i = 0u; i < 16u; ++i) {
            acc = fma(x[i], float((uint(q[i / 4u]) >> (2u * (i % 4u))) & 3u), acc);
            sum += x[i];
        }
        return gguf_half_at(b) * (acc - sum);
    }
};

// Expert-indexed GEMV over any of the block decoders above (decode path for
// the formats without a dedicated kernel): two SIMD groups per threadgroup,
// two rows per SIMD group, each lane decoding 16-value windows of both rows
// and reusing its activations. Same contract as nvfp4_gemv_ids
// (GGUFGemvIdsArgs); a flat grid of row tiles x slots.
template <typename Decoder, uint R>
static void gguf_gemv_blocks_impl(
    device const uchar* weights, device const float* x, device float* y,
    device const int* ids, device const float* scales, constant GGUFGemvIdsArgs& args,
    device const uchar* tables, uint groupIndex, uint sg, uint lane)
{
    const uint rowTiles = (args.rows + 2u * R - 1u) / (2u * R);
    const uint slot = groupIndex / rowTiles;
    const uint firstRow = ((groupIndex % rowTiles) * 2u + sg) * R;
    if (firstRow >= args.rows) return;
    const int expert = ids[slot];
    device const uchar* matrix = weights + ulong(expert) * args.expertStride;
    device const float* xs = x + slot * args.xSlotStride;
    const uint rowBytes = Decoder::rowBytes(args.columns);
    uint rows[R];
    float sums[R];
    for (uint r = 0u; r < R; ++r) {
        rows[r] = min(firstRow + r, args.rows - 1u);
        sums[r] = 0.0f;
    }
    // Each lane walks 16-value windows, its activations reused across the R
    // rows. (One flat list of row x window items, which balances the ragged
    // last pass of 2,304 and 1,152 columns, measured ~25% slower.)
    for (uint chunk = lane; chunk < args.columns / 16u; chunk += 32u) {
        device const float4* x4 = (device const float4*)(xs + chunk * 16u);
        float xv[16];
        for (uint i = 0u; i < 4u; ++i) {
            const float4 part = x4[i];
            xv[4u * i] = part.x;
            xv[4u * i + 1u] = part.y;
            xv[4u * i + 2u] = part.z;
            xv[4u * i + 3u] = part.w;
        }
        for (uint r = 0u; r < R; ++r) {
            sums[r] += GGUFDot16<Decoder>::dot(matrix, rowBytes, rows[r], chunk * 16u, xv, tables);
        }
    }
    const float scale = args.useScale != 0u ? scales[expert] : 1.0f;
    device float* out = y + slot * args.rows;
    for (uint r = 0u; r < R; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0u && firstRow + r < args.rows) out[firstRow + r] = total * scale;
    }
}

// Rows per SIMD group (two SIMD groups per threadgroup).
constant constexpr uint GGUF_GEMV_BLOCK_ROWS = 4u;

#define GGUF_GEMV_BLOCKS_KERNEL(name, decoder)                                       \
kernel void name(                                                                    \
    device const uchar* weights [[buffer(0)]],                                       \
    device const float* x [[buffer(1)]],                                             \
    device float* y [[buffer(2)]],                                                   \
    device const int* ids [[buffer(3)]],                                             \
    device const float* scales [[buffer(4)]],                                        \
    constant GGUFGemvIdsArgs& args [[buffer(5)]],                                    \
    device const uchar* tables [[buffer(6)]],                                        \
    uint groupIndex [[threadgroup_position_in_grid]],                                \
    uint sg [[simdgroup_index_in_threadgroup]],                                      \
    uint lane [[thread_index_in_simdgroup]])                                         \
{                                                                                    \
    gguf_gemv_blocks_impl<decoder, GGUF_GEMV_BLOCK_ROWS>(                            \
        weights, x, y, ids, scales, args, tables, groupIndex, sg, lane);             \
}

GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_iq2_xxs, GemmIQ2XXS)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_iq2_xs, GemmIQ2XS)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_iq2_s, GemmIQ2S)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_iq1_s, GemmIQ1S)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_iq3_xxs, GemmIQ3XXS)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_q2_0, GemmQ2_0)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_mxfp4, GemmMXFP4)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_q4_k, GemmQ4K)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_q5_k, GemmQ5K)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_q6_k, GemmQ6K)
GGUF_GEMV_BLOCKS_KERNEL(gemv_blocks_iq4_xs, GemmIQ4XS)

// One embedding row per token from an IQ4_XS table: 16 values per thread,
// grid (columns / 16, tokens).
kernel void embed_iq4_xs_row(
    device const uchar* weights [[buffer(0)]],
    device const int* token [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    device const uchar* tables [[buffer(4)]],
    uint2 index [[thread_position_in_grid]])
{
    if (index.x * 16u >= columns) return;
    float wv[16];
    GemmIQ4XS::decode(
        weights, GemmIQ4XS::rowBytes(columns), uint(token[index.y]), index.x * 16u, wv, tables);
    for (uint j = 0u; j < 16u; ++j) out[index.y * columns + index.x * 16u + j] = wv[j];
}
