#include <metal_stdlib>
using namespace metal;

// ============================================================================
// multirow_gemv — decode's single-row GEMVs, run for up to nine activation
// rows with one read of each weight row.
//
// The wide verify must reproduce scalar decode byte for byte, so every row
// here is reduced exactly the way the single-row decode kernel reduces it:
// one SIMD per output row, the same per-lane partial sums in the same order,
// the same simd_sum, the same conversions. The only change is that each lane
// keeps one accumulator per activation row and, having loaded a weight chunk
// and its scale and bias, folds it into every row's accumulator in turn. A
// row's value therefore cannot depend on how many other rows ride along.
//
// Three row bodies mirror the three decode kernels a dense projection can
// take:
//   int4   dequant_int4_gemv_simd_body (dequant_int4.metal) and its copy
//          gdn_dequant_int4_gemv_simd_body (gdn.metal)
//   affine affine_quant_gemv_simd (dequant_affine.metal), 2 or 8 bits
//   bf16   bf16_gemv_simd (bf16_gemv.metal) and gdn_bf16_gemv_body
// Keep each in lockstep with its decode original: a reordered sum here is a
// verify row that differs from the token decode would have produced.
//
// One dispatch covers up to four segments over a concatenated output-row
// space, like the fused GDN input projection (qkv/z/a/b) and the fused QKV:
// each segment has its own weights, width and output buffer, and all read the
// same activation rows.
//
// Function constants: 130 rows per dispatch (1...9), 131 the column count,
// 132-135 each segment's format (4 = int4, 16 = bf16, 2/8 = affine width).
// ============================================================================

constant constexpr uint kMRGroupSize = 64;
constant constexpr uint kMRMaxRows = 9;
constant constexpr uint kMRMaxSegments = 4;

constant uint FC_MR_ROWS [[function_constant(130)]];
constant uint FC_MR_N    [[function_constant(131)]];
constant uint FC_MR_FMT0 [[function_constant(132)]];
constant uint FC_MR_FMT1 [[function_constant(133)]];
constant uint FC_MR_FMT2 [[function_constant(134)]];
constant uint FC_MR_FMT3 [[function_constant(135)]];

struct MultiRowGEMVParams {
    uint segment_rows[kMRMaxSegments];
    uint y_stride[kMRMaxSegments];
    uint x_stride;
};

static inline uint mr_rows() {
    return is_function_constant_defined(FC_MR_ROWS) ? FC_MR_ROWS : 1u;
}

static inline uint mr_format(uint segment) {
    switch (segment) {
    case 0u: return is_function_constant_defined(FC_MR_FMT0) ? FC_MR_FMT0 : 4u;
    case 1u: return is_function_constant_defined(FC_MR_FMT1) ? FC_MR_FMT1 : 4u;
    case 2u: return is_function_constant_defined(FC_MR_FMT2) ? FC_MR_FMT2 : 4u;
    default: return is_function_constant_defined(FC_MR_FMT3) ? FC_MR_FMT3 : 4u;
    }
}

// Same packing as `affine_quant_value` in dequant_affine.metal. Duplicated so
// this module self-contains its dependencies (the shared library concatenates
// modules; see gdn.metal's K24 note).
static inline uint mr_affine_value(device const uint* packed,
                                   uint element,
                                   uint bits) {
    const ulong bit_offset = ulong(element) * ulong(bits);
    const uint word = uint(bit_offset >> 5);
    const uint shift = uint(bit_offset & 31u);
    const uint mask = (1u << bits) - 1u;
    uint value = packed[word] >> shift;
    if (shift + bits > 32u) {
        value |= packed[word + 1u] << (32u - shift);
    }
    return value & mask;
}

static inline void mr_store(thread float* acc, device half* y, uint y_stride,
                            uint row, uint lane) {
    const uint R = mr_rows();
    for (uint r = 0; r < R; ++r) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u) y[r * y_stride + row] = half(total);
    }
}

// dequant_int4_gemv_simd_body, per activation row.
static inline void mr_int4_rows(device const uint8_t* W,
                                device const bfloat* scales,
                                device const bfloat* biases,
                                device const half* x, uint x_stride,
                                device half* y, uint y_stride,
                                uint row, uint N, uint lane) {
    const uint R = mr_rows();
    const uint n_groups  = N / kMRGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc[kMRMaxRows];
    for (uint r = 0; r < kMRMaxRows; ++r) acc[r] = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // `ushort*` like the decode body: sub-tensor offsets are only
        // 2-byte aligned.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        for (uint r = 0; r < R; ++r) {
            device const half* xr = x + r * x_stride;
            const half4 xa = *((device const half4*)(xr + elem));
            const half4 xb = *((device const half4*)(xr + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
            float dot = 0.0f;
            dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
            dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
            dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
            dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            acc[r] = fma(s, dot, acc[r]);
            acc[r] = fma(b, sum, acc[r]);
        }
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kMRGroupSize / 2) + lane];
        for (uint r = 0; r < R; ++r) {
            device const half* xr = x + r * x_stride;
            const float x0 = float(xr[g * kMRGroupSize + lane * 2u]);
            const float x1 = float(xr[g * kMRGroupSize + lane * 2u + 1u]);
            float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(float(uint(byte >> 4)), x1, dot);
            const float sum = x0 + x1;
            acc[r] = fma(s, dot, acc[r]);
            acc[r] = fma(b, sum, acc[r]);
        }
    }
    mr_store(acc, y, y_stride, row, lane);
}

// affine_quant_gemv_simd, per activation row.
static inline void mr_affine_rows(device const uint* weights,
                                  device const bfloat* scales,
                                  device const bfloat* biases,
                                  device const half* x, uint x_stride,
                                  device half* y, uint y_stride,
                                  uint row, uint columns, uint bits, uint lane) {
    const uint R = mr_rows();
    const uint groups_per_row = columns / kMRGroupSize;
    const uint words_per_row = (columns * bits) / 32u;
    device const uint* row_weights = weights + row * words_per_row;
    device const bfloat* row_scales = scales + row * groups_per_row;
    device const bfloat* row_biases = biases + row * groups_per_row;

    float acc[kMRMaxRows];
    for (uint r = 0; r < kMRMaxRows; ++r) acc[r] = 0.0f;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint first = group * kMRGroupSize + lane * 2u;
        const float q0 = float(mr_affine_value(row_weights, first, bits));
        const float q1 = float(mr_affine_value(row_weights, first + 1u, bits));
        const float scale = float(row_scales[group]);
        const float bias = float(row_biases[group]);
        for (uint r = 0; r < R; ++r) {
            device const half* xr = x + r * x_stride;
            const float x0 = float(xr[first]);
            const float x1 = float(xr[first + 1u]);
            acc[r] = fma(scale, q0 * x0 + q1 * x1, acc[r]);
            acc[r] = fma(bias, x0 + x1, acc[r]);
        }
    }
    mr_store(acc, y, y_stride, row, lane);
}

// bf16_gemv_simd, per activation row.
static inline void mr_bf16_rows(device const bfloat* weights,
                                device const half* x, uint x_stride,
                                device half* y, uint y_stride,
                                uint row, uint columns, uint lane) {
    const uint R = mr_rows();
    device const bfloat* row_weights = weights + row * columns;
    float acc[kMRMaxRows];
    for (uint r = 0; r < kMRMaxRows; ++r) acc[r] = 0.0f;
    for (uint base = 0; base < columns; base += 64u) {
        const uint first = base + lane * 2u;
        const float w0 = float(row_weights[first]);
        const float w1 = float(row_weights[first + 1u]);
        for (uint r = 0; r < R; ++r) {
            device const half* xr = x + r * x_stride;
            acc[r] = fma(w0, float(xr[first]), acc[r]);
            acc[r] = fma(w1, float(xr[first + 1u]), acc[r]);
        }
    }
    mr_store(acc, y, y_stride, row, lane);
}

static inline void mr_segment_rows(uint format,
                                   device const uint8_t* W,
                                   device const bfloat* scales,
                                   device const bfloat* biases,
                                   device const half* x, uint x_stride,
                                   device half* y, uint y_stride,
                                   uint row, uint N, uint lane) {
    if (format == 16u) {
        mr_bf16_rows((device const bfloat*)W, x, x_stride, y, y_stride, row, N, lane);
    } else if (format == 4u) {
        mr_int4_rows(W, scales, biases, x, x_stride, y, y_stride, row, N, lane);
    } else {
        mr_affine_rows((device const uint*)W, scales, biases, x, x_stride,
                       y, y_stride, row, N, format, lane);
    }
}

// Eight output rows per threadgroup, one SIMD per output row (256 threads),
// the decode kernels' geometry.
kernel void multirow_gemv_simd(
    device const uint8_t* w0 [[buffer(0)]],
    device const bfloat*  s0 [[buffer(1)]],
    device const bfloat*  b0 [[buffer(2)]],
    device const uint8_t* w1 [[buffer(3)]],
    device const bfloat*  s1 [[buffer(4)]],
    device const bfloat*  b1 [[buffer(5)]],
    device const uint8_t* w2 [[buffer(6)]],
    device const bfloat*  s2 [[buffer(7)]],
    device const bfloat*  b2 [[buffer(8)]],
    device const uint8_t* w3 [[buffer(9)]],
    device const bfloat*  s3 [[buffer(10)]],
    device const bfloat*  b3 [[buffer(11)]],
    device const half*    x  [[buffer(12)]],
    device half*          y0 [[buffer(13)]],
    device half*          y1 [[buffer(14)]],
    device half*          y2 [[buffer(15)]],
    device half*          y3 [[buffer(16)]],
    constant MultiRowGEMVParams& p [[buffer(17)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint N = FC_MR_N;
    uint row = tg_idx * rows_per_tg + sg_idx;
    if (row < p.segment_rows[0]) {
        mr_segment_rows(mr_format(0u), w0, s0, b0, x, p.x_stride,
                        y0, p.y_stride[0], row, N, lane);
        return;
    }
    row -= p.segment_rows[0];
    if (row < p.segment_rows[1]) {
        mr_segment_rows(mr_format(1u), w1, s1, b1, x, p.x_stride,
                        y1, p.y_stride[1], row, N, lane);
        return;
    }
    row -= p.segment_rows[1];
    if (row < p.segment_rows[2]) {
        mr_segment_rows(mr_format(2u), w2, s2, b2, x, p.x_stride,
                        y2, p.y_stride[2], row, N, lane);
        return;
    }
    row -= p.segment_rows[2];
    if (row < p.segment_rows[3]) {
        mr_segment_rows(mr_format(3u), w3, s3, b3, x, p.x_stride,
                        y3, p.y_stride[3], row, N, lane);
    }
}
