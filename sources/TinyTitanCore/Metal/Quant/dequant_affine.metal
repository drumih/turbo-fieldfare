#include <metal_stdlib>
using namespace metal;

constant constexpr uint kAffineGroupSize = 64;
constant uint FC_AFFINE_BITS [[function_constant(100)]];

static inline uint affine_quant_bits() {
    return is_function_constant_defined(FC_AFFINE_BITS) ? FC_AFFINE_BITS : 4u;
}

static inline uint affine_quant_value(device const uint* packed,
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

kernel void affine_quant_gemv_simd(
    device const uint*   weights [[buffer(0)]],
    device const bfloat* scales  [[buffer(1)]],
    device const bfloat* biases  [[buffer(2)]],
    device const half*   x       [[buffer(3)]],
    device half*         y       [[buffer(4)]],
    constant uint&       rows    [[buffer(5)]],
    constant uint&       columns [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;

    const uint bits = affine_quant_bits();
    const uint groups_per_row = columns / kAffineGroupSize;
    const uint words_per_row = (columns * bits) / 32u;
    device const uint* row_weights = weights + row * words_per_row;
    device const bfloat* row_scales = scales + row * groups_per_row;
    device const bfloat* row_biases = biases + row * groups_per_row;

    float accumulator = 0.0f;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint first = group * kAffineGroupSize + lane * 2u;
        const float x0 = float(x[first]);
        const float x1 = float(x[first + 1u]);
        const float q0 = float(affine_quant_value(row_weights, first, bits));
        const float q1 = float(affine_quant_value(row_weights, first + 1u, bits));
        accumulator = fma(float(row_scales[group]), q0 * x0 + q1 * x1,
                          accumulator);
        accumulator = fma(float(row_biases[group]), x0 + x1, accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[row] = half(accumulator);
}

kernel void affine_quant_gemv2_simd(
    device const uint*   weights [[buffer(0)]],
    device const bfloat* scales  [[buffer(1)]],
    device const bfloat* biases  [[buffer(2)]],
    device const half*   x       [[buffer(3)]],
    device half*         y       [[buffer(4)]],
    constant uint&       rows    [[buffer(5)]],
    constant uint&       columns [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;
    const uint bits = affine_quant_bits();
    const uint groups_per_row = columns / kAffineGroupSize;
    const uint words_per_row = (columns * bits) / 32u;
    device const uint* row_weights = weights + row * words_per_row;
    device const bfloat* row_scales = scales + row * groups_per_row;
    device const bfloat* row_biases = biases + row * groups_per_row;
    device const half* x1 = x + columns;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint first = group * kAffineGroupSize + lane * 2u;
        const float q0 = float(affine_quant_value(row_weights, first, bits));
        const float q1 = float(affine_quant_value(row_weights, first + 1u, bits));
        const float x00 = float(x[first]), x01 = float(x[first + 1u]);
        const float x10 = float(x1[first]), x11 = float(x1[first + 1u]);
        const float scale = float(row_scales[group]);
        const float bias = float(row_biases[group]);
        acc0 = fma(scale, q0 * x00 + q1 * x01, acc0);
        acc0 = fma(bias, x00 + x01, acc0);
        acc1 = fma(scale, q0 * x10 + q1 * x11, acc1);
        acc1 = fma(bias, x10 + x11, acc1);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        y[row] = half(acc0);
        y[rows + row] = half(acc1);
    }
}

// Four-row variant: the same read of each output weight row serves four
// activation rows. Used by the batched decode's head at width 4, where two
// two-row launches would read the 254 MB head twice. Per row the accumulation
// order is the two-row kernel's (that kernel's head is the byte-identical
// reference the width-2 batch already ships), so row r's value cannot depend on
// how many other rows ride along.
kernel void affine_quant_gemv4_simd(
    device const uint*   weights [[buffer(0)]],
    device const bfloat* scales  [[buffer(1)]],
    device const bfloat* biases  [[buffer(2)]],
    device const half*   x       [[buffer(3)]],
    device half*         y       [[buffer(4)]],
    constant uint&       rows    [[buffer(5)]],
    constant uint&       columns [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;
    const uint bits = affine_quant_bits();
    const uint groups_per_row = columns / kAffineGroupSize;
    const uint words_per_row = (columns * bits) / 32u;
    device const uint* row_weights = weights + row * words_per_row;
    device const bfloat* row_scales = scales + row * groups_per_row;
    device const bfloat* row_biases = biases + row * groups_per_row;
    device const half* x1 = x + columns;
    device const half* x2 = x + 2u * columns;
    device const half* x3 = x + 3u * columns;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint first = group * kAffineGroupSize + lane * 2u;
        const float q0 = float(affine_quant_value(row_weights, first, bits));
        const float q1 = float(affine_quant_value(row_weights, first + 1u, bits));
        const float x00 = float(x[first]), x01 = float(x[first + 1u]);
        const float x10 = float(x1[first]), x11 = float(x1[first + 1u]);
        const float x20 = float(x2[first]), x21 = float(x2[first + 1u]);
        const float x30 = float(x3[first]), x31 = float(x3[first + 1u]);
        const float scale = float(row_scales[group]);
        const float bias = float(row_biases[group]);
        acc0 = fma(scale, q0 * x00 + q1 * x01, acc0);
        acc0 = fma(bias, x00 + x01, acc0);
        acc1 = fma(scale, q0 * x10 + q1 * x11, acc1);
        acc1 = fma(bias, x10 + x11, acc1);
        acc2 = fma(scale, q0 * x20 + q1 * x21, acc2);
        acc2 = fma(bias, x20 + x21, acc2);
        acc3 = fma(scale, q0 * x30 + q1 * x31, acc3);
        acc3 = fma(bias, x30 + x31, acc3);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[row] = half(acc0);
        y[rows + row] = half(acc1);
        y[2u * rows + row] = half(acc2);
        y[3u * rows + row] = half(acc3);
    }
}

kernel void affine_quant_embedding_lookup(
    device const uint*   table     [[buffer(0)]],
    device const bfloat* scales    [[buffer(1)]],
    device const bfloat* biases    [[buffer(2)]],
    device half*         output    [[buffer(3)]],
    constant uint&       token_id  [[buffer(4)]],
    constant uint&       dimension [[buffer(5)]],
    constant float&      out_scale [[buffer(6)]],
    constant uint&       vocab     [[buffer(7)]],
    uint element [[thread_position_in_grid]]
) {
    if (element >= dimension) return;
    // OOB token guard: never index past the vocab rows; write zeros so the
    // pipeline stays well-defined for out-of-range token ids.
    if (token_id >= vocab) {
        output[element] = half(0.0f);
        return;
    }
    const uint bits = affine_quant_bits();
    const uint groups_per_row = dimension / kAffineGroupSize;
    const uint words_per_row = (dimension * bits) / 32u;
    device const uint* row_weights = table + token_id * words_per_row;
    device const bfloat* row_scales = scales + token_id * groups_per_row;
    device const bfloat* row_biases = biases + token_id * groups_per_row;
    const uint group = element / kAffineGroupSize;
    const float value = float(affine_quant_value(row_weights, element, bits))
        * float(row_scales[group]) + float(row_biases[group]);
    output[element] = half(value * out_scale);
}

// Affine GEMM: one row of weights against a whole chunk of tokens.
//
// The single-token GEMV reads a row's weights for every token; this keeps the
// row in registers and loops over the chunk, so the weight bytes are read once
// per chunk instead of once per token. The lane mapping matches the coalesced
// GEMV: a 64-element group is covered by `64/elementsPerWord` lanes, each
// holding one 32-bit word and the group's scale and bias.
constant constexpr uint kAffineGEMMRowsPerThreadgroup = 8;
constant constexpr uint kAffineGEMMMaxIterations = 32;

kernel void affine_quant_gemm_simd(
    device const uint*   weights [[buffer(0)]],
    device const bfloat* scales  [[buffer(1)]],
    device const bfloat* biases  [[buffer(2)]],
    device const half*   x       [[buffer(3)]],   // [tokens][columns]
    device half*         y       [[buffer(4)]],   // [tokens][rows]
    constant uint&       rows    [[buffer(5)]],
    constant uint&       columns [[buffer(6)]],
    constant uint&       tokens  [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kAffineGEMMRowsPerThreadgroup + sg_idx;
    if (row >= rows || tokens == 0u) return;

    const uint bits = affine_quant_bits();
    const uint elements = 32u / bits;                              // 8 or 4
    const uint words_per_group = kAffineGroupSize / elements;      // 8 or 16
    const uint groups_per_iteration = 32u / words_per_group;       // 4 or 2
    const uint groups = columns / kAffineGroupSize;
    const uint iterations = groups / groups_per_iteration;
    if (iterations == 0u || iterations > kAffineGEMMMaxIterations) return;
    const uint mask = (1u << bits) - 1u;
    const uint lane_group = lane / words_per_group;
    const uint lane_word = lane - lane_group * words_per_group;

    const uint words_per_row = (columns * bits) / 32u;
    device const uint* row_weights = weights + row * words_per_row;
    device const bfloat* row_scales = scales + row * groups;
    device const bfloat* row_biases = biases + row * groups;

    // The row's words, scales and biases stay in registers for the whole chunk.
    uint held_words[kAffineGEMMMaxIterations];
    float held_scales[kAffineGEMMMaxIterations];
    float held_biases[kAffineGEMMMaxIterations];
    for (uint i = 0; i < iterations; ++i) {
        const uint group = i * groups_per_iteration + lane_group;
        held_words[i] = row_weights[group * words_per_group + lane_word];
        held_scales[i] = float(row_scales[group]);
        held_biases[i] = float(row_biases[group]);
    }

    for (uint token = 0; token < tokens; ++token) {
        device const half* xrow = x + token * columns;
        float accumulator = 0.0f;
        for (uint i = 0; i < iterations; ++i) {
            const uint group = i * groups_per_iteration + lane_group;
            const uint first =
                group * kAffineGroupSize + lane_word * elements;
            const uint word = held_words[i];
            float dot = 0.0f;
            float sum = 0.0f;
            for (uint e = 0; e < elements; ++e) {
                const float xv = float(xrow[first + e]);
                const float q = float((word >> (e * bits)) & mask);
                dot = fma(q, xv, dot);
                sum += xv;
            }
            accumulator = fma(held_scales[i], dot, accumulator);
            accumulator = fma(held_biases[i], sum, accumulator);
        }
        accumulator = simd_sum(accumulator);
        if (lane == 0u) {
            y[token * rows + row] = half(accumulator);
        }
    }
}
