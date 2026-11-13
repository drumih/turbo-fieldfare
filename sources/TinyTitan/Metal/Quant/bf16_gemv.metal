#include <metal_stdlib>
using namespace metal;

// Unquantized GEMV over a bf16 weight matrix.
//
// The quantized kernels beside this one exist because weights are normally
// stored packed with per-group scales and biases. A tensor kept at the
// checkpoint's own bf16 has none of that: no packing to unpick, no group to
// rescale, one multiply per element. It is here so a build can leave chosen
// tensors unquantized -- the 8-bit build promotes the families whose error
// does not average away -- without a separate code path at the call site.
//
// Layout matches the quantized kernels exactly: one simdgroup per row, each
// lane taking two adjacent columns, so a caller can swap between them on the
// tensor's dtype and change nothing else.
kernel void bf16_gemv_simd(
    device const bfloat* weights [[buffer(0)]],
    device const half*   x       [[buffer(1)]],
    device half*         y       [[buffer(2)]],
    constant uint&       rows    [[buffer(3)]],
    constant uint&       columns [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint lanes = 32;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;

    device const bfloat* row_weights = weights + row * columns;
    float accumulator = 0.0f;
    // Columns are a multiple of 64 for every tensor this serves, which is the
    // same guarantee the affine kernels rely on for their group stride.
    for (uint base = 0; base < columns; base += lanes * 2u) {
        const uint first = base + lane * 2u;
        accumulator = fma(float(row_weights[first]), float(x[first]),
                          accumulator);
        accumulator = fma(float(row_weights[first + 1u]), float(x[first + 1u]),
                          accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[row] = half(accumulator);
}

// The same GEMV, but the logits stay float.
//
// A router needs full float logits: K2-Horizon's selection applies sigmoid and
// a bias, and rounding the logits to half first would change the top-k order
// near a tie -- a different expert set, not a different last bit. Every other
// caller wants half activations and keeps `bf16_gemv_simd`.
kernel void bf16_gemv_float_simd(
    device const bfloat* weights [[buffer(0)]],
    device const half*   x       [[buffer(1)]],
    device float*        y       [[buffer(2)]],
    constant uint&       rows    [[buffer(3)]],
    constant uint&       columns [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint lanes = 32;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;

    device const bfloat* row_weights = weights + row * columns;
    float accumulator = 0.0f;
    for (uint base = 0; base < columns; base += lanes * 2u) {
        const uint first = base + lane * 2u;
        accumulator = fma(float(row_weights[first]), float(x[first]),
                          accumulator);
        accumulator = fma(float(row_weights[first + 1u]), float(x[first + 1u]),
                          accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[row] = accumulator;
}

// The same arithmetic for a prompt: one threadgroup per (row block, token), so
// a whole prefill chunk costs one dispatch instead of one per token per
// projection.
//
// This amortises launches, not bytes. A promoted projection is four times the
// traffic of the same projection at 4 bits (bf16 against packed nibbles), and
// prefill is bandwidth-bound, so the win here is the CPU-side encode and the
// round trip -- measured on a 1500-token prompt at about 1% of the run. It
// exists because the per-row path issued ~180k dispatches for that prompt, and
// dispatch cost is the part a caller can remove without changing the weights.
kernel void bf16_gemv_multirow_simd(
    device const bfloat* weights [[buffer(0)]],
    device const half*   x       [[buffer(1)]],
    device half*         y       [[buffer(2)]],
    constant uint&       rows    [[buffer(3)]],
    constant uint&       columns [[buffer(4)]],
    constant uint&       x_stride [[buffer(5)]],
    constant uint&       y_stride [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint lanes = 32;
    const uint row = tg.x * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;

    device const bfloat* row_weights = weights + row * columns;
    device const half* token_x = x + tg.y * x_stride;
    float accumulator = 0.0f;
    for (uint base = 0; base < columns; base += lanes * 2u) {
        const uint first = base + lane * 2u;
        accumulator = fma(float(row_weights[first]), float(token_x[first]),
                          accumulator);
        accumulator = fma(float(row_weights[first + 1u]), float(token_x[first + 1u]),
                          accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[tg.y * y_stride + row] = half(accumulator);
}

// One embedding row from a table kept at the checkpoint's bf16: the bf16
// counterpart of `embed_lookup_int4`, with the same buffers minus the scales
// and biases, the same fused output scale and the same out-of-range guard.
// Each element is one widen, one multiply and one rounding to half.
kernel void embed_lookup_bf16(
    device const bfloat* table     [[buffer(0)]],   // [V, D] BF16
    device half*         out       [[buffer(1)]],   // [D] FP16
    constant uint&       token_id  [[buffer(2)]],
    constant uint&       D         [[buffer(3)]],
    constant float&      out_scale [[buffer(4)]],   // pass 1.0 to disable
    constant uint&       vocab     [[buffer(5)]],   // [V] row count
    uint                 gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    if (token_id >= vocab) {
        out[gid] = half(0.0f);
        return;
    }
    out[gid] = half(float(table[ulong(token_id) * ulong(D) + gid]) * out_scale);
}

// The prompt form: one row per token of a chunk, `[T, D]` token-major, as
// `prefill_embed_lookup_affine_block` writes it. Per element it is exactly
// `embed_lookup_bf16`, so a prefilled and a decoded embedding agree.
kernel void prefill_embed_lookup_bf16_block(
    device const bfloat* table     [[buffer(0)]],
    device const uint*   tokens    [[buffer(1)]],
    device half*         out       [[buffer(2)]],
    constant uint&       T         [[buffer(3)]],
    constant uint&       D         [[buffer(4)]],
    constant float&      out_scale [[buffer(5)]],
    constant uint&       vocab     [[buffer(6)]],
    uint2                gid       [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;
    const uint token = tokens[t];
    if (token >= vocab) {
        out[t * D + d] = half(0.0f);
        return;
    }
    out[t * D + d] = half(float(table[ulong(token) * ulong(D) + d]) * out_scale);
}
