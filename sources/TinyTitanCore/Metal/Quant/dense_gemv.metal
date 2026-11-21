#include <metal_stdlib>
using namespace metal;

// BF16 weight matrix times FP32 activation vector, FP32 output. This keeps the
// attention projections in float math on both sides of the multiply.
kernel void bf16_gemv_floatx(
    device const bfloat* weights [[buffer(0)]],
    device const float*  x       [[buffer(1)]],
    device float*        y       [[buffer(2)]],
    constant uint&       rows    [[buffer(3)]],
    constant uint&       columns [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    constexpr uint rows_per_threadgroup = 8;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;
    device const bfloat* row_weights = weights + row * columns;
    float accumulator = 0.0f;
    for (uint base = 0; base < columns; base += 64u) {
        const uint first = base + lane * 2u;
        accumulator = fma(float(row_weights[first]), x[first], accumulator);
        accumulator = fma(float(row_weights[first + 1u]), x[first + 1u], accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[row] = accumulator;
}
