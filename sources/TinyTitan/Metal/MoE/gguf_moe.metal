#include <metal_stdlib>
using namespace metal;

// The two elementwise steps the as-is routed-MoE stage needs between fused
// GEMVs. Both run in half because that is what the GEMVs write and read.

kernel void gguf_silu_mul_half(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       act  [[buffer(2)]],
    constant uint&     n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const float g = float(gate[gid]);
    const float u = float(up[gid]);
    act[gid] = half(g / (1.0f + exp(-g)) * u);
}

kernel void gguf_accumulate_scaled_half(
    device half*       y   [[buffer(0)]],
    device const half* row [[buffer(1)]],
    constant float&    w   [[buffer(2)]],
    constant uint&     n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    y[gid] = half(float(y[gid]) + w * float(row[gid]));
}

// Shared-expert output with its sigmoid scalar gate: y += sigmoid(gate[0]) * row.
kernel void gguf_accumulate_sigmoid_half(
    device half*       y      [[buffer(0)]],
    device const half* row    [[buffer(1)]],
    device const half* scalar [[buffer(2)]],
    constant uint&     n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const float g = float(scalar[0]);
    const float s = 1.0f / (1.0f + exp(-g));
    y[gid] = half(float(y[gid]) + s * float(row[gid]));
}
