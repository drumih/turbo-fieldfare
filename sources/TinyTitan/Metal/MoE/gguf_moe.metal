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

// The embedding gather decodes Q6_K rows to bf16; the residual is half.
kernel void gguf_bf16_to_half(
    device const bfloat* src [[buffer(0)]],
    device half*         dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = half(float(src[gid]));
}

// Router logits in F32. The checkpoint stores `ffn_gate_inp` as F32 and the
// top-10 of 512 experts is decided by probability gaps of ~1e-4, so neither
// the weights (bf16 would cost ~0.4%) nor the logits (half) are rounded.
// One SIMD group per row; `cols` is a multiple of 4.
kernel void gguf_f32_gemv_rows(
    device const float* w    [[buffer(0)]],
    device const half*  x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      cols [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= rows) return;
    device const float4* w4 = (device const float4*)(w + row * cols);
    device const half4*  x4 = (device const half4*)x;
    float acc = 0.0f;
    for (uint i = lane; i < cols / 4u; i += 32u) {
        acc += dot(w4[i], float4(x4[i]));
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}
