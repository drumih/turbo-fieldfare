#include <metal_stdlib>
using namespace metal;

/// Quantize one affine group per threadgroup. The output token row is:
/// packed values, FP16 scales, FP16 biases. Scale/bias arrays contain one
/// entry per group and are deliberately row-local for append-only KV writes.
kernel void kv_cache_quantize_affine(
    device const half* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant uint& source_stride [[buffer(2)]],
    constant uint& destination_stride [[buffer(3)]],
    constant uint& values_bytes [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    constant uint& bits [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint3 group_token [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint lid = thread_position.x;
    threadgroup float group_min_parts[2];
    threadgroup float group_max_parts[2];
    threadgroup float group_scale;
    threadgroup float group_bias;

    const uint group = group_token.x;
    const uint token = group_token.y;
    const uint flat = group * group_size + lid;
    const bool active = flat < element_count;
    const float value = active ? float(source[token * source_stride + flat]) : 0.0f;
    const float local_min = simd_min(active ? value : INFINITY);
    const float local_max = simd_max(active ? value : -INFINITY);
    if (lane == 0) {
        group_min_parts[simd_group] = local_min;
        group_max_parts[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        const float minimum = min(group_min_parts[0], group_min_parts[1]);
        const float maximum = max(group_max_parts[0], group_max_parts[1]);
        const float levels = bits == 4u ? 15.0f : 255.0f;
        group_bias = minimum;
        group_scale = maximum > minimum ? (maximum - minimum) / levels : 0.0f;

        device uchar* row = destination + token * destination_stride;
        const uint groups = (element_count + group_size - 1u) / group_size;
        device half* scales = reinterpret_cast<device half*>(row + values_bytes);
        device half* biases = scales + groups;
        scales[group] = half(group_scale);
        biases[group] = half(group_bias);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uchar* row = destination + token * destination_stride;
    const float quantized = group_scale > 0.0f
        ? round(clamp((value - group_bias) / group_scale,
                      0.0f, bits == 4u ? 15.0f : 255.0f))
        : 0.0f;
    if (bits == 8u) {
        if (active) { row[flat] = uchar(quantized); }
    } else if ((lid & 1u) == 0u && flat < element_count) {
        const uchar low = uchar(quantized);
        uchar high = 0u;
        if (flat + 1u < element_count) {
            const float other = float(source[token * source_stride + flat + 1u]);
            high = uchar(group_scale > 0.0f
                ? round(clamp((other - group_bias) / group_scale, 0.0f, 15.0f))
                : 0.0f);
        }
        row[flat / 2u] = low | (high << 4u);
    }
}
