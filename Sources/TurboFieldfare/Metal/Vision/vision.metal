#include <metal_stdlib>
using namespace metal;

constant constexpr uint kVisionMaxSimdGroups = 8;

static inline float vision_row_inv(
    device const half* x,
    uint width,
    float eps,
    uint lid,
    uint threads,
    uint lane,
    uint simdgroup,
    uint simdgroups,
    threadgroup float* partial
) {
    float sum = 0.0f;
    for (uint column = lid; column < width; column += threads) {
        const float value = float(x[column]);
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float value = lane < simdgroups ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0u) partial[0] = rsqrt(value / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

kernel void vision_bf16_to_f16(
    device const bfloat* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) destination[index] = half(float(source[index]));
}

kernel void vision_add_axial_position_embedding(
    device half* hidden [[buffer(0)]],
    device const bfloat* positionRows [[buffer(1)]],
    constant uint& patchCount [[buffer(2)]],
    constant uint& patchColumns [[buffer(3)]],
    constant uint& hiddenSize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint column = gid.x;
    const uint patch = gid.y;
    if (patch >= patchCount || column >= hiddenSize) return;
    const uint x = patch % patchColumns;
    const uint y = patch / patchColumns;
    const float xPosition = float(positionRows[x * hiddenSize + column]);
    const float yPosition = float(positionRows[(patchColumns + y) * hiddenSize + column]);
    const uint index = patch * hiddenSize + column;
    hidden[index] = half(float(hidden[index]) + xPosition + yPosition);
}

kernel void vision_rmsnorm_bf16_rows(
    device const half* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rowCount [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    if (row >= rowCount) return;
    threadgroup float partial[kVisionMaxSimdGroups];
    device const half* sourceRow = input + row * width;
    device half* outputRow = output + row * width;
    const float inv = vision_row_inv(sourceRow, width, eps, lid, threads,
                                     lane, simdgroup, simdgroups, partial);
    for (uint column = lid; column < width; column += threads) {
        outputRow[column] = half(float(sourceRow[column]) * inv * float(weight[column]));
    }
}

kernel void vision_rmsnorm_no_scale_rows(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rowCount [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    if (row >= rowCount) return;
    threadgroup float partial[kVisionMaxSimdGroups];
    device const half* sourceRow = input + row * width;
    device half* outputRow = output + row * width;
    const float inv = vision_row_inv(sourceRow, width, eps, lid, threads,
                                     lane, simdgroup, simdgroups, partial);
    for (uint column = lid; column < width; column += threads) {
        outputRow[column] = half(float(sourceRow[column]) * inv);
    }
}

kernel void vision_rmsnorm_bf16_heads(
    device const half* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& tokenCount [[buffer(3)]],
    constant uint& headCount [[buffer(4)]],
    constant uint& headSize [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    if (group.x >= tokenCount || group.y >= headCount) return;
    threadgroup float partial[kVisionMaxSimdGroups];
    const uint offset = (group.x * headCount + group.y) * headSize;
    device const half* sourceHead = input + offset;
    device half* outputHead = output + offset;
    const float inv = vision_row_inv(sourceHead, headSize, eps, lid.x, threads.x,
                                     lane, simdgroup, simdgroups, partial);
    for (uint column = lid.x; column < headSize; column += threads.x) {
        outputHead[column] = half(float(sourceHead[column]) * inv * float(weight[column]));
    }
}

kernel void vision_rmsnorm_no_scale_heads(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& tokenCount [[buffer(2)]],
    constant uint& headCount [[buffer(3)]],
    constant uint& headSize [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    if (group.x >= tokenCount || group.y >= headCount) return;
    threadgroup float partial[kVisionMaxSimdGroups];
    const uint offset = (group.x * headCount + group.y) * headSize;
    device const half* sourceHead = input + offset;
    device half* outputHead = output + offset;
    const float inv = vision_row_inv(sourceHead, headSize, eps, lid.x, threads.x,
                                     lane, simdgroup, simdgroups, partial);
    for (uint column = lid.x; column < headSize; column += threads.x) {
        outputHead[column] = half(float(sourceHead[column]) * inv);
    }
}

kernel void vision_rope_2d_qk(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant uint& tokenCount [[buffer(2)]],
    constant uint& patchColumns [[buffer(3)]],
    constant uint& headCount [[buffer(4)]],
    constant uint& headSize [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint pairsPerHead = headSize / 2u;
    const uint totalPairs = tokenCount * headCount * pairsPerHead;
    if (index >= totalPairs) return;
    const uint pair = index % pairsPerHead;
    const uint headAndToken = index / pairsPerHead;
    const uint head = headAndToken % headCount;
    const uint token = headAndToken / headCount;
    const uint pairsPerAxis = pairsPerHead / 2u;
    const uint axisPair = pair % pairsPerAxis;
    const uint coordinate = pair < pairsPerAxis
        ? token % patchColumns
        : token / patchColumns;
    const float inverseFrequency = pow(theta, -float(axisPair) / float(pairsPerAxis));
    const float angle = float(coordinate) * inverseFrequency;
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const uint axisBase = pair < pairsPerAxis ? 0u : headSize / 2u;
    // Gemma's rotate_half pairs the first and second halves of each
    // 36-channel spatial partition: (0,18), (1,19), ... — not adjacent
    // channels.
    const uint firstChannel = axisBase + axisPair;
    const uint secondChannel = firstChannel + pairsPerAxis;
    const uint baseOffset = (token * headCount + head) * headSize;
    const uint firstOffset = baseOffset + firstChannel;
    const uint secondOffset = baseOffset + secondChannel;

    const float q0 = float(query[firstOffset]);
    const float q1 = float(query[secondOffset]);
    query[firstOffset] = half(q0 * cosine - q1 * sine);
    query[secondOffset] = half(q1 * cosine + q0 * sine);
    const float k0 = float(key[firstOffset]);
    const float k1 = float(key[secondOffset]);
    key[firstOffset] = half(k0 * cosine - k1 * sine);
    key[secondOffset] = half(k1 * cosine + k0 * sine);
}

static inline float vision_attention_sum(
    float value,
    uint lane,
    uint simdgroup,
    uint simdgroups,
    threadgroup float* partial) {
    value = simd_sum(value);
    if (lane == 0u) partial[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float merged = lane < simdgroups ? partial[lane] : 0.0f;
        merged = simd_sum(merged);
        if (lane == 0u) partial[0] = merged;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

/// Full, noncausal attention with online FP32 softmax. The kernel never
/// materializes a `[tokens,tokens]` score matrix.
kernel void vision_attention_noncausal_online(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& tokenCount [[buffer(4)]],
    constant uint& headCount [[buffer(5)]],
    constant uint& headSize [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= tokenCount || head >= headCount) return;
    threadgroup float partial[kVisionMaxSimdGroups];
    const bool ownsChannel = lid.x < headSize;
    const uint queryOffset = (token * headCount + head) * headSize;
    const float queryValue = ownsChannel ? float(query[queryOffset + lid.x]) : 0.0f;
    float rowMaximum = -INFINITY;
    float rowSum = 0.0f;
    float accumulator = 0.0f;
    for (uint sourceToken = 0; sourceToken < tokenCount; ++sourceToken) {
        const uint sourceOffset = (sourceToken * headCount + head) * headSize;
        const float keyValue = ownsChannel ? float(key[sourceOffset + lid.x]) : 0.0f;
        const float score = vision_attention_sum(
            queryValue * keyValue,
            lane, simdgroup, simdgroups, partial);
        const float newMaximum = max(rowMaximum, score);
        const float previousScale = rowSum > 0.0f ? fast::exp(rowMaximum - newMaximum) : 0.0f;
        const float sourceScale = fast::exp(score - newMaximum);
        if (ownsChannel) {
            accumulator = fma(sourceScale, float(value[sourceOffset + lid.x]),
                              accumulator * previousScale);
        }
        rowSum = rowSum * previousScale + sourceScale;
        rowMaximum = newMaximum;
    }
    if (ownsChannel) {
        output[queryOffset + lid.x] = rowSum > 0.0f
            ? half(accumulator / rowSum)
            : half(0.0f);
    }
}

kernel void vision_gelu_tanh_multiply(
    device half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = float(gate[index]);
    const float x3 = x * x * x;
    const float activated = 0.5f * x
        * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
    gate[index] = half(activated * float(up[index]));
}

kernel void vision_residual_add(
    device half* hidden [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) hidden[index] = half(float(hidden[index]) + float(branch[index]));
}

// BF16 can represent much larger finite values than FP16.  Keep the FP16
// execution path finite at operator boundaries; the next RMS norm restores
// the working scale.  NaNs cannot carry useful model information, so map them
// to zero rather than allowing one value to poison an entire attention row.
kernel void vision_sanitize_finite(
    device half* values [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float value = float(values[index]);
    constexpr float limit = 65504.0f;
    if (isnan(value)) {
        values[index] = half(0.0f);
    } else if (isinf(value)) {
        values[index] = half(value < 0.0f ? -limit : limit);
    }
}

kernel void vision_pool_and_standardize(
    device const half* hidden [[buffer(0)]],
    device const bfloat* standardBias [[buffer(1)]],
    device const bfloat* standardScale [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& patchColumns [[buffer(4)]],
    constant uint& patchRows [[buffer(5)]],
    constant uint& hiddenSize [[buffer(6)]],
    uint softToken [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const uint softColumns = patchColumns / 3u;
    const uint softRows = patchRows / 3u;
    if (softToken >= softColumns * softRows) return;
    const uint softX = softToken % softColumns;
    const uint softY = softToken / softColumns;
    for (uint column = lid; column < hiddenSize; column += threads) {
        float sum = 0.0f;
        for (uint dy = 0; dy < 3u; ++dy) {
            for (uint dx = 0; dx < 3u; ++dx) {
                const uint patchX = softX * 3u + dx;
                const uint patchY = softY * 3u + dy;
                const uint patch = patchY * patchColumns + patchX;
                sum += float(hidden[patch * hiddenSize + column]);
            }
        }
        const float pooled = (sum / 9.0f) * sqrt(float(hiddenSize));
        const float standardized = (pooled - float(standardBias[column]))
            * float(standardScale[column]);
        output[softToken * hiddenSize + column] = half(standardized);
    }
}
