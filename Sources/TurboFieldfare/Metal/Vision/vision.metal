#include <metal_stdlib>
using namespace metal;

constant constexpr uint kVisionHeadDim = 72;
constant constexpr uint kVisionPaddedHeadDim = 80;
constant constexpr uint kVisionMaxQueryTile = 16;
constant constexpr uint kVisionKeyTile = 32;
constant constexpr uint kVisionThreads = 128;

kernel void vision_pad_heads_72_to_80(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* paddedQ [[buffer(3)]],
    device bfloat* paddedK [[buffer(4)]],
    device bfloat* paddedV [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * heads * 80u) return;
    const uint dimension = index % 80u;
    const uint vector = index / 80u;
    const uint row = vector % rows;
    const uint head = vector / rows;
    const uint source = (row * heads + head) * 72u + dimension;
    paddedQ[index] = dimension < 72u ? q[source] : bfloat(0.0f);
    paddedK[index] = dimension < 72u ? k[source] : bfloat(0.0f);
    paddedV[index] = dimension < 72u ? v[source] : bfloat(0.0f);
}

kernel void vision_unpad_heads_80_to_72(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * heads * 72u) return;
    const uint dimension = index % 72u;
    const uint vector = index / 72u;
    const uint row = vector / heads;
    const uint head = vector % heads;
    output[index] = input[(head * rows + row) * 80u + dimension];
}

template <uint ComputeDim, uint QueryTile>
inline void vision_attention_online_impl(
    device const bfloat* q,
    device const bfloat* k,
    device const bfloat* v,
    device bfloat* output,
    constant uint& sequenceLength,
    constant uint& numHeads,
    uint2 group,
    uint tid,
    threadgroup bfloat* qTile,
    threadgroup bfloat* kTile,
    threadgroup bfloat* vTile,
    threadgroup float* probabilities,
    threadgroup float* accumulator,
    threadgroup float* oldScale,
    threadgroup float* rowSum,
    threadgroup float* rowMax) {

    const uint head = group.y;
    const uint queryBase = group.x * QueryTile;
    if (head >= numHeads) return;

    for (uint index = tid; index < QueryTile * kVisionHeadDim;
         index += kVisionThreads) {
        const uint localQuery = index / kVisionHeadDim;
        const uint dimension = index % kVisionHeadDim;
        const uint query = queryBase + localQuery;
        qTile[index] = query < sequenceLength
            ? q[(query * numHeads + head) * kVisionHeadDim + dimension]
            : bfloat(0.0f);
    }
    for (uint index = tid; index < QueryTile * kVisionPaddedHeadDim;
         index += kVisionThreads) {
        accumulator[index] = 0.0f;
    }
    if (tid < QueryTile) {
        rowMax[tid] = -INFINITY;
        rowSum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint keyBase = 0; keyBase < sequenceLength; keyBase += kVisionKeyTile) {
        for (uint index = tid; index < kVisionKeyTile * kVisionHeadDim;
             index += kVisionThreads) {
            const uint localKey = index / kVisionHeadDim;
            const uint dimension = index % kVisionHeadDim;
            const uint keyIndex = keyBase + localKey;
            const uint source = (keyIndex * numHeads + head) * kVisionHeadDim + dimension;
            kTile[index] = keyIndex < sequenceLength ? k[source] : bfloat(0.0f);
            vTile[index] = keyIndex < sequenceLength ? v[source] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint pair = tid; pair < QueryTile * kVisionKeyTile;
             pair += kVisionThreads) {
            const uint localQuery = pair / kVisionKeyTile;
            const uint localKey = pair % kVisionKeyTile;
            const uint query = queryBase + localQuery;
            const uint keyIndex = keyBase + localKey;
            float score = -INFINITY;
            if (query < sequenceLength && keyIndex < sequenceLength) {
                score = 0.0f;
                for (uint dimension = 0; dimension < ComputeDim; ++dimension) {
                    const float qValue = dimension < kVisionHeadDim
                        ? float(qTile[localQuery * kVisionHeadDim + dimension]) : 0.0f;
                    const float kValue = dimension < kVisionHeadDim
                        ? float(kTile[localKey * kVisionHeadDim + dimension]) : 0.0f;
                    score = fma(qValue, kValue, score);
                }
            }
            probabilities[pair] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < QueryTile) {
            float tileMax = -INFINITY;
            for (uint localKey = 0; localKey < kVisionKeyTile; ++localKey) {
                tileMax = max(tileMax,
                              probabilities[tid * kVisionKeyTile + localKey]);
            }
            const float nextMax = max(rowMax[tid], tileMax);
            const float scale = rowMax[tid] == -INFINITY
                ? 0.0f : exp(rowMax[tid] - nextMax);
            float tileSum = 0.0f;
            for (uint localKey = 0; localKey < kVisionKeyTile; ++localKey) {
                const uint keyIndex = keyBase + localKey;
                const float probability = keyIndex < sequenceLength
                    ? exp(probabilities[tid * kVisionKeyTile + localKey] - nextMax)
                    : 0.0f;
                probabilities[tid * kVisionKeyTile + localKey] = probability;
                tileSum += probability;
            }
            oldScale[tid] = scale;
            rowSum[tid] = rowSum[tid] * scale + tileSum;
            rowMax[tid] = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = tid; index < QueryTile * kVisionPaddedHeadDim;
             index += kVisionThreads) {
            const uint localQuery = index / kVisionPaddedHeadDim;
            const uint dimension = index % kVisionPaddedHeadDim;
            float value = accumulator[index] * oldScale[localQuery];
            if (dimension < kVisionHeadDim) {
                for (uint localKey = 0; localKey < kVisionKeyTile; ++localKey) {
                    value = fma(
                        probabilities[localQuery * kVisionKeyTile + localKey],
                        float(vTile[localKey * kVisionHeadDim + dimension]), value);
                }
            }
            accumulator[index] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = tid; index < QueryTile * kVisionHeadDim;
         index += kVisionThreads) {
        const uint localQuery = index / kVisionHeadDim;
        const uint dimension = index % kVisionHeadDim;
        const uint query = queryBase + localQuery;
        if (query < sequenceLength) {
            output[(query * numHeads + head) * kVisionHeadDim + dimension] =
                bfloat(accumulator[localQuery * kVisionPaddedHeadDim + dimension]
                       / rowSum[localQuery]);
        }
    }
}

kernel void vision_attention_online_72(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadPosition [[thread_position_in_threadgroup]]) {
    threadgroup bfloat qTile[kVisionMaxQueryTile * kVisionHeadDim];
    threadgroup bfloat kTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup bfloat vTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup float probabilities[kVisionMaxQueryTile * kVisionKeyTile];
    threadgroup float accumulator[kVisionMaxQueryTile * kVisionPaddedHeadDim];
    threadgroup float oldScale[kVisionMaxQueryTile];
    threadgroup float rowSum[kVisionMaxQueryTile];
    threadgroup float rowMax[kVisionMaxQueryTile];
    vision_attention_online_impl<kVisionHeadDim, 8>(
        q, k, v, output, sequenceLength, numHeads, group.xy, threadPosition.x,
        qTile, kTile, vTile, probabilities, accumulator,
        oldScale, rowSum, rowMax);
}

kernel void vision_attention_online_80(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadPosition [[thread_position_in_threadgroup]]) {
    threadgroup bfloat qTile[kVisionMaxQueryTile * kVisionHeadDim];
    threadgroup bfloat kTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup bfloat vTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup float probabilities[kVisionMaxQueryTile * kVisionKeyTile];
    threadgroup float accumulator[kVisionMaxQueryTile * kVisionPaddedHeadDim];
    threadgroup float oldScale[kVisionMaxQueryTile];
    threadgroup float rowSum[kVisionMaxQueryTile];
    threadgroup float rowMax[kVisionMaxQueryTile];
    vision_attention_online_impl<kVisionPaddedHeadDim, 8>(
        q, k, v, output, sequenceLength, numHeads, group.xy, threadPosition.x,
        qTile, kTile, vTile, probabilities, accumulator,
        oldScale, rowSum, rowMax);
}

kernel void vision_attention_online_72_q16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadPosition [[thread_position_in_threadgroup]]) {
    threadgroup bfloat qTile[kVisionMaxQueryTile * kVisionHeadDim];
    threadgroup bfloat kTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup bfloat vTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup float probabilities[kVisionMaxQueryTile * kVisionKeyTile];
    threadgroup float accumulator[kVisionMaxQueryTile * kVisionPaddedHeadDim];
    threadgroup float oldScale[kVisionMaxQueryTile];
    threadgroup float rowSum[kVisionMaxQueryTile];
    threadgroup float rowMax[kVisionMaxQueryTile];
    vision_attention_online_impl<kVisionHeadDim, 16>(
        q, k, v, output, sequenceLength, numHeads, group.xy, threadPosition.x,
        qTile, kTile, vTile, probabilities, accumulator,
        oldScale, rowSum, rowMax);
}

kernel void vision_normalize_patches(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& realRows [[buffer(2)]],
    constant uint& paddedRows [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = paddedRows * 768u;
    if (index >= count) return;
    output[index] = index < realRows * 768u
        ? bfloat(2.0f * (float(input[index]) - 0.5f))
        : bfloat(0.0f);
}

kernel void vision_add_position(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* table [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * 1152u;
    if (index >= count) return;
    const uint row = index / 1152u;
    const uint dimension = index % 1152u;
    const int2 position = positions[row];
    const uint xIndex = uint(position.x) * 1152u + dimension;
    const uint yIndex = (10240u + uint(position.y)) * 1152u + dimension;
    hidden[index] = bfloat(float(hidden[index])
                           + float(table[xIndex]) + float(table[yIndex]));
}

inline float vision_simdgroup_sum(float value,
                                  uint lane,
                                  uint simdgroup,
                                  threadgroup float* partials) {
    const float subtotal = simd_sum(value);
    if (lane == 0) partials[simdgroup] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float merged = lane < 8u ? partials[lane] : 0.0f;
        merged = simd_sum(merged);
        if (lane == 0) partials[0] = merged;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partials[0];
}

kernel void vision_rmsnorm_rows(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float inverseRMS[1];
    float values[4];
    float sum = 0.0f;
    const uint base = tid * 4u;
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        values[index] = dimension < width
            ? float(input[row * width + dimension]) : 0.0f;
        sum += values[index] * values[index];
    }
    sum = simd_sum(sum);
    if (simdgroup == 0) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        sum = simd_sum(partials[lane]);
        if (lane == 0) {
            inverseRMS[0] = metal::precise::rsqrt(sum / float(width) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        if (dimension < width) {
            const bfloat normalized = bfloat(values[index] * inverseRMS[0]);
            output[row * width + dimension] = bfloat(
                float(weight[dimension]) * float(normalized));
        }
    }
}

kernel void vision_postnorm_residual(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float inverseRMS[1];
    float values[4];
    float sum = 0.0f;
    const uint base = tid * 4u;
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        values[index] = dimension < width
            ? float(branch[row * width + dimension]) : 0.0f;
        sum += values[index] * values[index];
    }
    sum = simd_sum(sum);
    if (simdgroup == 0) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        sum = simd_sum(partials[lane]);
        if (lane == 0) {
            inverseRMS[0] = metal::precise::rsqrt(sum / float(width) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint element = 0; element < 4u; ++element) {
        const uint dimension = base + element;
        if (dimension < width) {
            const uint index = row * width + dimension;
            const bfloat scaled = bfloat(values[element] * inverseRMS[0]);
            const bfloat normalized = bfloat(
                float(weight[dimension]) * float(scaled));
            output[index] = bfloat(float(residual[index]) + float(normalized));
        }
    }
}

inline float2 vision_rope_pair(float2 value, int position, uint pairInAxis) {
    const float exponent = (2.0f / 36.0f) * float(pairInAxis);
    const float angle = float(position) / powr(100.0f, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float2(value.x * cosine - value.y * sine,
                  value.y * cosine + value.x * sine);
}

kernel void vision_qkv_norm_rope(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qWeight [[buffer(3)]],
    device const bfloat* kWeight [[buffer(4)]],
    device const int2* positions [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    const uint base = vectorIndex * kVisionHeadDim;
    float qSquares = 0.0f;
    float kSquares = 0.0f;
    float vSquares = 0.0f;
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        const float qValue = float(q[base + dimension]);
        const float kValue = float(k[base + dimension]);
        const float vValue = float(v[base + dimension]);
        qSquares = fma(qValue, qValue, qSquares);
        kSquares = fma(kValue, kValue, kSquares);
        vSquares = fma(vValue, vValue, vSquares);
    }
    const float qInverse = rsqrt(qSquares / 72.0f + 1e-6f);
    const float kInverse = rsqrt(kSquares / 72.0f + 1e-6f);
    const float vInverse = rsqrt(vSquares / 72.0f + 1e-6f);
    const int2 position = positions[vectorIndex / heads];
    for (uint axis = 0; axis < 2u; ++axis) {
        const uint axisBase = axis * 36u;
        const int coordinate = axis == 0u ? position.x : position.y;
        for (uint pair = 0; pair < 18u; ++pair) {
            const uint first = base + axisBase + pair;
            const uint second = first + 18u;
            float2 qPair = float2(float(q[first]) * qInverse * float(qWeight[axisBase + pair]),
                                  float(q[second]) * qInverse * float(qWeight[axisBase + pair + 18u]));
            float2 kPair = float2(float(k[first]) * kInverse * float(kWeight[axisBase + pair]),
                                  float(k[second]) * kInverse * float(kWeight[axisBase + pair + 18u]));
            qPair = vision_rope_pair(qPair, coordinate, pair);
            kPair = vision_rope_pair(kPair, coordinate, pair);
            q[first] = bfloat(qPair.x);
            q[second] = bfloat(qPair.y);
            k[first] = bfloat(kPair.x);
            k[second] = bfloat(kPair.y);
        }
    }
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        v[base + dimension] = bfloat(float(v[base + dimension]) * vInverse);
    }
}

/// Same arithmetic as `vision_qkv_norm_rope` on the head-major padded layout the
/// QKV projections now write: vector `(head, row)` starts at
/// `(head * rowStride + row) * 80`. Threads walk rows within a head so the
/// strided-by-80 accesses stay coalesced.
kernel void vision_qkv_norm_rope_padded(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qWeight [[buffer(3)]],
    device const bfloat* kWeight [[buffer(4)]],
    device const int2* positions [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& rowStride [[buffer(8)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    const uint head = vectorIndex / rows;
    const uint row = vectorIndex % rows;
    const uint base = (head * rowStride + row) * kVisionPaddedHeadDim;
    float qSquares = 0.0f;
    float kSquares = 0.0f;
    float vSquares = 0.0f;
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        const float qValue = float(q[base + dimension]);
        const float kValue = float(k[base + dimension]);
        const float vValue = float(v[base + dimension]);
        qSquares = fma(qValue, qValue, qSquares);
        kSquares = fma(kValue, kValue, kSquares);
        vSquares = fma(vValue, vValue, vSquares);
    }
    const float qInverse = rsqrt(qSquares / 72.0f + 1e-6f);
    const float kInverse = rsqrt(kSquares / 72.0f + 1e-6f);
    const float vInverse = rsqrt(vSquares / 72.0f + 1e-6f);
    const int2 position = positions[row];
    for (uint axis = 0; axis < 2u; ++axis) {
        const uint axisBase = axis * 36u;
        const int coordinate = axis == 0u ? position.x : position.y;
        for (uint pair = 0; pair < 18u; ++pair) {
            const uint first = base + axisBase + pair;
            const uint second = first + 18u;
            float2 qPair = float2(float(q[first]) * qInverse * float(qWeight[axisBase + pair]),
                                  float(q[second]) * qInverse * float(qWeight[axisBase + pair + 18u]));
            float2 kPair = float2(float(k[first]) * kInverse * float(kWeight[axisBase + pair]),
                                  float(k[second]) * kInverse * float(kWeight[axisBase + pair + 18u]));
            qPair = vision_rope_pair(qPair, coordinate, pair);
            kPair = vision_rope_pair(kPair, coordinate, pair);
            q[first] = bfloat(qPair.x);
            q[second] = bfloat(qPair.y);
            k[first] = bfloat(kPair.x);
            k[second] = bfloat(kPair.y);
        }
    }
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        v[base + dimension] = bfloat(float(v[base + dimension]) * vInverse);
    }
}

kernel void vision_geglu(
    device bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = float(gate[index]);
    const float inner = clamp(
        0.7978845608f * (x + 0.044715f * x * x * x), -20.0f, 20.0f);
    const float gelu = 0.5f * x * (1.0f + tanh(inner));
    gate[index] = bfloat(gelu * float(up[index]));
}

kernel void vision_pool_standardize_norm(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* stdBias [[buffer(1)]],
    device const bfloat* stdScale [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& patchWidth [[buffer(4)]],
    constant uint& patchHeight [[buffer(5)]],
    uint outputRow [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint outputWidth = patchWidth / 3u;
    const uint outputHeight = patchHeight / 3u;
    if (outputRow >= outputWidth * outputHeight) return;
    const uint outputX = outputRow % outputWidth;
    const uint outputY = outputRow / outputWidth;
    threadgroup float partials[8];
    float squares = 0.0f;
    for (uint dimension = tid; dimension < 1152u; dimension += 256u) {
        float pooled = 0.0f;
        for (uint dy = 0; dy < 3u; ++dy) {
            for (uint dx = 0; dx < 3u; ++dx) {
                const uint inputRow = (outputY * 3u + dy) * patchWidth
                    + outputX * 3u + dx;
                pooled += float(input[inputRow * 1152u + dimension]);
            }
        }
        pooled *= sqrt(1152.0f) / 9.0f;
        const float standardized = (pooled - float(stdBias[dimension]))
            * float(stdScale[dimension]);
        output[outputRow * 1152u + dimension] = bfloat(standardized);
        squares = fma(standardized, standardized, squares);
    }
    const float total = vision_simdgroup_sum(squares, lane, simdgroup, partials);
    const float inverse = rsqrt(total / 1152.0f + 1e-6f);
    for (uint dimension = tid; dimension < 1152u; dimension += 256u) {
        const uint index = outputRow * 1152u + dimension;
        output[index] = bfloat(float(output[index]) * inverse);
    }
}

kernel void vision_bfloat_to_half(
    device const bfloat* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = half(input[index]);
}
