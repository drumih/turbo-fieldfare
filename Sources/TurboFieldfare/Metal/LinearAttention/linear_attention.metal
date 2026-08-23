#include <metal_stdlib>
using namespace metal;

constant float kQwenDeltaNormEpsilon = 1.0e-6f;

static inline float qwen_gated_delta_silu(float value) {
    return value / (1.0f + exp(-value));
}

kernel void qwen_gated_delta_causal_conv(
    device const half* input [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device half* state [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= channels || kernel_size < 2u) return;

    const uint state_width = kernel_size - 1u;
    const uint state_base = channel * state_width;
    const uint weight_base = channel * kernel_size;
    float result = 0.0f;
    for (uint tap = 0; tap < state_width; ++tap) {
        result = fma(float(weights[weight_base + tap]),
                     float(state[state_base + tap]), result);
    }
    result = fma(float(weights[weight_base + state_width]),
                 float(input[channel]), result);

    for (uint tap = 0; tap + 1u < state_width; ++tap) {
        state[state_base + tap] = state[state_base + tap + 1u];
    }
    state[state_base + state_width - 1u] = input[channel];
    output[channel] = half(qwen_gated_delta_silu(result));
}

kernel void qwen_prefill_gated_delta_causal_conv(
    device const half* input [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device half* state [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    constant uint& token_count [[buffer(6)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= channels || kernel_size < 2u) return;

    const uint state_width = kernel_size - 1u;
    const uint state_base = channel * state_width;
    const uint weight_base = channel * kernel_size;
    for (uint token = 0; token < token_count; ++token) {
        const uint input_base = token * channels;
        float result = 0.0f;
        for (uint tap = 0; tap < state_width; ++tap) {
            result = fma(float(weights[weight_base + tap]),
                         float(state[state_base + tap]), result);
        }
        result = fma(float(weights[weight_base + state_width]),
                     float(input[input_base + channel]), result);

        for (uint tap = 0; tap + 1u < state_width; ++tap) {
            state[state_base + tap] = state[state_base + tap + 1u];
        }
        state[state_base + state_width - 1u] = input[input_base + channel];
        output[input_base + channel] = half(qwen_gated_delta_silu(result));
    }
}

kernel void qwen_prefill_split_qkv(
    device const half* input [[buffer(0)]],
    device half* query [[buffer(1)]],
    device half* key [[buffer(2)]],
    device half* value [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& key_width [[buffer(5)]],
    constant uint& value_width [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= token_count) return;
    const uint key_base = gid.y * key_width;
    const uint value_base = gid.y * value_width;
    const uint input_base = gid.y * (key_width * 2u + value_width);
    if (gid.x < key_width) {
        query[key_base + gid.x] = input[input_base + gid.x];
        key[key_base + gid.x] = input[input_base + key_width + gid.x];
    }
    if (gid.x < value_width) {
        value[value_base + gid.x] = input[input_base + key_width * 2u + gid.x];
    }
}

kernel void qwen_gated_delta_recurrent(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant uint& key_heads [[buffer(7)]],
    constant uint& value_heads [[buffer(8)]],
    constant uint& key_dim [[buffer(9)]],
    constant uint& value_dim [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint value_index = gid.x;
    const uint head = gid.y;
    if (head >= value_heads || value_index >= value_dim) return;

    const uint key_head = head * key_heads / value_heads;
    const uint q_base = key_head * key_dim;
    const uint k_base = key_head * key_dim;
    const uint v_base = head * value_dim;
    const uint state_base = head * key_dim * value_dim;

    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (uint i = 0; i < key_dim; ++i) {
        const float q = float(query[q_base + i]);
        const float k = float(key[k_base + i]);
        q_norm = fma(q, q, q_norm);
        k_norm = fma(k, k, k_norm);
    }
    q_norm = rsqrt(q_norm + kQwenDeltaNormEpsilon);
    k_norm = rsqrt(k_norm + kQwenDeltaNormEpsilon);
    const float q_scale = rsqrt(float(key_dim));
    const float state_decay = exp(decay[head]);
    const float state_beta = beta[head];

    for (uint i = 0; i < key_dim; ++i) {
        state[state_base + i * value_dim + value_index] *= state_decay;
    }

    float memory = 0.0f;
    for (uint i = 0; i < key_dim; ++i) {
        memory = fma(state[state_base + i * value_dim + value_index],
                     float(key[k_base + i]) * k_norm, memory);
    }
    const float delta = (float(value[v_base + value_index]) - memory) * state_beta;
    for (uint i = 0; i < key_dim; ++i) {
        state[state_base + i * value_dim + value_index] +=
            float(key[k_base + i]) * k_norm * delta;
    }

    float result = 0.0f;
    for (uint i = 0; i < key_dim; ++i) {
        result = fma(state[state_base + i * value_dim + value_index],
                     float(query[q_base + i]) * q_norm * q_scale, result);
    }
    output[v_base + value_index] = half(result);
}

kernel void qwen_prefill_gated_delta_recurrent(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    constant uint& key_heads [[buffer(8)]],
    constant uint& value_heads [[buffer(9)]],
    constant uint& key_dim [[buffer(10)]],
    constant uint& value_dim [[buffer(11)]],
    constant uint& query_stride [[buffer(12)]],
    constant uint& key_stride [[buffer(13)]],
    constant uint& value_stride [[buffer(14)]],
    constant uint& output_stride [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint value_index = gid.x;
    const uint head = gid.y;
    if (head >= value_heads || value_index >= value_dim) return;

    const uint key_head = head * key_heads / value_heads;
    const uint q_base = key_head * key_dim;
    const uint k_base = key_head * key_dim;
    const uint v_base = head * value_dim;
    const uint state_base = head * key_dim * value_dim;
    const float q_scale = rsqrt(float(key_dim));

    for (uint token = 0; token < token_count; ++token) {
        const uint q_token = token * query_stride;
        const uint k_token = token * key_stride;
        const uint v_token = token * value_stride;
        const uint out_token = token * output_stride;
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            const float q = float(query[q_token + q_base + i]);
            const float k = float(key[k_token + k_base + i]);
            q_norm = fma(q, q, q_norm);
            k_norm = fma(k, k, k_norm);
        }
        q_norm = rsqrt(q_norm + kQwenDeltaNormEpsilon);
        k_norm = rsqrt(k_norm + kQwenDeltaNormEpsilon);
        const float state_decay = exp(decay[token * value_heads + head]);
        const float state_beta = beta[token * value_heads + head];

        for (uint i = 0; i < key_dim; ++i) {
            state[state_base + i * value_dim + value_index] *= state_decay;
        }

        float memory = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            memory = fma(state[state_base + i * value_dim + value_index],
                         float(key[k_token + k_base + i]) * k_norm, memory);
        }
        const float delta =
            (float(value[v_token + v_base + value_index]) - memory) * state_beta;
        for (uint i = 0; i < key_dim; ++i) {
            state[state_base + i * value_dim + value_index] +=
                float(key[k_token + k_base + i]) * k_norm * delta;
        }

        float result = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            result = fma(state[state_base + i * value_dim + value_index],
                         float(query[q_token + q_base + i]) * q_norm * q_scale,
                         result);
        }
        output[out_token + v_base + value_index] = half(result);
    }
}