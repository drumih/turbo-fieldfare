#include <metal_stdlib>
using namespace metal;

constant uint FC_ROPE_HEAD_DIM [[function_constant(50)]];
constant uint FC_ROPE_NUM_HEADS [[function_constant(51)]];
constant uint FC_ROPE_ROTATED_PAIRS [[function_constant(52)]];
constant bool FC_ROPE_USE_FC [[function_constant(53)]];

static inline uint rope_head_dim(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_HEAD_DIM))
        ? FC_ROPE_HEAD_DIM
        : runtime_value;
}
static inline uint rope_num_heads(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_NUM_HEADS))
        ? FC_ROPE_NUM_HEADS
        : runtime_value;
}

static inline uint rope_rotated_pairs(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_ROTATED_PAIRS))
        ? FC_ROPE_ROTATED_PAIRS
        : runtime_value;
}

static inline void apply_neox_pair(
    device half* head,
    uint pair,
    uint half_dim,
    uint frequency_divisor,
    float position,
    float theta
) {
    const float exponent = -float(2u * pair) / float(frequency_divisor);
    const float angle = position * pow(theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const uint lower = pair;
    const uint upper = half_dim + pair;
    const float x0 = float(head[lower]);
    const float x1 = float(head[upper]);
    head[lower] = half(x0 * cosine - x1 * sine);
    head[upper] = half(x0 * sine + x1 * cosine);
}

kernel void rope_default_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_dimension = dimension / 2u;
    if (pair >= half_dimension || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}

// Qwen-style partial RoPE: rotation confined to the first `rotary_dim`
// elements of each head, NeoX pairing (i, rotary_dim/2 + i) inside that
// window, frequency divisor = rotary_dim. Elements >= rotary_dim pass
// through untouched. (The proportional variant instead pairs across the
// full head and divides frequencies by head_dim — a different element set.)
kernel void rope_neox_subdim(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_rotary = rotary_dim / 2u;
    if (pair >= half_rotary || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_rotary, rotary_dim,
                    float(position), theta);
}

// NeoX sub-dim rope where each row along the token axis carries its own
// position: `position + token_index * stride`.
//
// The scalar-position kernel above covers a decode step and a contiguous
// prefill chunk read at one offset. The pooled indexer blocks are neither:
// block b sits at position b * compress_ratio, so their positions advance by
// the ratio, not by one.
kernel void rope_neox_subdim_strided(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_rotary = rotary_dim / 2u;
    if (pair >= half_rotary || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_rotary, rotary_dim,
                    float(position + token_index * stride), theta);
}

kernel void rope_yarn_neox_subdim(
    device half* data [[buffer(0)]],
    device const float* inverse_frequencies [[buffer(1)]],
    constant uint& position [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    constant float& attention_factor [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint half_rotary = rotary_dim / 2u;
    if (pair >= half_rotary || head_index >= num_heads) return;
    device half* head = data
        + token_index * num_heads * head_dim
        + head_index * head_dim;
    const float angle = float(position + token_index) * inverse_frequencies[pair];
    const float cosine = cos(angle) * attention_factor;
    const float sine = sin(angle) * attention_factor;
    const uint lower = pair;
    const uint upper = half_rotary + pair;
    const float x0 = float(head[lower]);
    const float x1 = float(head[upper]);
    head[lower] = half(x0 * cosine - x1 * sine);
    head[upper] = half(x0 * sine + x1 * cosine);
}

kernel void rope_proportional_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotated_pairs [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint active_pairs = rope_rotated_pairs(rotated_pairs);
    if (pair >= active_pairs || head_index >= heads) return;

    const uint half_dimension = dimension / 2u;
    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}
