#include <metal_stdlib>
using namespace metal;

// Decode and prefill ops for a depth-mixed, looped, relative-attention MoE
// decoder (llama.cpp's inkling2 graph), for a chunk of `tokens` consecutive
// positions starting at `DepthStep.position` (one token at decode). Vectors of
// a chunk are token-major: element c of token t is at [t * width + c]. Ops
// with a recurrence over positions (the short convolutions, the KV store) walk
// the chunk's tokens in order; everything else is parallel over tokens.

struct DepthStep {
    uint position;
};

static inline float depth_threadgroup_sum(
    float value, threadgroup float* scratch, uint sg, uint lane, uint simdgroups)
{
    value = simd_sum(value);
    if (lane == 0u) scratch[sg] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < simdgroups; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// out[t][c] = dequant(row token[t] of a Q8_0 matrix)[c]; grid (columns, tokens).
kernel void embed_q8_0_row(
    device const uchar* weights [[buffer(0)]],
    device const int* token [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    uint2 index [[thread_position_in_grid]])
{
    if (index.x >= columns) return;
    device const uchar* row = weights + ulong(token[index.y]) * (columns / 32u) * 34u;
    device const uchar* b = row + (index.x / 32u) * 34u;
    const float d = float(as_type<half>(*(device const ushort*)b));
    out[index.y * columns + index.x] = d * float(((device const char*)(b + 2u))[index.x % 32u]);
}

// out[t][c] = row token[t] of a BF16 matrix, as F32 (the Inkling 2 checkpoint's
// embedding, TT-125); grid (columns, tokens).
kernel void embed_bf16_row(
    device const ushort* weights [[buffer(0)]],
    device const int* token [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    uint2 index [[thread_position_in_grid]])
{
    if (index.x >= columns) return;
    const ushort bits = weights[ulong(token[index.y]) * columns + index.x];
    out[index.y * columns + index.x] = as_type<float>(uint(bits) << 16);
}

struct DepthRMSArgs {
    uint width;
    uint inStride;
    uint outStride;
    uint hasWeight;
    float epsilon;
};

// One threadgroup per vector: out = x * rsqrt(mean(x^2) + eps) (* weight).
kernel void rms_norm_rows(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant DepthRMSArgs& args [[buffer(3)]],
    uint vector [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    device const float* xs = x + vector * args.inStride;
    device float* os = out + vector * args.outStride;
    float sum = 0.0f;
    for (uint i = tid; i < args.width; i += threads) sum += xs[i] * xs[i];
    const float total = depth_threadgroup_sum(sum, scratch, sg, lane, (threads + 31u) / 32u);
    const float inv = rsqrt(total / float(args.width) + args.epsilon);
    for (uint i = tid; i < args.width; i += threads) {
        os[i] = xs[i] * inv * (args.hasWeight != 0u ? weight[i] : 1.0f);
    }
}

struct DepthGemvArgs {
    uint rows;
    uint columns;
};

// y[t][r] = W[r] . x[t] for an f32 (isBF16 = 0) or bf16 row-major matrix; one
// SIMD group per (row, token), four rows per threadgroup, a flat grid of
// ceil(rows / 4) threadgroups per token. Used at decode and, at prefill, for
// the small matrices (router, sinks, conv taps, loop controllers) whose few
// row tiles would leave the tiled GEMM a handful of threadgroups; the matrix
// stays in cache across the chunk's tokens.
kernel void dense_gemv_f32_bf16(
    device const uchar* weights [[buffer(0)]],
    device const float* xAll [[buffer(1)]],
    device float* yAll [[buffer(2)]],
    constant DepthGemvArgs& args [[buffer(3)]],
    constant uint& isBF16 [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint groupsPerToken = (args.rows + 3u) / 4u;
    const uint token = groupIndex / groupsPerToken;
    const uint row = (groupIndex % groupsPerToken) * 4u + sg;
    if (row >= args.rows) return;
    device const float* x = xAll + ulong(token) * args.columns;
    device float* y = yAll + ulong(token) * args.rows;
    float sum = 0.0f;
    if (isBF16 != 0u) {
        device const ushort* w = (device const ushort*)weights + ulong(row) * args.columns;
        for (uint i = lane; i < args.columns; i += 32u) {
            sum += as_type<float>(uint(w[i]) << 16u) * x[i];
        }
    } else {
        device const float* w = (device const float*)weights + ulong(row) * args.columns;
        for (uint i = lane; i < args.columns; i += 32u) sum += w[i] * x[i];
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}

struct DepthMixArgs {
    uint width;
    uint stateCount;
    uint queryCount;
    float epsilon;
    uint normWeighted;  // 1: RMS-normalise the output and multiply by `normWeight`
    float normEpsilon;
    uint tokens;
    uint slab;          // floats per pool slab (tokens * width at prefill)
};

// Attention over depth, per (query, token): out = sum_s softmax_s(rms(s) . q /
// sqrt(width)) * rms(s). `rowsOf` lists the pool slabs holding the states
// (earlier blocks, then cur); query q's output is slab q of `out`.
kernel void depth_mix(
    device const float* pool [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant DepthMixArgs& args [[buffer(3)]],
    constant uint* rowsOf [[buffer(4)]],
    device const float* normWeight [[buffer(5)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    threadgroup float partialSq[32][32];   // [state][SIMD group]
    threadgroup float partialDot[32][32];
    threadgroup float inv[32];
    threadgroup float score[32];
    // A flat grid of (query, token) pairs: the SIMD-group attributes are
    // scalar, so the threadgroup position must be too.
    const uint query = groupIndex % args.queryCount;
    const uint token = groupIndex / args.queryCount;
    const uint simdgroups = (threads + 31u) / 32u;
    device const float* q = queries + query * args.width;
    // Per-state partial sums go to threadgroup memory without a barrier; one
    // barrier then totals every state, adding the SIMD groups in order as
    // depth_threadgroup_sum does (two barriers in place of four per state;
    // the output expression is the original's, term for term).
    for (uint s = 0u; s < args.stateCount; ++s) {
        device const float* xs = pool + ulong(rowsOf[s]) * args.slab + token * args.width;
        float sq = 0.0f;
        float dot = 0.0f;
        for (uint i = tid; i < args.width; i += threads) {
            sq += xs[i] * xs[i];
            dot += xs[i] * q[i];
        }
        sq = simd_sum(sq);
        dot = simd_sum(dot);
        if (lane == 0u) {
            partialSq[s][sg] = sq;
            partialDot[s][sg] = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint s = 0u; s < args.stateCount; ++s) {
            float sqTotal = 0.0f;
            float dotTotal = 0.0f;
            for (uint i = 0u; i < simdgroups; ++i) {
                sqTotal += partialSq[s][i];
                dotTotal += partialDot[s][i];
            }
            const float r = rsqrt(sqTotal / float(args.width) + args.epsilon);
            inv[s] = r;
            score[s] = dotTotal * r * rsqrt(float(args.width));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peak = -INFINITY;
    for (uint s = 0u; s < args.stateCount; ++s) peak = max(peak, score[s]);
    float denominator = 0.0f;
    for (uint s = 0u; s < args.stateCount; ++s) denominator += exp(score[s] - peak);
    device float* o = out + ulong(query) * args.slab + token * args.width;
    float outSquares = 0.0f;
    for (uint i = tid; i < args.width; i += threads) {
        float value = 0.0f;
        for (uint s = 0u; s < args.stateCount; ++s) {
            value += exp(score[s] - peak) / denominator * inv[s]
                * pool[ulong(rowsOf[s]) * args.slab + token * args.width + i];
        }
        o[i] = value;
        outSquares += value * value;
    }
    if (args.normWeighted == 0u) return;
    // Each thread rescales the elements it wrote, so only the sum is shared.
    const float total = depth_threadgroup_sum(outSquares, scratch, sg, lane, simdgroups);
    const float r = rsqrt(total / float(args.width) + args.normEpsilon);
    for (uint i = tid; i < args.width; i += threads) o[i] = o[i] * r * normWeight[i];
}

struct DepthControlArgs {
    uint width;
    uint rank;
    float epsilon;
};

struct DepthConvArgs {
    uint channels;
    uint slots;        // > 0: the input is sum_s weights[t][s] * x[t * slots + s][c]
    uint hasResidual;  // add `residual` after the scale
    float scale;
    uint tokens;
};

// Causal 4-tap short convolution with residual, one channel per thread, over
// the chunk's tokens in order: out[t] = (x[t] + sum_k w[k] * [s0 s1 s2 x[t]][k])
// * scale (+ residual[t]), state <- [s1 s2 x[t]]. `state` is [channels][3];
// weight is [channels][4].
kernel void short_conv_step(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* state [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const float* routeWeights [[buffer(4)]],
    device const float* residual [[buffer(5)]],
    constant DepthConvArgs& args [[buffer(6)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= args.channels) return;
    float s0 = state[c * 3u + 0u];
    float s1 = state[c * 3u + 1u];
    float s2 = state[c * 3u + 2u];
    const float w0 = weight[c * 4u + 0u];
    const float w1 = weight[c * 4u + 1u];
    const float w2 = weight[c * 4u + 2u];
    const float w3 = weight[c * 4u + 3u];
    for (uint t = 0u; t < args.tokens; ++t) {
        float xv;
        if (args.slots > 0u) {
            xv = 0.0f;
            for (uint s = 0u; s < args.slots; ++s) {
                xv += routeWeights[t * args.slots + s]
                    * x[(t * args.slots + s) * args.channels + c];
            }
        } else {
            xv = x[t * args.channels + c];
        }
        const float conv = s0 * w0 + s1 * w1 + s2 * w2 + xv * w3;
        s0 = s1;
        s1 = s2;
        s2 = xv;
        out[t * args.channels + c] = (xv + conv) * args.scale
            + (args.hasResidual != 0u ? residual[t * args.channels + c] : 0.0f);
    }
    state[c * 3u + 0u] = s0;
    state[c * 3u + 1u] = s1;
    state[c * 3u + 2u] = s2;
}

struct DepthKVFinishArgs {
    uint headDim;
    uint kvHeads;
    uint hasDynamic;
    uint capacity;
    uint ring;
    float threshold;  // K clip
    float epsilon;    // V RMS norm
    float scale;      // convolution residual scale
    uint tokens;
};

// K and V after their projections, in one kernel: one threadgroup per head
// (first kvHeads for K, then V), one thread per channel, over the chunk's
// tokens in order. Each runs its causal convolution (with the per-head dynamic
// taps of full layers), then K is L2-clipped and V RMS-normalised per head, and
// both are stored as half at that position's cache slot. `state` holds K
// channels then V channels; dynamic taps are [tokens][kvHeads][4], lag 0 the
// current input.
kernel void kv_finish(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* kWeight [[buffer(2)]],
    device const float* vWeight [[buffer(3)]],
    device float* state [[buffer(4)]],
    device const float* kDynamic [[buffer(5)]],
    device const float* vDynamic [[buffer(6)]],
    device half* kCache [[buffer(7)]],
    device half* vCache [[buffer(8)]],
    device const DepthStep* step [[buffer(9)]],
    constant DepthKVFinishArgs& args [[buffer(10)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    const bool isV = groupIndex >= args.kvHeads;
    const uint head = groupIndex % args.kvHeads;
    const uint width = args.kvHeads * args.headDim;
    const uint c = head * args.headDim + tid;
    device const float* xs = isV ? v : k;
    device const float* weight = isV ? vWeight : kWeight;
    device const float* dynamic = isV ? vDynamic : kDynamic;
    device float* st = state + (isV ? width : 0u) * 3u;
    device half* cache = isV ? vCache : kCache;
    float s0 = st[c * 3u + 0u];
    float s1 = st[c * 3u + 1u];
    float s2 = st[c * 3u + 2u];
    const float w0 = weight[c * 4u + 0u];
    const float w1 = weight[c * 4u + 1u];
    const float w2 = weight[c * 4u + 2u];
    const float w3 = weight[c * 4u + 3u];
    const uint simdgroups = (threads + 31u) / 32u;
    for (uint t = 0u; t < args.tokens; ++t) {
        const float xv = xs[t * width + c];
        float conv = s0 * w0 + s1 * w1 + s2 * w2 + xv * w3;
        if (args.hasDynamic != 0u) {
            device const float* taps = dynamic + (t * args.kvHeads + head) * 4u;
            conv += xv * taps[0];
            conv += s2 * taps[1];
            conv += s1 * taps[2];
            conv += s0 * taps[3];
        }
        s0 = s1;
        s1 = s2;
        s2 = xv;
        const float value = (xv + conv) * args.scale;
        const float sq = depth_threadgroup_sum(value * value, scratch, sg, lane, simdgroups);
        const float factor = isV
            ? rsqrt(sq / float(args.headDim) + args.epsilon)
            : clamp(args.threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
        const uint position = step[0].position + t;
        const uint slot = args.ring != 0u ? position % args.capacity : position;
        cache[ulong(slot) * width + c] = half(value * factor);
    }
    st[c * 3u + 0u] = s0;
    st[c * 3u + 1u] = s1;
    st[c * 3u + 2u] = s2;
}

struct DepthClipArgs {
    uint headDim;
    uint hasScale;
    uint applyTau;
    float threshold;
    float tauFloor;
    float tauAlpha;
    uint heads;
};

// Per (head, token), one SIMD group: v *= scale; v *= clamp(t / (|v| + 1e-6), 0, 1);
// then, for full-attention layers, v *= 1 + alpha * log(max(1, (pos + 1) / floor)).
// A flat grid of heads x tokens; v is [tokens][heads][headDim].
kernel void clip_heads(
    device float* v [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const DepthStep* step [[buffer(2)]],
    constant DepthClipArgs& args [[buffer(3)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint head = groupIndex % args.heads;
    const uint token = groupIndex / args.heads;
    device float* h = v + (token * args.heads + head) * args.headDim;
    float sq = 0.0f;
    for (uint i = lane; i < args.headDim; i += 32u) {
        const float value = h[i] * (args.hasScale != 0u ? scale[head * args.headDim + i] : 1.0f);
        sq += value * value;
    }
    const float norm = sqrt(simd_sum(sq)) + 1e-6f;
    float factor = clamp(args.threshold / norm, 0.0f, 1.0f);
    if (args.applyTau != 0u) {
        const float position = float(step[0].position + token);
        const float ratio = max(1.0f, (position + 1.0f) / args.tauFloor);
        factor *= 1.0f + args.tauAlpha * log(ratio);
    }
    for (uint i = lane; i < args.headDim; i += 32u) {
        h[i] = h[i] * (args.hasScale != 0u ? scale[head * args.headDim + i] : 1.0f) * factor;
    }
}
