#include <metal_stdlib>
using namespace metal;

// Single-token decode ops for a depth-mixed, looped, relative-attention MoE
// decoder (llama.cpp's inkling2 graph). Every kernel works on f32 vectors that
// stay on the GPU; position-dependent values come from `DepthStep` in a device
// buffer, so one token's command sequence is the same at every position.

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

// out[t][c] = dequant(row token of a Q8_0 matrix)[c]
kernel void embed_q8_0_row(
    device const uchar* weights [[buffer(0)]],
    device const int* token [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= columns) return;
    device const uchar* row = weights + ulong(token[0]) * (columns / 32u) * 34u;
    device const uchar* b = row + (index / 32u) * 34u;
    const float d = float(as_type<half>(*(device const ushort*)b));
    out[index] = d * float(((device const char*)(b + 2u))[index % 32u]);
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

// y[r] = W[r] . x for an f32 (isBF16 = 0) or bf16 row-major matrix; one SIMD
// group per row, four per threadgroup.
kernel void dense_gemv_f32_bf16(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant DepthGemvArgs& args [[buffer(3)]],
    constant uint& isBF16 [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = groupIndex * 4u + sg;
    if (row >= args.rows) return;
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
};

// Attention over depth: for query q, out[q] = sum_s softmax_s(rms(s) . q / sqrt(width)) * rms(s).
// `rowsOf` lists the pool rows holding the states (earlier blocks, then cur).
kernel void depth_mix(
    device const float* pool [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant DepthMixArgs& args [[buffer(3)]],
    constant uint* rowsOf [[buffer(4)]],
    device const float* normWeight [[buffer(5)]],
    uint query [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    threadgroup float inv[32];
    threadgroup float score[32];
    const uint simdgroups = (threads + 31u) / 32u;
    device const float* q = queries + query * args.width;
    for (uint s = 0u; s < args.stateCount; ++s) {
        device const float* xs = pool + rowsOf[s] * args.width;
        float sq = 0.0f;
        float dot = 0.0f;
        for (uint i = tid; i < args.width; i += threads) {
            sq += xs[i] * xs[i];
            dot += xs[i] * q[i];
        }
        const float sqTotal = depth_threadgroup_sum(sq, scratch, sg, lane, simdgroups);
        const float dotTotal = depth_threadgroup_sum(dot, scratch, sg, lane, simdgroups);
        if (tid == 0u) {
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
    device float* o = out + query * args.width;
    float outSquares = 0.0f;
    for (uint i = tid; i < args.width; i += threads) {
        float value = 0.0f;
        for (uint s = 0u; s < args.stateCount; ++s) {
            value += exp(score[s] - peak) / denominator * inv[s] * pool[rowsOf[s] * args.width + i];
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

// joined = [rms(cur); rms(message)], the controller's 2 * width input; the
// projection h = D . joined then runs on the parallel dense GEMV.
kernel void control_join(
    device const float* cur [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device float* joined [[buffer(2)]],
    constant DepthControlArgs& args [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    const uint simdgroups = (threads + 31u) / 32u;
    float a = 0.0f;
    float b = 0.0f;
    for (uint i = tid; i < args.width; i += threads) {
        a += cur[i] * cur[i];
        b += message[i] * message[i];
    }
    const float invCur = rsqrt(
        depth_threadgroup_sum(a, scratch, sg, lane, simdgroups) / float(args.width) + args.epsilon);
    const float invMessage = rsqrt(
        depth_threadgroup_sum(b, scratch, sg, lane, simdgroups) / float(args.width) + args.epsilon);
    for (uint i = tid; i < args.width; i += threads) {
        joined[i] = cur[i] * invCur;
        joined[args.width + i] = message[i] * invMessage;
    }
}

// out = (1 - tanh(F . h)) * cur + tanh(G . h) * message; F, G are [width][rank].
// One SIMD group per output reads the two contiguous rank-length rows.
kernel void control_apply(
    device const float* cur [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device const float* h [[buffer(2)]],
    device const float* gain [[buffer(3)]],
    device const float* forget [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant DepthControlArgs& args [[buffer(6)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint index = groupIndex * 4u + sg;
    if (index >= args.width) return;
    float g = 0.0f;
    float f = 0.0f;
    for (uint r = lane; r < args.rank; r += 32u) {
        g += gain[index * args.rank + r] * h[r];
        f += forget[index * args.rank + r] * h[r];
    }
    g = simd_sum(g);
    f = simd_sum(f);
    if (lane == 0u) out[index] = (1.0f - tanh(f)) * cur[index] + tanh(g) * message[index];
}

struct DepthConvArgs {
    uint channels;
    uint slots;        // > 0: the input is sum_s weights[s] * x[s][c] (routed experts)
    uint hasResidual;  // add `residual` after the scale
    float scale;
};

// Causal 4-tap short convolution with residual, one channel per thread:
// out = (x + sum_k w[k] * [s0 s1 s2 x][k]) * scale (+ residual), state <- [s1 s2 x].
// `state` is [channels][3]; weight is [channels][4].
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
    float xv;
    if (args.slots > 0u) {
        xv = 0.0f;
        for (uint s = 0u; s < args.slots; ++s) xv += routeWeights[s] * x[s * args.channels + c];
    } else {
        xv = x[c];
    }
    const float window[4] = {state[c * 3u + 0u], state[c * 3u + 1u], state[c * 3u + 2u], xv};
    float conv = 0.0f;
    for (uint k = 0u; k < 4u; ++k) conv += window[k] * weight[c * 4u + k];
    state[c * 3u + 0u] = window[1];
    state[c * 3u + 1u] = window[2];
    state[c * 3u + 2u] = xv;
    out[c] = (xv + conv) * args.scale + (args.hasResidual != 0u ? residual[c] : 0.0f);
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
};

// K and V after their projections, in one kernel: one threadgroup per head
// (first kvHeads for K, then V), one thread per channel. Each runs its
// causal convolution (with the per-head dynamic taps of full layers), then K
// is L2-clipped and V RMS-normalised per head, and both are stored as half
// at this position's cache slot. `state` holds K channels then V channels.
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

    const float xv = xs[c];
    const float window[4] = {st[c * 3u + 0u], st[c * 3u + 1u], st[c * 3u + 2u], xv};
    float conv = 0.0f;
    for (uint t = 0u; t < 4u; ++t) conv += window[t] * weight[c * 4u + t];
    if (args.hasDynamic != 0u) {
        for (uint lag = 0u; lag < 4u; ++lag) conv += window[3u - lag] * dynamic[head * 4u + lag];
    }
    st[c * 3u + 0u] = window[1];
    st[c * 3u + 1u] = window[2];
    st[c * 3u + 2u] = xv;
    const float value = (xv + conv) * args.scale;

    const float sq = depth_threadgroup_sum(value * value, scratch, sg, lane, (threads + 31u) / 32u);
    const float factor = isV
        ? rsqrt(sq / float(args.headDim) + args.epsilon)
        : clamp(args.threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
    const uint slot = args.ring != 0u ? step[0].position % args.capacity : step[0].position;
    device half* cache = isV ? vCache : kCache;
    cache[ulong(slot) * width + c] = half(value * factor);
}

struct DepthClipArgs {
    uint headDim;
    uint hasScale;
    uint applyTau;
    float threshold;
    float tauFloor;
    float tauAlpha;
};

// Per head (one SIMD group each): v *= scale; v *= clamp(t / (|v| + 1e-6), 0, 1);
// then, for full-attention layers, v *= 1 + alpha * log(max(1, (pos + 1) / floor)).
kernel void clip_heads(
    device float* v [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const DepthStep* step [[buffer(2)]],
    constant DepthClipArgs& args [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    device float* h = v + head * args.headDim;
    float sq = 0.0f;
    for (uint i = lane; i < args.headDim; i += 32u) {
        const float value = h[i] * (args.hasScale != 0u ? scale[head * args.headDim + i] : 1.0f);
        sq += value * value;
    }
    const float norm = sqrt(simd_sum(sq)) + 1e-6f;
    float factor = clamp(args.threshold / norm, 0.0f, 1.0f);
    if (args.applyTau != 0u) {
        const float ratio = max(1.0f, (float(step[0].position) + 1.0f) / args.tauFloor);
        factor *= 1.0f + args.tauAlpha * log(ratio);
    }
    for (uint i = lane; i < args.headDim; i += 32u) {
        h[i] = h[i] * (args.hasScale != 0u ? scale[head * args.headDim + i] : 1.0f) * factor;
    }
}
