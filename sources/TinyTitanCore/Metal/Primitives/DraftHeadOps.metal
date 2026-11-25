#include <metal_stdlib>
using namespace metal;

// Generic block-parallel draft model and recurrent Markov head ops (TT-129).
// Model-free: operates on arbitrary hidden/head/vocab/rank dimensions.

static inline float draft_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}

static inline float draft_tg_sum(
    float val, threadgroup float* scratch, uint sg, uint lane, uint sgs)
{
    val = simd_sum(val);
    if (lane == 0u) scratch[sg] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < sgs; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

struct DraftMixRMSArgs {
    uint taps;
    uint hidden;
    float epsilon;
};

// Weighted sum of `taps` target feature vectors + RMSNorm with BF16 weight.
// `targets` is [rows][taps][hidden] FP32, `alpha` is [taps] FP32,
// `weight` is [hidden] BF16, `out` is [rows][hidden] FP32.
kernel void draft_mix_rmsnorm(
    device const float* targets [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device const ushort* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant DraftMixRMSArgs& args [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    const ulong rowBase = ulong(row) * args.taps * args.hidden;
    device float* dst = out + ulong(row) * args.hidden;
    float sq = 0.0f;
    for (uint i = tid; i < args.hidden; i += threads) {
        float acc = 0.0f;
        for (uint t = 0u; t < args.taps; ++t) {
            acc += alpha[t] * targets[rowBase + ulong(t) * args.hidden + i];
        }
        dst[i] = acc;
        sq += acc * acc;
    }
    const float total = draft_tg_sum(sq, scratch, sg, lane, (threads + 31u) / 32u);
    const float inv = rsqrt(total / float(args.hidden) + args.epsilon);
    for (uint i = tid; i < args.hidden; i += threads) {
        dst[i] = dst[i] * inv * draft_bf16_to_f32(weight[i]);
    }
}

struct DraftRMSBF16Args {
    uint width;
    uint hasResidual;
    float epsilon;
};

// RMSNorm with BF16 weight, optionally fused with in-place residual add:
// if hasResidual: residual += x; out = rmsnorm(residual) * bf16(weight).
kernel void draft_rms_norm_bf16(
    device const float* x [[buffer(0)]],
    device float* residual [[buffer(1)]],
    device const ushort* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant DraftRMSBF16Args& args [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    const ulong base = ulong(row) * args.width;
    device const float* src = x + base;
    device float* res = residual + base;
    device float* dst = out + base;
    float sq = 0.0f;
    for (uint i = tid; i < args.width; i += threads) {
        float v = src[i];
        if (args.hasResidual != 0u) {
            v += res[i];
            res[i] = v;
        }
        sq += v * v;
    }
    const float total = draft_tg_sum(sq, scratch, sg, lane, (threads + 31u) / 32u);
    const float inv = rsqrt(total / float(args.width) + args.epsilon);
    for (uint i = tid; i < args.width; i += threads) {
        const float v = (args.hasResidual != 0u) ? res[i] : src[i];
        dst[i] = v * inv * draft_bf16_to_f32(weight[i]);
    }
}

struct DraftQKNormRoPEArgs {
    uint heads;
    uint headDim;
    uint basePosition;
    uint hasNorm;
    float epsilon;
    float ropeTheta;
};

// In-place per-head RMSNorm (BF16 weight) + NeoX RoPE on [tokens, heads, headDim].
// One threadgroup per (token, head); threads = headDim / 2 (64 for headDim=128).
kernel void draft_qk_norm_rope(
    device float* vec [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    constant DraftQKNormRoPEArgs& args [[buffer(2)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    const uint halfDim = args.headDim / 2u;
    if (tid >= halfDim) return;
    const uint token = groupIndex / args.heads;
    const uint head = groupIndex % args.heads;
    device float* h = vec + (ulong(token) * args.heads + head) * args.headDim;
    float x1 = h[tid];
    float x2 = h[tid + halfDim];
    if (args.hasNorm != 0u) {
        const float sq = x1 * x1 + x2 * x2;
        const float total = draft_tg_sum(sq, scratch, sg, lane, (threads + 31u) / 32u);
        const float inv = rsqrt(total / float(args.headDim) + args.epsilon);
        x1 = x1 * inv * draft_bf16_to_f32(weight[tid]);
        x2 = x2 * inv * draft_bf16_to_f32(weight[tid + halfDim]);
    }
    const float pos = float(args.basePosition + token);
    const float exponent = -2.0f * float(tid) / float(args.headDim);
    const float angle = pos * pow(args.ropeTheta, exponent);
    const float c = cos(angle);
    const float s = sin(angle);
    h[tid] = x1 * c - x2 * s;
    h[tid + halfDim] = x2 * c + x1 * s;
}

struct DraftStoreKVArgs {
    uint kvWidth;
    uint basePosition;
    uint capacity;
    uint ring;
};

// Stores [tokens, kvWidth] FP32 K and V into FP16 cache buffers.
kernel void draft_store_kv_fp16(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device half* kCache [[buffer(2)]],
    device half* vCache [[buffer(3)]],
    constant DraftStoreKVArgs& args [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= args.kvWidth) return;
    const uint pos = args.basePosition + gid.y;
    const uint slot = args.ring != 0u ? (pos % args.capacity) : pos;
    const ulong src = ulong(gid.y) * args.kvWidth + gid.x;
    const ulong dst = ulong(slot) * args.kvWidth + gid.x;
    kCache[dst] = half(k[src]);
    vCache[dst] = half(v[src]);
}

struct DraftBlockAttnArgs {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint ctxLength;     // number of committed context tokens before block
    uint blockTokens;   // B (8)
    uint capacity;
    uint ring;
    uint window;        // sliding window for context (0 = full global)
    float scale;
};

// Bidirectional block attention over [context KV cache (FP16), block KV (FP32)].
// One threadgroup per (blockToken, qHead), 128 threads (1 per headDim lane).
kernel void draft_block_attention(
    device const float* q [[buffer(0)]],
    device const half* kCache [[buffer(1)]],
    device const half* vCache [[buffer(2)]],
    device const float* blockK [[buffer(3)]],
    device const float* blockV [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant DraftBlockAttnArgs& args [[buffer(6)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    const uint qToken = groupIndex / args.numHeads;
    const uint qHead = groupIndex % args.numHeads;
    const uint kvHead = qHead / (args.numHeads / args.numKVHeads);
    const uint kvWidth = args.numKVHeads * args.headDim;
    const uint sgs = (threads + 31u) / 32u;

    const float qVal = tid < args.headDim
        ? q[(ulong(qToken) * args.numHeads + qHead) * args.headDim + tid]
        : 0.0f;

    const uint ctxStart = (args.window > 0u && args.ctxLength > args.window)
        ? (args.ctxLength - args.window)
        : 0u;
    const uint ctxCount = args.ctxLength - ctxStart;
    const uint totalKeys = ctxCount + args.blockTokens;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint j = 0u; j < totalKeys; ++j) {
        float kVal = 0.0f;
        float vVal = 0.0f;
        if (tid < args.headDim) {
            if (j < ctxCount) {
                const uint pos = ctxStart + j;
                const uint slot = args.ring != 0u ? (pos % args.capacity) : pos;
                const ulong idx = ulong(slot) * kvWidth + kvHead * args.headDim + tid;
                kVal = float(kCache[idx]);
                vVal = float(vCache[idx]);
            } else {
                const uint bt = j - ctxCount;
                const ulong idx = ulong(bt) * kvWidth + kvHead * args.headDim + tid;
                kVal = blockK[idx];
                vVal = blockV[idx];
            }
        }
        const float dot = draft_tg_sum(qVal * kVal, scratch, sg, lane, sgs) * args.scale;
        const float mNew = max(m, dot);
        const float alpha = exp(m - mNew);
        const float p = exp(dot - mNew);
        acc = acc * alpha + p * vVal;
        l = l * alpha + p;
        m = mNew;
    }

    if (tid < args.headDim) {
        out[(ulong(qToken) * args.numHeads + qHead) * args.headDim + tid] =
            (l > 0.0f) ? (acc / l) : 0.0f;
    }
}

// Elementwise SwiGLU: out[i] = silu(gate[i]) * up[i].
kernel void draft_silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// Copy [width] BF16 vector into `rows` rows of FP32 `dst` starting at `rowOffset`.
kernel void draft_copy_bf16_broadcast(
    device const ushort* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& rowOffset [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width) return;
    dst[ulong(rowOffset + gid.y) * width + gid.x] = draft_bf16_to_f32(src[gid.x]);
}

// Recurrent Markov head gate: raw[3 * rank], stateIn[rank] -> stateOut[rank], latentOut[rank].
kernel void draft_markov_gate(
    device const float* raw [[buffer(0)]],
    device const float* stateIn [[buffer(1)]],
    device float* stateOut [[buffer(2)]],
    device float* latentOut [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= rank) return;
    const float gate = 1.0f / (1.0f + exp(-raw[c]));
    const float cand = tanh(raw[rank + c]);
    const float outv = raw[2u * rank + c];
    stateOut[c] = gate * stateIn[c] + (1.0f - gate) * cand;
    latentOut[c] = tanh(outv);
}

// Add base + bias + foldBias and apply optional softcap: out[i] = softcap * tanh(s / softcap).
kernel void draft_add_softcap(
    device const float* base [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const float* foldBias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant float& softcap [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const float s = base[gid] + bias[gid] + foldBias[gid];
    out[gid] = softcap > 0.0f ? (softcap * tanh(s / softcap)) : s;
}

struct DraftConvSnapshotArgs {
    uint channels;
    uint slots;
    uint tokens;
    uint snapStrideFloats;  // stride between per-token snapshots (convChannels * 3)
    uint channelOffset;     // starting channel inside the snapshot (0, kvWidth, 2*kvWidth, 2*kvWidth+hidden)
};

// Captures the exact 3-tap short-conv state after each token `a` in 0..<tokens
// from pre-chunk `state` ([channels][3]) and chunk inputs `x`.
kernel void draft_conv_snapshots(
    device const float* x [[buffer(0)]],
    device const float* state [[buffer(1)]],
    device const float* routeWeights [[buffer(2)]],
    device float* snapshots [[buffer(3)]],
    constant DraftConvSnapshotArgs& args [[buffer(4)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= args.channels) return;
    float s0 = state[c * 3u + 0u];
    float s1 = state[c * 3u + 1u];
    float s2 = state[c * 3u + 2u];
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
        s0 = s1;
        s1 = s2;
        s2 = xv;
        device float* dst = snapshots + ulong(t) * args.snapStrideFloats
            + (args.channelOffset + c) * 3u;
        dst[0] = s0;
        dst[1] = s1;
        dst[2] = s2;
    }
}

// Copy [tokens, width] from `src` (with row stride `srcStride`) into `dst` at tap index `tap`
// of `[tokens, taps, width]`.
kernel void draft_capture_tap(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& taps [[buffer(3)]],
    constant uint& tapIndex [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width) return;
    const uint t = gid.y;
    dst[(ulong(t) * taps + tapIndex) * width + gid.x] = src[ulong(t) * width + gid.x];
}

// Add BF16 bias vector to each row of [rows, width] FP32.
kernel void draft_add_bf16_bias(
    device float* y [[buffer(0)]],
    device const ushort* bias [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width) return;
    y[ulong(gid.y) * width + gid.x] += draft_bf16_to_f32(bias[gid.x]);
}

// Copy [width] FP32 vector from `src` to `dst` at `dstOffset` floats.
kernel void draft_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& dstOffset [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= width) return;
    dst[dstOffset + gid] = src[gid];
}
