#include <metal_stdlib>
using namespace metal;

// Small dense f32/bf16 matrices of the depth decoder (router, conv taps,
// sinks) and the loop message controllers. Compiled after depth_decode_ops
// (one combined source), whose DepthGemvArgs and DepthControlArgs these use.
//
// At one token the kernels give each output row its own threadgroup or SIMD
// group; at a chunk they tile rows x tokens in registers, because a
// per-(row, token) kernel re-reads the weight row for every token.

// The same GEMV for matrices of few rows (router 128, conv taps 32, sinks
// 24): one threadgroup of 128 threads per (row, token), four values per
// thread per step, so a 24-row matrix is not six threadgroups of serial lanes.
// Needs columns % 4 == 0; a flat grid of rows x tokens.
kernel void dense_gemv_wide_rows(
    device const uchar* weights [[buffer(0)]],
    device const float* xAll [[buffer(1)]],
    device float* yAll [[buffer(2)]],
    constant DepthGemvArgs& args [[buffer(3)]],
    constant uint& isBF16 [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[4];
    const uint row = groupIndex % args.rows;
    const uint token = groupIndex / args.rows;
    device const float4* x = (device const float4*)(xAll + ulong(token) * args.columns);
    const uint quads = args.columns / 4u;
    float sum = 0.0f;
    if (isBF16 != 0u) {
        device const ushort4* w = (device const ushort4*)((device const ushort*)weights + ulong(row) * args.columns);
        for (uint i = tid; i < quads; i += 128u) {
            const uint4 bits = uint4(w[i]) << 16u;
            sum += dot(as_type<float4>(bits), x[i]);
        }
    } else {
        device const float4* w = (device const float4*)((device const float*)weights + ulong(row) * args.columns);
        for (uint i = tid; i < quads; i += 128u) sum += dot(w[i], x[i]);
    }
    sum = simd_sum(sum);
    if (lane == 0u) scratch[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        yAll[ulong(token) * args.rows + row] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    }
}

// y[t][r] = W[r] . x[t] for few rows over a chunk (router, conv taps, sinks,
// controller downs at prefill). A threadgroup holds an 8-row x 8-token tile;
// its four SIMD groups split the columns (quads q = lane + 32 * sg, stride
// 128) with an 8 x 8 register block each, reduced through threadgroup memory,
// so each weight and activation quad feeds eight FMAs and even a 24-row
// matrix at 64 tokens is 24 threadgroups. Needs columns % 4 == 0; flat grid of
// row tiles x token tiles, 128 threads.
kernel void dense_gemm_small_rows(
    device const uchar* weights [[buffer(0)]],
    device const float* xAll [[buffer(1)]],
    device float* yAll [[buffer(2)]],
    constant DepthGemvArgs& args [[buffer(3)]],
    constant uint& isBF16 [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[4][64];
    const uint rowTiles = (args.rows + 7u) / 8u;
    const uint row0 = (groupIndex % rowTiles) * 8u;
    const uint token0 = (groupIndex / rowTiles) * 8u;
    const uint quads = args.columns / 4u;
    float acc[8][8];
    for (uint r = 0u; r < 8u; ++r) {
        for (uint t = 0u; t < 8u; ++t) acc[r][t] = 0.0f;
    }
    for (uint q = lane + 32u * sg; q < quads; q += 128u) {
        float4 xv[8];
        for (uint t = 0u; t < 8u; ++t) {
            const uint token = min(token0 + t, tokens - 1u);
            xv[t] = ((device const float4*)(xAll + ulong(token) * args.columns))[q];
        }
        for (uint r = 0u; r < 8u; ++r) {
            const ulong row = min(row0 + r, args.rows - 1u);
            float4 wv;
            if (isBF16 != 0u) {
                wv = as_type<float4>(uint4(((device const ushort4*)((device const ushort*)weights + row * args.columns))[q]) << 16u);
            } else {
                wv = ((device const float4*)((device const float*)weights + row * args.columns))[q];
            }
            for (uint t = 0u; t < 8u; ++t) acc[r][t] += dot(wv, xv[t]);
        }
    }
    for (uint r = 0u; r < 8u; ++r) {
        for (uint t = 0u; t < 8u; ++t) {
            const float total = simd_sum(acc[r][t]);
            if (lane == 0u) partial[sg][r * 8u + t] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64u) {
        const uint r = tid / 8u;
        const uint t = tid % 8u;
        if (row0 + r < args.rows && token0 + t < tokens) {
            yAll[ulong(token0 + t) * args.rows + row0 + r] =
                partial[0][tid] + partial[1][tid] + partial[2][tid] + partial[3][tid];
        }
    }
}

// joined[t] = [rms(cur[t]); rms(message[t])], the controller's 2 * width input,
// one threadgroup per token; at a chunk h = D . joined then runs on dense_gemm_small_rows.
kernel void control_join(
    device const float* cur [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device float* joined [[buffer(2)]],
    constant DepthControlArgs& args [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    const uint simdgroups = (threads + 31u) / 32u;
    device const float* c = cur + token * args.width;
    device const float* m = message + token * args.width;
    device float* j = joined + token * 2u * args.width;
    float a = 0.0f;
    float b = 0.0f;
    for (uint i = tid; i < args.width; i += threads) {
        a += c[i] * c[i];
        b += m[i] * m[i];
    }
    const float invCur = rsqrt(
        depth_threadgroup_sum(a, scratch, sg, lane, simdgroups) / float(args.width) + args.epsilon);
    const float invMessage = rsqrt(
        depth_threadgroup_sum(b, scratch, sg, lane, simdgroups) / float(args.width) + args.epsilon);
    for (uint i = tid; i < args.width; i += threads) {
        j[i] = c[i] * invCur;
        j[args.width + i] = m[i] * invMessage;
    }
}

// h[t][r] = D[r] . [rms(cur[t]), rms(message[t])] for the loop controller's
// rank-r down projection D ([rank][2 * width], f32), computed as
// rsqrt(mean(cur^2) + eps) * (D[r][:w] . cur) + the same for message: one
// threadgroup per (row, token), so all its threads share the 2 * width dot
// (16 threadgroups of serial lanes was ~60 us per controller) and the join
// pass disappears. Decode only: at a chunk it would re-read D per token.
kernel void control_down(
    device const float* cur [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device const float* down [[buffer(2)]],
    device float* h [[buffer(3)]],
    constant DepthControlArgs& args [[buffer(4)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[4][32];
    const uint r = groupIndex % args.rank;
    const uint token = groupIndex / args.rank;
    device const float* c = cur + token * args.width;
    device const float* m = message + token * args.width;
    device const float* w = down + ulong(r) * 2u * args.width;
    float dc = 0.0f, dm = 0.0f, cc = 0.0f, mm = 0.0f;
    for (uint i = tid; i < args.width; i += threads) {
        const float ci = c[i];
        const float mi = m[i];
        dc = fma(w[i], ci, dc);
        dm = fma(w[args.width + i], mi, dm);
        cc = fma(ci, ci, cc);
        mm = fma(mi, mi, mm);
    }
    dc = simd_sum(dc);
    dm = simd_sum(dm);
    cc = simd_sum(cc);
    mm = simd_sum(mm);
    if (lane == 0u) {
        scratch[0][sg] = dc;
        scratch[1][sg] = dm;
        scratch[2][sg] = cc;
        scratch[3][sg] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float4 total = 0.0f;
        for (uint s = 0u; s < (threads + 31u) / 32u; ++s) {
            total += float4(scratch[0][s], scratch[1][s], scratch[2][s], scratch[3][s]);
        }
        const float width = float(args.width);
        h[token * args.rank + r] = total.x * rsqrt(total.z / width + args.epsilon)
            + total.y * rsqrt(total.w / width + args.epsilon);
    }
}

// out = (1 - tanh(F . h)) * cur + tanh(G . h) * message; F, G are [width][rank].
// One SIMD group per output reads the two contiguous rank-length rows; a flat
// grid of width / 4 threadgroups per token.
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
    const uint groupsPerToken = (args.width + 3u) / 4u;
    const uint index = (groupIndex % groupsPerToken) * 4u + sg;
    const uint token = groupIndex / groupsPerToken;
    if (index >= args.width) return;
    device const float* ht = h + token * args.rank;
    float g = 0.0f;
    float f = 0.0f;
    for (uint r = lane; r < args.rank; r += 32u) {
        g += gain[index * args.rank + r] * ht[r];
        f += forget[index * args.rank + r] * ht[r];
    }
    g = simd_sum(g);
    f = simd_sum(f);
    const uint at = token * args.width + index;
    if (lane == 0u) out[at] = (1.0f - tanh(f)) * cur[at] + tanh(g) * message[at];
}

// control_apply for a chunk, bit for bit: one SIMD group per (output index,
// block of 16 tokens) loads its gain and forget rows once (lanes hold r = lane
// and lane + 32, the per-token kernel's order) and reuses them for every token
// of the block, with the same per-lane sums and simd_sum. The per-token kernel
// re-read both 2,304 x 64 matrices for every token (~75 MB per controller at
// 64 tokens). Needs rank <= 64; a flat grid of (width / 4) x token blocks,
// 128 threads.
constant constexpr uint CONTROL_TOKEN_BLOCK = 16u;

kernel void control_apply_chunk(
    device const float* cur [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device const float* h [[buffer(2)]],
    device const float* gain [[buffer(3)]],
    device const float* forget [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant DepthControlArgs& args [[buffer(6)]],
    constant uint& tokens [[buffer(7)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint groupsPerBlock = (args.width + 3u) / 4u;
    const uint index = (groupIndex % groupsPerBlock) * 4u + sg;
    const uint token0 = (groupIndex / groupsPerBlock) * CONTROL_TOKEN_BLOCK;
    if (index >= args.width) return;
    const bool has0 = lane < args.rank;
    const bool has1 = lane + 32u < args.rank;
    const float g0 = has0 ? gain[index * args.rank + lane] : 0.0f;
    const float g1 = has1 ? gain[index * args.rank + lane + 32u] : 0.0f;
    const float f0 = has0 ? forget[index * args.rank + lane] : 0.0f;
    const float f1 = has1 ? forget[index * args.rank + lane + 32u] : 0.0f;
    const uint end = min(token0 + CONTROL_TOKEN_BLOCK, tokens);
    for (uint token = token0; token < end; ++token) {
        device const float* ht = h + token * args.rank;
        float g = 0.0f;
        float f = 0.0f;
        if (has0) {
            g += g0 * ht[lane];
            f += f0 * ht[lane];
        }
        if (has1) {
            g += g1 * ht[lane + 32u];
            f += f1 * ht[lane + 32u];
        }
        g = simd_sum(g);
        f = simd_sum(f);
        const uint at = token * args.width + index;
        if (lane == 0u) out[at] = (1.0f - tanh(f)) * cur[at] + tanh(g) * message[at];
    }
}

// short_conv_step for a chunk, in parallel: the 4-tap window covers inputs
// only, so out[t][c] needs x[t - 3 ... t] (the state for positions before the
// chunk), never an earlier output. One thread per (token, channel) computes
// out[t][c] with short_conv_step's expressions; short_conv_state then shifts
// the last three inputs into the state (a separate dispatch, after every read
// of the old state). The sequential kernel walked 64 tokens per channel on
// 6,656 threads. Grid: tokens x channels.
static inline float depth_conv_input(
    device const float* x, device const float* routeWeights, constant DepthConvArgs& args,
    uint t, uint c)
{
    if (args.slots == 0u) return x[t * args.channels + c];
    float xv = 0.0f;
    for (uint s = 0u; s < args.slots; ++s) {
        xv += routeWeights[t * args.slots + s] * x[(t * args.slots + s) * args.channels + c];
    }
    return xv;
}

kernel void short_conv_chunk(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const float* routeWeights [[buffer(4)]],
    device const float* residual [[buffer(5)]],
    constant DepthConvArgs& args [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    const uint c = index % args.channels;
    const uint t = index / args.channels;
    if (t >= args.tokens) return;
    float window[3];
    for (uint k = 0u; k < 3u; ++k) {
        const int source = int(t) - 3 + int(k);
        window[k] = source >= 0
            ? depth_conv_input(x, routeWeights, args, uint(source), c)
            : state[c * 3u + uint(source + 3)];
    }
    const float xv = depth_conv_input(x, routeWeights, args, t, c);
    const float conv = window[0] * weight[c * 4u + 0u] + window[1] * weight[c * 4u + 1u]
        + window[2] * weight[c * 4u + 2u] + xv * weight[c * 4u + 3u];
    out[t * args.channels + c] = (xv + conv) * args.scale
        + (args.hasResidual != 0u ? residual[t * args.channels + c] : 0.0f);
}

kernel void short_conv_state(
    device const float* x [[buffer(0)]],
    device float* state [[buffer(2)]],
    device const float* routeWeights [[buffer(4)]],
    constant DepthConvArgs& args [[buffer(6)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= args.channels) return;
    float next[3];
    for (uint k = 0u; k < 3u; ++k) {
        const int source = int(args.tokens) - 3 + int(k);
        next[k] = source >= 0
            ? depth_conv_input(x, routeWeights, args, uint(source), c)
            : state[c * 3u + uint(source + 3)];
    }
    for (uint k = 0u; k < 3u; ++k) state[c * 3u + k] = next[k];
}

// kv_finish for a chunk, in parallel: the K/V convolutions' windows cover
// inputs only, so each (K or V head, token) is its own threadgroup with
// kv_finish's expressions and reduction; kv_finish_state then shifts the last
// three inputs into the state once every read of the old state is done. The
// sequential kernel ran 16 threadgroups walking the chunk with two barriers
// per token. Grid: 2 * kvHeads * tokens threadgroups of headDim threads.
kernel void kv_finish_chunk(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* kWeight [[buffer(2)]],
    device const float* vWeight [[buffer(3)]],
    device const float* state [[buffer(4)]],
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
    const uint t = groupIndex / (2u * args.kvHeads);
    const uint unit = groupIndex % (2u * args.kvHeads);
    const bool isV = unit >= args.kvHeads;
    const uint head = unit % args.kvHeads;
    const uint width = args.kvHeads * args.headDim;
    const uint c = head * args.headDim + tid;
    device const float* xs = isV ? v : k;
    device const float* weight = isV ? vWeight : kWeight;
    device const float* dynamic = isV ? vDynamic : kDynamic;
    device const float* st = state + (isV ? width : 0u) * 3u;
    device half* cache = isV ? vCache : kCache;
    float window[3];
    for (uint i = 0u; i < 3u; ++i) {
        const int source = int(t) - 3 + int(i);
        window[i] = source >= 0 ? xs[uint(source) * width + c] : st[c * 3u + uint(source + 3)];
    }
    const float s0 = window[0];
    const float s1 = window[1];
    const float s2 = window[2];
    const float xv = xs[t * width + c];
    float conv = s0 * weight[c * 4u + 0u] + s1 * weight[c * 4u + 1u]
        + s2 * weight[c * 4u + 2u] + xv * weight[c * 4u + 3u];
    if (args.hasDynamic != 0u) {
        device const float* taps = dynamic + (t * args.kvHeads + head) * 4u;
        conv += xv * taps[0];
        conv += s2 * taps[1];
        conv += s1 * taps[2];
        conv += s0 * taps[3];
    }
    const float value = (xv + conv) * args.scale;
    const uint simdgroups = (threads + 31u) / 32u;
    const float sq = depth_threadgroup_sum(value * value, scratch, sg, lane, simdgroups);
    const float factor = isV
        ? rsqrt(sq / float(args.headDim) + args.epsilon)
        : clamp(args.threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
    const uint position = step[0].position + t;
    const uint slot = args.ring != 0u ? position % args.capacity : position;
    cache[ulong(slot) * width + c] = half(value * factor);
}

kernel void kv_finish_state(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* state [[buffer(4)]],
    constant DepthKVFinishArgs& args [[buffer(10)]],
    uint index [[thread_position_in_grid]])
{
    const uint width = args.kvHeads * args.headDim;
    if (index >= 2u * width) return;
    const bool isV = index >= width;
    const uint c = index % width;
    device const float* xs = isV ? v : k;
    device float* st = state + (isV ? width : 0u) * 3u;
    float next[3];
    for (uint i = 0u; i < 3u; ++i) {
        const int source = int(args.tokens) - 3 + int(i);
        next[i] = source >= 0 ? xs[uint(source) * width + c] : st[c * 3u + uint(source + 3)];
    }
    for (uint i = 0u; i < 3u; ++i) st[c * 3u + i] = next[i];
}
