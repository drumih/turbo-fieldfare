#include <metal_stdlib>
using namespace metal;

// Attention, routing and elementwise ops of the depth decoder, for a chunk of
// tokens (one at decode); see depth_decode_ops.metal for the step buffer.

struct DepthAttentionArgs {
    uint heads;
    uint kvHeads;
    uint headDim;
    uint relative;   // relative bias rank (16)
    uint extent;     // relative bias extent; distances past it get bias 0
    uint capacity;   // cache slots
    uint ring;       // 1: a ring of `capacity` slots (sliding-window layers)
    float scale;     // 1 / sqrt(headDim), applied to q.k only
    uint window;     // sliding window: keys at distance >= window are masked
    uint tokens;     // chunk length; the chunk wrote positions p0 ..< p0 + tokens
};

constant uint ATTENTION_CHUNK = 1024u;

// One threadgroup per (query head, chunk token). For key j at distance d,
// logit_j = scale * q.k_j + b(d) - sink, b(d) = R . bias[d] for d < extent else
// 0, plus one extra logit 0 (the sink). The bias and sink are not scaled: this
// is llama.cpp's flash-attention path, where they form the additive mask, and
// the checkpoint's own reference. out = sigmoid(gate) * sum_j softmax_j * v_j,
// with an online softmax over 1,024-key chunks. K/V caches are half
// [capacity][kvHeads][headDim]; q, r, sinks, gate and out are token-major.
kernel void relative_attention_decode(
    device const float* q [[buffer(0)]],
    device const half* kCache [[buffer(1)]],
    device const half* vCache [[buffer(2)]],
    device const float* r [[buffer(3)]],
    device const float* relativeBias [[buffer(4)]],
    device const float* sinks [[buffer(5)]],
    device float* out [[buffer(6)]],
    device const uint* step [[buffer(7)]],
    constant DepthAttentionArgs& args [[buffer(8)]],
    device const float* gate [[buffer(9)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    threadgroup float probabilities[ATTENTION_CHUNK];
    threadgroup float partial[8][128];
    const uint head = groupIndex % args.heads;   // flat grid of heads x tokens
    const uint token = groupIndex / args.heads;
    const uint simdgroups = (threads + 31u) / 32u;
    const uint position = step[0] + token;
    const uint last = step[0] + args.tokens - 1u;  // latest position in the cache
    const uint kvHead = head / (args.heads / args.kvHeads);
    const uint rowWidth = args.kvHeads * args.headDim;
    const uint qWidth = args.heads * args.headDim;
    device const float* qh = q + token * qWidth + head * args.headDim;
    device const float* rh = r + (token * args.heads + head) * args.relative;
    const float sink = sinks[token * args.heads + head];
    // Slots to scan: every slot written so far (ring), or positions 0 ... p.
    const uint slots = args.ring != 0u ? min(last + 1u, args.capacity) : position + 1u;

    for (uint d = lane; d < args.headDim; d += 32u) partial[sg][d] = 0.0f;
    float running = 0.0f;   // running max, starting at the sink logit 0
    float total = 0.0f;     // running sum of exp(logit - running)
    for (uint first = 0u; first < slots; first += ATTENTION_CHUNK) {
        const uint count = min(ATTENTION_CHUNK, slots - first);
        float chunkPeak = -INFINITY;
        for (uint i = tid; i < count; i += threads) {
            const uint j = first + i;
            // The position this slot holds, then its distance to the query.
            const uint held = args.ring != 0u ? last - (last + args.capacity - j) % args.capacity : j;
            float logit = -INFINITY;
            if (held <= position && position - held < args.window) {
                const uint distance = position - held;
                device const half* kj = kCache + ulong(j) * rowWidth + kvHead * args.headDim;
                float dot = 0.0f;
                for (uint e = 0u; e < args.headDim; ++e) dot += qh[e] * float(kj[e]);
                float bias = 0.0f;
                if (distance < args.extent) {
                    device const float* b = relativeBias + distance * args.relative;
                    for (uint e = 0u; e < args.relative; ++e) bias += rh[e] * b[e];
                }
                logit = args.scale * dot + bias - sink;
            }
            probabilities[i] = logit;
            chunkPeak = max(chunkPeak, logit);
        }
        chunkPeak = simd_max(chunkPeak);
        if (lane == 0u) scratch[sg] = chunkPeak;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float peak = running;
        for (uint i = 0u; i < simdgroups; ++i) peak = max(peak, scratch[i]);
        const float rescale = exp(running - peak);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        for (uint i = tid; i < count; i += threads) {
            const float p = exp(probabilities[i] - peak);
            probabilities[i] = p;
            sum += p;
        }
        sum = simd_sum(sum);
        if (lane == 0u) scratch[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float chunkSum = 0.0f;
        for (uint i = 0u; i < simdgroups; ++i) chunkSum += scratch[i];
        total = total * rescale + chunkSum;
        running = peak;
        // Each SIMD group accumulates a stride of positions for all dimensions.
        for (uint d = lane; d < args.headDim; d += 32u) {
            float acc = partial[sg][d] * rescale;
            for (uint i = sg; i < count; i += simdgroups) {
                if (probabilities[i] == 0.0f) continue;
                acc += probabilities[i]
                    * float(vCache[ulong(first + i) * rowWidth + kvHead * args.headDim + d]);
            }
            partial[sg][d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denominator = total + exp(-running);  // the sink's exp(0 - running)
    for (uint d = tid; d < args.headDim; d += threads) {
        float acc = 0.0f;
        for (uint i = 0u; i < simdgroups; ++i) acc += partial[i][d];
        const uint at = token * qWidth + head * args.headDim + d;
        out[at] = acc / denominator / (1.0f + exp(-gate[at]));
    }
}

// The same attention for a prefill chunk, tiled: one threadgroup per (query
// head, block of 16 chunk tokens), keys in blocks of 16. S = Q . K^T and
// O += P . V run on fp32 simdgroup MMA; each query row keeps an online softmax
// (max starting at the sink logit 0) exactly as relative_attention_decode,
// with the same bias, sink, window and ring masking. The per-(head, token)
// kernel re-read every key per query and dotted serially: 16.8 ms per
// 512-token visit. Needs headDim == 128; 128 threads (four SIMD groups).
constant constexpr uint ATTN_QB = 16u;   // queries per threadgroup
constant constexpr uint ATTN_KB = 16u;   // keys per block
constant constexpr uint ATTN_D = 128u;   // head dimension

kernel void relative_attention_chunk(
    device const float* q [[buffer(0)]],
    device const half* kCache [[buffer(1)]],
    device const half* vCache [[buffer(2)]],
    device const float* r [[buffer(3)]],
    device const float* relativeBias [[buffer(4)]],
    device const float* sinks [[buffer(5)]],
    device float* out [[buffer(6)]],
    device const uint* step [[buffer(7)]],
    constant DepthAttentionArgs& args [[buffer(8)]],
    device const float* gate [[buffer(9)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sRunning[ATTN_QB];
    threadgroup float sTotal[ATTN_QB];
    threadgroup float sQ[ATTN_QB * ATTN_D];
    threadgroup float sKV[ATTN_KB * ATTN_D];   // K block, then V block
    threadgroup float sO[ATTN_QB * ATTN_D];
    threadgroup float sS[ATTN_QB * ATTN_KB];   // scores, then probabilities
    threadgroup float sScale[ATTN_QB];
    const uint head = groupIndex % args.heads;
    const uint qBase = (groupIndex / args.heads) * ATTN_QB;
    const uint kvHead = head / (args.heads / args.kvHeads);
    const uint rowWidth = args.kvHeads * ATTN_D;
    const uint qWidth = args.heads * ATTN_D;
    const uint first = step[0];
    const uint last = first + args.tokens - 1u;
    const uint queries = min(ATTN_QB, args.tokens - qBase);
    const uint newest = first + qBase + queries - 1u;
    // Keys past the block's newest query are masked, so the scan stops there
    // unless the ring has wrapped (then slot order is not position order).
    const bool wrapped = args.ring != 0u && last + 1u > args.capacity;
    const uint slots = wrapped ? args.capacity : newest + 1u;

    for (uint i = tid; i < ATTN_QB * ATTN_D; i += 128u) {
        const uint t = i / ATTN_D;
        sQ[i] = t < queries ? q[(qBase + t) * qWidth + head * ATTN_D + i % ATTN_D] : 0.0f;
        sO[i] = 0.0f;
    }
    // Row state: running max (from the sink's logit 0) and sum per query row.
    if (tid < ATTN_QB) {
        sRunning[tid] = 0.0f;
        sTotal[tid] = 0.0f;
    }
    // Each thread scores (row tid / 16 + 8 * pass, key tid % 16): a row's 16
    // keys are one half of a SIMD group, reduced with xor shuffles.
    const uint key = tid % ATTN_KB;

    for (uint keyBase = 0u; keyBase < slots; keyBase += ATTN_KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < ATTN_KB * ATTN_D; i += 128u) {
            const uint j = keyBase + i / ATTN_D;
            sKV[i] = j < slots ? float(kCache[ulong(j) * rowWidth + kvHead * ATTN_D + i % ATTN_D]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // S tile (sg / 2, sg % 2) of the 16 x 16 scores.
            const uint tr = (sg / 2u) * 8u;
            const uint tc = (sg % 2u) * 8u;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0u; d < ATTN_D; d += 8u) {
                simdgroup_float8x8 mq;
                simdgroup_float8x8 mk;
                simdgroup_load(mq, sQ + tr * ATTN_D + d, ATTN_D);
                simdgroup_load(mk, sKV + tc * ATTN_D + d, ATTN_D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, mq, mk, acc);
            }
            simdgroup_store(acc, sS + tr * ATTN_KB + tc, ATTN_KB);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pass = 0u; pass < 2u; ++pass) {
            const uint row = tid / ATTN_KB + 8u * pass;
            const uint rowToken = qBase + row;
            const uint rowPosition = first + rowToken;
            const uint j = keyBase + key;
            float logit = -INFINITY;
            if (row < queries && j < slots) {
                const uint held = args.ring != 0u
                    ? last - (last + args.capacity - j) % args.capacity : j;
                if (held <= rowPosition && rowPosition - held < args.window) {
                    const uint distance = rowPosition - held;
                    float bias = 0.0f;
                    if (distance < args.extent) {
                        device const float* rh = r + (rowToken * args.heads + head) * args.relative;
                        device const float* b = relativeBias + distance * args.relative;
                        for (uint e = 0u; e < args.relative; ++e) bias += rh[e] * b[e];
                    }
                    logit = args.scale * sS[row * ATTN_KB + key] + bias
                        - sinks[rowToken * args.heads + head];
                }
            }
            float rowPeak = logit;
            for (ushort m = 8u; m >= 1u; m >>= 1u) rowPeak = max(rowPeak, simd_shuffle_xor(rowPeak, m));
            const float running = sRunning[row];
            const float peak = max(running, rowPeak);
            const float pr = exp(logit - peak);
            float rowSum = pr;
            for (ushort m = 8u; m >= 1u; m >>= 1u) rowSum += simd_shuffle_xor(rowSum, m);
            sS[row * ATTN_KB + key] = pr;
            if (key == 0u) {
                const float rescale = exp(running - peak);
                sTotal[row] = sTotal[row] * rescale + rowSum;
                sRunning[row] = peak;
                sScale[row] = rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < ATTN_KB * ATTN_D; i += 128u) {
            const uint j = keyBase + i / ATTN_D;
            sKV[i] = j < slots ? float(vCache[ulong(j) * rowWidth + kvHead * ATTN_D + i % ATTN_D]) : 0.0f;
        }
        for (uint i = tid; i < ATTN_QB * ATTN_D; i += 128u) sO[i] *= sScale[i / ATTN_D];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O columns sg * 32 ..< + 32: 2 row tiles x 4 column tiles.
        for (uint rt = 0u; rt < 2u; ++rt) {
            for (uint ct = 0u; ct < 4u; ++ct) {
                threadgroup float* o = sO + rt * 8u * ATTN_D + sg * 32u + ct * 8u;
                simdgroup_float8x8 acc;
                simdgroup_load(acc, o, ATTN_D);
                for (uint ks = 0u; ks < ATTN_KB; ks += 8u) {
                    simdgroup_float8x8 mp;
                    simdgroup_float8x8 mv;
                    simdgroup_load(mp, sS + rt * 8u * ATTN_KB + ks, ATTN_KB);
                    simdgroup_load(mv, sKV + ks * ATTN_D + sg * 32u + ct * 8u, ATTN_D);
                    simdgroup_multiply_accumulate(acc, mp, mv, acc);
                }
                simdgroup_store(acc, o, ATTN_D);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ATTN_QB) sScale[tid] = 1.0f / (sTotal[tid] + exp(-sRunning[tid]));  // + the sink's exp(0 - max)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < queries * ATTN_D; i += 128u) {
        const uint t = i / ATTN_D;
        const uint at = (qBase + t) * qWidth + head * ATTN_D + i % ATTN_D;
        out[at] = sO[i] * sScale[t] / (1.0f + exp(-gate[at]));
    }
}

struct DepthRouterArgs {
    uint experts;
    uint width;
    uint used;
    float weightScale;
    float normEpsilon;
};

// Sigmoid top-k over router logits (computed by a parallel GEMV), in one SIMD
// group: p = sigmoid(logit); top-`used` by p + bias, ties to the lower index;
// weights are the chosen p normalised by their sum (+ eps) times weightScale.
// Up to 32 * 8 experts (each lane holds `experts / 32` of them). One SIMD
// group per token: logits [tokens][experts], ids and weights [tokens][used].
kernel void sigmoid_topk(
    device const float* allLogits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device int* allIds [[buffer(2)]],
    device float* allWeights [[buffer(3)]],
    constant DepthRouterArgs& args [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    device const float* logits = allLogits + token * args.experts;
    device int* ids = allIds + token * args.used;
    device float* routeWeights = allWeights + token * args.used;
    const uint perLane = (args.experts + 31u) / 32u;
    float probs[8];
    float selection[8];
    for (uint i = 0u; i < perLane; ++i) {
        const uint e = lane * perLane + i;
        probs[i] = e < args.experts ? 1.0f / (1.0f + exp(-logits[e])) : 0.0f;
        selection[i] = e < args.experts ? probs[i] + bias[e] : -INFINITY;
    }
    float total = 0.0f;
    int chosen = 0;
    float chosenProb = 0.0f;
    for (uint k = 0u; k < args.used; ++k) {
        float best = -INFINITY;
        uint bestIndex = 0xFFFFFFFFu;
        for (uint i = 0u; i < perLane; ++i) {
            if (selection[i] > best) {
                best = selection[i];
                bestIndex = lane * perLane + i;
            }
        }
        for (uint offset = 16u; offset > 0u; offset /= 2u) {
            const float otherValue = simd_shuffle_xor(best, offset);
            const uint otherIndex = simd_shuffle_xor(bestIndex, offset);
            if (otherValue > best || (otherValue == best && otherIndex < bestIndex)) {
                best = otherValue;
                bestIndex = otherIndex;
            }
        }
        // The owning lane retires the winner and reports its probability.
        float prob = 0.0f;
        if (bestIndex / perLane == lane) {
            prob = probs[bestIndex % perLane];
            selection[bestIndex % perLane] = -INFINITY;
        }
        prob = simd_sum(prob);
        total += prob;
        if (lane == k) {
            chosen = int(bestIndex);
            chosenProb = prob;
        }
    }
    if (lane < args.used) {
        ids[lane] = chosen;
        routeWeights[lane] = chosenProb / (total + args.normEpsilon) * args.weightScale;
    }
}

struct DepthSituArgs {
    uint feedForward;
    uint slots;
    float beta;
    float linearBeta;
};

// act[s][j] = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), g and u the two
// halves of slot s's gate_up output.
kernel void situ_activation(
    device const float* gateUp [[buffer(0)]],
    device float* act [[buffer(1)]],
    constant DepthSituArgs& args [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= args.feedForward * args.slots) return;
    const uint slot = index / args.feedForward;
    const uint j = index % args.feedForward;
    const float g = gateUp[slot * 2u * args.feedForward + j];
    const float u = gateUp[slot * 2u * args.feedForward + args.feedForward + j];
    const float gated = args.beta * tanh(g / args.beta) / (1.0f + exp(-g));
    act[index] = gated * args.linearBeta * tanh(u / args.linearBeta);
}

// Two-pass argmax over `count` logits: pass 1 writes one (value, index) per
// threadgroup, pass 2 (one threadgroup, `partials` entries) reduces them.
kernel void argmax_partial(
    device const float* logits [[buffer(0)]],
    device float2* partials [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint groupIndex [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint groups [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float bestValue[32];
    threadgroup uint bestIndex[32];
    float value = -INFINITY;
    uint index = 0u;
    for (uint i = groupIndex * threads + tid; i < count; i += threads * groups) {
        if (logits[i] > value) {
            value = logits[i];
            index = i;
        }
    }
    for (uint offset = 16u; offset > 0u; offset /= 2u) {
        const float otherValue = simd_shuffle_down(value, offset);
        const uint otherIndex = simd_shuffle_down(index, offset);
        if (otherValue > value || (otherValue == value && otherIndex < index)) {
            value = otherValue;
            index = otherIndex;
        }
    }
    if (lane == 0u) {
        bestValue[sg] = value;
        bestIndex[sg] = index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint i = 1u; i < (threads + 31u) / 32u; ++i) {
            if (bestValue[i] > value || (bestValue[i] == value && bestIndex[i] < index)) {
                value = bestValue[i];
                index = bestIndex[i];
            }
        }
        partials[groupIndex] = float2(value, as_type<float>(index));
    }
}

kernel void argmax_final(
    device const float2* partials [[buffer(0)]],
    device int* token [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tid != 0u) return;
    float value = -INFINITY;
    uint index = 0u;
    for (uint i = 0u; i < count; ++i) {
        const uint candidate = as_type<uint>(partials[i].y);
        if (partials[i].x > value || (partials[i].x == value && candidate < index)) {
            value = partials[i].x;
            index = candidate;
        }
    }
    token[0] = int(index);
}
