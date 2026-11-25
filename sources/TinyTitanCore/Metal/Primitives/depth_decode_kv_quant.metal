#include <metal_stdlib>
using namespace metal;

// Affine per-32-channel quantized K/V cache kernels for the Inkling 2 depth
// decoder (--kv-bits 8 and --kv-bits 4). Compiled after depth_decode_ops,
// depth_decode_dense and depth_decode_attention so DepthStep,
// DepthKVFinishArgs, DepthAttentionArgs, depth_threadgroup_sum and the
// attention tiling constants are already in scope.

struct DepthKVGroupQ8 {
    half scale;
    half bias;
    uchar q[32];
};

struct DepthKVGroupQ4 {
    half scale;
    half bias;
    uchar q[8];
};

inline void depth_kv_store_q8(device DepthKVGroupQ8* dst, float val, uint lane) {
    const float vMin = simd_min(val);
    const float vMax = simd_max(val);
    const half hBias = half(vMin);
    const float fBias = float(hBias);
    const half hScale = half((vMax - fBias) / 255.0f);
    const float fScale = float(hScale);
    const float invScale = fScale > 0.0f ? (1.0f / fScale) : 0.0f;
    const uchar q = uchar(clamp(rint((val - fBias) * invScale), 0.0f, 255.0f));
    if (lane == 0u) {
        dst->scale = hScale;
        dst->bias = hBias;
    }
    dst->q[lane] = q;
}

inline void depth_kv_store_q4(device DepthKVGroupQ4* dst, float val, uint lane) {
    float vMin = val;
    float vMax = val;
    vMin = min(vMin, simd_shuffle_xor(vMin, 1u));
    vMax = max(vMax, simd_shuffle_xor(vMax, 1u));
    vMin = min(vMin, simd_shuffle_xor(vMin, 2u));
    vMax = max(vMax, simd_shuffle_xor(vMax, 2u));
    vMin = min(vMin, simd_shuffle_xor(vMin, 4u));
    vMax = max(vMax, simd_shuffle_xor(vMax, 4u));
    vMin = min(vMin, simd_shuffle_xor(vMin, 8u));
    vMax = max(vMax, simd_shuffle_xor(vMax, 8u));
    const half hBias = half(vMin);
    const float fBias = float(hBias);
    const half hScale = half((vMax - fBias) / 15.0f);
    const float fScale = float(hScale);
    const float invScale = fScale > 0.0f ? (1.0f / fScale) : 0.0f;
    const uint q = uint(clamp(rint((val - fBias) * invScale), 0.0f, 15.0f));
    const uint hi = simd_shuffle_xor(q, 1u);
    const uint localLane = lane & 15u;
    if (localLane == 0u) {
        dst->scale = hScale;
        dst->bias = hBias;
    }
    if ((localLane & 1u) == 0u) {
        dst->q[localLane >> 1] = uchar((q & 0xFu) | ((hi & 0xFu) << 4));
    }
}

inline float depth_kv_load_q8(device const DepthKVGroupQ8* headRow, uint ch) {
    device const DepthKVGroupQ8& g = headRow[ch >> 5];
    return float(g.bias) + float(g.scale) * float(g.q[ch & 31u]);
}

inline float depth_kv_load_q4(device const DepthKVGroupQ4* headRow, uint ch) {
    device const DepthKVGroupQ4& g = headRow[ch >> 4];
    const uint off = ch & 15u;
    const uchar packed = g.q[off >> 1];
    const uint nibble = (off & 1u) == 0u ? (packed & 0xFu) : (packed >> 4);
    return float(g.bias) + float(g.scale) * float(nibble);
}

inline float depth_kv_dot_q8(
    device const float* qh, device const DepthKVGroupQ8* kj, uint groupsPerHead)
{
    float dot = 0.0f;
    for (uint g = 0u; g < groupsPerHead; ++g) {
        const float sc = float(kj[g].scale);
        const float bi = float(kj[g].bias);
        const uint base = g << 5;
        for (uint e = 0u; e < 32u; ++e) {
            dot += qh[base + e] * (bi + sc * float(kj[g].q[e]));
        }
    }
    return dot;
}

inline float depth_kv_dot_q4(
    device const float* qh, device const DepthKVGroupQ4* kj, uint groupsPerHead)
{
    float dot = 0.0f;
    for (uint g = 0u; g < groupsPerHead; ++g) {
        const float sc = float(kj[g].scale);
        const float bi = float(kj[g].bias);
        const uint base = g << 4;
        for (uint e = 0u; e < 8u; ++e) {
            const uchar packed = kj[g].q[e];
            dot += qh[base + 2u * e] * (bi + sc * float(packed & 0xFu));
            dot += qh[base + 2u * e + 1u] * (bi + sc * float(packed >> 4));
        }
    }
    return dot;
}

template <typename GroupT, bool IsQ4>
inline void depth_kv_finish_impl(
    device const float* k, device const float* v,
    device const float* kWeight, device const float* vWeight,
    device float* state, device const float* kDynamic, device const float* vDynamic,
    device GroupT* kCache, device GroupT* vCache,
    device const DepthStep* step, constant DepthKVFinishArgs& args,
    threadgroup float* scratch,
    uint groupIndex, uint tid, uint threads, uint sg, uint lane)
{
    const bool isV = groupIndex >= args.kvHeads;
    const uint head = groupIndex % args.kvHeads;
    const uint width = args.kvHeads * args.headDim;
    const uint groupsPerHead = IsQ4 ? (args.headDim >> 4) : (args.headDim >> 5);
    const uint groupsPerRow = args.kvHeads * groupsPerHead;
    const uint c = head * args.headDim + tid;
    device const float* xs = isV ? v : k;
    device const float* weight = isV ? vWeight : kWeight;
    device const float* dynamic = isV ? vDynamic : kDynamic;
    device float* st = state + (isV ? width : 0u) * 3u;
    device GroupT* cache = isV ? vCache : kCache;
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
            conv += xv * taps[0] + s2 * taps[1] + s1 * taps[2] + s0 * taps[3];
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
        const uint groupIdx = IsQ4 ? (sg * 2u + (lane >> 4)) : sg;
        device GroupT* dst = cache + ulong(slot) * groupsPerRow + head * groupsPerHead + groupIdx;
        if (IsQ4) {
            depth_kv_store_q4(reinterpret_cast<device DepthKVGroupQ4*>(dst), value * factor, lane);
        } else {
            depth_kv_store_q8(reinterpret_cast<device DepthKVGroupQ8*>(dst), value * factor, lane);
        }
    }
    st[c * 3u + 0u] = s0;
    st[c * 3u + 1u] = s1;
    st[c * 3u + 2u] = s2;
}

template <typename GroupT, bool IsQ4>
inline void depth_kv_finish_chunk_impl(
    device const float* k, device const float* v,
    device const float* kWeight, device const float* vWeight,
    device const float* state, device const float* kDynamic, device const float* vDynamic,
    device GroupT* kCache, device GroupT* vCache,
    device const DepthStep* step, constant DepthKVFinishArgs& args,
    threadgroup float* scratch,
    uint groupIndex, uint tid, uint threads, uint sg, uint lane)
{
    const uint t = groupIndex / (2u * args.kvHeads);
    const uint unit = groupIndex % (2u * args.kvHeads);
    const bool isV = unit >= args.kvHeads;
    const uint head = unit % args.kvHeads;
    const uint width = args.kvHeads * args.headDim;
    const uint groupsPerHead = IsQ4 ? (args.headDim >> 4) : (args.headDim >> 5);
    const uint groupsPerRow = args.kvHeads * groupsPerHead;
    const uint c = head * args.headDim + tid;
    device const float* xs = isV ? v : k;
    device const float* weight = isV ? vWeight : kWeight;
    device const float* dynamic = isV ? vDynamic : kDynamic;
    device const float* st = state + (isV ? width : 0u) * 3u;
    device GroupT* cache = isV ? vCache : kCache;
    float window[3];
    for (uint i = 0u; i < 3u; ++i) {
        const int source = int(t) - 3 + int(i);
        window[i] = source >= 0 ? xs[uint(source) * width + c] : st[c * 3u + uint(source + 3)];
    }
    const float xv = xs[t * width + c];
    float conv = window[0] * weight[c * 4u + 0u] + window[1] * weight[c * 4u + 1u]
        + window[2] * weight[c * 4u + 2u] + xv * weight[c * 4u + 3u];
    if (args.hasDynamic != 0u) {
        device const float* taps = dynamic + (t * args.kvHeads + head) * 4u;
        conv += xv * taps[0] + window[2] * taps[1] + window[1] * taps[2] + window[0] * taps[3];
    }
    const float value = (xv + conv) * args.scale;
    const uint simdgroups = (threads + 31u) / 32u;
    const float sq = depth_threadgroup_sum(value * value, scratch, sg, lane, simdgroups);
    const float factor = isV
        ? rsqrt(sq / float(args.headDim) + args.epsilon)
        : clamp(args.threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
    const uint position = step[0].position + t;
    const uint slot = args.ring != 0u ? position % args.capacity : position;
    const uint groupIdx = IsQ4 ? (sg * 2u + (lane >> 4)) : sg;
    device GroupT* dst = cache + ulong(slot) * groupsPerRow + head * groupsPerHead + groupIdx;
    if (IsQ4) {
        depth_kv_store_q4(reinterpret_cast<device DepthKVGroupQ4*>(dst), value * factor, lane);
    } else {
        depth_kv_store_q8(reinterpret_cast<device DepthKVGroupQ8*>(dst), value * factor, lane);
    }
}

template <typename GroupT, bool IsQ4>
inline void relative_attention_decode_q_impl(
    device const float* q, device const GroupT* kCache, device const GroupT* vCache,
    device const float* r, device const float* relativeBias, device const float* sinks,
    device float* out, device const uint* step, constant DepthAttentionArgs& args,
    device const float* gate,
    threadgroup float* scratch, threadgroup float* probabilities, threadgroup float* partial,
    uint groupIndex, uint tid, uint threads, uint sg, uint lane)
{
    const uint head = groupIndex % args.heads;
    const uint token = groupIndex / args.heads;
    const uint simdgroups = (threads + 31u) / 32u;
    const uint position = step[0] + token;
    const uint last = step[0] + args.tokens - 1u;
    const uint kvHead = head / (args.heads / args.kvHeads);
    const uint groupsPerHead = IsQ4 ? (args.headDim >> 4) : (args.headDim >> 5);
    const uint groupsPerRow = args.kvHeads * groupsPerHead;
    const uint qWidth = args.heads * args.headDim;
    device const float* qh = q + token * qWidth + head * args.headDim;
    device const float* rh = r + (token * args.heads + head) * args.relative;
    const float sink = sinks[token * args.heads + head];
    const uint slots = args.ring != 0u ? min(last + 1u, args.capacity) : position + 1u;

    for (uint d = lane; d < args.headDim; d += 32u) partial[sg * 128u + d] = 0.0f;
    float running = 0.0f;
    float total = 0.0f;
    for (uint first = 0u; first < slots; first += ATTENTION_CHUNK) {
        const uint count = min(ATTENTION_CHUNK, slots - first);
        float chunkPeak = -INFINITY;
        for (uint i = tid; i < count; i += threads) {
            const uint j = first + i;
            const uint held = args.ring != 0u ? last - (last + args.capacity - j) % args.capacity : j;
            float logit = -INFINITY;
            if (held <= position && position - held < args.window) {
                const uint distance = position - held;
                device const GroupT* kj = kCache + ulong(j) * groupsPerRow + kvHead * groupsPerHead;
                const float dot = IsQ4
                    ? depth_kv_dot_q4(qh, reinterpret_cast<device const DepthKVGroupQ4*>(kj), groupsPerHead)
                    : depth_kv_dot_q8(qh, reinterpret_cast<device const DepthKVGroupQ8*>(kj), groupsPerHead);
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
        for (uint d = lane; d < args.headDim; d += 32u) {
            float acc = partial[sg * 128u + d] * rescale;
            for (uint i = sg; i < count; i += simdgroups) {
                if (probabilities[i] == 0.0f) continue;
                device const GroupT* vj =
                    vCache + ulong(first + i) * groupsPerRow + kvHead * groupsPerHead;
                const float vVal = IsQ4
                    ? depth_kv_load_q4(reinterpret_cast<device const DepthKVGroupQ4*>(vj), d)
                    : depth_kv_load_q8(reinterpret_cast<device const DepthKVGroupQ8*>(vj), d);
                acc += probabilities[i] * vVal;
            }
            partial[sg * 128u + d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denominator = total + exp(-running);
    for (uint d = tid; d < args.headDim; d += threads) {
        float acc = 0.0f;
        for (uint i = 0u; i < simdgroups; ++i) acc += partial[i * 128u + d];
        const uint at = token * qWidth + head * args.headDim + d;
        out[at] = acc / denominator / (1.0f + exp(-gate[at]));
    }
}

template <typename GroupT, bool IsQ4>
inline void relative_attention_chunk_q_impl(
    device const float* q, device const GroupT* kCache, device const GroupT* vCache,
    device const float* r, device const float* relativeBias, device const float* sinks,
    device float* out, device const uint* step, constant DepthAttentionArgs& args,
    device const float* gate,
    threadgroup float* sRunning, threadgroup float* sTotal,
    threadgroup float* sQ, threadgroup float* sKV,
    threadgroup float* sS, threadgroup float* sDiag,
    uint groupIndex, uint tid, uint sg, uint lane)
{
    const uint head = groupIndex % args.heads;
    const uint qBase = (groupIndex / args.heads) * ATTN_QB;
    const uint kvHead = head / (args.heads / args.kvHeads);
    const uint groupsPerHead = IsQ4 ? (ATTN_D >> 4) : (ATTN_D >> 5);
    const uint groupsPerRow = args.kvHeads * groupsPerHead;
    const uint qWidth = args.heads * ATTN_D;
    const uint first = step[0];
    const uint last = first + args.tokens - 1u;
    const uint queries = min(ATTN_QB, args.tokens - qBase);
    const uint newest = first + qBase + queries - 1u;
    const bool wrapped = args.ring != 0u && last + 1u > args.capacity;
    const uint slots = wrapped ? args.capacity : newest + 1u;

    for (uint i = tid; i < ATTN_QB * ATTN_D; i += 128u) {
        const uint t = i / ATTN_D;
        sQ[i] = t < queries ? q[(qBase + t) * qWidth + head * ATTN_D + i % ATTN_D] : 0.0f;
    }
    for (uint i = tid; i < 2u * 64u; i += 128u) sDiag[i] = 0.0f;
    if (tid < ATTN_QB) {
        sRunning[tid] = 0.0f;
        sTotal[tid] = 0.0f;
    }
    simdgroup_float8x8 o[8];
    for (uint i = 0u; i < 8u; ++i) o[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint keyBase = 0u; keyBase < slots; keyBase += ATTN_KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < ATTN_KB * ATTN_D; i += 128u) {
            const uint j = keyBase + i / ATTN_D;
            if (j < slots) {
                device const GroupT* kj = kCache + ulong(j) * groupsPerRow + kvHead * groupsPerHead;
                sKV[i] = IsQ4
                    ? depth_kv_load_q4(reinterpret_cast<device const DepthKVGroupQ4*>(kj), i % ATTN_D)
                    : depth_kv_load_q8(reinterpret_cast<device const DepthKVGroupQ8*>(kj), i % ATTN_D);
            } else {
                sKV[i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0u; c < 2u; ++c) {
            const uint tr = (sg / 2u) * 8u;
            const uint tc = ((sg % 2u) * 2u + c) * 8u;
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
        for (uint pass = 0u; pass < 4u; ++pass) {
            const uint row = sg + 4u * pass;
            const uint rowToken = qBase + row;
            const uint rowPosition = first + rowToken;
            const uint j = keyBase + lane;
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
                    logit = args.scale * sS[row * ATTN_KB + lane] + bias
                        - sinks[rowToken * args.heads + head];
                }
            }
            const float running = sRunning[row];
            const float peak = max(running, simd_max(logit));
            const float pr = exp(logit - peak);
            const float rowSum = simd_sum(pr);
            sS[row * ATTN_KB + lane] = pr;
            if (lane == 0u) {
                const float rescale = exp(running - peak);
                sTotal[row] = sTotal[row] * rescale + rowSum;
                sRunning[row] = peak;
                sDiag[(row / 8u) * 64u + (row % 8u) * 9u] = rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < ATTN_KB * ATTN_D; i += 128u) {
            const uint j = keyBase + i / ATTN_D;
            if (j < slots) {
                device const GroupT* vj = vCache + ulong(j) * groupsPerRow + kvHead * groupsPerHead;
                sKV[i] = IsQ4
                    ? depth_kv_load_q4(reinterpret_cast<device const DepthKVGroupQ4*>(vj), i % ATTN_D)
                    : depth_kv_load_q8(reinterpret_cast<device const DepthKVGroupQ8*>(vj), i % ATTN_D);
            } else {
                sKV[i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint rt = 0u; rt < 2u; ++rt) {
            simdgroup_float8x8 diag;
            simdgroup_load(diag, sDiag + rt * 64u, 8);
            for (uint ct = 0u; ct < 4u; ++ct) {
                simdgroup_float8x8 acc;
                simdgroup_multiply(acc, diag, o[rt * 4u + ct]);
                for (uint ks = 0u; ks < ATTN_KB; ks += 8u) {
                    simdgroup_float8x8 mp;
                    simdgroup_float8x8 mv;
                    simdgroup_load(mp, sS + rt * 8u * ATTN_KB + ks, ATTN_KB);
                    simdgroup_load(mv, sKV + ks * ATTN_D + sg * 32u + ct * 8u, ATTN_D);
                    simdgroup_multiply_accumulate(acc, mp, mv, acc);
                }
                o[rt * 4u + ct] = acc;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rt = 0u; rt < 2u; ++rt) {
        for (uint ct = 0u; ct < 4u; ++ct) {
            simdgroup_store(o[rt * 4u + ct], sKV + rt * 8u * ATTN_D + sg * 32u + ct * 8u, ATTN_D);
        }
    }
    if (tid < ATTN_QB) sS[tid] = 1.0f / (sTotal[tid] + exp(-sRunning[tid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < queries * ATTN_D; i += 128u) {
        const uint t = i / ATTN_D;
        const uint at = (qBase + t) * qWidth + head * ATTN_D + i % ATTN_D;
        out[at] = sKV[i] * sS[t] / (1.0f + exp(-gate[at]));
    }
}

kernel void kv_finish_q8(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* kWeight [[buffer(2)]], device const float* vWeight [[buffer(3)]],
    device float* state [[buffer(4)]], device const float* kDynamic [[buffer(5)]],
    device const float* vDynamic [[buffer(6)]], device DepthKVGroupQ8* kCache [[buffer(7)]],
    device DepthKVGroupQ8* vCache [[buffer(8)]], device const DepthStep* step [[buffer(9)]],
    constant DepthKVFinishArgs& args [[buffer(10)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    depth_kv_finish_impl<DepthKVGroupQ8, false>(
        k, v, kWeight, vWeight, state, kDynamic, vDynamic, kCache, vCache, step, args,
        scratch, groupIndex, tid, threads, sg, lane);
}

kernel void kv_finish_q4(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* kWeight [[buffer(2)]], device const float* vWeight [[buffer(3)]],
    device float* state [[buffer(4)]], device const float* kDynamic [[buffer(5)]],
    device const float* vDynamic [[buffer(6)]], device DepthKVGroupQ4* kCache [[buffer(7)]],
    device DepthKVGroupQ4* vCache [[buffer(8)]], device const DepthStep* step [[buffer(9)]],
    constant DepthKVFinishArgs& args [[buffer(10)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    depth_kv_finish_impl<DepthKVGroupQ4, true>(
        k, v, kWeight, vWeight, state, kDynamic, vDynamic, kCache, vCache, step, args,
        scratch, groupIndex, tid, threads, sg, lane);
}

kernel void kv_finish_chunk_q8(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* kWeight [[buffer(2)]], device const float* vWeight [[buffer(3)]],
    device const float* state [[buffer(4)]], device const float* kDynamic [[buffer(5)]],
    device const float* vDynamic [[buffer(6)]], device DepthKVGroupQ8* kCache [[buffer(7)]],
    device DepthKVGroupQ8* vCache [[buffer(8)]], device const DepthStep* step [[buffer(9)]],
    constant DepthKVFinishArgs& args [[buffer(10)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    depth_kv_finish_chunk_impl<DepthKVGroupQ8, false>(
        k, v, kWeight, vWeight, state, kDynamic, vDynamic, kCache, vCache, step, args,
        scratch, groupIndex, tid, threads, sg, lane);
}

kernel void kv_finish_chunk_q4(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* kWeight [[buffer(2)]], device const float* vWeight [[buffer(3)]],
    device const float* state [[buffer(4)]], device const float* kDynamic [[buffer(5)]],
    device const float* vDynamic [[buffer(6)]], device DepthKVGroupQ4* kCache [[buffer(7)]],
    device DepthKVGroupQ4* vCache [[buffer(8)]], device const DepthStep* step [[buffer(9)]],
    constant DepthKVFinishArgs& args [[buffer(10)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    depth_kv_finish_chunk_impl<DepthKVGroupQ4, true>(
        k, v, kWeight, vWeight, state, kDynamic, vDynamic, kCache, vCache, step, args,
        scratch, groupIndex, tid, threads, sg, lane);
}

kernel void relative_attention_decode_q8(
    device const float* q [[buffer(0)]], device const DepthKVGroupQ8* kCache [[buffer(1)]],
    device const DepthKVGroupQ8* vCache [[buffer(2)]], device const float* r [[buffer(3)]],
    device const float* relativeBias [[buffer(4)]], device const float* sinks [[buffer(5)]],
    device float* out [[buffer(6)]], device const uint* step [[buffer(7)]],
    constant DepthAttentionArgs& args [[buffer(8)]], device const float* gate [[buffer(9)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    threadgroup float probabilities[ATTENTION_CHUNK];
    threadgroup float partial[8 * 128];
    relative_attention_decode_q_impl<DepthKVGroupQ8, false>(
        q, kCache, vCache, r, relativeBias, sinks, out, step, args, gate,
        scratch, probabilities, partial, groupIndex, tid, threads, sg, lane);
}

kernel void relative_attention_decode_q4(
    device const float* q [[buffer(0)]], device const DepthKVGroupQ4* kCache [[buffer(1)]],
    device const DepthKVGroupQ4* vCache [[buffer(2)]], device const float* r [[buffer(3)]],
    device const float* relativeBias [[buffer(4)]], device const float* sinks [[buffer(5)]],
    device float* out [[buffer(6)]], device const uint* step [[buffer(7)]],
    constant DepthAttentionArgs& args [[buffer(8)]], device const float* gate [[buffer(9)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[32];
    threadgroup float probabilities[ATTENTION_CHUNK];
    threadgroup float partial[8 * 128];
    relative_attention_decode_q_impl<DepthKVGroupQ4, true>(
        q, kCache, vCache, r, relativeBias, sinks, out, step, args, gate,
        scratch, probabilities, partial, groupIndex, tid, threads, sg, lane);
}

kernel void relative_attention_chunk_q8(
    device const float* q [[buffer(0)]], device const DepthKVGroupQ8* kCache [[buffer(1)]],
    device const DepthKVGroupQ8* vCache [[buffer(2)]], device const float* r [[buffer(3)]],
    device const float* relativeBias [[buffer(4)]], device const float* sinks [[buffer(5)]],
    device float* out [[buffer(6)]], device const uint* step [[buffer(7)]],
    constant DepthAttentionArgs& args [[buffer(8)]], device const float* gate [[buffer(9)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sRunning[ATTN_QB];
    threadgroup float sTotal[ATTN_QB];
    threadgroup float sQ[ATTN_QB * ATTN_D];
    threadgroup float sKV[ATTN_KB * ATTN_D];
    threadgroup float sS[ATTN_QB * ATTN_KB];
    threadgroup float sDiag[2 * 64];
    relative_attention_chunk_q_impl<DepthKVGroupQ8, false>(
        q, kCache, vCache, r, relativeBias, sinks, out, step, args, gate,
        sRunning, sTotal, sQ, sKV, sS, sDiag, groupIndex, tid, sg, lane);
}

kernel void relative_attention_chunk_q4(
    device const float* q [[buffer(0)]], device const DepthKVGroupQ4* kCache [[buffer(1)]],
    device const DepthKVGroupQ4* vCache [[buffer(2)]], device const float* r [[buffer(3)]],
    device const float* relativeBias [[buffer(4)]], device const float* sinks [[buffer(5)]],
    device float* out [[buffer(6)]], device const uint* step [[buffer(7)]],
    constant DepthAttentionArgs& args [[buffer(8)]], device const float* gate [[buffer(9)]],
    uint groupIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sRunning[ATTN_QB];
    threadgroup float sTotal[ATTN_QB];
    threadgroup float sQ[ATTN_QB * ATTN_D];
    threadgroup float sKV[ATTN_KB * ATTN_D];
    threadgroup float sS[ATTN_QB * ATTN_KB];
    threadgroup float sDiag[2 * 64];
    relative_attention_chunk_q_impl<DepthKVGroupQ4, true>(
        q, kCache, vCache, r, relativeBias, sinks, out, step, args, gate,
        sRunning, sTotal, sQ, sKV, sS, sDiag, groupIndex, tid, sg, lane);
}
