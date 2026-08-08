#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_k3 — Stage-E1 chunked-prefill kernels for Kimi K3 (docs/
// K3_DATAFLOW.md). Every kernel here is a BATCHED mirror of a decode kernel:
// the per-position math (reduction structure, fp32 accumulation, fp16 storage
// points) is identical to the single-token path, with an added token dimension
// `t`. That keeps chunked prefill bit-comparable to serial replay up to the
// GEMM / attention reduction order (the fp16-chained tolerance).
//
//   Fallback trunk GEMMs (pre-M5 / forced-fallback path):
//     k3_gemm_int4 / k3_gemm_int8 / k3_gemm_bf16 / k3_gemm_f32
//       out[t][n] = sum_k W[n][k] . x[t][k], one SIMD per output element,
//       affine factoring s*sum(qx)+b*sum(x) (matches dequant_int4.metal).
//   Router:            k3_router_prefill        sigmoid scores [T][numExperts]
//   Norms (one TG/pos): k3_rmsnorm_{bf16w,f32w,f32x}_prefill
//   AttnRes:           k3_attnres_prefill       one TG per position
//   Embed:             k3_embed_gather_prefill  int8 row gather [T][H]
//   Block append:      k3_block_append_prefill  strided per-position row copy
//   KDA:               k3_kda_conv_prefill (+ _state), k3_kda_prefill_serial,
//                      k3_kda_onorm_prefill, k3_addbias_prefill
//   MLA:               k3_mla_absorb_q_prefill, k3_mla_cache_append_prefill,
//                      k3_mla_prefill_attention, k3_mla_out_project_prefill
//   MoE:               k3_moe_prefill_phase1 / _phase2 / _reduce
//
// No function constants: prefill runs at most a few chunks per generate, and
// the runtime dims keep one pipeline per kernel. The NAX tensor-op QMM lives
// in tensorops_k3.metal (__HAVE_TENSOR__ guarded); these are the portable
// fallbacks and the irregular (non-GEMM) stages.
// ============================================================================

constant constexpr uint kK3PfGroup = 64;           // affine group size
constant constexpr uint kK3PfThreads = 256;
constant constexpr uint kK3PfMaxSimd = kK3PfThreads / 32;
constant constexpr uint kK3PfMxfp4Group = 32;
constant constexpr uint kK3PfMxfp4GroupBytes = 16;

static inline float k3_pf_sigmoid(float v) { return 1.0f / (1.0f + exp(-v)); }

// House two-stage block reduction (per-SIMD simd_sum, scratch, one SIMD
// merges, broadcast slot doubles as the return value).
static inline float k3_pf_block_reduce_sum(
    float v, uint lane, uint sg, uint simdgroups,
    threadgroup float* scratch, threadgroup float* bcast) {
    float s = simd_sum(v);
    if (lane == 0) { scratch[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = (lane < simdgroups) ? scratch[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}

// ============================================================================
// Fallback trunk GEMMs. Grid = (ceil(N/8), T): one SIMD per output element,
// eight per threadgroup. FP32 accumulation, fp16 out (except f32 -> f32).
// ============================================================================

kernel void k3_gemm_int4(
    device const uint8_t* W      [[buffer(0)]],   // [N, K/2] nibbles
    device const bfloat*  scales [[buffer(1)]],   // [N, K/64]
    device const bfloat*  biases [[buffer(2)]],   // [N, K/64]
    device const half*    x      [[buffer(3)]],   // [T, K]
    device       half*    y      [[buffer(4)]],   // [T, N]
    constant     uint&    T      [[buffer(5)]],
    constant     uint&    N      [[buffer(6)]],
    constant     uint&    K      [[buffer(7)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    const uint n = tgid.x * 8u + sg_idx;
    const uint t = tgid.y;
    if (n >= N || t >= T) return;
    const uint groups = K / kK3PfGroup;
    const uint rowBytes = K / 2u;
    device const uint8_t* wRow = W + uint(n) * rowBytes;
    device const bfloat* sRow = scales + uint(n) * groups;
    device const bfloat* bRow = biases + uint(n) * groups;
    device const half* xRow = x + uint(t) * K;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const uint e0 = g * kK3PfGroup + lane * 2u;
        const uint8_t byte = wRow[e0 >> 1];
        const float x0 = float(xRow[e0]);
        const float x1 = float(xRow[e0 + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(float(sRow[g]), dot, acc);
        acc = fma(float(bRow[g]), sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[uint(t) * N + n] = half(acc); }
}

kernel void k3_gemm_int8(
    device const uint8_t* W      [[buffer(0)]],   // [N, K]
    device const bfloat*  scales [[buffer(1)]],   // [N, K/64]
    device const bfloat*  biases [[buffer(2)]],   // [N, K/64]
    device const half*    x      [[buffer(3)]],   // [T, K]
    device       half*    y      [[buffer(4)]],   // [T, N]
    constant     uint&    T      [[buffer(5)]],
    constant     uint&    N      [[buffer(6)]],
    constant     uint&    K      [[buffer(7)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    const uint n = tgid.x * 8u + sg_idx;
    const uint t = tgid.y;
    if (n >= N || t >= T) return;
    const uint groups = K / kK3PfGroup;
    device const uint8_t* wRow = W + uint(n) * K;
    device const bfloat* sRow = scales + uint(n) * groups;
    device const bfloat* bRow = biases + uint(n) * groups;
    device const half* xRow = x + uint(t) * K;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const uint e0 = g * kK3PfGroup + lane * 2u;
        const float x0 = float(xRow[e0]);
        const float x1 = float(xRow[e0 + 1u]);
        float dot = fma(float(uint(wRow[e0])), x0, 0.0f);
        dot = fma(float(uint(wRow[e0 + 1u])), x1, dot);
        const float sum = x0 + x1;
        acc = fma(float(sRow[g]), dot, acc);
        acc = fma(float(bRow[g]), sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[uint(t) * N + n] = half(acc); }
}

kernel void k3_gemm_bf16(
    device const bfloat* W [[buffer(0)]],   // [N, K]
    device const half*   x [[buffer(1)]],   // [T, K]
    device       half*   y [[buffer(2)]],   // [T, N]
    constant     uint&   T [[buffer(3)]],
    constant     uint&   N [[buffer(4)]],
    constant     uint&   K [[buffer(5)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    const uint n = tgid.x * 8u + sg_idx;
    const uint t = tgid.y;
    if (n >= N || t >= T) return;
    device const bfloat* wRow = W + uint(n) * K;
    device const half* xRow = x + uint(t) * K;
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32u) {
        acc = fma(float(wRow[k]), float(xRow[k]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[uint(t) * N + n] = half(acc); }
}

kernel void k3_gemm_f32(
    device const float* W [[buffer(0)]],   // [N, K]
    device const float* x [[buffer(1)]],   // [T, K]
    device       float* y [[buffer(2)]],   // [T, N]
    constant     uint&  T [[buffer(3)]],
    constant     uint&  N [[buffer(4)]],
    constant     uint&  K [[buffer(5)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    const uint n = tgid.x * 8u + sg_idx;
    const uint t = tgid.y;
    if (n >= N || t >= T) return;
    device const float* wRow = W + uint(n) * K;
    device const float* xRow = x + uint(t) * K;
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32u) {
        acc = fma(wRow[k], xRow[k], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[uint(t) * N + n] = acc; }
}

// Router: scores[t][e] = sigmoid(gate[e] . x32[t]), fp32 in/out. One SIMD per
// (t, e). The bias is added on the CPU during top-k (matches the decode path).
kernel void k3_router_prefill(
    device const float* gate   [[buffer(0)]],   // [numExperts, H] fp32
    device const float* x      [[buffer(1)]],   // [T, H] fp32
    device       float* scores [[buffer(2)]],   // [T, numExperts] fp32
    constant     uint&  T      [[buffer(3)]],
    constant     uint&  numExperts [[buffer(4)]],
    constant     uint&  H      [[buffer(5)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    const uint e = tgid.x * 8u + sg_idx;
    const uint t = tgid.y;
    if (e >= numExperts || t >= T) return;
    device const float* gRow = gate + uint(e) * H;
    device const float* xRow = x + uint(t) * H;
    float acc = 0.0f;
    for (uint k = lane; k < H; k += 32u) {
        acc = fma(gRow[k], xRow[k], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { scores[uint(t) * numExperts + e] = k3_pf_sigmoid(acc); }
}

// ============================================================================
// Batched RMSNorm. One threadgroup per position, 256 threads, identical
// reduction to the decode norm kernels. Grid = (T).
// ============================================================================

// bf16 weight, fp16 in/out (input/post-attention layernorm). Matches the
// house rmsnorm_bf16w element order (x * inv * w).
kernel void k3_rmsnorm_bf16w_prefill(
    device const half*   x      [[buffer(0)]],   // [T, D] fp16
    device const bfloat* weight [[buffer(1)]],   // [D] bf16
    device       half*   out    [[buffer(2)]],   // [T, D] fp16
    constant     uint&   D      [[buffer(3)]],
    constant     float&  eps    [[buffer(4)]],
    uint t            [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint lane         [[thread_index_in_simdgroup]],
    uint sg           [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[kK3PfMaxSimd];
    threadgroup float bcast;
    device const half* xr = x + uint(t) * D;
    device half* outr = out + uint(t) * D;
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = float(xr[i]);
        ss = fma(v, v, ss);
    }
    const float total = k3_pf_block_reduce_sum(ss, lane, sg, simdgroups,
                                               scratch, &bcast);
    const float inv = rsqrt(total / float(D) + eps);
    for (uint i = lid; i < D; i += lsize) {
        outr[i] = half(float(xr[i]) * inv * float(weight[i]));
    }
}

// fp32 weight, fp16 in/out (MLA q_a_layernorm). Matches k3_rmsnorm_f32w.
kernel void k3_rmsnorm_f32w_prefill(
    device const half*  x      [[buffer(0)]],   // [T, D] fp16
    device const float* weight [[buffer(1)]],   // [D] fp32
    device       half*  out    [[buffer(2)]],   // [T, D] fp16
    constant     uint&  D      [[buffer(3)]],
    constant     float& eps    [[buffer(4)]],
    uint t            [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint lane         [[thread_index_in_simdgroup]],
    uint sg           [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[kK3PfMaxSimd];
    threadgroup float bcast;
    device const half* xr = x + uint(t) * D;
    device half* outr = out + uint(t) * D;
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = float(xr[i]);
        ss = fma(v, v, ss);
    }
    const float total = k3_pf_block_reduce_sum(ss, lane, sg, simdgroups,
                                               scratch, &bcast);
    const float inv = rsqrt(total / float(D) + eps);
    for (uint i = lid; i < D; i += lsize) {
        outr[i] = half(weight[i] * float(xr[i]) * inv);
    }
}

// fp32 in, bf16 weight, fp16 out (routed_expert_norm on y_lat). Matches
// k3_rmsnorm_f32x.
kernel void k3_rmsnorm_f32x_prefill(
    device const float*  x      [[buffer(0)]],   // [T, D] fp32
    device const bfloat* weight [[buffer(1)]],   // [D] bf16
    device       half*   out    [[buffer(2)]],   // [T, D] fp16
    constant     uint&   D      [[buffer(3)]],
    constant     float&  eps    [[buffer(4)]],
    uint t            [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint lane         [[thread_index_in_simdgroup]],
    uint sg           [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[kK3PfMaxSimd];
    threadgroup float bcast;
    device const float* xr = x + uint(t) * D;
    device half* outr = out + uint(t) * D;
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = xr[i];
        ss = fma(v, v, ss);
    }
    const float total = k3_pf_block_reduce_sum(ss, lane, sg, simdgroups,
                                               scratch, &bcast);
    const float inv = rsqrt(total / float(D) + eps);
    for (uint i = lid; i < D; i += lsize) {
        outr[i] = half(float(weight[i]) * xr[i] * inv);
    }
}

// ============================================================================
// Batched AttnRes. One threadgroup per position; identical math to
// k3_attnres. blocks is [T][maxBlocks][H] (per-position block list), prefix
// [T][H], out [T][H]. numBlocks is uniform across the chunk.
// ============================================================================
kernel void k3_attnres_prefill(
    device const half*  blocks    [[buffer(0)]],   // [T][maxBlocks][H] fp16
    device const half*  prefix    [[buffer(1)]],   // [T][H] fp16
    device const float* score_vec [[buffer(2)]],   // [H] fp32 fused
    device       half*  out       [[buffer(3)]],   // [T][H] fp16
    constant     uint&  hidden    [[buffer(4)]],
    constant     uint&  numBlocks [[buffer(5)]],
    constant     uint&  maxBlocks [[buffer(6)]],   // row stride of blocks
    constant     float& eps       [[buffer(7)]],
    uint t            [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint lane         [[thread_index_in_simdgroup]],
    uint sg           [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]) {
    threadgroup float scores[16];
    threadgroup float probs[16];
    threadgroup float scratch[kK3PfMaxSimd];
    threadgroup float bcast;
    const uint H = hidden;
    const uint B = numBlocks;
    if (B + 1u > 16u) return;
    const uint numVectors = B + 1u;
    device const half* blockBase = blocks + uint(t) * uint(maxBlocks) * H;
    device const half* prefixRow = prefix + uint(t) * H;

    for (uint vi = 0; vi < numVectors; ++vi) {
        device const half* v = (vi < B)
            ? (blockBase + uint(vi) * H) : prefixRow;
        float ss = 0.0f;
        for (uint j = lid; j < H; j += lsize) {
            const float xv = float(v[j]);
            ss = fma(xv, xv, ss);
        }
        const float total = k3_pf_block_reduce_sum(ss, lane, sg, simdgroups,
                                                   scratch, &bcast);
        const float inv = rsqrt(total / float(H) + eps);
        float sc = 0.0f;
        for (uint j = lid; j < H; j += lsize) {
            sc = fma(float(v[j]) * inv, score_vec[j], sc);
        }
        const float score = k3_pf_block_reduce_sum(sc, lane, sg, simdgroups,
                                                   scratch, &bcast);
        if (lid == 0) { scores[vi] = score; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        float m = scores[0];
        for (uint vi = 1; vi < numVectors; ++vi) { m = max(m, scores[vi]); }
        float sum = 0.0f;
        for (uint vi = 0; vi < numVectors; ++vi) {
            const float e = exp(scores[vi] - m);
            probs[vi] = e;
            sum += e;
        }
        const float invSum = 1.0f / sum;
        for (uint vi = 0; vi < numVectors; ++vi) { probs[vi] *= invSum; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* outr = out + uint(t) * H;
    for (uint j = lid; j < H; j += lsize) {
        float acc = 0.0f;
        for (uint vi = 0; vi < numVectors; ++vi) {
            device const half* v = (vi < B)
                ? (blockBase + uint(vi) * H) : prefixRow;
            acc = fma(probs[vi], float(v[j]), acc);
        }
        outr[j] = half(acc);
    }
}

// ============================================================================
// Batched int8 embedding gather: out[t][d] = dequant(table[token[t]][d]).
// One thread per (t, d). Mirrors k3_embed_gather_int8 (affine g64, fp16 out).
// ============================================================================
kernel void k3_embed_gather_prefill(
    device const uint8_t* table  [[buffer(0)]],   // [V, H] int8
    device const bfloat*  scales [[buffer(1)]],   // [V, H/64]
    device const bfloat*  biases [[buffer(2)]],   // [V, H/64]
    device const int*     tokens [[buffer(3)]],   // [T]
    device       half*    out    [[buffer(4)]],   // [T, H] fp16
    constant     uint&    H      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= H) return;
    const uint token = uint(tokens[t]);
    const uint groups = H / kK3PfGroup;
    device const uint8_t* row = table + uint(token) * H;
    device const bfloat* sRow = scales + uint(token) * groups;
    device const bfloat* bRow = biases + uint(token) * groups;
    const uint g = d / kK3PfGroup;
    const float w = float(uint(row[d])) * float(sRow[g]) + float(bRow[g]);
    out[uint(t) * H + d] = half(w);
}

// Strided per-position row copy: dst[t][sel][0:H] = src[t][0:H]. Used for the
// AttnRes block append (sel = blockCount, dst row stride = maxBlocks*H).
kernel void k3_block_append_prefill(
    device const half* src       [[buffer(0)]],   // [T, H] fp16
    device       half* dst       [[buffer(1)]],   // [T, maxBlocks, H] fp16
    constant     uint& H         [[buffer(2)]],
    constant     uint& sel       [[buffer(3)]],   // block slot index
    constant     uint& maxBlocks [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= H) return;
    dst[(uint(t) * uint(maxBlocks) + sel) * H + d] = src[uint(t) * H + d];
}

// ============================================================================
// KDA prefill.
// ============================================================================

// Batched causal depthwise conv width 4 + SiLU over the chunk. One thread per
// (which*c, t); the conv state supplies the 3 pre-chunk inputs (oldest first)
// and is NOT written here (see k3_kda_conv_prefill_state). Inputs q/k/v are
// the [T][P] fp16 projections; outputs are [T][P] fp32.
kernel void k3_kda_conv_prefill(
    device const half*  xq         [[buffer(0)]],   // [T, P] fp16
    device const half*  xk         [[buffer(1)]],   // [T, P] fp16
    device const half*  xv         [[buffer(2)]],   // [T, P] fp16
    device const float* weights    [[buffer(3)]],   // [3][P][4] fp32
    device const float* convStates [[buffer(4)]],   // [3][P][3] fp32 (history)
    device       float* qOut       [[buffer(5)]],   // [T, P] fp32
    device       float* kOut       [[buffer(6)]],   // [T, P] fp32
    device       float* vOut       [[buffer(7)]],   // [T, P] fp32
    constant     uint&  channels   [[buffer(8)]],   // P
    constant     uint&  T          [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint P = channels;
    const uint whichc = gid.x;      // [0, 3P)
    const uint t = gid.y;           // [0, T)
    if (whichc >= 3u * P || t >= T) return;
    const uint which = whichc / P;
    const uint c = whichc % P;
    device const half* x = (which == 0u) ? xq : (which == 1u) ? xk : xv;
    device float* y = (which == 0u) ? qOut : (which == 1u) ? kOut : vOut;
    device const float* w = weights + which * P * 4u + c * 4u;
    device const float* st = convStates + which * P * 3u + c * 3u;
    // Taps oldest..newest; tap 3 multiplies the current input. Input at
    // relative position t-3..t; negative indices read the conv history.
    float in[4];
    for (uint j = 0; j < 4u; ++j) {
        const int rel = int(t) - 3 + int(j);
        in[j] = (rel >= 0) ? float(x[uint(rel) * P + c]) : st[3u + rel];
    }
    float acc = w[3] * in[3];
    acc = fma(w[0], in[0], acc);
    acc = fma(w[1], in[1], acc);
    acc = fma(w[2], in[2], acc);
    y[uint(t) * P + c] = acc * k3_pf_sigmoid(acc);
}

// Conv state update: new history = last 3 of [old history, chunk projections].
// One thread per (which, c). Runs AFTER k3_kda_conv_prefill (ordered pass).
kernel void k3_kda_conv_prefill_state(
    device const half*  xq         [[buffer(0)]],   // [T, P] fp16
    device const half*  xk         [[buffer(1)]],   // [T, P] fp16
    device const half*  xv         [[buffer(2)]],   // [T, P] fp16
    device       float* convStates [[buffer(3)]],   // [3][P][3] fp32, in place
    constant     uint&  channels   [[buffer(4)]],   // P
    constant     uint&  T          [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const uint P = channels;
    if (gid >= 3u * P) return;
    const uint which = gid / P;
    const uint c = gid % P;
    device const half* x = (which == 0u) ? xq : (which == 1u) ? xk : xv;
    device float* st = convStates + which * P * 3u + c * 3u;
    // Combined sequence: st[0], st[1], st[2], proj[0..T-1]; keep the last 3.
    // Element i of the combined window (i in [0, T+3)) is proj[i-3] if i>=3
    // else st[i]. The new history holds combined indices T, T+1, T+2.
    float newHist[3];
    for (uint j = 0; j < 3u; ++j) {
        const uint combined = uint(T) + j;      // T, T+1, T+2
        if (combined >= 3u) {
            newHist[j] = float(x[(combined - 3u) * P + c]);
        } else {
            newHist[j] = st[combined];
        }
    }
    st[0] = newHist[0];
    st[1] = newHist[1];
    st[2] = newHist[2];
}

// GPU-resident serial KDA recurrence over the chunk. One threadgroup per head
// (128 threads, thread j owns state column j — the k3_kda_step structure),
// looping t = 0..T-1 with the delta rule applied in order, emitting o[t].
kernel void k3_kda_prefill_serial(
    device       float* state       [[buffer(0)]],   // [H][D][D] fp32, in place
    device const float* q           [[buffer(1)]],   // [T, P] fp32 (post-conv)
    device const float* k           [[buffer(2)]],   // [T, P] fp32
    device const float* v           [[buffer(3)]],   // [T, P] fp32
    device const float* z           [[buffer(4)]],   // [T, P] fp32 (+dt_bias)
    device const float* betaLogits  [[buffer(5)]],   // [T, H] fp32
    device const float* aLog        [[buffer(6)]],   // [H] fp32
    device       float* o           [[buffer(7)]],   // [T, P] fp32 out
    constant     uint&  numHeads    [[buffer(8)]],
    constant     uint&  headDim     [[buffer(9)]],
    constant     uint&  T           [[buffer(10)]],
    uint head         [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint lane         [[thread_index_in_simdgroup]],
    uint sg           [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]) {
    threadgroup float qS[128];
    threadgroup float kS[128];
    threadgroup float alphaS[128];
    threadgroup float scratch[4];
    threadgroup float bcast;
    const uint H = numHeads;
    const uint D = headDim;
    if (head >= H || D > 128u) return;
    const uint P = H * D;
    device float* S = state + uint(head) * D * D;
    const float a = exp(aLog[head]);

    for (uint t = 0; t < T; ++t) {
        device const float* qHead = q + uint(t) * P + uint(head) * D;
        device const float* kHead = k + uint(t) * P + uint(head) * D;
        device const float* vHead = v + uint(t) * P + uint(head) * D;
        device const float* zHead = z + uint(t) * P + uint(head) * D;

        float sq = 0.0f;
        float sk = 0.0f;
        for (uint i = lid; i < D; i += lsize) {
            const float qv = qHead[i];
            const float kv = kHead[i];
            sq = fma(qv, qv, sq);
            sk = fma(kv, kv, sk);
        }
        const float ssq = k3_pf_block_reduce_sum(sq, lane, sg, simdgroups,
                                                 scratch, &bcast);
        const float ssk = k3_pf_block_reduce_sum(sk, lane, sg, simdgroups,
                                                 scratch, &bcast);
        const float invQ = rsqrt(ssq + 1e-6f);
        const float invK = rsqrt(ssk + 1e-6f);
        const float qScale = invQ * rsqrt(float(D));
        const float beta = k3_pf_sigmoid(betaLogits[uint(t) * H + head]);
        for (uint i = lid; i < D; i += lsize) {
            qS[i] = qHead[i] * qScale;
            kS[i] = kHead[i] * invK;
            alphaS[i] = exp(-5.0f * k3_pf_sigmoid(a * zHead[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = lid; j < D; j += lsize) {
            float u = 0.0f;
            for (uint i = 0; i < D; ++i) {
                float s = S[i * D + j] * alphaS[i];
                S[i * D + j] = s;
                u = fma(kS[i], s, u);
            }
            const float err = vHead[j] - u;
            for (uint i = 0; i < D; ++i) {
                S[i * D + j] = fma(beta * kS[i], err, S[i * D + j]);
            }
            float outVal = 0.0f;
            for (uint i = 0; i < D; ++i) {
                outVal = fma(qS[i], S[i * D + j], outVal);
            }
            o[uint(t) * P + uint(head) * D + j] = outVal;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Batched KDA output norm: per (position, head) RMSNorm + sigmoid gate.
// One threadgroup per (head, t). Matches k3_kda_onorm.
kernel void k3_kda_onorm_prefill(
    device const float* x      [[buffer(0)]],   // [T, P] fp32 (step out)
    device const half*  gate   [[buffer(1)]],   // [T, P] fp16 (g_proj)
    device const float* weight [[buffer(2)]],   // [D] fp32 (o_norm)
    device       half*  out    [[buffer(3)]],   // [T, P] fp16
    constant     uint&  numHeads [[buffer(4)]],
    constant     uint&  headDim [[buffer(5)]],
    constant     float& eps    [[buffer(6)]],
    uint3 tg3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 lsize3 [[threads_per_threadgroup]]) {
    const uint2 tg = tg3.xy;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    const uint lane = lid3.x % 32u;
    const uint sg = lid3.x / 32u;
    const uint simdgroups = lsize3.x / 32u;
    threadgroup float scratch[4];
    threadgroup float bcast;
    const uint H = numHeads;
    const uint D = headDim;
    const uint head = tg.x;
    const uint t = tg.y;
    if (head >= H) return;
    const uint P = H * D;
    device const float* xHead = x + uint(t) * P + uint(head) * D;
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = xHead[i];
        ss = fma(v, v, ss);
    }
    const float total = k3_pf_block_reduce_sum(ss, lane, sg, simdgroups,
                                               scratch, &bcast);
    const float inv = rsqrt(total / float(D) + eps);
    device const half* gateHead = gate + uint(t) * P + uint(head) * D;
    device half* outHead = out + uint(t) * P + uint(head) * D;
    for (uint i = lid; i < D; i += lsize) {
        const float normed = weight[i] * xHead[i] * inv;
        outHead[i] = half(normed * k3_pf_sigmoid(float(gateHead[i])));
    }
}

// z[t][c] = fb[t][c] + dt_bias[c]; fp16 fb in, fp32 out, bias broadcast over
// the chunk. One thread per (c, t).
kernel void k3_addbias_prefill(
    device const half*  x    [[buffer(0)]],   // [T, P] fp16
    device const float* bias [[buffer(1)]],   // [P] fp32
    device       float* out  [[buffer(2)]],   // [T, P] fp32
    constant     uint&  P    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= P) return;
    out[uint(t) * P + c] = float(x[uint(t) * P + c]) + bias[c];
}

// ============================================================================
// MLA prefill (absorbed; NoPE — rope channels cached/consumed raw).
// ============================================================================

// q~ per position: qAbs[t][h] = [kT_h^T q_nope | q_rope], fp32. One SIMD per
// (h, l) dot over N, batched over t. Mirrors k3_mla_absorb_q.
kernel void k3_mla_absorb_q_prefill(
    device const half* kT    [[buffer(0)]],   // [H][L][N] fp16
    device const half* q     [[buffer(1)]],   // [T][H][N+R] fp16
    device       float* qAbs [[buffer(2)]],   // [T][H][L+R] fp32
    constant     uint& numHeads [[buffer(3)]],
    constant     uint& latent [[buffer(4)]],
    constant     uint& rope   [[buffer(5)]],
    constant     uint& nope   [[buffer(6)]],
    constant     uint& T      [[buffer(7)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    constexpr uint rowsPerTg = 8;
    const uint H = numHeads;
    const uint L = latent;
    const uint R = rope;
    const uint N = nope;
    const uint row = tgid.x * rowsPerTg + sg_idx;   // (h, l) within a position
    const uint t = tgid.y;
    if (row >= H * (L + R) || t >= T) return;
    const uint h = row / (L + R);
    const uint l = row % (L + R);
    const uint outIdx = uint(t) * H * (L + R) + row;
    if (l >= L) {
        if (lane == 0) {
            qAbs[outIdx] = float(q[(uint(t) * H + h) * (N + R) + N + (l - L)]);
        }
        return;
    }
    device const half* kTrow = kT + (uint(h) * L + l) * N;
    device const half* qRow = q + (uint(t) * H + h) * (N + R);
    float acc = 0.0f;
    for (uint i = lane; i < N; i += 32u) {
        acc = fma(float(kTrow[i]), float(qRow[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { qAbs[outIdx] = acc; }
}

// Batched cache append: position (chunkStart + t) gets
// [kv_a_layernorm(latent) | rope]. One threadgroup per position. Mirrors
// k3_mla_cache_append.
kernel void k3_mla_cache_append_prefill(
    device const half*  kvA        [[buffer(0)]],   // [T, L+R] fp16
    device const float* normWeight [[buffer(1)]],   // [L] fp32
    device       half*  cache      [[buffer(2)]],   // [cap, L+R] fp16
    constant     uint&  chunkStart [[buffer(3)]],
    constant     uint&  latent     [[buffer(4)]],
    constant     uint&  rope       [[buffer(5)]],
    constant     float& eps        [[buffer(6)]],
    uint t            [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint lane         [[thread_index_in_simdgroup]],
    uint sg           [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[kK3PfMaxSimd];
    threadgroup float bcast;
    const uint L = latent;
    const uint R = rope;
    device const half* kvRow = kvA + uint(t) * (L + R);
    float ss = 0.0f;
    for (uint i = lid; i < L; i += lsize) {
        const float v = float(kvRow[i]);
        ss = fma(v, v, ss);
    }
    const float total = k3_pf_block_reduce_sum(ss, lane, sg, simdgroups,
                                               scratch, &bcast);
    const float inv = rsqrt(total / float(L) + eps);
    device half* row = cache + uint(chunkStart + t) * (L + R);
    for (uint i = lid; i < L; i += lsize) {
        row[i] = half(normWeight[i] * float(kvRow[i]) * inv);
    }
    for (uint i = L + lid; i < L + R; i += lsize) {
        row[i] = kvRow[i];   // rope part cached raw, never rotated
    }
}

// Causal absorbed attention over the latent cache. One threadgroup per
// (head, position); position p attends cache rows [0, chunkStart + p]. Online
// softmax, fp32, latent-only output accumulation (the absorbed value side is
// applied by k3_mla_out_project_prefill). Single-pass (no split-KV); the
// recurrence is algebraically the decode partial+combine.
kernel void k3_mla_prefill_attention(
    device const half*  cache      [[buffer(0)]],   // [cap, L+R] fp16
    device const float* qAbs       [[buffer(1)]],   // [T, H, L+R] fp32
    device       float* outLat     [[buffer(2)]],   // [T, H, L] fp32
    constant     uint&  numHeads   [[buffer(3)]],
    constant     uint&  latent     [[buffer(4)]],
    constant     uint&  rope       [[buffer(5)]],
    constant     uint&  chunkStart [[buffer(6)]],
    constant     float& scale      [[buffer(7)]],
    uint3 tg3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 lsize3 [[threads_per_threadgroup]]) {
    const uint2 tg = tg3.xy;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    const uint lane = lid3.x % 32u;
    const uint sg = lid3.x / 32u;
    const uint simdgroups = lsize3.x / 32u;
    threadgroup float qSmem[576];
    threadgroup float scratch[kK3PfMaxSimd];
    threadgroup float bcast;
    const uint H = numHeads;
    const uint L = latent;
    const uint R = rope;
    const uint rowLen = L + R;
    const uint h = tg.x;
    const uint p = tg.y;
    if (h >= H) return;
    const uint kvValid = chunkStart + p + 1u;

    device const float* qRow = qAbs + (uint(p) * H + h) * rowLen;
    for (uint i = lid; i < rowLen; i += lsize) { qSmem[i] = qRow[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint perThread = (512 + kK3PfThreads - 1) / kK3PfThreads;
    float oLocal[perThread];
    for (uint k = 0; k < perThread; ++k) { oLocal[k] = 0.0f; }
    float mRun = -INFINITY;
    float dRun = 0.0f;
    for (uint key = 0; key < kvValid; ++key) {
        device const half* cRow = cache + uint(key) * rowLen;
        float partial = 0.0f;
        for (uint i = lid; i < rowLen; i += lsize) {
            partial = fma(qSmem[i], float(cRow[i]), partial);
        }
        float s = k3_pf_block_reduce_sum(partial, lane, sg, simdgroups,
                                         scratch, &bcast);
        s *= scale;
        const float mNew = max(mRun, s);
        const float alpha = exp(mRun - mNew);
        const float pExp = exp(s - mNew);
        dRun = dRun * alpha + pExp;
        uint slot = 0;
        for (uint i = lid; i < L; i += lsize) {
            oLocal[slot] = oLocal[slot] * alpha + pExp * float(cRow[i]);
            slot += 1;
        }
        mRun = mNew;
    }
    const float invD = (dRun > 0.0f) ? (1.0f / dRun) : 0.0f;
    device float* outRow = outLat + (uint(p) * H + h) * L;
    uint slot = 0;
    for (uint i = lid; i < L; i += lsize) {
        outRow[i] = oLocal[slot] * invD;
        slot += 1;
    }
}

// Batched out-project + gate: out[t][h*V+i] = sigmoid(gate) * (v_h . outLat).
// One SIMD per (h, i) dot over L, batched over t. Mirrors k3_mla_out_project.
kernel void k3_mla_out_project_prefill(
    device const half*  vW     [[buffer(0)]],   // [H][V][L] fp16
    device const float* outLat [[buffer(1)]],   // [T][H][L] fp32
    device const half*  gate   [[buffer(2)]],   // [T][H*V] fp16
    device       half*  out    [[buffer(3)]],   // [T][H*V] fp16
    constant     uint&  numHeads [[buffer(4)]],
    constant     uint&  latent [[buffer(5)]],
    constant     uint&  vHead  [[buffer(6)]],
    constant     uint&  T      [[buffer(7)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    constexpr uint rowsPerTg = 8;
    const uint H = numHeads;
    const uint L = latent;
    const uint V = vHead;
    const uint row = tgid.x * rowsPerTg + sg_idx;   // (h, i)
    const uint t = tgid.y;
    if (row >= H * V || t >= T) return;
    const uint h = row / V;
    device const half* wRow = vW + uint(row) * L;
    device const float* xRow = outLat + (uint(t) * H + h) * L;
    float acc = 0.0f;
    for (uint i = lane; i < L; i += 32u) {
        acc = fma(float(wRow[i]), xRow[i], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        const uint outIdx = uint(t) * H * V + row;
        out[outIdx] = half(acc * k3_pf_sigmoid(float(gate[outIdx])));
    }
}

// ============================================================================
// MoE prefill (grouped MXFP4, per-pair batched). Pairs carry their token-major
// global index so phase outputs land deterministically for the reduce.
// ============================================================================

struct K3PrefillPair {
    uint  globalPair;   // token-major pair index = token * topK + rank
    uint  token;        // row in xLat
    uint  poolSlot;     // slot in the prefill expert pool
    float weight;       // router weight (applied in phase 2)
};

struct K3PrefillExpertOffsets {
    uint w1PackedOff;
    uint w1ScalesOff;
    uint w2PackedOff;
    uint w2ScalesOff;
    uint w3PackedOff;
    uint w3ScalesOff;
};

constant constexpr float kK3PfE2M1LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float k3_pf_e2m1(uint nibble) {
    return kK3PfE2M1LUT[nibble & 0xFu];
}

static inline float k3_pf_e8m0(uint8_t b) {
    if (b == 255u) { return 0.0f; }
    if (b == 0u)   { return as_type<float>(0x00400000u); }
    return as_type<float>(uint(b) << 23);
}

// Unscaled 32-element MXFP4 group dot against an fp16 x (mirrors
// k3_mxfp4_group_dot_f16x).
static inline float k3_pf_mxfp4_group_dot_f16x(
    device const uint8_t* wGroup, device const half* xGroup) {
    device const ushort* wp = (device const ushort*)wGroup;
    float dot = 0.0f;
    for (uint b = 0; b < 8u; ++b) {
        const uint w2 = uint(wp[b]);
        dot = fma(k3_pf_e2m1(w2 & 0xFu),         float(xGroup[b * 4u]),      dot);
        dot = fma(k3_pf_e2m1((w2 >> 4) & 0xFu),  float(xGroup[b * 4u + 1u]), dot);
        dot = fma(k3_pf_e2m1((w2 >> 8) & 0xFu),  float(xGroup[b * 4u + 2u]), dot);
        dot = fma(k3_pf_e2m1(w2 >> 12),          float(xGroup[b * 4u + 3u]), dot);
    }
    return dot;
}

static inline float k3_pf_mxfp4_row_dot_f16x(
    device const uint8_t* wRow, device const uint8_t* sRow,
    device const half* x, uint nGroups, uint lane) {
    float acc = 0.0f;
    for (uint g = lane; g < nGroups; g += 32u) {
        const float s = k3_pf_e8m0(sRow[g]);
        const float dot = k3_pf_mxfp4_group_dot_f16x(
            wRow + g * kK3PfMxfp4GroupBytes, x + g * kK3PfMxfp4Group);
        acc = fma(s, dot, acc);
    }
    return acc;
}

static inline float k3_pf_mxfp4_row_dot_f32x(
    device const uint8_t* wRow, device const uint8_t* sRow,
    device const float* x, uint nGroups, uint lane) {
    float acc = 0.0f;
    for (uint g = lane; g < nGroups; g += 32u) {
        const float s = k3_pf_e8m0(sRow[g]);
        device const uint8_t* wg = wRow + g * kK3PfMxfp4GroupBytes;
        device const float* xg = x + g * kK3PfMxfp4Group;
        device const ushort* wp = (device const ushort*)wg;
        float dot = 0.0f;
        for (uint b = 0; b < 8u; ++b) {
            const uint w2 = uint(wp[b]);
            dot = fma(k3_pf_e2m1(w2 & 0xFu),         xg[b * 4u],      dot);
            dot = fma(k3_pf_e2m1((w2 >> 4) & 0xFu),  xg[b * 4u + 1u], dot);
            dot = fma(k3_pf_e2m1((w2 >> 8) & 0xFu),  xg[b * 4u + 2u], dot);
            dot = fma(k3_pf_e2m1(w2 >> 12),          xg[b * 4u + 3u], dot);
        }
        acc = fma(s, dot, acc);
    }
    return acc;
}

static inline float k3_pf_situ(float g, float u) {
    const float t1 = tanh(clamp(g / 4.0f, -20.0f, 20.0f));
    const float sg = 1.0f / (1.0f + exp(-g));
    const float t2 = tanh(clamp(u / 25.0f, -20.0f, 20.0f));
    return (4.0f * t1 * sg) * (25.0f * t2);
}

// Phase 1: h[globalPair*F + f] = SiTU(w1[f].xLat[token], w3[f].xLat[token]).
// One SIMD per (tilePair, f). Mirrors k3_moe_phase1_situ.
kernel void k3_moe_prefill_phase1(
    device const uint8_t* experts   [[buffer(0)]],   // prefill pool
    device const K3PrefillPair* pairs [[buffer(1)]], // [tilePairCount]
    device const half*    xLat      [[buffer(2)]],   // [T, dLat] fp16
    device       float*   h         [[buffer(3)]],   // [numPairs, F] fp32
    constant     K3PrefillExpertOffsets& subtensor [[buffer(4)]],
    constant     uint&    expertStride [[buffer(5)]],
    constant     uint&    dLat      [[buffer(6)]],
    constant     uint&    F         [[buffer(7)]],
    constant     uint&    tilePairCount [[buffer(8)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    constexpr uint rowsPerTg = 8;
    const uint DD = dLat;
    const uint FF = F;
    const uint rowg = tgid.x * rowsPerTg + sg_idx;
    if (rowg >= tilePairCount * FF) return;
    const uint tilePair = rowg / FF;
    const uint f = rowg % FF;
    const K3PrefillPair pair = pairs[tilePair];
    device const uint8_t* base = experts + uint(pair.poolSlot) * expertStride;
    device const uint8_t* w1 = base + subtensor.w1PackedOff;
    device const uint8_t* s1 = base + subtensor.w1ScalesOff;
    device const uint8_t* w3 = base + subtensor.w3PackedOff;
    device const uint8_t* s3 = base + subtensor.w3ScalesOff;
    device const half* xRow = xLat + uint(pair.token) * DD;
    const uint nGroups = DD / kK3PfMxfp4Group;
    const uint rowBytes = DD / 2u;
    const float g = simd_sum(k3_pf_mxfp4_row_dot_f16x(
        w1 + uint(f) * rowBytes, s1 + uint(f) * nGroups, xRow, nGroups, lane));
    const float u = simd_sum(k3_pf_mxfp4_row_dot_f16x(
        w3 + uint(f) * rowBytes, s3 + uint(f) * nGroups, xRow, nGroups, lane));
    if (lane == 0) {
        h[uint(pair.globalPair) * FF + f] = k3_pf_situ(g, u);
    }
}

// Phase 2: pairOut[globalPair*d + d] = weight * (w2[d].h[globalPair]).
// One SIMD per (tilePair, d). Mirrors k3_moe_phase2_reduce's per-expert dot.
kernel void k3_moe_prefill_phase2(
    device const uint8_t* experts   [[buffer(0)]],
    device const K3PrefillPair* pairs [[buffer(1)]],
    device const float*   h         [[buffer(2)]],   // [numPairs, F] fp32
    device       float*   pairOut   [[buffer(3)]],   // [numPairs, dLat] fp32
    constant     K3PrefillExpertOffsets& subtensor [[buffer(4)]],
    constant     uint&    expertStride [[buffer(5)]],
    constant     uint&    dLat      [[buffer(6)]],
    constant     uint&    F         [[buffer(7)]],
    constant     uint&    tilePairCount [[buffer(8)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    const uint2 tgid = tgid3.xy;
    const uint sg_idx = lid3.x / 32u;
    const uint lane = lid3.x % 32u;
    constexpr uint rowsPerTg = 8;
    const uint DD = dLat;
    const uint FF = F;
    const uint rowg = tgid.x * rowsPerTg + sg_idx;
    if (rowg >= tilePairCount * DD) return;
    const uint tilePair = rowg / DD;
    const uint d = rowg % DD;
    const K3PrefillPair pair = pairs[tilePair];
    device const uint8_t* base = experts + uint(pair.poolSlot) * expertStride;
    device const uint8_t* w2 = base + subtensor.w2PackedOff;
    device const uint8_t* s2 = base + subtensor.w2ScalesOff;
    const uint nGroups = FF / kK3PfMxfp4Group;
    const uint rowBytes = FF / 2u;
    const float acc = simd_sum(k3_pf_mxfp4_row_dot_f32x(
        w2 + uint(d) * rowBytes, s2 + uint(d) * nGroups,
        h + uint(pair.globalPair) * FF, nGroups, lane));
    if (lane == 0) {
        pairOut[uint(pair.globalPair) * DD + d] = pair.weight * acc;
    }
}

// Token-major weighted reduce: yLat[t][d] = sum_rank pairOut[(t*topK+rank)][d].
// One thread per (t, d). Deterministic (fixed rank order), fp32.
kernel void k3_moe_prefill_reduce(
    device const float* pairOut [[buffer(0)]],   // [numPairs, dLat] fp32
    device       float* yLat    [[buffer(1)]],   // [T, dLat] fp32
    constant     uint&  dLat    [[buffer(2)]],
    constant     uint&  topK    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= dLat) return;
    float acc = 0.0f;
    for (uint rank = 0; rank < topK; ++rank) {
        acc += pairOut[(uint(t) * topK + rank) * dLat + d];
    }
    yLat[uint(t) * dLat + d] = acc;
}
