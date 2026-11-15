#include <metal_stdlib>
using namespace metal;

// ============================================================================
// fused.metal — current decode fusions:
//   fused_qkv_epilogue:     per-head Q/K/V RMSNorm + NeoX RoPE after projection.
// ============================================================================

constant constexpr uint kFusedThreads        = 256;
constant constexpr uint kFusedMaxSimdGroups  = kFusedThreads / 32;  // 8
constant constexpr uint kFusedMaxHeadDim     = 512;
constant uint FC_FUSED_D [[function_constant(80)]];
constant uint FC_FUSED_N [[function_constant(81)]];
constant uint FC_FUSED_HEAD_DIM [[function_constant(82)]];
constant uint FC_FUSED_NUM_Q_HEADS [[function_constant(83)]];
constant uint FC_FUSED_NUM_KV_HEADS [[function_constant(84)]];
constant uint FC_FUSED_ROTARY [[function_constant(85)]];
constant bool FC_FUSED_USE_FC [[function_constant(86)]];

static inline bool fused_use_fc() {
    return is_function_constant_defined(FC_FUSED_USE_FC) && FC_FUSED_USE_FC;
}

static inline uint fused_fc_head_dim(constant uint& head_dim) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_HEAD_DIM)) ? FC_FUSED_HEAD_DIM : head_dim;
}

static inline uint fused_fc_num_q_heads(constant uint& num_q_heads) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_NUM_Q_HEADS)) ? FC_FUSED_NUM_Q_HEADS : num_q_heads;
}

static inline uint fused_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_NUM_KV_HEADS)) ? FC_FUSED_NUM_KV_HEADS : num_kv_heads;
}

static inline uint fused_fc_rotary(constant uint& rotary) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_ROTARY)) ? FC_FUSED_ROTARY : rotary;
}

inline void fused_rope_neox_pair(thread float& x0,
                                 thread float& x1,
                                 uint pair_index,
                                 uint head_dim,
                                 float position,
                                 float theta_base)
{
    const float exponent = -float(2u * pair_index) / float(head_dim);
    const float freq     = pow(theta_base, exponent);
    const float angle    = position * freq;
    const float c = cos(angle);
    const float s = sin(angle);
    const float r0 = x0 * c - x1 * s;
    const float r1 = x0 * s + x1 * c;
    x0 = r0;
    x1 = r1;
}

// ============================================================================
// fused_qkv_epilogue — per-head Q/K/V post-processing.
//
// Replaces the cb1 chain after Q/K/V projection:
//     rmsnorm_bf16w_perhead(Q, q_norm) -> rmsnorm_bf16w_perhead(K, k_norm)
//     -> rmsnorm_no_scale_perhead(V) -> rope_neox(Q) -> rope_neox(K)
//
// One threadgroup owns one logical head. Q and K use BF16 weights shared across
// heads; V uses no-scale RMSNorm. For Q/K, the normalized per-head values are
// rounded to half in threadgroup memory before RoPE, preserving the standalone
// kernel boundary where RMSNorm writes FP16 and RoPE reads FP16.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kFusedThreads)]]
void fused_qkv_epilogue(
    device       half*   q              [[buffer(0)]],  // [num_q_heads, head_dim]
    device       half*   k              [[buffer(1)]],  // [num_kv_heads, head_dim]
    device       half*   v              [[buffer(2)]],  // [num_kv_heads, head_dim]
    device const bfloat* q_weight       [[buffer(3)]],  // [head_dim]
    device const bfloat* k_weight       [[buffer(4)]],  // [head_dim]
    constant     uint&   head_dim       [[buffer(5)]],
    constant     uint&   num_q_heads    [[buffer(6)]],
    constant     uint&   num_kv_heads   [[buffer(7)]],
    constant     uint&   position       [[buffer(8)]],
    constant     float&  theta_base     [[buffer(9)]],
    constant     uint&   rotated_pairs  [[buffer(10)]],
    constant     float&  rms_eps        [[buffer(11)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]],
    uint  head_group       [[threadgroup_position_in_grid]]
) {
    // K23: the half4 reinterpretation below requires 16-byte alignment of
    // these threadgroup arrays; declare it explicitly so the alignment holds
    // by construction instead of depending on the compiler's placement.
    alignas(16) threadgroup half  head_tg[kFusedMaxHeadDim];
    threadgroup float partial[kFusedMaxSimdGroups];

    const uint HD = fused_fc_head_dim(head_dim);
    const uint NQ = fused_fc_num_q_heads(num_q_heads);
    const uint NKV = fused_fc_num_kv_heads(num_kv_heads);
    const uint RP = fused_fc_rotary(rotated_pairs);

    const bool is_q = head_group < NQ;
    const bool is_k = !is_q && head_group < (NQ + NKV);
    const bool is_v = !is_q && !is_k && head_group < (NQ + 2u * NKV);
    if (!is_q && !is_k && !is_v) return;

    const uint local_head = is_q ? head_group : (head_group - NQ) % NKV;
    device half* dst = is_q ? (q + local_head * HD)
                    : (is_k ? (k + local_head * HD)
                            : (v + local_head * HD));
    device const half* src = dst;
    device const bfloat* w = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(src[i]);
        acc = fma(xv, xv, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(sum / float(HD) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = partial[0];
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(src[i]) * inv;
        if (!is_v) {
            xv *= float(w[i]);
        }
        head_tg[i] = half(xv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_v) {
        for (uint i = lid; i < HD; i += lsize) {
            dst[i] = head_tg[i];
        }
        return;
    }

    const uint half_dim = HD / 2u;
    for (uint pair = lid; pair < half_dim; pair += lsize) {
        float x0 = float(head_tg[pair]);
        float x1 = float(head_tg[half_dim + pair]);
        if (pair < RP) {
            fused_rope_neox_pair(x0, x1, pair, HD, float(position), theta_base);
        }
        dst[pair] = half(x0);
        dst[half_dim + pair] = half(x1);
    }
}

// ===========================================================================
// Fused hyper-connection gates (Qwen3.8-Flash-Next decode).
//
// The composed read is five dispatches -- grouped RMSNorm, down-GEMV, silu,
// up-GEMV, stream-reduce -- and the write is two. Measured at ~20 us each on
// this machine, that is launch cost, not work: the whole read moves ~3 MB.
// These three kernels do the same arithmetic in three dispatches. Every
// rounding point of the composed path is reproduced -- GEMV outputs to half,
// silu and sigmoid to half, acc/S to half -- so the result is bit-identical,
// which the golden checks because hyper-connections run on every decode layer.
//
// Helpers are copied rather than shared: each shader module compiles on its
// own, and matching the reduction tree exactly (256 threads, 8 simdgroups,
// same fma order) is what makes the RMS inverse identical to rmsnorm's.
// ===========================================================================

constant constexpr uint kHCThreads  = 256u;
constant constexpr uint kHCGroup    = 64u;      // affine group size
constant constexpr uint kHCMaxWide  = 10240u;   // streams * dim upper bound

static inline float hc_rms_inv(device const half* x, uint D, float eps,
                               uint lid, uint lsize, uint lane, uint sg, uint sgs,
                               threadgroup float* partial) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < sgs) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = rsqrt(v / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);   // partial reused by the next group
    return inv;
}

// One int4-affine row dot, x in threadgroup memory. Mirrors
// dequant_int4_gemv_simd_body term for term: same lane split, same fma order,
// same s*dot + b*sum factoring, so the float result is identical.
static inline float hc_int4_row_dot_tg(device const uint8_t* W_row,
                                       device const bfloat* s_row,
                                       device const bfloat* b_row,
                                       threadgroup const half* x,
                                       uint N, uint lane) {
    const uint n_groups = N / kHCGroup;
    const uint full_blocks = n_groups / 4u;
    float acc = 0.0f;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((threadgroup const half4*)(x + elem));
        const half4 xb = *((threadgroup const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kHCGroup / 2u) + lane];
        const float x0 = float(x[g * kHCGroup + lane * 2u]);
        const float x1 = float(x[g * kHCGroup + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// Same dot with x in device memory (the [lowRank] half vector).
static inline float hc_int4_row_dot_dev(device const uint8_t* W_row,
                                        device const bfloat* s_row,
                                        device const bfloat* b_row,
                                        device const half* x,
                                        uint N, uint lane) {
    const uint n_groups = N / kHCGroup;
    const uint full_blocks = n_groups / 4u;
    float acc = 0.0f;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kHCGroup / 2u) + lane];
        const float x0 = float(x[g * kHCGroup + lane * 2u]);
        const float x1 = float(x[g * kHCGroup + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

/// Phase 1: grouped RMSNorm of the S streams, then the down-projection and
/// its silu. Every threadgroup builds the whole normed vector in threadgroup
/// memory, because its GEMV rows need all of it and a dispatch cannot wait on
/// another threadgroup's slice; threadgroup 0 also publishes it for phase 2
/// and the inject. 8 rows per threadgroup, one simdgroup each.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_read_phase1_int4(
    device const half*    streams   [[buffer(0)]],   // [S * D]
    device const bfloat*  hcNorm    [[buffer(1)]],   // [S * D]
    device const uint8_t* downW     [[buffer(2)]],   // [lowRank, S*D] int4
    device const bfloat*  downS     [[buffer(3)]],
    device const bfloat*  downB     [[buffer(4)]],
    device       half*    normed    [[buffer(5)]],   // [S * D] out
    device       half*    lowRank   [[buffer(6)]],   // [lowRank] out, silu applied
    constant     uint&    D         [[buffer(7)]],
    constant     uint&    S         [[buffer(8)]],
    constant     uint&    R         [[buffer(9)]],   // lowRank
    constant     float&   eps       [[buffer(10)]],
    constant     float&   inScale   [[buffer(11)]],  // 1/S inside the silu
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint sgs  [[simdgroups_per_threadgroup]]
) {
    // One staging array, not two: `xs` used to hold a threadgroup copy of
    // `streams`, which put `nrm` + `xs` (2 x 10,240 halves = 40,992 bytes) over
    // Apple's 32,768-byte threadgroup limit, so this kernel could not build at
    // all and the fused read path was dead code. The copy bought nothing: every
    // thread reads only the elements it wrote (same stride in the staging loop
    // and in the reduction), so `streams` is read from device memory instead.
    // The values are identical, so the arithmetic and its rounding are
    // unchanged; what the fused path loses is one device->threadgroup staging
    // pass, which is why it stays opt-in (`TINYTITAN_HC_FUSED`) until it is measured
    // against the unfused path on the family that uses it.
    threadgroup half  nrm[kHCMaxWide];
    threadgroup float partial[kHCThreads / 32u];
    const uint wide = S * D;
    // Per-stream inverse RMS, identical reduction to rmsnorm_bf16w_grouped.
    for (uint s = 0; s < S; ++s) {
        const float inv = hc_rms_inv(streams + s * D, D, eps, lid, kHCThreads, lane, sg, sgs, partial);
        for (uint i = lid; i < D; i += kHCThreads) {
            const float xv = float(streams[s * D + i]);
            const float wv = float(hcNorm[s * D + i]);
            nrm[s * D + i] = half(xv * inv * wv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tg == 0) {
        for (uint i = lid; i < wide; i += kHCThreads) normed[i] = nrm[i];
    }
    const uint row = tg * 8u + sg;
    if (row >= R) return;
    const uint n_groups = wide / kHCGroup;
    const float acc = hc_int4_row_dot_tg(downW + row * (wide / 2u),
                                         downS + row * n_groups,
                                         downB + row * n_groups,
                                         nrm, wide, lane);
    if (lane == 0) {
        // GEMV writes half, silu reads it back: reproduce both roundings.
        const float v = float(half(acc)) * inScale;
        lowRank[row] = half(v / (1.0f + exp(-v)));
    }
}

/// Phase 2: the up-projection with its sigmoid, reduced over streams into the
/// block input. Rows are (stream, d) pairs; a threadgroup takes 2 d-values x S
/// streams so the reduce over streams needs no second dispatch.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_read_phase2_int4(
    device const uint8_t* upW       [[buffer(0)]],   // [S*D, lowRank] int4
    device const bfloat*  upS       [[buffer(1)]],
    device const bfloat*  upB       [[buffer(2)]],
    device const half*    lowRank   [[buffer(3)]],   // [lowRank]
    device const half*    normed    [[buffer(4)]],   // [S * D]
    device       half*    blockIn   [[buffer(5)]],   // [D] out
    constant     uint&    D         [[buffer(6)]],
    constant     uint&    S         [[buffer(7)]],
    constant     uint&    R         [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float mixv[8];
    const uint dPer = 8u / S;                 // d-values per threadgroup
    const uint dLocal = sg / S;
    const uint s = sg - dLocal * S;
    const uint d = tg * dPer + dLocal;
    const uint n_groups = R / kHCGroup;
    float m = 0.0f;
    if (d < D) {
        const uint row = s * D + d;
        m = hc_int4_row_dot_dev(upW + row * (R / 2u), upS + row * n_groups,
                                upB + row * n_groups, lowRank, R, lane);
    }
    if (lane == 0) mixv[sg] = float(half(m));   // GEMV output rounds to half
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < dPer) {
        const uint dd = tg * dPer + lid;
        if (dd < D) {
            float acc = 0.0f;
            for (uint ss = 0; ss < S; ++ss) {
                // read gate: half(sigmoid(mix)) as hc_stream_mix_reduce_fp16
                const half g = half(1.0f / (1.0f + exp(-mixv[lid * S + ss])));
                acc += float(g) * float(normed[ss * D + dd]);
            }
            blockIn[dd] = half(acc / float(S));
        }
    }
}

/// Inject: the write gate's GEMV (S rows over the normed vector) and the gated
/// outer-product add into every stream, one dispatch. Each threadgroup
/// recomputes the S-row GEMV -- it reads 40 KB -- which is cheaper than the
/// launch it replaces.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_inject_int4(
    device       half*    streams   [[buffer(0)]],   // [S * D], updated
    device const half*    normed    [[buffer(1)]],   // [S * D]
    device const uint8_t* injW      [[buffer(2)]],   // [S, S*D] int4
    device const bfloat*  injS      [[buffer(3)]],
    device const bfloat*  injB      [[buffer(4)]],
    device const half*    blockOut  [[buffer(5)]],   // [D]
    constant     uint&    D         [[buffer(6)]],
    constant     uint&    S         [[buffer(7)]],
    constant     float&   inScale   [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half inj[8];
    const uint wide = S * D;
    const uint n_groups = wide / kHCGroup;
    if (sg < S) {
        const float acc = hc_int4_row_dot_dev(injW + sg * (wide / 2u),
                                              injS + sg * n_groups,
                                              injB + sg * n_groups,
                                              normed, wide, lane);
        if (lane == 0) inj[sg] = half(acc);   // GEMV output rounds to half
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint e = tg * kHCThreads + lid;
    if (e >= wide) return;
    const uint s = e / D;
    const uint d = e - s * D;
    const half g = half(2.0f / (1.0f + exp(-float(inj[s]) * inScale)));
    streams[e] = half(float(streams[e]) + float(blockOut[d]) * float(g));
}
