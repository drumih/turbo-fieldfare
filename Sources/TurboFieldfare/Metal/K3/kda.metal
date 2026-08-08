#include <metal_stdlib>
using namespace metal;

// ============================================================================
// kda — Kimi Delta Attention single-token decode step (docs/K3_DATAFLOW.md
// "KDA layer"), ported against the fla-verified k3_ops.c (k3_shortconv,
// k3_kda_decay, k3_kda_step, l2norm_).
//
// Per token, per head (H heads, D=head_dim, P = H*D):
//   1. k3_kda_conv   : causal depthwise conv width 4 + SiLU on the projected
//                      q/k/v, carrying the last-3-inputs state per channel.
//   2. k3_kda_step   : L2-normalize q,k per head (eps 1e-6 added to the SUM
//                      of squares), scale q by D^-0.5, per-channel decay
//                      alpha = exp(-5 * sigmoid(exp(A_log[h]) * z)) (z already
//                      includes dt_bias), beta_h = sigmoid(beta_logit), then
//                      the delta rule on the fp32 state S (D x D, row = key
//                      channel):
//                        S <- Diag(alpha) S
//                        u = S^T k                       (post-decay)
//                        S <- S + beta * k (v - u)^T
//                        o = S^T q                       (AFTER the update)
//   3. k3_kda_onorm  : per-head RMSNorm with o_norm.weight (eps = model
//                      rms_norm_eps), then elementwise * sigmoid(gate)
//                      (gate = full-rank g_proj output), fp16 out for the
//                      external o_proj trunk GEMV.
//
// Conv weights are FP32 device buffers ([3][P][4], q|k|v, taps ordered
// oldest..newest so tap 3 multiplies the current input) — the loader
// converts them once; they are tiny (147K parameters for canonical K3).
// Conv states are FP32 [3][P][3], oldest first, updated in place. The KDA
// checkpoint lists conv biases (fla's ShortConvolution default); the
// canonical task shapes define the bias-free decode contract, so the kernel
// takes no bias — a bias, if present, is folded by the loader stage.
// ============================================================================

constant constexpr uint kK3KdaConvWidth = 4;
constant constexpr uint kK3KdaConvHist = kK3KdaConvWidth - 1;
constant constexpr uint kK3KdaMaxHeadDim = 128;
constant constexpr uint kK3KdaThreads = 128;
constant constexpr uint kK3KdaMaxSimdGroups = kK3KdaThreads / 32;
constant constexpr float kK3KdaL2NormEps = 1e-6f;
constant constexpr float kK3KdaGateLowerBound = -5.0f;

constant uint FC_K3_KDA_NUM_HEADS [[function_constant(70)]];
constant uint FC_K3_KDA_HEAD_DIM [[function_constant(71)]];
constant uint FC_K3_KDA_CHANNELS [[function_constant(72)]];
constant bool FC_K3_KDA_USE_FC [[function_constant(73)]];

static inline uint k3_kda_fc_num_heads(constant uint& H) {
    return (is_function_constant_defined(FC_K3_KDA_USE_FC) &&
            FC_K3_KDA_USE_FC &&
            is_function_constant_defined(FC_K3_KDA_NUM_HEADS))
        ? FC_K3_KDA_NUM_HEADS : H;
}

static inline uint k3_kda_fc_head_dim(constant uint& D) {
    return (is_function_constant_defined(FC_K3_KDA_USE_FC) &&
            FC_K3_KDA_USE_FC &&
            is_function_constant_defined(FC_K3_KDA_HEAD_DIM))
        ? FC_K3_KDA_HEAD_DIM : D;
}

static inline uint k3_kda_fc_channels(constant uint& P) {
    return (is_function_constant_defined(FC_K3_KDA_USE_FC) &&
            FC_K3_KDA_USE_FC &&
            is_function_constant_defined(FC_K3_KDA_CHANNELS))
        ? FC_K3_KDA_CHANNELS : P;
}

static inline float k3_sigmoid_(float v) {
    return 1.0f / (1.0f + exp(-v));
}

// Block reduction over one 128-thread threadgroup (4 SIMDs), the house
// two-stage pattern: per-SIMD simd_sum, scratch, one SIMD merges, broadcast
// slot doubles as the return value.
static inline float k3_kda_block_reduce_sum(
    float v,
    uint simd_lane_id,
    uint simd_group_id,
    uint simdgroups,
    threadgroup float* scratch,
    threadgroup float* bcast
) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}

// Depthwise causal conv width 4 + SiLU for q, k, v in one dispatch. One
// thread per (tensor, channel); the state row for a channel is touched by
// exactly one thread, so the in-place shift needs no synchronization.
kernel void k3_kda_conv(
    device const half*  xq          [[buffer(0)]],   // [P] fp16 projected q
    device const half*  xk          [[buffer(1)]],   // [P] fp16 projected k
    device const half*  xv          [[buffer(2)]],   // [P] fp16 projected v
    device const float* weights     [[buffer(3)]],   // [3][P][4] fp32, q|k|v
    device float*       conv_states [[buffer(4)]],   // [3][P][3] fp32, in place
    device float*       q_out       [[buffer(5)]],   // [P] fp32
    device float*       k_out       [[buffer(6)]],   // [P] fp32
    device float*       v_out       [[buffer(7)]],   // [P] fp32
    constant uint&      channels    [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint P = k3_kda_fc_channels(channels);
    if (gid >= 3u * P) return;
    const uint which = gid / P;
    const uint c = gid % P;

    device const half* x = (which == 0u) ? xq : (which == 1u) ? xk : xv;
    device float* y = (which == 0u) ? q_out : (which == 1u) ? k_out : v_out;
    device const float* w = weights + which * P * kK3KdaConvWidth + c * kK3KdaConvWidth;
    device float* st = conv_states + which * P * kK3KdaConvHist + c * kK3KdaConvHist;

    const float cur = float(x[c]);
    // Taps oldest..newest: w[3] multiplies the current input, w[0..2] the
    // history (oldest first).
    float acc = w[3] * cur;
    acc = fma(w[0], st[0], acc);
    acc = fma(w[1], st[1], acc);
    acc = fma(w[2], st[2], acc);
    // Shift the history left and append the current (pre-conv) input.
    st[0] = st[1];
    st[1] = st[2];
    st[2] = cur;
    y[c] = acc * k3_sigmoid_(acc);   // SiLU, fused
}

// Fused per-head recurrence step. One threadgroup per head, kK3KdaThreads
// threads; thread j owns state COLUMN j through the whole step (decay, delta
// update, output), which serializes every per-column dot in the same
// i-ascending order as k3_kda_step and needs a single barrier (for the
// shared q/k/alpha staging).
kernel void k3_kda_step(
    device float*       state        [[buffer(0)]],   // [H][D][D] fp32, in place
    device const float* q_in         [[buffer(1)]],   // [P] fp32, post-conv
    device const float* k_in         [[buffer(2)]],   // [P] fp32, post-conv
    device const float* v_in         [[buffer(3)]],   // [P] fp32, post-conv
    device const float* z            [[buffer(4)]],   // [P] fp32, f_b(f_a(x)) + dt_bias
    device const float* beta_logits  [[buffer(5)]],   // [H] fp32
    device const float* a_log        [[buffer(6)]],   // [H] fp32
    device float*       o            [[buffer(7)]],   // [P] fp32 out
    constant uint&      num_heads    [[buffer(8)]],
    constant uint&      head_dim     [[buffer(9)]],
    uint head         [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_s[kK3KdaMaxHeadDim];
    threadgroup float k_s[kK3KdaMaxHeadDim];
    threadgroup float alpha_s[kK3KdaMaxHeadDim];
    threadgroup float reduce_scratch[kK3KdaMaxSimdGroups];
    threadgroup float bcast;

    const uint H = k3_kda_fc_num_heads(num_heads);
    const uint D = k3_kda_fc_head_dim(head_dim);
    if (head >= H) return;
    const uint P = H * D;
    device float* S = state + uint(head) * D * D;
    device const float* q_head = q_in + uint(head) * D;
    device const float* k_head = k_in + uint(head) * D;
    device const float* v_head = v_in + uint(head) * D;
    device const float* z_head = z + uint(head) * D;

    // L2 norms of q and k (eps added to the SUM of squares, per l2norm_).
    float sq = 0.0f;
    float sk = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float qv = q_head[i];
        const float kv = k_head[i];
        sq = fma(qv, qv, sq);
        sk = fma(kv, kv, sk);
    }
    const float ssq = k3_kda_block_reduce_sum(
        sq, simd_lane_id, simd_group_id, simdgroups, reduce_scratch, &bcast);
    const float ssk = k3_kda_block_reduce_sum(
        sk, simd_lane_id, simd_group_id, simdgroups, reduce_scratch, &bcast);
    const float inv_q = rsqrt(ssq + kK3KdaL2NormEps);
    const float inv_k = rsqrt(ssk + kK3KdaL2NormEps);
    // q is scaled by D^-0.5 before the recurrence (fla default scale).
    const float q_scale = inv_q * rsqrt(float(D));

    // Per-channel decay gate and per-head beta. beta is recomputed per
    // thread — one sigmoid is cheaper than another barrier.
    const float a = exp(a_log[head]);
    const float beta = k3_sigmoid_(beta_logits[head]);
    for (uint i = lid; i < D; i += lsize) {
        q_s[i] = q_head[i] * q_scale;
        k_s[i] = k_head[i] * inv_k;
        // alpha = exp(-5 * sigmoid(a * z)), z already carries dt_bias.
        alpha_s[i] = exp(kK3KdaGateLowerBound * k3_sigmoid_(a * z_head[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Column-local: decay + u = S^T k, then the rank-one delta write, then
    // o = S^T q from the updated state. No further barriers: every S[i][j]
    // access for a given j is made by the same thread.
    for (uint j = lid; j < D; j += lsize) {
        float u = 0.0f;
        for (uint i = 0; i < D; ++i) {
            float s = S[i * D + j] * alpha_s[i];
            S[i * D + j] = s;
            u = fma(k_s[i], s, u);
        }
        const float err = v_head[j] - u;
        for (uint i = 0; i < D; ++i) {
            S[i * D + j] = fma(beta * k_s[i], err, S[i * D + j]);
        }
        float out = 0.0f;
        for (uint i = 0; i < D; ++i) {
            out = fma(q_s[i], S[i * D + j], out);
        }
        o[uint(head) * D + j] = out;
    }
}

// Per-head gated output RMSNorm: y = (x * rsqrt(mean(x^2) + eps) * w)
// . sigmoid(gate), fp16 out. One threadgroup per head.
kernel void k3_kda_onorm(
    device const float* x        [[buffer(0)]],   // [P] fp32 (k3_kda_step o)
    device const half*  gate     [[buffer(1)]],   // [P] fp16 (g_proj output)
    device const float* weight   [[buffer(2)]],   // [D] fp32 (o_norm.weight)
    device half*        out      [[buffer(3)]],   // [P] fp16
    constant uint&      num_heads [[buffer(4)]],
    constant uint&      head_dim [[buffer(5)]],
    constant float&     eps      [[buffer(6)]],
    uint head         [[threadgroup_position_in_grid]],
    uint lid          [[thread_position_in_threadgroup]],
    uint lsize        [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups   [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[kK3KdaMaxSimdGroups];
    threadgroup float bcast;
    const uint H = k3_kda_fc_num_heads(num_heads);
    const uint D = k3_kda_fc_head_dim(head_dim);
    if (head >= H) return;

    device const float* x_head = x + uint(head) * D;
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = x_head[i];
        ss = fma(v, v, ss);
    }
    const float total = k3_kda_block_reduce_sum(
        ss, simd_lane_id, simd_group_id, simdgroups, reduce_scratch, &bcast);
    const float inv = rsqrt(total / float(D) + eps);

    device const half* gate_head = gate + uint(head) * D;
    device half* out_head = out + uint(head) * D;
    for (uint i = lid; i < D; i += lsize) {
        const float normed = weight[i] * x_head[i] * inv;
        out_head[i] = half(normed * k3_sigmoid_(float(gate_head[i])));
    }
}
