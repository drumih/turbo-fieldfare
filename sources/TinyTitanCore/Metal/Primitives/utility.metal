#include <metal_stdlib>
using namespace metal;

// Local copy of the tanh-GELU used by the shared-expert kernels. Kept local
// (rather than a cross-module static inline from moe.metal) so reordering the
// concatenated shader library can never break this file's resolution.
static inline float utility_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same gelu_pytorch_tanh activation without compiling a private shader
// module.
[[kernel, max_total_threads_per_threadgroup(256)]]
void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(utility_gelu_pytorch_tanh(g) * u);
}

// SwiGLU counterpart for architectures with silu hidden activation (Qwen 3.6).
[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half((g / (1.0f + exp(-g))) * u);
}

// out[i] *= sigmoid(gate[i]) — Qwen 3.6 full-attention output gate.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_gate_mul_fp16(
    device half*       out  [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    out[tid] = half(float(out[tid]) / (1.0f + exp(-g)));
}

// out[i] *= softplus(gate[i]; beta = ln 2) — K2-Horizon's full-attention gate.
//
// softplus(x; beta) = log(1 + exp(beta*x)) / beta, evaluated in the
// overflow-safe branch. At beta = ln 2 this is log2(1 + 2^x), which is what
// the checkpoint's `attention_gate_func: softplus` computes; it is not the
// sigmoid the Qwen families use, and substituting one for the other yields a
// bounded, positive, wrong gate rather than a shape error.
[[kernel, max_total_threads_per_threadgroup(256)]]
void softplus_gate_mul_fp16(
    device half*       out  [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float beta = 0.6931471805599453f;
    const float s = beta * float(gate[tid]);
    const float sp = s > 0.0f ? s + log(1.0f + exp(-s)) : log(1.0f + exp(s));
    out[tid] = half(float(out[tid]) * (sp / beta));
}

// out[i] *= softplus(gate[i / head_dim]) — Laguna-S-2.1's per-head gate.
//
// One gate value per query head, broadcast across that head's head_dim
// channels. Plain softplus log(1 + exp(x)), not K2's beta = ln 2 form: the
// checkpoints differ and the gate is applied before o_proj in both.
[[kernel, max_total_threads_per_threadgroup(256)]]
void softplus_gate_per_head_mul_fp16(
    device half*       out      [[buffer(0)]],
    device const half* gate     [[buffer(1)]],
    constant uint&     count    [[buffer(2)]],
    constant uint&     head_dim [[buffer(3)]],
    uint               tid      [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float s = float(gate[tid / head_dim]);
    const float sp = s > 0.0f ? s + log(1.0f + exp(-s)) : log(1.0f + exp(s));
    out[tid] = half(float(out[tid]) * sp);
}

// out[i] = x[i] * factor — the residual-stream scale.
[[kernel, max_total_threads_per_threadgroup(256)]]
void scale_fp16(
    device half*       out    [[buffer(0)]],
    device const half* x      [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    constant float&    factor [[buffer(3)]],
    uint               tid    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    out[tid] = half(float(x[tid]) * factor);
}

// hidden[i] += factor * delta[i] — the scaled residual add.
[[kernel, max_total_threads_per_threadgroup(256)]]
void residual_add_scaled_fp16(
    device half*       hidden [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    constant float&    factor [[buffer(3)]],
    uint               tid    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    hidden[tid] = half(float(hidden[tid]) + factor * float(delta[tid]));
}

// y[i] *= sigmoid(gate[0]) — Qwen 3.6 shared-expert scalar gate.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_scalar_mul_fp16(
    device half*       y    [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[0]);
    y[tid] = half(float(y[tid]) / (1.0f + exp(-g)));
}

// Qwen 3.6 q_proj emits per-head [query(D) ; gate(D)] pairs. Split them into
// contiguous q [H, D] and gate [H, D] so the per-head norm, RoPE, and
// attention kernels see their usual layout.
[[kernel, max_total_threads_per_threadgroup(256)]]
void split_q_gate_fp16(
    device const half* packed [[buffer(0)]],   // [H, 2*D]
    device half*       q      [[buffer(1)]],   // [H, D]
    device half*       gate   [[buffer(2)]],   // [H, D]
    constant uint&     heads  [[buffer(3)]],
    constant uint&     dim    [[buffer(4)]],
    uint               tid   [[thread_position_in_grid]]
) {
    const uint total = heads * dim;
    if (tid >= total) return;
    const uint h = tid / dim;
    const uint d = tid % dim;
    q[tid] = packed[h * 2u * dim + d];
    gate[tid] = packed[h * 2u * dim + dim + d];
}

// hidden[i] += delta[i] — plain pre-norm residual add for architectures
// without a fused sandwich tail.
[[kernel, max_total_threads_per_threadgroup(256)]]
void residual_add_fp16(
    device half*       hidden [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    hidden[tid] = half(float(hidden[tid]) + float(delta[tid]));
}

// Row-wise concatenation used by Qwen3.5-MoE MTP's 4096 -> 2048 adapter.
[[kernel, max_total_threads_per_threadgroup(256)]]
void concat_rows_fp16(
    device const half* lhs  [[buffer(0)]],
    device const half* rhs  [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     rows [[buffer(3)]],
    constant uint&     dim  [[buffer(4)]],
    uint               tid  [[thread_position_in_grid]]
) {
    const uint total = rows * dim;
    if (tid >= total) return;
    const uint row = tid / dim;
    const uint col = tid % dim;
    out[row * 2u * dim + col] = lhs[tid];
    out[row * 2u * dim + dim + col] = rhs[tid];
}

// ============================================================================
// Hyper-connection (Gated Residual) stream plumbing — Qwen3.8-Flash-Next.
//
// The residual is S parallel streams of D (4 x 2560). Everything else in the
// block is per-D; only these two steps see the stream axis, so they are the
// only new kernels the residual needs — the projections and gating reuse the
// existing INT4 GEMV and elementwise primitives.
// ============================================================================

// out[d] = (1/S) * sum_s mix[s*D + d] * normed[s*D + d]
//
// The gated read: each stream's normalized value is scaled by its own mix
// weight, then the streams are averaged into the single D-wide vector the
// attention or MLP block consumes.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_stream_mix_reduce_fp16(
    device const half* mix     [[buffer(0)]],   // [T, S, D]
    device const half* normed  [[buffer(1)]],   // [T, S, D]
    device       half* out     [[buffer(2)]],   // [T, D]
    constant     uint& D       [[buffer(3)]],
    constant     uint& S       [[buffer(4)]],
    // Rows: 1 for a decode step, the chunk length in prefill. The residual is
    // [T, S, D], so a token's streams stay contiguous and the same index
    // arithmetic serves both.
    constant     uint& T       [[buffer(5)]],
    constant    float& inScale [[buffer(6)]],
    uint               tid     [[thread_position_in_grid]]
) {
    if (tid >= D * T) return;
    const uint t = tid / D;
    const uint d = tid - t * D;
    device const half* mixRow = mix + t * S * D;
    device const half* normRow = normed + t * S * D;
    float acc = 0.0f;
    for (uint s = 0; s < S; ++s) {
        const uint i = s * D + d;
        // The read gate used to be a separate sigmoid pass over the whole
        // [S, D] mix before this reduce read it back. Folding it here removes
        // one dispatch per sublayer -- 96 per token at 48 layers -- and costs
        // nothing: this loop already touches each element exactly once, so the
        // sigmoid count is unchanged. Rounding to half before use reproduces
        // the separate pass bit for bit.
        const half g = half(1.0f / (1.0f + exp(-float(mixRow[i]) * inScale)));
        acc += float(g) * float(normRow[i]);
    }
    out[tid] = half(acc / float(S));
}

// streams[s*D + d] += block_out[d] * inject[s]
//
// The gated write: one block output is injected back into every stream, each
// with its own learned weight. Accumulates in place, so the residual carries
// forward exactly as a plain transformer's would -- the streams differ only in
// how much of each block they take.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_stream_inject_fp16(
    device       half* streams   [[buffer(0)]],  // [S * D], updated in place
    device const half* block_out [[buffer(1)]],  // [D]
    device const half* inject    [[buffer(2)]],  // [S], pre-sigmoid
    constant     uint& D         [[buffer(3)]],
    constant     uint& S         [[buffer(4)]],
    constant     uint& T         [[buffer(5)]],
    constant    float& inScale   [[buffer(6)]],
    uint               tid       [[thread_position_in_grid]]
) {
    if (tid >= D * S * T) return;
    const uint t = tid / (D * S);
    const uint rest = tid - t * D * S;
    const uint s = rest / D;
    const uint d = rest - s * D;
    // Write gate folded in: this was a sigmoid pass over S values (four, on
    // this family) in its own dispatch and its own command encoder. The
    // recomputation per thread is free on a kernel this memory-bound, and
    // rounding to half before use reproduces the separate pass exactly.
    const half g = half(2.0f / (1.0f + exp(-float(inject[t * S + s]) * inScale)));
    streams[tid] = half(float(streams[tid])
                        + float(block_out[t * D + d]) * float(g));
}

// out[i] = outScale * sigmoid(inScale * x[i]) — standalone gate, distinct
// from the fused sigmoid_gate_mul which multiplies into an existing value.
//
// The scales are not decoration: the Gated Residual computes its read gate as
// sigmoid(W @ normed / S) and its write gate as 2 * sigmoid(W @ normed / S).
// Folding them here keeps the chain to one kernel per step instead of adding
// separate scale passes over a 10,240-wide vector.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_fp16(
    device const half* x        [[buffer(0)]],
    device       half* out      [[buffer(1)]],
    constant     uint& count    [[buffer(2)]],
    constant    float& inScale  [[buffer(3)]],
    constant    float& outScale [[buffer(4)]],
    uint               tid      [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float v = float(x[tid]) * inScale;
    out[tid] = half(outScale / (1.0f + exp(-v)));
}

// out[i] = silu(inScale * x[i]) — standalone, where silu_mul_fp16 fuses a
// second operand this path does not have. The scale must be applied INSIDE
// the nonlinearity: silu(x/S) is not silu(x)/S, so it cannot be folded away
// afterwards.
[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_fp16(
    device const half* x       [[buffer(0)]],
    device       half* out     [[buffer(1)]],
    constant     uint& count   [[buffer(2)]],
    constant    float& inScale [[buffer(3)]],
    uint               tid     [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float v = float(x[tid]) * inScale;
    out[tid] = half(v / (1.0f + exp(-v)));
}

// ============================================================================
// PLE (n-gram) block — Qwen3.8-Flash-Next. Shapes match mlx_qwen4exp/ple.py.
// ============================================================================

// s[stream] = sum over d of key[s,d] * query[s,d] / sqrt(D)
//
// One score per residual stream, from the normalized n-gram key against the
// normalized residual. Distinct from the hyper-connection reduce, which
// produces a D-wide vector; this collapses each stream to a scalar.
[[kernel, max_total_threads_per_threadgroup(256)]]
void ple_stream_score_fp16(
    device const half* key    [[buffer(0)]],   // [T, S, D]
    device const half* query  [[buffer(1)]],   // [T, S, D]
    device       half* out    [[buffer(2)]],   // [T, S]
    constant     uint& D      [[buffer(3)]],
    constant     uint& S      [[buffer(4)]],
    constant     uint& T      [[buffer(5)]],
    uint               tid    [[thread_position_in_grid]]
) {
    if (tid >= S * T) return;
    float acc = 0.0f;
    const uint base = tid * D;
    for (uint d = 0; d < D; ++d) {
        acc += float(key[base + d]) * float(query[base + d]);
    }
    out[tid] = half(acc / sqrt(float(D)));
}

// gate[s] = sigmoid(sign(x) * sqrt(max(|x|, floor)))
//
// A SIGNED square root before the sigmoid, with a floor on the magnitude. The
// floor is not decoration: without it the derivative is unbounded as x -> 0,
// and the reference carries 1e-6 there.
[[kernel, max_total_threads_per_threadgroup(256)]]
void ple_signed_sqrt_gate_fp16(
    device const half* x      [[buffer(0)]],
    device       half* out    [[buffer(1)]],
    constant     uint& count  [[buffer(2)]],
    constant    float& floorV [[buffer(3)]],
    uint               tid    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float v = float(x[tid]);
    const float m = sqrt(max(fabs(v), floorV));
    const float signed_root = v < 0.0f ? -m : m;
    out[tid] = half(1.0f / (1.0f + exp(-signed_root)));
}

// out[s*D + d] = value[d] * gate[s]
//
// One D-wide value broadcast across streams, each scaled by its own gate.
// Writes rather than accumulates -- the block adds this to the residual
// separately, alongside the convolution branch.
[[kernel, max_total_threads_per_threadgroup(256)]]
void ple_broadcast_scale_fp16(
    device const half* value [[buffer(0)]],  // [T, D]
    device const half* gate  [[buffer(1)]],  // [T, S]
    device       half* out   [[buffer(2)]],  // [T, S, D]
    constant     uint& D     [[buffer(3)]],
    constant     uint& S     [[buffer(4)]],
    constant     uint& T     [[buffer(5)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= D * S * T) return;
    const uint t = tid / (D * S);
    const uint rest = tid - t * D * S;
    const uint s = rest / D;
    const uint d = rest - s * D;
    out[tid] = half(float(value[t * D + d]) * float(gate[t * S + s]));
}


// Depthwise causal convolution over time, dilated:
//   out[t,c] = sum over k of w[c,k] * xpad[t + k*dilation, c]
//
// `xpad` is the carried state followed by this chunk, so tap k reads
// (K-1-k)*dilation positions back from t -- the newest tap is the last one,
// not the first. Getting that backwards still produces smooth, plausible
// output, which is why it is spelled out here and pinned in a test.
[[kernel, max_total_threads_per_threadgroup(256)]]
void ple_dilated_depthwise_conv_fp16(
    device const half* xpad     [[buffer(0)]],  // [(history + T) * C]
    // bfloat, not half: `ple.conv1d` is a bf16 checkpoint tensor like the
    // norms. Reading its bytes as fp16 yields plausible small numbers rather
    // than an error, which is exactly how this stays hidden.
    device const bfloat* weight [[buffer(1)]],  // [C * K], channel-major
    device       half* out      [[buffer(2)]],  // [T * C]
    constant     uint& C        [[buffer(3)]],
    constant     uint& T        [[buffer(4)]],
    constant     uint& K        [[buffer(5)]],
    constant     uint& dilation [[buffer(6)]],
    uint               tid      [[thread_position_in_grid]]
) {
    if (tid >= T * C) return;
    const uint t = tid / C;
    const uint c = tid - t * C;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        const uint row = t + k * dilation;
        acc += float(weight[c * K + k]) * float(xpad[row * C + c]);
    }
    // silu, as the reference applies before the residual add.
    out[tid] = half(acc / (1.0f + exp(-acc)));
}

// Replicate stream 0 across the remaining residual streams, in place.
//
// Entry to a hyper-connection stack: the token embedding is written once, then
// every stream starts from that same vector. Only slots at or past D are
// written and only slots below D are read, so this is safe in place.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_stream_broadcast_fp16(
    device       half* streams [[buffer(0)]],  // [S * D], stream 0 populated
    constant     uint& D       [[buffer(1)]],
    constant     uint& S       [[buffer(2)]],
    uint               tid     [[thread_position_in_grid]]
) {
    const uint total = D * S;
    if (tid < D || tid >= total) return;
    streams[tid] = streams[tid % D];
}

// dst[t, s, d] = src[t, d] -- widen a [T, D] block into the [T, S, D]
// residual.
//
// Separate from the in-place broadcast above because the two layouts differ:
// a prefill chunk's embeddings are [T, D] contiguous, and token t's row lands
// at t * S * D in the residual, so expanding in place would overwrite rows
// that have not been read yet.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_stream_expand_fp16(
    device const half* src     [[buffer(0)]],  // [T, D]
    device       half* dst     [[buffer(1)]],  // [T, S, D]
    constant     uint& D       [[buffer(2)]],
    constant     uint& S       [[buffer(3)]],
    constant     uint& T       [[buffer(4)]],
    uint               tid     [[thread_position_in_grid]]
) {
    if (tid >= D * S * T) return;
    const uint t = tid / (D * S);
    const uint rest = tid - t * D * S;
    const uint d = rest % D;
    dst[tid] = src[t * D + d];
}

// out[t, d] = mean over streams of streams[t, s, d].
//
// The unweighted collapse of the wide residual. Distinct from
// hc_stream_mix_reduce, which weights each stream by a learned gate first:
// the MTP draft's fusion takes a plain mean, because the gate that would
// weight it belongs to a mixer that runs after the draft's own layer, not
// before it.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_stream_mean_fp16(
    device const half* streams [[buffer(0)]],  // [T, S, D]
    device       half* out     [[buffer(1)]],  // [T, D]
    constant     uint& D       [[buffer(2)]],
    constant     uint& S       [[buffer(3)]],
    constant     uint& T       [[buffer(4)]],
    uint               tid     [[thread_position_in_grid]]
) {
    if (tid >= D * T) return;
    const uint t = tid / D;
    const uint d = tid - t * D;
    device const half* row = streams + t * S * D;
    float acc = 0.0f;
    for (uint s = 0; s < S; ++s) { acc += float(row[s * D + d]); }
    out[tid] = half(acc / float(S));
}
