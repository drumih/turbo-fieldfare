#include <metal_stdlib>

/// Widest activation the Gated-DeltaNet staging tile covers. The dense
/// Qwen 3.5 9B is 4096 wide; 8192 bytes of threadgroup memory, well
/// inside the 32 KB limit, and this kernel is the only one that stages
/// an activation of that width.
constant constexpr uint kGDNActivationMaxD = 4096;
using namespace metal;

// ============================================================================
// gdn.metal — gated-DeltaNet linear attention (Qwen 3.6 layers with layer-mask
// value 2). Decode processes one token; prefill processes a bounded chunk of
// rows with the recurrence sequential inside the kernel.
//
// Dataflow per layer (dimensions are runtime parameters; Qwen 3.6 uses
// C = 8192 conv channels, K = 4 taps, Hk = 16 x Dk = 128, Hv = 32 x Dv = 128):
//   mixed_qkv = in_proj_qkv(x)                       (existing INT4 GEMV)
//   conv_out  = silu(causal_depthwise_conv(mixed))    gdn_conv_mix_*
//   q, k      = per-head no-weight RMS norm + scale   gdn_qk_norm_*
//   y, S      = gated delta rule recurrence           gdn_delta_*
//   out       = rmsnorm(y) * gate(z)                  gdn_gated_norm_*
//               gate is silu or sigmoid per FC_GDN_SIGMOID_GATE
//   out_proj(out)                                     (existing INT4 GEMV)
//
// The recurrence follows the mlx-vlm reference (gated_delta.py): per value
// head h with state S[Dv, Dk] (FP32, persistent across tokens):
//   g    = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
//   beta = sigmoid(b[h])
//   S    = S * g
//   kv   = S @ k
//   S   += outer((v - kv) * beta, k)
//   y    = S @ q
// ============================================================================

constant constexpr float kGdnRmsEps = 1e-6f;
constant constexpr uint kGdnGroupSize = 64;

constant uint FC_GDN_IN_QKV    [[function_constant(90)]];
constant uint FC_GDN_IN_Z      [[function_constant(91)]];
constant uint FC_GDN_IN_AB     [[function_constant(92)]];
constant uint FC_GDN_IN_N      [[function_constant(93)]];
constant bool FC_GDN_IN_USE_FC [[function_constant(94)]];
// Threads per threadgroup for the norm kernels (gdn_qk_norm/gdn_gated_norm).
// The wrapper passes the same value it dispatches with, so the partial-count
// loop below can never drift from the actual SIMD-group count.
constant uint FC_GDN_TG_THREADS [[function_constant(95)]];
// Output gate of gdn_gated_norm. Qwen 3.6 / Qwen3-Next gate the normalized
// delta readout with silu; Qwen3.8-Flash-Next gates it with sigmoid
// (`output_gate_type` in its config). The two agree in sign and shape, so
// using the wrong one costs no error and no crash -- it just produces a
// different model. It has to be selected, not assumed.
constant bool FC_GDN_SIGMOID_GATE [[function_constant(96)]];

// Set when in_proj_a and in_proj_b were promoted to the checkpoint's own bf16.
// Only those two: they are 48 rows each against qkv's 10,240, so the fused
// dispatch keeps its shape and only the two narrow slots read differently.
constant bool FC_GDN_AB_BF16 [[function_constant(97)]];
// The decay multiplier the delta rule reads: the engine's installs store the
// raw A_log, so the kernel exponentiates (`exp(-exp(A) * sp)`). GGUF
// checkpoints carry `ssm_a` already as `-exp(A_log)` — llama.cpp consumes it
// raw — so the GGUF route compiles the pre-exponentiated variant and the
// kernel multiplies directly. Measured on the UD-IQ3_XXS checkpoint
// (2026-11-10): blk.0.ssm_a holds values like -9.05 and -77.0, which are
// `-exp(A_log)` and not log-space.
constant bool FC_GDN_A_LOG_PREEXP [[function_constant(98)]];
// Which key head a value head reads when there are more value heads than key
// heads (48 against 16 here). The engine's installs keep the Hugging Face
// order, value heads grouped by key head, so head h reads key head
// h / (Hv / Hk). llama.cpp's converter (`_LinearAttentionVReorderBase`)
// reorders every per-value-head tensor of a GGUF to the tiled order ggml's
// broadcast wants, so there head h reads key head h % Hk.
constant bool FC_GDN_V_TILED [[function_constant(99)]];

static inline bool gdn_ab_bf16() {
    return is_function_constant_defined(FC_GDN_AB_BF16) && FC_GDN_AB_BF16;
}

static inline void gdn_bf16_gemv_body(device const bfloat* W,
                                      device const half* x,
                                      device half* y,
                                      uint N,
                                      uint local_row,
                                      uint lane) {
    device const bfloat* row = W + local_row * N;
    float acc = 0.0f;
    for (uint base = 0; base < N; base += 64u) {
        const uint idx = base + lane * 2u;
        acc = fma(float(row[idx]), float(x[idx]), acc);
        acc = fma(float(row[idx + 1u]), float(x[idx + 1u]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[local_row] = half(acc);
}

static inline uint gdn_tg_threads() {
    return is_function_constant_defined(FC_GDN_TG_THREADS)
        ? FC_GDN_TG_THREADS : 128u;
}

static inline bool gdn_in_use_fc() {
    return is_function_constant_defined(FC_GDN_IN_USE_FC) && FC_GDN_IN_USE_FC;
}

static inline uint gdn_in_fc_qkv(constant uint& rows) {
    return (gdn_in_use_fc() && is_function_constant_defined(FC_GDN_IN_QKV))
        ? FC_GDN_IN_QKV : rows;
}

static inline uint gdn_in_fc_z(constant uint& rows) {
    return (gdn_in_use_fc() && is_function_constant_defined(FC_GDN_IN_Z))
        ? FC_GDN_IN_Z : rows;
}

static inline uint gdn_in_fc_ab(constant uint& rows) {
    return (gdn_in_use_fc() && is_function_constant_defined(FC_GDN_IN_AB))
        ? FC_GDN_IN_AB : rows;
}

static inline uint gdn_in_fc_n(constant uint& n) {
    return (gdn_in_use_fc() && is_function_constant_defined(FC_GDN_IN_N))
        ? FC_GDN_IN_N : n;
}

static inline float gdn_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float gdn_out_gate(float x) {
    const bool sigmoidGate = is_function_constant_defined(FC_GDN_SIGMOID_GATE)
        && FC_GDN_SIGMOID_GATE;
    return sigmoidGate ? (1.0f / (1.0f + exp(-x))) : gdn_silu(x);
}

static inline float gdn_softplus(float x) {
    // log1p(exp(x)) with the standard large-x shortcut; matches
    // mlx.nn.softplus to FP32 precision.
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

/// The decay factor `g` from the gate multiplier `a` and the softplus'd time
/// step. The storage semantics of `a` are selected by FC_GDN_A_LOG_PREEXP:
/// raw A_log (engine installs) vs `-exp(A_log)` (GGUF `ssm_a`).
/// The key head value head `h` pairs with, under FC_GDN_V_TILED.
static inline uint gdn_key_head(uint h, uint Hv, uint Hk) {
    if (is_function_constant_defined(FC_GDN_V_TILED) && FC_GDN_V_TILED) {
        return h % Hk;
    }
    return h / (Hv / Hk);
}

static inline float gdn_decay(float a, float dtBias, float aLog) {
    const float sp = gdn_softplus(a + dtBias);
    if (is_function_constant_defined(FC_GDN_A_LOG_PREEXP) && FC_GDN_A_LOG_PREEXP) {
        return exp(aLog * sp);
    }
    return exp(-exp(aLog) * sp);
}

// ----------------------------------------------------------------------------
// One-SIMD-per-row affine INT4 GEMV body. This is the same math as
// `dequant_int4_gemv_simd_body` in dequant_int4.metal, duplicated here so
// gdn.metal self-contains its dependencies (the combined shader library
// concatenates modules, so relying on another module's static inline would
// silently break if the module order or set changes). K24: keep both copies
// in lockstep; the row body must keep reading packed weights via `ushort*`
// (sub-tensor offsets are only 2-byte aligned).
//
//   y[row] = sum_n W[row, n] * x[n],  W row is N/2 bytes of nibbles with
//   N/64 BF16 scale/bias pairs (MLX affine), one SIMD (32 threads) per row.
// ----------------------------------------------------------------------------
static inline void gdn_dequant_int4_gemv_simd_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGdnGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Read the 4-byte weight chunk as two ushorts: row stride N/2,
        // sub-tensor weightsOffset, and byte_base are all even, so a
        // `ushort*` load is always aligned even when the row is only
        // 2-byte aligned overall.
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
        const uint8_t byte = W_row[g * (kGdnGroupSize / 2) + lane];
        const float x0 = float(x[g * kGdnGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGdnGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// ----------------------------------------------------------------------------
// Fused GDN input projection. Decode issues four INT4 GEMVs that all read the
// same hidden vector `x` (N = hiddenSize): in_proj_qkv (qkvRows), in_proj_z
// (zRows), in_proj_a and in_proj_b (abRows each — 32 rows for Qwen 3.6, i.e.
// four near-empty threadgroups apiece). This kernel dispatches over the
// concatenated row space and routes each row to its own weight/scale/bias base
// and output buffer with a 4-way compare on the global row index.
//
// The per-row math is `gdn_dequant_int4_gemv_simd_body` (defined above in this
// module — see K24; the module must self-contain the body rather than borrow
// dequant_int4.metal's static inline), called with the sub-matrix local row,
// so results are bit-identical to the four separate dispatches.
// That body reads packed weights via `ushort*`: sub-tensor offsets are only
// 2-byte aligned, so the ushort-pair path must stay.
//
// Eight rows per threadgroup, one SIMD per row (256 threads).
// ----------------------------------------------------------------------------
kernel void gdn_in_proj_gemv_simd(
    device const uint8_t* qkvW     [[buffer(0)]],
    device const bfloat*  qkvS     [[buffer(1)]],
    device const bfloat*  qkvB     [[buffer(2)]],
    device const uint8_t* zW       [[buffer(3)]],
    device const bfloat*  zS       [[buffer(4)]],
    device const bfloat*  zB       [[buffer(5)]],
    device const uint8_t* aW       [[buffer(6)]],
    device const bfloat*  aS       [[buffer(7)]],
    device const bfloat*  aB       [[buffer(8)]],
    device const uint8_t* bW       [[buffer(9)]],
    device const bfloat*  bS       [[buffer(10)]],
    device const bfloat*  bB       [[buffer(11)]],
    device const half*    x        [[buffer(12)]],
    device half*          qkvY     [[buffer(13)]],
    device half*          zY       [[buffer(14)]],
    device half*          aY       [[buffer(15)]],
    device half*          bY       [[buffer(16)]],
    constant uint&        qkvRows  [[buffer(17)]],
    constant uint&        zRows    [[buffer(18)]],
    constant uint&        abRows   [[buffer(19)]],
    constant uint&        N        [[buffer(20)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QKV = gdn_in_fc_qkv(qkvRows);
    const uint Z   = gdn_in_fc_z(zRows);
    const uint AB  = gdn_in_fc_ab(abRows);
    const uint NN  = gdn_in_fc_n(N);

    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    if (global_row >= QKV + Z + 2u * AB) return;

    device const uint8_t* W;
    device const bfloat*  scales;
    device const bfloat*  biases;
    device half*          y;
    uint local_row;
    uint M;
    if (global_row < QKV) {
        W = qkvW; scales = qkvS; biases = qkvB; y = qkvY;
        local_row = global_row;
        M = QKV;
    } else if (global_row < QKV + Z) {
        W = zW; scales = zS; biases = zB; y = zY;
        local_row = global_row - QKV;
        M = Z;
    } else if (global_row < QKV + Z + AB) {
        W = aW; scales = aS; biases = aB; y = aY;
        local_row = global_row - QKV - Z;
        M = AB;
    } else {
        W = bW; scales = bS; biases = bB; y = bY;
        local_row = global_row - QKV - Z - AB;
        M = AB;
    }
    if (global_row >= QKV + Z && gdn_ab_bf16()) {
        // a and b only; qkv and z keep whatever width the slot declares.
        gdn_bf16_gemv_body((device const bfloat*)W, x, y, NN, local_row, lane);
        return;
    }
    gdn_dequant_int4_gemv_simd_body(W, scales, biases, x, y, M, NN,
                                    1u, local_row, 0u, lane);
}

// ----------------------------------------------------------------------------
// Causal depthwise conv, decode. One thread per channel. The tail buffer holds
// the previous K-1 pre-activation rows and is shifted in place (each thread
// owns its channel column exclusively).
// ----------------------------------------------------------------------------
kernel void gdn_conv_mix_decode(
    device half*        tail     [[buffer(0)]],   // [K-1, C] shifted in place
    device const half*  qkv      [[buffer(1)]],   // [C] current raw row
    device const bfloat* conv_w  [[buffer(2)]],   // [C, K]
    device half*        out      [[buffer(3)]],   // [C] silu(conv)
    constant uint&      channels [[buffer(4)]],
    constant uint&      taps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint K = taps;
    if (tid >= C) return;

    const uint history = K - 1u;
    float acc = float(qkv[tid]) * float(conv_w[tid * K + (K - 1u)]);
    for (uint j = 0; j < history; ++j) {
        acc = fma(float(tail[j * C + tid]), float(conv_w[tid * K + j]), acc);
    }
    out[tid] = half(gdn_silu(acc));

    for (uint j = 0; j + 1u < history; ++j) {
        tail[j * C + tid] = tail[(j + 1u) * C + tid];
    }
    if (history > 0u) {
        tail[(history - 1u) * C + tid] = qkv[tid];
    }
}

// ----------------------------------------------------------------------------
// Causal depthwise conv, prefill. Threads (channel, row); the incoming tail is
// read-only here — gdn_conv_tail_update refreshes it after the chunk.
// ----------------------------------------------------------------------------
kernel void gdn_conv_mix_prefill(
    device const half*  tail     [[buffer(0)]],   // [K-1, C] state entering chunk
    device const half*  qkv      [[buffer(1)]],   // [T, C] raw rows
    device const bfloat* conv_w  [[buffer(2)]],   // [C, K]
    device half*        out      [[buffer(3)]],   // [T, C]
    constant uint&      channels [[buffer(4)]],
    constant uint&      taps     [[buffer(5)]],
    constant uint&      rows     [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint K = taps;
    const uint T = rows;
    const uint ch = gid.x;
    const uint t = gid.y;
    if (ch >= C || t >= T) return;

    const uint history = K - 1u;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        // Input row index in the virtual sequence [tail rows | chunk rows].
        const int src = int(t) + int(j) - int(history);
        float value;
        if (src >= 0) {
            value = float(qkv[uint(src) * C + ch]);
        } else {
            value = float(tail[uint(int(history) + src) * C + ch]);
        }
        acc = fma(value, float(conv_w[ch * K + j]), acc);
    }
    out[t * C + ch] = half(gdn_silu(acc));
}

// After a prefill chunk: tail := last K-1 raw rows of [old tail | chunk].
kernel void gdn_conv_tail_update(
    device half*        tail     [[buffer(0)]],   // [K-1, C]
    device const half*  qkv      [[buffer(1)]],   // [T, C] raw rows
    constant uint&      channels [[buffer(2)]],
    constant uint&      taps     [[buffer(3)]],
    constant uint&      rows     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint history = taps - 1u;
    const uint T = rows;
    const uint ch = gid.x;
    const uint j = gid.y;
    if (ch >= C || j >= history) return;

    // New tail row j corresponds to virtual row (T + j) - history.
    const int src = int(T) + int(j) - int(history);
    half value;
    if (src >= 0) {
        value = qkv[uint(src) * C + ch];
    } else {
        value = tail[uint(int(history) + src) * C + ch];
    }
    // Threads for row j read only tail rows < j's source (all sources are
    // < history), and every source row index is strictly less than every
    // destination row index only when T >= history. For T < history the
    // shift moves rows downward (src < j), so read-before-write hazards
    // would occur between threads in different threadgroups. Stage through
    // a register and rely on the separate raw-row case being the common
    // path; for the tiny T < history case the kernel is dispatched with a
    // single threadgroup per channel column (grid.y == history <= 8), and
    // the loop below performs the ordered shift instead.
    if (T >= history) {
        tail[j * C + ch] = value;
        return;
    }
    // T < history: only thread j == 0 performs the whole ordered shift for
    // this channel to avoid cross-thread hazards.
    if (j != 0u) return;
    for (uint row = 0; row < history; ++row) {
        const int rowSrc = int(T) + int(row) - int(history);
        half rowValue;
        if (rowSrc >= 0) {
            rowValue = qkv[uint(rowSrc) * C + ch];
        } else {
            rowValue = tail[uint(int(history) + rowSrc) * C + ch];
        }
        tail[row * C + ch] = rowValue;
    }
}

// Snapshot tail after the first row: [old tail | qkv[0]]'s last K-1 rows.
kernel void gdn_conv_tail_checkpoint(
    device const half* tail       [[buffer(0)]],
    device const half* qkv        [[buffer(1)]],
    device half*       checkpoint [[buffer(2)]],
    constant uint&     channels   [[buffer(3)]],
    constant uint&     taps       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint history = taps - 1u;
    const uint total = history * channels;
    if (tid >= total) return;
    const uint row = tid / channels;
    const uint ch = tid - row * channels;
    checkpoint[tid] = (row + 1u < history)
        ? tail[(row + 1u) * channels + ch]
        : qkv[ch];
}

// ----------------------------------------------------------------------------
// Per-head no-weight RMS norm over q and k slices of conv_out, with the
// delta-rule scales folded in: q *= 1/Dk (inv_scale^2), k *= 1/sqrt(Dk).
// One threadgroup per (head, row); head_dim <= 1024 threads.
// conv_out layout per row: [q: Hk*Dk][k: Hk*Dk][v: Hv*Dv].
// ----------------------------------------------------------------------------
kernel void gdn_qk_norm(
    device half*   conv_out  [[buffer(0)]],
    constant uint& kHeads    [[buffer(1)]],
    constant uint& keyDim    [[buffer(2)]],
    constant uint& rowStride [[buffer(3)]],   // C, elements per row
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_idx  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    const uint tid = tpos.x;
    const uint Hk = kHeads;
    const uint Dk = keyDim;
    const uint headIndex = tg.x;          // 0..<2*Hk: q heads then k heads
    const uint row = tg.y;
    if (headIndex >= 2u * Hk) return;

    // Threadgroup size and SIMD-partial count are a function-constant pair
    // (FC_GDN_TG_THREADS); the wrapper dispatches exactly tgThreads threads.
    const uint tgThreads = gdn_tg_threads();
    const uint partialCount = tgThreads / 32u;

    const bool isQ = headIndex < Hk;
    const uint head = isQ ? headIndex : headIndex - Hk;
    const uint base = row * rowStride + (isQ ? 0u : Hk * Dk) + head * Dk;

    float sumsq = 0.0f;
    for (uint i = tid; i < Dk; i += tgThreads) {
        const float x = float(conv_out[base + i]);
        sumsq = fma(x, x, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (simd_lane == 0) partial[simd_idx] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < partialCount; ++i) total += partial[i];
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0] / float(Dk);
    const float invRms = rsqrt(mean + kGdnRmsEps);
    const float scale = isQ ? (1.0f / float(Dk)) : rsqrt(float(Dk));

    for (uint i = tid; i < Dk; i += tgThreads) {
        const float x = float(conv_out[base + i]);
        conv_out[base + i] = half(x * invRms * scale);
    }
}

// ----------------------------------------------------------------------------
// Gated delta rule. Decode: one token. Threadgroups (Hv, Dv/4); threads
// (32, 4): thread (lane, dvSub) owns state[dv][lane*4 .. lane*4+3].
// State is FP32 [Hv, Dv, Dk], persistent. a/b are the projection outputs
// [Hv] half; A_log/dt_bias are BF16 [Hv].
// ----------------------------------------------------------------------------
kernel void gdn_delta_step_decode(
    device const half*   conv_out [[buffer(0)]],   // [C] normed q,k + raw v
    device const half*   a_proj   [[buffer(1)]],   // [Hv]
    device const half*   b_proj   [[buffer(2)]],   // [Hv]
    device const bfloat* A_log    [[buffer(3)]],   // [Hv]
    device const bfloat* dt_bias  [[buffer(4)]],   // [Hv]
    device float*        state    [[buffer(5)]],   // [Hv, Dv, Dk]
    device half*         y        [[buffer(6)]],   // [Hv * Dv]
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv) return;

    const uint hk = gdn_key_head(h, Hv, Hk);
    device const half* q = conv_out + hk * Dk;
    device const half* k = conv_out + Hk * Dk + hk * Dk;
    device const half* v = conv_out + 2u * Hk * Dk + h * Dv;

    const float g = gdn_decay(
        float(a_proj[h]), float(dt_bias[h]), float(A_log[h]));
    const float beta = 1.0f / (1.0f + exp(-float(b_proj[h])));

    const uint nPerLane = Dk / 32u;
    device float* srow = state + (uint(h) * Dv + dv) * Dk;

    float s[8];
    float kv = 0.0f;
    for (uint i = 0; i < nPerLane; ++i) {
        const uint idx = lane * nPerLane + i;
        s[i] = srow[idx] * g;
        kv = fma(s[i], float(k[idx]), kv);
    }
    kv = simd_sum(kv);

    const float delta = (float(v[dv]) - kv) * beta;

    float out = 0.0f;
    for (uint i = 0; i < nPerLane; ++i) {
        const uint idx = lane * nPerLane + i;
        s[i] = fma(float(k[idx]), delta, s[i]);
        out = fma(s[i], float(q[idx]), out);
        srow[idx] = s[i];
    }
    out = simd_sum(out);
    if (lane == 0) y[h * Dv + dv] = half(out);
}

// Prefill: identical math with the token loop inside the kernel. q/k/v/a/b
// advance by their row strides each step; state persists in registers across
// the whole chunk and is written back once.
kernel void gdn_delta_step_prefill(
    device const half*   conv_out [[buffer(0)]],   // [T, C]
    device const half*   a_proj   [[buffer(1)]],   // [T, Hv]
    device const half*   b_proj   [[buffer(2)]],   // [T, Hv]
    device const bfloat* A_log    [[buffer(3)]],   // [Hv]
    device const bfloat* dt_bias  [[buffer(4)]],   // [Hv]
    device float*        state    [[buffer(5)]],   // [Hv, Dv, Dk]
    device half*         y        [[buffer(6)]],   // [T, Hv * Dv]
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    constant uint&       rows     [[buffer(11)]],
    constant uint&       rowStride [[buffer(12)]], // C, conv_out elements/row
    device float*        checkpointState [[buffer(13)]],
    constant bool&       checkpointEnabled [[buffer(14)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint T = rows;
    const uint C = rowStride;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv) return;

    const uint hk = gdn_key_head(h, Hv, Hk);
    const float dtb = float(dt_bias[h]);

    const uint nPerLane = Dk / 32u;
    device float* srow = state + (uint(h) * Dv + dv) * Dk;

    float s[8];
    for (uint i = 0; i < nPerLane; ++i) {
        s[i] = srow[lane * nPerLane + i];
    }

    for (uint t = 0; t < T; ++t) {
        device const half* q = conv_out + t * C + hk * Dk;
        device const half* k = conv_out + t * C + Hk * Dk + hk * Dk;
        device const half* v = conv_out + t * C + 2u * Hk * Dk + h * Dv;

        const float g = gdn_decay(
            float(a_proj[t * Hv + h]), dtb, float(A_log[h]));
        const float beta = 1.0f / (1.0f + exp(-float(b_proj[t * Hv + h])));

        float kv = 0.0f;
        for (uint i = 0; i < nPerLane; ++i) {
            const uint idx = lane * nPerLane + i;
            s[i] *= g;
            kv = fma(s[i], float(k[idx]), kv);
        }
        kv = simd_sum(kv);

        const float delta = (float(v[dv]) - kv) * beta;

        float out = 0.0f;
        for (uint i = 0; i < nPerLane; ++i) {
            const uint idx = lane * nPerLane + i;
            s[i] = fma(float(k[idx]), delta, s[i]);
            out = fma(s[i], float(q[idx]), out);
        }
        out = simd_sum(out);
        if (lane == 0) y[t * Hv * Dv + h * Dv + dv] = half(out);
        if (checkpointEnabled && t == 0u) {
            for (uint i = 0; i < nPerLane; ++i) {
                checkpointState[(uint(h) * Dv + dv) * Dk
                    + lane * nPerLane + i] = s[i];
            }
        }
    }

    for (uint i = 0; i < nPerLane; ++i) {
        srow[lane * nPerLane + i] = s[i];
    }
}

// ----------------------------------------------------------------------------
// Gated output norm: out = rmsnorm(y; weight, eps) * gate(z), per value head,
// where gate is silu or sigmoid (FC_GDN_SIGMOID_GATE).
// One threadgroup per (head, row), 128 threads. Norm statistics span one
// head's Dv elements; silu/product in FP32 (matches the reference's
// _precise_swiglu).
// ----------------------------------------------------------------------------
kernel void gdn_gated_norm(
    device const half*   y        [[buffer(0)]],   // [T, Hv * Dv]
    device const half*   z        [[buffer(1)]],   // [T, Hv * Dv]
    device const bfloat* weight   [[buffer(2)]],   // [Dv]
    device half*         out      [[buffer(3)]],   // [T, Hv * Dv]
    constant uint&       vHeads   [[buffer(4)]],
    constant uint&       valueDim [[buffer(5)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_idx  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    const uint tid = tpos.x;
    const uint Hv = vHeads;
    const uint Dv = valueDim;
    const uint head = tg.x;
    const uint row = tg.y;
    if (head >= Hv) return;

    // Threadgroup size and SIMD-partial count are a function-constant pair
    // (FC_GDN_TG_THREADS); the wrapper dispatches exactly tgThreads threads.
    const uint tgThreads = gdn_tg_threads();
    const uint partialCount = tgThreads / 32u;

    const uint base = row * Hv * Dv + head * Dv;

    float sumsq = 0.0f;
    for (uint i = tid; i < Dv; i += tgThreads) {
        const float x = float(y[base + i]);
        sumsq = fma(x, x, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (simd_lane == 0) partial[simd_idx] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < partialCount; ++i) total += partial[i];
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0] / float(Dv);
    const float invRms = rsqrt(mean + kGdnRmsEps);

    for (uint i = tid; i < Dv; i += tgThreads) {
        const float normed = float(y[base + i]) * invRms * float(weight[i]);
        const float gate = gdn_out_gate(float(z[base + i]));
        out[base + i] = half(normed * gate);
    }
}

// ---- bench variants (TinyTitanBench gdn_inproj_*) ----
// Access-pattern sweep for the fused in-proj GEMV: xsh8 (8 rows/TG +
// threadgroup-staged x), r16 (16 rows/TG, device x), xsh16 (16 rows + tgmem).

static inline float gdn_dequant_int4_gemv_simd_body_tgmem(
    threadgroup const half* x,
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return 0.0f;
    const uint n_groups  = N / kGdnGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const float e0 = float(x[elem]), e1 = float(x[elem + 1u]);
        const float e2 = float(x[elem + 2u]), e3 = float(x[elem + 3u]);
        const float e4 = float(x[elem + 4u]), e5 = float(x[elem + 5u]);
        const float e6 = float(x[elem + 6u]), e7 = float(x[elem + 7u]);
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
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
        const uint8_t byte = W_row[g * (kGdnGroupSize / 2) + lane];
        const float x0 = float(x[g * kGdnGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGdnGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
    return acc;
}

static inline void gdn_in_proj_rows(
    device const half* xd, threadgroup const half* xt, bool tgmem,
    device const uint8_t* qkvW, device const bfloat* qkvS, device const bfloat* qkvB,
    device const uint8_t* zW,   device const bfloat* zS,   device const bfloat* zB,
    device const uint8_t* aW,   device const bfloat* aS,   device const bfloat* aB,
    device const uint8_t* bW,   device const bfloat* bS,   device const bfloat* bB,
    device half* qkvY, device half* zY, device half* aY, device half* bY,
    uint QKV, uint Z, uint AB, uint NN, uint global_row, uint lane
) {
    if (global_row >= QKV + Z + 2u * AB) return;
    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QKV) {
        W = qkvW; scales = qkvS; biases = qkvB; y = qkvY;
        local_row = global_row;
        M = QKV;
    } else if (global_row < QKV + Z) {
        W = zW; scales = zS; biases = zB; y = zY;
        local_row = global_row - QKV;
        M = Z;
    } else if (global_row < QKV + Z + AB) {
        W = aW; scales = aS; biases = aB; y = aY;
        local_row = global_row - QKV - Z;
        M = AB;
    } else {
        W = bW; scales = bS; biases = bB; y = bY;
        local_row = global_row - QKV - Z - AB;
        M = AB;
    }
    // Same promotion branch as gdn_in_proj_gemv_simd. This is a second
    // implementation of the identical slot selection -- one for the plain
    // dispatch, one for the threadgroup-memory variant -- and patching only
    // the first left this one reading bf16 weights through the INT4 body,
    // whose scales pointer is then the file header read as bfloat: NaN.
    if (global_row >= QKV + Z && gdn_ab_bf16()) {
        device const bfloat* row = (device const bfloat*)W + local_row * NN;
        float acc = 0.0f;
        for (uint base = 0; base < NN; base += 64u) {
            const uint idx = base + lane * 2u;
            const float x0 = tgmem ? float(xt[idx]) : float(xd[idx]);
            const float x1 = tgmem ? float(xt[idx + 1u]) : float(xd[idx + 1u]);
            acc = fma(float(row[idx]), x0, acc);
            acc = fma(float(row[idx + 1u]), x1, acc);
        }
        acc = simd_sum(acc);
        if (lane == 0u) y[local_row] = half(acc);
        return;
    }
    if (tgmem) {
        gdn_dequant_int4_gemv_simd_body_tgmem(xt, W, scales, biases, y, M, NN,
                                              1u, local_row, 0u, lane);
    } else {
        gdn_dequant_int4_gemv_simd_body(W, scales, biases, xd, y, M, NN,
                                        1u, local_row, 0u, lane);
    }
}

#define GDN_IN_PROJ_VARIANT(NAME, RPT, XSH)                                \
kernel void NAME(                                                          \
    device const uint8_t* qkvW [[buffer(0)]],                              \
    device const bfloat*  qkvS [[buffer(1)]],                              \
    device const bfloat*  qkvB [[buffer(2)]],                              \
    device const uint8_t* zW   [[buffer(3)]],                              \
    device const bfloat*  zS   [[buffer(4)]],                              \
    device const bfloat*  zB   [[buffer(5)]],                              \
    device const uint8_t* aW   [[buffer(6)]],                              \
    device const bfloat*  aS   [[buffer(7)]],                              \
    device const bfloat*  aB   [[buffer(8)]],                              \
    device const uint8_t* bW   [[buffer(9)]],                              \
    device const bfloat*  bS   [[buffer(10)]],                             \
    device const bfloat*  bB   [[buffer(11)]],                             \
    device const half*    x    [[buffer(12)]],                             \
    device half* qkvY [[buffer(13)]], device half* zY [[buffer(14)]],      \
    device half* aY   [[buffer(15)]], device half* bY [[buffer(16)]],      \
    constant uint& qkvRows [[buffer(17)]], constant uint& zRows [[buffer(18)]], \
    constant uint& abRows  [[buffer(19)]], constant uint& N      [[buffer(20)]], \
    uint tg_idx [[threadgroup_position_in_grid]],                          \
    uint sg_idx [[simdgroup_index_in_threadgroup]],                        \
    uint lane   [[thread_index_in_simdgroup]]) {                           \
    constexpr uint rows_per_tg = RPT;                                      \
    const uint QKV = gdn_in_fc_qkv(qkvRows);                               \
    const uint Z   = gdn_in_fc_z(zRows);                                   \
    const uint AB  = gdn_in_fc_ab(abRows);                                 \
    const uint NN  = gdn_in_fc_n(N);                                       \
    threadgroup half xt[kGDNActivationMaxD];                               \
    if (XSH) {                                                             \
        for (uint i = lane; i < NN; i += 32u) { xt[i] = x[i]; }            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        gdn_in_proj_rows(x, xt, true, qkvW, qkvS, qkvB, zW, zS, zB, aW, aS, aB, \
                         bW, bS, bB, qkvY, zY, aY, bY, QKV, Z, AB, NN,     \
                         tg_idx * rows_per_tg + sg_idx, lane);             \
    } else {                                                               \
        gdn_in_proj_rows(x, nullptr, false, qkvW, qkvS, qkvB, zW, zS, zB, aW, aS, aB, \
                         bW, bS, bB, qkvY, zY, aY, bY, QKV, Z, AB, NN,     \
                         tg_idx * rows_per_tg + sg_idx, lane);             \
    }                                                                      \
}

GDN_IN_PROJ_VARIANT(gdn_in_proj_gemv_simd_xsh8, 8,  1)
GDN_IN_PROJ_VARIANT(gdn_in_proj_gemv_simd_r16, 16, 0)
GDN_IN_PROJ_VARIANT(gdn_in_proj_gemv_simd_xsh16, 16, 1)

// ----------------------------------------------------------------------------
// Micro-benchmark candidates for the in_proj GEMV (TinyTitanBench gdn_inproj_*).
// The production kernel runs one simdgroup per row with one 4-byte weight
// load per lane per iteration -- about 40 KB in flight across the whole
// dispatch, which is what a latency-bound 50 GB/s looks like on a 100 GB/s
// part. These test the two ways out and the ceiling:
//   u4  -- same arithmetic, block loop unrolled 4x so four loads are in
//          flight per lane (bit-exact with the production kernel);
//   sk4 -- four simdgroups per row, each every 4th block, partials summed in
//          simdgroup order (changes float association: a candidate, not a
//          drop-in);
//   bw  -- reads the rows with 16-byte loads and no dequant, the ceiling for
//          this access pattern.
// ----------------------------------------------------------------------------
static inline void gdn_dequant_int4_gemv_simd_body_u4(
    device const uint8_t* W, device const bfloat* scales, device const bfloat* biases,
    device const half* x, device half* y, uint M, uint N,
    uint rows_per_tg, uint tg_idx, uint sg_idx, uint lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGdnGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;
    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    #pragma unroll 4
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
        const uint8_t byte = W_row[g * (kGdnGroupSize / 2) + lane];
        const float x0 = float(x[g * kGdnGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGdnGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = half(acc);
}

#define GDN_IN_PROJ_BUFFERS                                                  \
    device const uint8_t* qkvW     [[buffer(0)]],                            \
    device const bfloat*  qkvS     [[buffer(1)]],                            \
    device const bfloat*  qkvB     [[buffer(2)]],                            \
    device const uint8_t* zW       [[buffer(3)]],                            \
    device const bfloat*  zS       [[buffer(4)]],                            \
    device const bfloat*  zB       [[buffer(5)]],                            \
    device const uint8_t* aW       [[buffer(6)]],                            \
    device const bfloat*  aS       [[buffer(7)]],                            \
    device const bfloat*  aB       [[buffer(8)]],                            \
    device const uint8_t* bW       [[buffer(9)]],                            \
    device const bfloat*  bS       [[buffer(10)]],                           \
    device const bfloat*  bB       [[buffer(11)]],                           \
    device const half*    x        [[buffer(12)]],                           \
    device half*          qkvY     [[buffer(13)]],                           \
    device half*          zY       [[buffer(14)]],                           \
    device half*          aY       [[buffer(15)]],                           \
    device half*          bY       [[buffer(16)]],                           \
    constant uint&        qkvRows  [[buffer(17)]],                           \
    constant uint&        zRows    [[buffer(18)]],                           \
    constant uint&        abRows   [[buffer(19)]],                           \
    constant uint&        N        [[buffer(20)]]

// Slot selection shared by the candidates; mirrors gdn_in_proj_gemv_simd.
#define GDN_IN_PROJ_SELECT(global_row)                                       \
    device const uint8_t* W; device const bfloat* scales;                    \
    device const bfloat* biases; device half* y; uint local_row; uint M;     \
    if (global_row < QKV) {                                                  \
        W = qkvW; scales = qkvS; biases = qkvB; y = qkvY;                    \
        local_row = global_row; M = QKV;                                     \
    } else if (global_row < QKV + Z) {                                       \
        W = zW; scales = zS; biases = zB; y = zY;                            \
        local_row = global_row - QKV; M = Z;                                 \
    } else if (global_row < QKV + Z + AB) {                                  \
        W = aW; scales = aS; biases = aB; y = aY;                            \
        local_row = global_row - QKV - Z; M = AB;                            \
    } else {                                                                 \
        W = bW; scales = bS; biases = bB; y = bY;                            \
        local_row = global_row - QKV - Z - AB; M = AB;                       \
    }

kernel void gdn_in_proj_gemv_simd_u4(
    GDN_IN_PROJ_BUFFERS,
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QKV = gdn_in_fc_qkv(qkvRows);
    const uint Z   = gdn_in_fc_z(zRows);
    const uint AB  = gdn_in_fc_ab(abRows);
    const uint NN  = gdn_in_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    if (global_row >= QKV + Z + 2u * AB) return;
    GDN_IN_PROJ_SELECT(global_row)
    if (global_row >= QKV + Z && gdn_ab_bf16()) {
        gdn_bf16_gemv_body((device const bfloat*)W, x, y, NN, local_row, lane);
        return;
    }
    gdn_dequant_int4_gemv_simd_body_u4(W, scales, biases, x, y, M, NN,
                                       1u, local_row, 0u, lane);
}

kernel void gdn_in_proj_gemv_simd_sk4(
    GDN_IN_PROJ_BUFFERS,
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    constexpr uint rows_per_tg = 2, sg_per_row = 4;
    const uint QKV = gdn_in_fc_qkv(qkvRows);
    const uint Z   = gdn_in_fc_z(zRows);
    const uint AB  = gdn_in_fc_ab(abRows);
    const uint NN  = gdn_in_fc_n(N);
    const uint total = QKV + Z + 2u * AB;
    const uint row_in_tg = sg_idx / sg_per_row;
    const uint part = sg_idx - row_in_tg * sg_per_row;
    const uint global_row = min(tg_idx * rows_per_tg + row_in_tg, total - 1u);
    const bool valid = tg_idx * rows_per_tg + row_in_tg < total;
    GDN_IN_PROJ_SELECT(global_row)
    const bool bf16_row = global_row >= QKV + Z && gdn_ab_bf16();
    float acc = 0.0f;
    if (valid && bf16_row) {
        if (part == 0) gdn_bf16_gemv_body((device const bfloat*)W, x, y, NN, local_row, lane);
    } else if (valid) {
        const uint n_groups  = NN / kGdnGroupSize;
        const uint row_bytes = NN / 2;
        device const uint8_t* W_row = W      + uint(local_row) * row_bytes;
        device const bfloat*  s_row = scales + uint(local_row) * n_groups;
        device const bfloat*  b_row = biases + uint(local_row) * n_groups;
        const uint full_blocks = n_groups / 4;
        for (uint blk = part; blk < full_blocks; blk += sg_per_row) {
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
        if (part == 0) {
            for (uint g = full_blocks * 4u; g < n_groups; ++g) {
                const float s = float(s_row[g]);
                const float b = float(b_row[g]);
                const uint8_t byte = W_row[g * (kGdnGroupSize / 2) + lane];
                const float x0 = float(x[g * kGdnGroupSize + lane * 2u]);
                const float x1 = float(x[g * kGdnGroupSize + lane * 2u + 1u]);
                float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
                dot = fma(float(uint(byte >> 4)), x1, dot);
                acc = fma(s, dot, acc);
                acc = fma(b, x0 + x1, acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) partial[sg_idx] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid && !bf16_row && part == 0 && lane == 0) {
        const uint base = row_in_tg * sg_per_row;
        const float t = ((partial[base] + partial[base + 1u]) + partial[base + 2u]) + partial[base + 3u];
        y[local_row] = half(t);
    }
}

kernel void gdn_in_proj_gemv_simd_bw(
    GDN_IN_PROJ_BUFFERS,
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QKV = gdn_in_fc_qkv(qkvRows);
    const uint Z   = gdn_in_fc_z(zRows);
    const uint AB  = gdn_in_fc_ab(abRows);
    const uint NN  = gdn_in_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    if (global_row >= QKV + Z + 2u * AB) return;
    GDN_IN_PROJ_SELECT(global_row)
    const uint row_bytes = NN / 2;
    device const uint4* W_row = (device const uint4*)(W + uint(local_row) * row_bytes);
    const uint n16 = row_bytes / 16u;
    uint4 acc = uint4(0u);
    for (uint i = lane; i < n16; i += 32u) acc ^= W_row[i];
    const uint v = simd_sum(acc.x ^ acc.y ^ acc.z ^ acc.w);
    if (lane == 0) y[local_row] = half(float(v & 0xFFu) * float(scales[0]) + float(biases[0]) * float(M));
}
