#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int4 — MLX `affine` 4-bit dequant.
//
// Layout (per row of length N):
//   W       : N/2 bytes. Low nibble of byte k = component 2k (unsigned 0..15),
//             high nibble = component 2k+1.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[i] = float(nibble[i]) * scale[i/64] + bias[i/64].
//
// Affine factoring for GEMV (sum over a group of 64):
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias each cost one mul + one FMA per group instead of per
// element; the per-element inner loop keeps the scalar path's FMA count.
// ============================================================================

constant constexpr uint kGroupSize = 64;
constant uint FC_INT4_M [[function_constant(20)]];
constant uint FC_INT4_N [[function_constant(21)]];
constant bool FC_INT4_USE_FC [[function_constant(22)]];
constant uint FC_INT4_QKV_MQ [[function_constant(23)]];
constant uint FC_INT4_QKV_MKV [[function_constant(24)]];
constant uint FC_INT4_QKV_N [[function_constant(25)]];
constant bool FC_INT4_QKV_USE_FC [[function_constant(26)]];

static inline uint int4_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_M)) ? FC_INT4_M : M;
}

static inline uint int4_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_N)) ? FC_INT4_N : N;
}

static inline uint int4_qkv_fc_mq(constant uint& Mq) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MQ)) ? FC_INT4_QKV_MQ : Mq;
}

static inline uint int4_qkv_fc_mkv(constant uint& Mkv) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MKV)) ? FC_INT4_QKV_MKV : Mkv;
}

static inline uint int4_qkv_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_N)) ? FC_INT4_QKV_N : N;
}

inline uint nib_lo(uint8_t b) { return uint(b & 0x0F); }
inline uint nib_hi(uint8_t b) { return uint(b >> 4); }


kernel void embed_lookup_int4(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/2] nibbles
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/64] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/64] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    constant uint&        vocab     [[buffer(7)]],   // [V] row count
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    // OOB token guard: the CPU clamps routing token ids, but a clamped id can
    // still land at the table edge; never index past the vocab rows. Zero
    // embeddings keep the rest of the pipeline well-defined for OOB tokens.
    if (token_id >= vocab) {
        out[gid] = half(0.0f);
        return;
    }
    const uint groups_per_row = D / kGroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * (D / 2u);
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;
    uint8_t byte = row_q[gid >> 1];
    uint    q    = (gid & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    float   s    = float(row_s[gid / kGroupSize]);
    float   b    = float(row_b[gid / kGroupSize]);
    out[gid] = half((float(q) * s + b) * out_scale);
}

// y[m] = sum_{n} W[m, n] * x[n]. One-SIMD-per-row variant: 32 threads
// cooperate on a single output row, each handling 2 elements per group of 64
// (one byte → two nibbles). simd_sum reduces across the group; lane 0 writes.
//
// Requires N % 64 == 0 (per group of 64). Validated at the wrapper.
// Each threadgroup handles eight consecutive rows, one SIMD per row. The
// larger work unit gives the scheduler enough independent rows while sharing
// the L1-cached input-vector reads.
static inline void dequant_int4_gemv_simd_body(
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
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    // The vectorized row path reads
    // weights a uint (4 bytes = 8 nibbles) and x as half4 in 4-group (128-byte)
    // blocks, with a scalar byte-per-lane remainder. Within a block the 32
    // lanes split 8-per-group, each handling 8 contiguous elements of one
    // 64-element group, so the affine factoring s·Σqx + b·Σx is preserved
    // (simd_sum aggregates; s/b are constant within a group). Aligned: row
    // stride N/2 and weightsOffset are multiples of 4; x is
    // half4-aligned (lane*8 elements). N=2816/4096/8192 → 44/64/128 groups, all
    // exact 4-blocks; the remainder covers any non-multiple-of-4 group count.
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Read the 4-byte weight chunk as two ushorts. The resident weight
        // tensors are 2-byte aligned but NOT 4-byte aligned (BF16 scale/bias
        // regions leave a 2-aligned weightsOffset), so a `uint*` load would be
        // misaligned (undefined → garbage); a `ushort*` load is safe (row stride
        // N/2, weightsOffset, and byte_base are all even) and halves the loads
        // vs byte-by-byte.
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
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
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

kernel void dequant_int4_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, MM, NN,
                                rows_per_tg, tg_idx, sg_idx, lane);
}

// Two-row variant for native-MTP verification. A SIMD dequantizes each output
// weight row once and accumulates both activation rows before writing [2, M].
kernel void dequant_int4_gemv2_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= MM) return;
    const uint n_groups = NN / kGroupSize;
    const uint row_bytes = NN / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = scales + row * n_groups;
    device const bfloat* b_row = biases + row * n_groups;
    device const half* x1 = x + NN;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint full_blocks = n_groups / 4u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 a0 = *((device const half4*)(x + elem));
        const half4 b0 = *((device const half4*)(x + elem + 4u));
        const half4 a1 = *((device const half4*)(x1 + elem));
        const half4 b1 = *((device const half4*)(x1 + elem + 4u));
        const uint q0 = w4 & 0xFFu, q1 = (w4 >> 8) & 0xFFu;
        const uint q2 = (w4 >> 16) & 0xFFu, q3 = (w4 >> 24) & 0xFFu;
        const float v0[8] = {float(a0.x), float(a0.y), float(a0.z), float(a0.w),
                             float(b0.x), float(b0.y), float(b0.z), float(b0.w)};
        const float v1[8] = {float(a1.x), float(a1.y), float(a1.z), float(a1.w),
                             float(b1.x), float(b1.y), float(b1.z), float(b1.w)};
        const uint qs[8] = {q0 & 0xFu, q0 >> 4, q1 & 0xFu, q1 >> 4,
                            q2 & 0xFu, q2 >> 4, q3 & 0xFu, q3 >> 4};
        float dot0 = 0.0f, dot1 = 0.0f, sum0 = 0.0f, sum1 = 0.0f;
        for (uint i = 0; i < 8u; ++i) {
            dot0 = fma(float(qs[i]), v0[i], dot0);
            dot1 = fma(float(qs[i]), v1[i], dot1);
            sum0 += v0[i];
            sum1 += v1[i];
        }
        acc0 = fma(s, dot0, acc0);
        acc0 = fma(b, sum0, acc0);
        acc1 = fma(s, dot1, acc1);
        acc1 = fma(b, sum1, acc1);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2u) + lane];
        const uint elem = g * kGroupSize + lane * 2u;
        const float x00 = float(x[elem]), x01 = float(x[elem + 1u]);
        const float x10 = float(x1[elem]), x11 = float(x1[elem + 1u]);
        const float lo = float(uint(byte & 0xFu)), hi = float(uint(byte >> 4));
        acc0 = fma(s, fma(lo, x00, hi * x01), acc0);
        acc0 = fma(b, x00 + x01, acc0);
        acc1 = fma(s, fma(lo, x10, hi * x11), acc1);
        acc1 = fma(b, x10 + x11, acc1);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0u) {
        y[row] = half(acc0);
        y[MM + row] = half(acc1);
    }
}


kernel void dequant_int4_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, M, NN,
                                1u, local_row, 0u, lane);
}

// ---------------------------------------------------------------------------
// Micro-benchmark variants (TinyTitanBench): same buffer contract and dispatch as
// dequant_int4_qkv_gemv_simd, so the achieved bandwidth can be compared
// directly. `bandwidth` preserves the loads but drops the dequant ALU (tests
// whether the dequant math is the limiter); `unroll2` issues two independent
// weight loads per iteration (tests memory-level parallelism / latency
// hiding).

kernel void dequant_int4_qkv_gemv_simd_bandwidth(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; y = qY; local_row = global_row; M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; y = kY; local_row = global_row - QQ; M = KK;
    } else {
        W = vW; y = vY; local_row = global_row - QQ - KK; M = KK;
    }
    if (local_row >= M) { return; }
    const uint row_bytes = NN / 2;
    device const uint8_t* W_row = W + uint(local_row) * row_bytes;

    float acc = 0.0f;
    const uint n_groups = NN / kGroupSize;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Preserve the loads; skip the dequant math (raw-sum accumulator).
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        acc += float(w4 & 0xFFu) + float(xa.x) + float(xa.y)
            + float(xa.z) + float(xa.w) + float(xb.x) + float(xb.y)
            + float(xb.z) + float(xb.w);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[local_row] = half(acc);
    }
}

kernel void dequant_int4_qkv_gemv_simd_unroll2(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row; M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ; M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK; M = KK;
    }
    if (local_row >= M) { return; }
    const uint n_groups = NN / kGroupSize;
    const uint row_bytes = NN / 2;
    device const uint8_t* W_row = W + uint(local_row) * row_bytes;
    device const bfloat* s_row = scales + uint(local_row) * n_groups;
    device const bfloat* b_row = biases + uint(local_row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    const uint paired = full_blocks - (full_blocks % 2u);
    for (uint blk = 0; blk < paired; blk += 2u) {
        const uint base_a = blk * 128u + lane * 4u;
        const uint base_b = base_a + 128u;
        // Two independent weight loads before the ALU chains (latency hiding).
        const uint w4a = *((device const uint*)(W_row + base_a));
        const uint w4b = *((device const uint*)(W_row + base_b));
        const uint elem_a = base_a * 2u;
        const uint elem_b = base_b * 2u;
        const half4 xa = *((device const half4*)(x + elem_a));
        const half4 xb = *((device const half4*)(x + elem_a + 4u));
        const half4 xc = *((device const half4*)(x + elem_b));
        const half4 xd = *((device const half4*)(x + elem_b + 4u));
        const uint ga = blk * 4u + (lane >> 3);
        const float sa = float(s_row[ga]);
        const float ba = float(b_row[ga]);
        const uint gb = ga + 4u;
        const float sb = float(s_row[gb]);
        const float bb = float(b_row[gb]);
        const uint b0 = w4a & 0xFFu, b1 = (w4a >> 8) & 0xFFu;
        const uint b2 = (w4a >> 16) & 0xFFu, b3 = (w4a >> 24) & 0xFFu;
        const uint c0 = w4b & 0xFFu, c1 = (w4b >> 8) & 0xFFu;
        const uint c2 = (w4b >> 16) & 0xFFu, c3 = (w4b >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        const float f0 = float(xc.x), f1 = float(xc.y), f2 = float(xc.z), f3 = float(xc.w);
        const float f4 = float(xd.x), f5 = float(xd.y), f6 = float(xd.z), f7 = float(xd.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(sa, dot, acc);
        acc = fma(ba, sum, acc);
        dot = 0.0f;
        dot = fma(float(c0 & 0x0Fu), f0, dot); dot = fma(float(c0 >> 4), f1, dot);
        dot = fma(float(c1 & 0x0Fu), f2, dot); dot = fma(float(c1 >> 4), f3, dot);
        dot = fma(float(c2 & 0x0Fu), f4, dot); dot = fma(float(c2 >> 4), f5, dot);
        dot = fma(float(c3 & 0x0Fu), f6, dot); dot = fma(float(c3 >> 4), f7, dot);
        const float sum2 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
        acc = fma(sb, dot, acc);
        acc = fma(bb, sum2, acc);
    }
    for (uint blk = paired; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu, b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu, b3 = (w4 >> 24) & 0xFFu;
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
    acc = simd_sum(acc);
    if (lane == 0) {
        y[local_row] = half(acc);
    }
}

// unroll2 + half-precision block dots: the 8-element dot products accumulate
// in `half` (the GPU's fp16 ALU is wider and avoids the per-element fp32
// converts), the affine s/b and the cross-block accumulator stay in float.
kernel void dequant_int4_qkv_gemv_simd_unroll2_half(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row; M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ; M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK; M = KK;
    }
    if (local_row >= M) { return; }
    const uint n_groups = NN / kGroupSize;
    const uint row_bytes = NN / 2;
    device const uint8_t* W_row = W + uint(local_row) * row_bytes;
    device const bfloat* s_row = scales + uint(local_row) * n_groups;
    device const bfloat* b_row = biases + uint(local_row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    const uint paired = full_blocks - (full_blocks % 2u);
    for (uint blk = 0; blk < paired; blk += 2u) {
        const uint base_a = blk * 128u + lane * 4u;
        const uint base_b = base_a + 128u;
        const uint w4a = *((device const uint*)(W_row + base_a));
        const uint w4b = *((device const uint*)(W_row + base_b));
        const uint elem_a = base_a * 2u;
        const uint elem_b = base_b * 2u;
        const half4 xa = *((device const half4*)(x + elem_a));
        const half4 xb = *((device const half4*)(x + elem_a + 4u));
        const half4 xc = *((device const half4*)(x + elem_b));
        const half4 xd = *((device const half4*)(x + elem_b + 4u));
        const uint ga = blk * 4u + (lane >> 3);
        const float sa = float(s_row[ga]);
        const float ba = float(b_row[ga]);
        const uint gb = ga + 4u;
        const float sb = float(s_row[gb]);
        const float bb = float(b_row[gb]);
        const uchar4 na = as_type<uchar4>(w4a);
        const uchar4 nb = as_type<uchar4>(w4b);
        const half hdot_a = fma(half(na.x & 0x0Fu), xa.x,
                          fma(half(na.x >> 4), xa.y,
                          fma(half(na.y & 0x0Fu), xa.z,
                          fma(half(na.y >> 4), xa.w,
                          fma(half(na.z & 0x0Fu), xb.x,
                          fma(half(na.z >> 4), xb.y,
                          fma(half(na.w & 0x0Fu), xb.z,
                              half(na.w >> 4) * xb.w)))))));
        const half hdot_b = fma(half(nb.x & 0x0Fu), xc.x,
                          fma(half(nb.x >> 4), xc.y,
                          fma(half(nb.y & 0x0Fu), xc.z,
                          fma(half(nb.y >> 4), xc.w,
                          fma(half(nb.z & 0x0Fu), xd.x,
                          fma(half(nb.z >> 4), xd.y,
                          fma(half(nb.w & 0x0Fu), xd.z,
                              half(nb.w >> 4) * xd.w)))))));
        const half hsum_a = (xa.x + xa.y) + (xa.z + xa.w)
                          + (xb.x + xb.y) + (xb.z + xb.w);
        const half hsum_b = (xc.x + xc.y) + (xc.z + xc.w)
                          + (xd.x + xd.y) + (xd.z + xd.w);
        acc = fma(sa, float(hdot_a), acc);
        acc = fma(ba, float(hsum_a), acc);
        acc = fma(sb, float(hdot_b), acc);
        acc = fma(bb, float(hsum_b), acc);
    }
    for (uint blk = paired; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uchar4 na = as_type<uchar4>(w4);
        const half hdot = fma(half(na.x & 0x0Fu), xa.x,
                        fma(half(na.x >> 4), xa.y,
                        fma(half(na.y & 0x0Fu), xa.z,
                        fma(half(na.y >> 4), xa.w,
                        fma(half(na.z & 0x0Fu), xb.x,
                        fma(half(na.z >> 4), xb.y,
                        fma(half(na.w & 0x0Fu), xb.z,
                            half(na.w >> 4) * xb.w)))))));
        const half hsum = (xa.x + xa.y) + (xa.z + xa.w)
                        + (xb.x + xb.y) + (xb.z + xb.w);
        acc = fma(s, float(hdot), acc);
        acc = fma(b, float(hsum), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[local_row] = half(acc);
    }
}
