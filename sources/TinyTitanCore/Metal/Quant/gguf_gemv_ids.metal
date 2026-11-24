#include <metal_stdlib>
using namespace metal;

// Expert-indexed GEMVs over GGUF blocks read in place.
//
// One dispatch computes `slots` GEMVs: slot s multiplies the matrix of expert
// ids[s] (at `ids[s] * expertStride` bytes from `weights`) by the activation at
// `x + s * xSlotStride` (a stride of 0 shares one activation across slots) and
// writes `rows` floats at `y + s * rows`, times `scales[ids[s]]` when
// `useScale` is set. A dense matrix is the same call with one slot, ids = {0}.
//
// Geometry follows llama.cpp's ggml-metal mul_mv kernels for the block type:
//   NVFP4  two SIMD groups per threadgroup, two rows per SIMD group; lane pairs
//          walk every 16th 64-value block, each lane taking two 16-value
//          sub-blocks (kernel_mul_mv_nvfp4_f32), decoded arithmetically.
//   Q8_0   four SIMD groups share two rows, each lane eight quants of every
//          32nd block, reduced across the SIMD groups in threadgroup memory
//          (kernel_mul_mv_q8_0_f32).
// Rows past the end are computed on a clamped row and never stored.

struct GGUFGemvIdsArgs {
    uint rows;
    uint columns;
    ulong expertStride;
    uint xSlotStride;
    uint useScale;
};

constant uint GEMV_IDS_NVFP4_SIMDGROUPS = 2u;
constant uint GEMV_IDS_NVFP4_ROWS = 2u;
constant uint GEMV_IDS_Q8_0_SIMDGROUPS = 4u;
constant uint GEMV_IDS_Q8_0_ROWS = 2u;

// Unsigned E4M3 (bias 7): the seven low bits placed in an fp16 give the value
// times 2^-8, subnormals included; 0x7F decodes to 0 as ggml does.
static inline float gemv_ids_ue4m3(uchar code) {
    const float v = float(as_type<half>(ushort(ushort(code & 0x7Fu) << 7u))) * 256.0f;
    return code == 0x7Fu ? 0.0f : v;
}

// Four E2M1 codes (one per byte of `codes`, low nibble used) as floats times
// 2^-14: the three magnitude bits shifted into an fp16's exponent/mantissa
// give exactly value * 2^-14, the subnormal 0.5 included, and bit 3 is the
// sign. Callers fold 2^14 into the block scale.
static inline float4 gemv_ids_e2m1x4(uint codes) {
    const uint even = codes & 0x000F000Fu;          // bytes 0 and 2
    const uint odd = (codes >> 8u) & 0x000F000Fu;   // bytes 1 and 3
    const half2 a = as_type<half2>(((even & 0x00070007u) << 9u) | ((even & 0x00080008u) << 12u));
    const half2 b = as_type<half2>(((odd & 0x00070007u) << 9u) | ((odd & 0x00080008u) << 12u));
    return float4(float(a.x), float(b.x), float(a.y), float(b.y));
}

// NVFP4 GEMV ids: llama.cpp's geometry with word loads and arithmetic E2M1
// decode instead of its threadgroup lookup table (one routed decode layer,
// 8 gate_up + 8 down over 128 experts: 0.41 ms / 87 GB/s against the table's
// 0.51 ms / 70 GB/s, paired, M3 24 GB).
kernel void nvfp4_gemv_ids(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    constant GGUFGemvIdsArgs& args [[buffer(5)]],
    uint2 groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint slot = groupIndex.y;
    const int expert = ids[slot];
    const uint firstRow =
        (groupIndex.x * GEMV_IDS_NVFP4_SIMDGROUPS + sg) * GEMV_IDS_NVFP4_ROWS;
    if (firstRow >= args.rows) return;
    const uint blockCount = args.columns / 64u;
    const uint rowBytes = blockCount * 36u;
    device const uchar* matrix = weights + ulong(expert) * args.expertStride;
    device const float* xs = x + slot * args.xSlotStride;
    const uint pairIndex = lane / 2u;
    const uint subPair = lane % 2u;

    device const uchar* rowBase[GEMV_IDS_NVFP4_ROWS];
    for (uint r = 0u; r < GEMV_IDS_NVFP4_ROWS; ++r) {
        rowBase[r] = matrix + min(firstRow + r, args.rows - 1u) * rowBytes;
    }
    float4 yl[8];
    float sums[GEMV_IDS_NVFP4_ROWS] = {0.0f};
    for (uint blockIndex = pairIndex; blockIndex < blockCount; blockIndex += 16u) {
        device const float4* x4 =
            (device const float4*)(xs + blockIndex * 64u + subPair * 32u);
        for (uint i = 0u; i < 8u; ++i) yl[i] = x4[i];
        for (uint r = 0u; r < GEMV_IDS_NVFP4_ROWS; ++r) {
            device const uchar* b = rowBase[r] + blockIndex * 36u;
            // Blocks are 36 bytes, so every word below is 4-byte aligned.
            device const uint* q = (device const uint*)(b + 4u + subPair * 16u);
            const uint scalesWord = *(device const ushort*)(b + 2u * subPair);
            float sumq = 0.0f;
            for (uint s = 0u; s < 2u; ++s) {
                const uint w0 = q[2u * s];
                const uint w1 = q[2u * s + 1u];
                float4 acc = yl[4u * s + 0u] * gemv_ids_e2m1x4(w0)
                    + yl[4u * s + 1u] * gemv_ids_e2m1x4(w1)
                    + yl[4u * s + 2u] * gemv_ids_e2m1x4(w0 >> 4u)
                    + yl[4u * s + 3u] * gemv_ids_e2m1x4(w1 >> 4u);
                const uchar code = uchar((scalesWord >> (8u * s)) & 0xFFu);
                sumq += gemv_ids_ue4m3(code) * ((acc[0] + acc[1]) + (acc[2] + acc[3]));
            }
            sums[r] += sumq * 16384.0f;
        }
    }
    const float scale = args.useScale != 0u ? scales[expert] : 1.0f;
    device float* out = y + slot * args.rows;
    for (uint r = 0u; r < GEMV_IDS_NVFP4_ROWS; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0u && firstRow + r < args.rows) out[firstRow + r] = total * scale;
    }
}

kernel void q8_0_gemv_ids(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    constant GGUFGemvIdsArgs& args [[buffer(5)]],
    uint2 groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[GEMV_IDS_Q8_0_ROWS][GEMV_IDS_Q8_0_SIMDGROUPS];

    const uint slot = groupIndex.y;
    const int expert = ids[slot];
    const uint firstRow = groupIndex.x * GEMV_IDS_Q8_0_ROWS;
    const uint blockCount = args.columns / 32u;
    const uint rowBytes = blockCount * 34u;
    device const uchar* matrix = weights + ulong(expert) * args.expertStride;
    device const float* xs = x + slot * args.xSlotStride;

    const uint blockLane = lane / 4u;  // 0...7
    const uint quarter = lane % 4u;    // eight quants of 32
    const uint firstBlock = sg * 8u + blockLane;

    device const uchar* rowBase[GEMV_IDS_Q8_0_ROWS];
    for (uint r = 0u; r < GEMV_IDS_Q8_0_ROWS; ++r) {
        rowBase[r] = matrix + min(firstRow + r, args.rows - 1u) * rowBytes;
    }

    float sums[GEMV_IDS_Q8_0_ROWS] = {0.0f};
    float yl[8];
    for (uint blockIndex = firstBlock; blockIndex < blockCount;
        blockIndex += GEMV_IDS_Q8_0_SIMDGROUPS * 8u)
    {
        device const float* xb = xs + blockIndex * 32u + quarter * 8u;
        for (uint i = 0u; i < 8u; ++i) yl[i] = xb[i];
        for (uint r = 0u; r < GEMV_IDS_Q8_0_ROWS; ++r) {
            device const uchar* b = rowBase[r] + blockIndex * 34u;
            device const char* qs = (device const char*)(b + 2u) + quarter * 8u;
            float sumq = 0.0f;
            for (uint i = 0u; i < 8u; ++i) sumq += float(qs[i]) * yl[i];
            sums[r] += sumq * float(as_type<half>(*(device const ushort*)b));
        }
    }

    for (uint r = 0u; r < GEMV_IDS_Q8_0_ROWS; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0u) partial[r][sg] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < GEMV_IDS_Q8_0_ROWS && firstRow + lane < args.rows) {
        float total = 0.0f;
        for (uint i = 0u; i < GEMV_IDS_Q8_0_SIMDGROUPS; ++i) total += partial[lane][i];
        const float scale = args.useScale != 0u ? scales[expert] : 1.0f;
        y[slot * args.rows + firstRow + lane] = total * scale;
    }
}

// OCP E4M3 (bias 7, no infinities; 0x7F/0xFF are NaN, read as 0) by table:
// every code's value, generated from the format's definition (TT-135; the
// branchy exp2 decode cost an ALU-heavy loop on every MXFP8 weight).
constant float gemv_ids_e4m3_table[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, 0.0f,
};

static inline float gemv_ids_e4m3(uchar bits) { return gemv_ids_e4m3_table[bits]; }

// The Inkling 2 checkpoint's MXFP8 experts as the NVFP4 plugin repacks them
// (TT-125): q8_0_gemv_ids over 34-byte blocks whose first byte is an E8M0 scale
// and whose values are E4M3 instead of int8.
kernel void mxfp8_gemv_ids(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    constant GGUFGemvIdsArgs& args [[buffer(5)]],
    uint2 groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[GEMV_IDS_Q8_0_ROWS][GEMV_IDS_Q8_0_SIMDGROUPS];

    const uint slot = groupIndex.y;
    const int expert = ids[slot];
    const uint firstRow = groupIndex.x * GEMV_IDS_Q8_0_ROWS;
    const uint blockCount = args.columns / 32u;
    const uint rowBytes = blockCount * 34u;
    device const uchar* matrix = weights + ulong(expert) * args.expertStride;
    device const float* xs = x + slot * args.xSlotStride;

    const uint blockLane = lane / 4u;  // 0...7
    const uint quarter = lane % 4u;    // eight quants of 32
    const uint firstBlock = sg * 8u + blockLane;

    device const uchar* rowBase[GEMV_IDS_Q8_0_ROWS];
    for (uint r = 0u; r < GEMV_IDS_Q8_0_ROWS; ++r) {
        rowBase[r] = matrix + min(firstRow + r, args.rows - 1u) * rowBytes;
    }

    float sums[GEMV_IDS_Q8_0_ROWS] = {0.0f};
    float yl[8];
    for (uint blockIndex = firstBlock; blockIndex < blockCount;
        blockIndex += GEMV_IDS_Q8_0_SIMDGROUPS * 8u)
    {
        device const float* xb = xs + blockIndex * 32u + quarter * 8u;
        for (uint i = 0u; i < 8u; ++i) yl[i] = xb[i];
        for (uint r = 0u; r < GEMV_IDS_Q8_0_ROWS; ++r) {
            device const uchar* b = rowBase[r] + blockIndex * 34u;
            device const uchar* qs = b + 2u + quarter * 8u;
            float sumq = 0.0f;
            for (uint i = 0u; i < 8u; ++i) sumq += gemv_ids_e4m3(qs[i]) * yl[i];
            sums[r] += sumq * exp2(float(b[0]) - 127.0f);
        }
    }

    for (uint r = 0u; r < GEMV_IDS_Q8_0_ROWS; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0u) partial[r][sg] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < GEMV_IDS_Q8_0_ROWS && firstRow + lane < args.rows) {
        float total = 0.0f;
        for (uint i = 0u; i < GEMV_IDS_Q8_0_SIMDGROUPS; ++i) total += partial[lane][i];
        const float scale = args.useScale != 0u ? scales[expert] : 1.0f;
        y[slot * args.rows + firstRow + lane] = total * scale;
    }
}
