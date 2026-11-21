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
//          sub-blocks (kernel_mul_mv_nvfp4_f32).
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

// E2M1 values (not doubled), so the UE4M3 scale needs no factor of one half.
constant float GEMV_IDS_FP4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Unsigned E4M3 (bias 7): the seven low bits placed in an fp16 give the value
// times 2^-8, subnormals included; 0x7F decodes to 0 as ggml does.
static inline float gemv_ids_ue4m3(uchar code) {
    const float v = float(as_type<half>(ushort(ushort(code & 0x7Fu) << 7u))) * 256.0f;
    return code == 0x7Fu ? 0.0f : v;
}

kernel void nvfp4_gemv_ids(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    constant GGUFGemvIdsArgs& args [[buffer(5)]],
    uint2 groupIndex [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threadIndex [[thread_index_in_threadgroup]])
{
    threadgroup float table[32];
    if (threadIndex < 32u) table[threadIndex] = GEMV_IDS_FP4[threadIndex % 16u];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint slot = groupIndex.y;
    const int expert = ids[slot];
    const uint firstRow =
        (groupIndex.x * GEMV_IDS_NVFP4_SIMDGROUPS + sg) * GEMV_IDS_NVFP4_ROWS;
    if (firstRow >= args.rows) return;
    const uint blockCount = args.columns / 64u;
    const uint rowBytes = blockCount * 36u;
    device const uchar* matrix = weights + ulong(expert) * args.expertStride;
    device const float* xs = x + slot * args.xSlotStride;

    const uint pairIndex = lane / 2u;  // 0...15: which block of sixteen
    const uint subPair = lane % 2u;      // sub-blocks 2*subPair and 2*subPair + 1

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
            device const uchar* q2 = b + 4u + subPair * 16u;
            float sumq = 0.0f;
            for (uint s = 0u; s < 2u; ++s) {
                device const uchar* q = q2 + 8u * s;
                float4 acc1 = yl[4u * s + 0u] * float4(
                    table[q[0] & 0x0Fu], table[q[1] & 0x0Fu],
                    table[q[2] & 0x0Fu], table[q[3] & 0x0Fu]);
                float4 acc2 = yl[4u * s + 1u] * float4(
                    table[q[4] & 0x0Fu], table[q[5] & 0x0Fu],
                    table[q[6] & 0x0Fu], table[q[7] & 0x0Fu]);
                float4 acc3 = yl[4u * s + 2u] * float4(
                    table[q[0] >> 4u], table[q[1] >> 4u],
                    table[q[2] >> 4u], table[q[3] >> 4u]);
                float4 acc4 = yl[4u * s + 3u] * float4(
                    table[q[4] >> 4u], table[q[5] >> 4u],
                    table[q[6] >> 4u], table[q[7] >> 4u]);
                acc1 = (acc1 + acc3) + (acc2 + acc4);
                sumq += gemv_ids_ue4m3(b[2u * subPair + s])
                    * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
            }
            sums[r] += sumq;
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
