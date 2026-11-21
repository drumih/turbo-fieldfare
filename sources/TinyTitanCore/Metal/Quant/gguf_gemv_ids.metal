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
