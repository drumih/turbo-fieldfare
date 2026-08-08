#include <metal_stdlib>
using namespace metal;

// ============================================================================
// tensorops_k3 — M5 Neural Accelerator (NAX) prefill QMM for the K3 trunk.
//
// Stage-E1 chunked prefill turns every affine int4/int8 trunk GEMV into a
// [chunkTokens x K] @ [K x N] matmul. On M5 (MSL 4.0 `__HAVE_TENSOR__`) these
// run on `mpp::tensor_ops::matmul2d` exactly like the house
// `mpp_prefill_affine_threadgroup_f16` (Metal/TensorCore/tensorops.metal):
// each threadgroup stages a dequantized-to-fp16 weight tile in threadgroup
// memory and feeds fp16 activation tiles to the tensor op, accumulating in
// fp32. The output rounds to fp16 — the same storage dtype the decode GEMV
// writes, so the CPU oracle's fp16-rounding contract is preserved.
//
// Weight layout (MLX-style affine, group 64; matches dequant_int4.metal):
//   int4: W[N][K/2] nibbles (low nibble = even element), scales/biases
//         [N][K/64] BF16. w = float(q) * s + b.
//   int8: W[N][K] bytes, same scale/bias layout.
// K is always a multiple of 64 (the affine group size) — enforced by the
// dispatcher, same precondition as the decode GEMV.
//
// This module is compiled separately from the concatenated K3 library (the
// house keeps tensorops out of the shared module list too) and guarded by
// `__HAVE_TENSOR__`: on pre-M5 silicon the kernels simply do not exist and
// `K3PrefillGEMM` falls back to the batched simd GEMMs in prefill_k3.metal.
// ============================================================================

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr uint kK3QMMGroupSize = 64;
constant constexpr int kK3QMMTileM = 64;
constant constexpr int kK3QMMTileN = 32;
constant constexpr int kK3QMMTileK = 64;

// One affine QMM, staged exactly like the house kernel: a [kN x kK] fp16
// weight tile, one K-group at a time, accumulated into an fp32 cooperative
// tensor. `isInt4` selects nibble vs byte dequant (uniform, so the branch is
// off the tensor-op critical path). The threadgroup tile is declared by the
// CALLING kernel and passed down — Metal forbids threadgroup address-space
// locals in non-kernel functions.
static inline void k3_tensorop_affine_qmm_body(
    device const uint8_t* packedWeights,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device half*          activations,
    device half*          output,
    uint M, uint N, uint K,
    bool isInt4,
    threadgroup half*     weightTile,
    uint3 tgid, uint3 lid3, uint3 threads3) {
    constexpr auto descriptor = matmul2d_descriptor(
        kK3QMMTileM, kK3QMMTileN, kK3QMMTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kK3QMMTileK, kK3QMMTileN),
        array<int32_t, 2>({1, kK3QMMTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kK3QMMTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * kK3QMMTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint groupsPerRow = K / kK3QMMGroupSize;
    const uint weightRowStride = isInt4 ? (K / 2u) : K;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kK3QMMTileN * kK3QMMTileK);
             linear += threads) {
            const uint localN = linear / uint(kK3QMMTileK);
            const uint localK = linear % uint(kK3QMMTileK);
            const uint globalN = tgid.x * uint(kK3QMMTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kK3QMMTileK) + localK;
                uint q;
                if (isInt4) {
                    const uint8_t packed =
                        packedWeights[globalN * weightRowStride + (globalK >> 1)];
                    q = (globalK & 1u) == 0u ? uint(packed & 0x0Fu)
                                             : uint(packed >> 4);
                } else {
                    q = uint(packedWeights[globalN * weightRowStride + globalK]);
                }
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kK3QMMTileK),
            dextents<int32_t, 2>(kK3QMMTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(0, int32_t(tgid.y) * kK3QMMTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kK3QMMTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kK3QMMTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

kernel void k3_tensorop_affine_int4_qmm(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat*  scales        [[buffer(1)]],
    device const bfloat*  biases        [[buffer(2)]],
    device half*          activations   [[buffer(3)]],
    device half*          output        [[buffer(4)]],
    constant uint&        M             [[buffer(5)]],
    constant uint&        N             [[buffer(6)]],
    constant uint&        K             [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]) {
    threadgroup half weightTile[kK3QMMTileN * kK3QMMTileK];
    k3_tensorop_affine_qmm_body(packedWeights, scales, biases, activations,
                                output, M, N, K, true, weightTile,
                                tgid, lid3, threads3);
}

kernel void k3_tensorop_affine_int8_qmm(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat*  scales        [[buffer(1)]],
    device const bfloat*  biases        [[buffer(2)]],
    device half*          activations   [[buffer(3)]],
    device half*          output        [[buffer(4)]],
    constant uint&        M             [[buffer(5)]],
    constant uint&        N             [[buffer(6)]],
    constant uint&        K             [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]) {
    threadgroup half weightTile[kK3QMMTileN * kK3QMMTileK];
    k3_tensorop_affine_qmm_body(packedWeights, scales, biases, activations,
                                output, M, N, K, false, weightTile,
                                tgid, lid3, threads3);
}

#endif // __HAVE_TENSOR__
