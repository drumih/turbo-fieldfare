import Foundation

/// Which fused kernels may run, decided from the *tensors* rather than the slot.
///
/// Every fused decode kernel reads packed affine weights: the QKV kernel unpacks
/// nibbles for q/k/v, the GDN input kernel does the same for qkv/z/a/b, and the
/// shared-expert kernels read scales and biases beside the weights. A promoted
/// (bf16) tensor has none of that, so handing it to one of them reads whatever
/// bytes precede it as scales and produces NaN logits rather than an error.
///
/// The slot cannot answer this. Promotion is per tensor, so a bf16 projection
/// sits inside a slot that still says 4 bits, and a guard written against the
/// slot lets it through -- which is the defect this type exists to close: the
/// 35B bf16 attention build loaded, verified and answered NaN until the guard
/// asked the tensor.
///
/// Lives on its own, model-free, so the truth table can be tested without a
/// Metal device or an install.
package enum FusedKernelPolicy {
    /// dtype 0 is packed affine weights, dtype 1 is the checkpoint's bf16.
    package static let affineDType = 0
    package static let bf16DType = 1

    /// The full-attention QKV kernel: `fused_qkv_gemv_simd`.
    ///
    /// Both width conditions are kept as well as the dtype ones: the kernel is
    /// built for 4-bit only, so an 8-bit slot must not reach it either.
    package static func mayUseFusedQKV(
        qoProjectionWeightBits: Int,
        kvProjectionWeightBits: Int,
        qDType: Int,
        kDType: Int,
        vDType: Int
    ) -> Bool {
        qoProjectionWeightBits == 4
            && kvProjectionWeightBits == 4
            && qDType == affineDType
            && kDType == affineDType
            && vDType == affineDType
    }

    /// The GDN input-projection kernel: `gdn_in_proj_gemv_simd`.
    ///
    /// It fuses qkv, z, a and b in one dispatch, so one promoted tensor among
    /// the four makes the whole fusion unusable; the caller falls back to four
    /// per-projection GEMVs, which dispatch on each tensor's dtype.
    package static func mayUseFusedGDNInputProjections(
        attentionWeightBits: Int,
        dTypes: [Int]
    ) -> Bool {
        attentionWeightBits == 4 && dTypes.count == 4
            && dTypes.allSatisfy { $0 == affineDType }
    }

    /// The shared-expert batched prefill block, which reads packed affine
    /// weights through a `weightBits` function constant.
    package static func mayUseBatchedSharedExpert(dTypes: [Int]) -> Bool {
        !dTypes.isEmpty && dTypes.allSatisfy { $0 == affineDType }
    }
}
