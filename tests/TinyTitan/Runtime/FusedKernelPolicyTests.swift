import Testing

@testable import TinyTitan
import TinyTitanCore

/// Every fused decode kernel reads packed affine weights. A promoted (bf16)
/// tensor inside a slot that still says 4 bits must never reach one: the 35B
/// bf16 attention build loaded, verified and answered NaN because the guard
/// asked the slot instead of the tensor. These pin the truth table the three
/// call sites now share.
@Suite struct FusedKernelPolicyTests {
    private let affine = FusedKernelPolicy.affineDType
    private let bf16 = FusedKernelPolicy.bf16DType

    // MARK: full-attention QKV

    @Test func fusedQKVRunsWhenEveryTensorIsFourBit() {
        #expect(
            FusedKernelPolicy.mayUseFusedQKV(
                qoProjectionWeightBits: 4, kvProjectionWeightBits: 4,
                qDType: affine, kDType: affine, vDType: affine))
    }

    /// The defect: the slot says 4 (so the old guard passed) while a tensor is
    /// bf16 (so the kernel would read it as nibbles).
    @Test func fusedQKVIsRefusedByAPromotedTensorEvenWhenTheSlotSaysFour() {
        for promoted in 0..<3 {
            var dTypes = [affine, affine, affine]
            dTypes[promoted] = bf16
            #expect(
                !FusedKernelPolicy.mayUseFusedQKV(
                    qoProjectionWeightBits: 4, kvProjectionWeightBits: 4,
                    qDType: dTypes[0], kDType: dTypes[1], vDType: dTypes[2]),
                "promoting tensor \(promoted) must refuse the fused kernel")
        }
    }

    @Test func fusedQKVIsRefusedAtEightBits() {
        #expect(
            !FusedKernelPolicy.mayUseFusedQKV(
                qoProjectionWeightBits: 8, kvProjectionWeightBits: 4,
                qDType: affine, kDType: affine, vDType: affine))
        #expect(
            !FusedKernelPolicy.mayUseFusedQKV(
                qoProjectionWeightBits: 4, kvProjectionWeightBits: 8,
                qDType: affine, kDType: affine, vDType: affine))
    }

    // MARK: GDN input projections

    @Test func fusedGDNRunsWhenAllFourAreFourBitAffine() {
        #expect(
            FusedKernelPolicy.mayUseFusedGDNInputProjections(
                attentionWeightBits: 4, dTypes: [affine, affine, affine, affine]))
    }

    @Test func fusedGDNIsRefusedByAnyPromotedOneOfTheFour() {
        for promoted in 0..<4 {
            var dTypes = [affine, affine, affine, affine]
            dTypes[promoted] = bf16
            #expect(
                !FusedKernelPolicy.mayUseFusedGDNInputProjections(
                    attentionWeightBits: 4, dTypes: dTypes),
                "promoting tensor \(promoted) must refuse the fused kernel")
        }
    }

    @Test func fusedGDNIsRefusedAtEightBitsAndForTheWrongTensorCount() {
        #expect(
            !FusedKernelPolicy.mayUseFusedGDNInputProjections(
                attentionWeightBits: 8, dTypes: [affine, affine, affine, affine]))
        // The kernel is written for exactly four projections; a caller that
        // hands it three has not described the fusion it asks for.
        #expect(
            !FusedKernelPolicy.mayUseFusedGDNInputProjections(
                attentionWeightBits: 4, dTypes: [affine, affine, affine]))
    }

    // MARK: shared-expert prefill block

    @Test func batchedSharedExpertRunsOnlyOnAllAffine() {
        #expect(
            FusedKernelPolicy.mayUseBatchedSharedExpert(
                dTypes: [affine, affine, affine]))
        for promoted in 0..<3 {
            var dTypes = [affine, affine, affine]
            dTypes[promoted] = bf16
            #expect(
                !FusedKernelPolicy.mayUseBatchedSharedExpert(dTypes: dTypes),
                "promoting tensor \(promoted) must refuse the batched block")
        }
    }

    @Test func batchedSharedExpertRefusesAnEmptySet() {
        #expect(!FusedKernelPolicy.mayUseBatchedSharedExpert(dTypes: []))
    }
}
