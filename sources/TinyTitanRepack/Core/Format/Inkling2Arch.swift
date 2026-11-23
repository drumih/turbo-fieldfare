import Foundation
import TinyTitanFormat

/// The `ArchInfo` of an Inkling 2 install, built from one set of model
/// constants whichever source supplied them (TT-121): the GGUF's `inkling2.*`
/// metadata (`Inkling2GGUFArch`) or the original checkpoint's `config.json`
/// (`Inkling2CheckpointArch`). Both installs therefore describe the model in
/// the same words, and the variant reads one manifest shape.
///
/// The shared fields carry what every reader looks at; everything only the
/// Inkling variant reads goes in `variantGeometry`, under the names
/// `docs/plan-inkling2-7x-engine.md` lists. Inkling has no rotary embedding:
/// position comes from its relative-attention bias, so the rotary fields record
/// "none" (factor 0, theta 1) rather than a borrowed value.
enum Inkling2Arch {
    struct Parameters {
        var hiddenSize: Int
        var intermediateSize: Int
        var expertIntermediateSize: Int
        var heads: Int
        var kvHeads: Int
        var headDim: Int
        var paddedVocabulary: Int
        var vocabulary: Int
        var slidingWindow: Int
        var finalLogitSoftcap: Double
        var storedLayers: Int
        var experts: Int
        var expertsPerToken: Int
        /// Stored layers that use sliding-window attention.
        var localLayers: [Int]
        var tiedEmbeddings: Bool
        var loopStart: Int
        var loopEnd: Int
        var loopPasses: Int
        var loopGateRank: Double
        var attentionResidualBlock: Double
        var relativeLength: Double
        var relativeExtent: Double
        var relativeClip: Double
        var convolutionTaps: Double
        var rmsEpsilon: Double
        var qkClip: Double
        var logScaleFloor: Double
        var logScaleAlpha: Double
        var expertWeightsScale: Double
        var situBeta: Double
        var situLinearBeta: Double
        var contextLength: Double
        /// Values one routed-expert scale covers: 32 for the GGUF's blocks, 16
        /// for the checkpoint's NVFP4 scales.
        var quantGroupSize: Int

        /// Layer visits the loop unrolls to.
        var unrolledLayers: Int { storedLayers + (loopPasses - 1) * (loopEnd - loopStart) }
    }

    static func archInfo(_ p: Parameters) throws -> ArchInfo {
        guard p.loopStart >= 0, p.loopEnd > p.loopStart, p.loopPasses >= 1,
            p.storedLayers > 0, p.loopEnd <= p.storedLayers
        else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 loop geometry \(p.loopStart)..<\(p.loopEnd) x \(p.loopPasses) "
                    + "does not fit \(p.storedLayers) stored layers")
        }
        guard p.localLayers.allSatisfy({ (0..<p.storedLayers).contains($0) }) else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 local layers \(p.localLayers) name a layer past "
                    + "\(p.storedLayers)")
        }
        let scalars: [String: Double] = [
            "loopStart": Double(p.loopStart), "loopEnd": Double(p.loopEnd),
            "loopPasses": Double(p.loopPasses), "loopGateRank": p.loopGateRank,
            "unrolledLayers": Double(p.unrolledLayers),
            "attentionResidualBlock": p.attentionResidualBlock,
            "relativeLength": p.relativeLength, "relativeExtent": p.relativeExtent,
            "relativeClip": p.relativeClip, "convolutionTaps": p.convolutionTaps,
            "paddedVocabulary": Double(p.paddedVocabulary),
            "vocabulary": Double(p.vocabulary), "rmsEpsilon": p.rmsEpsilon,
            "qkClip": p.qkClip, "logScaleFloor": p.logScaleFloor,
            "logScaleAlpha": p.logScaleAlpha, "expertWeightsScale": p.expertWeightsScale,
            "situBeta": p.situBeta, "situLinearBeta": p.situLinearBeta,
            "contextLength": p.contextLength,
        ]
        let mask = (0..<p.storedLayers).map { p.localLayers.contains($0) ? UInt8(0) : UInt8(1) }
        return ArchInfo(
            hiddenSize: p.hiddenSize,
            intermediateSize: p.intermediateSize,
            moeIntermediateSize: p.expertIntermediateSize,
            numHeads: p.heads,
            numKVHeads: p.kvHeads,
            numFullKVHeads: p.kvHeads,
            headDim: p.headDim,
            fullHeadDim: p.headDim,
            vocabSize: p.paddedVocabulary,
            slidingWindow: p.slidingWindow,
            finalLogitSoftcap: p.finalLogitSoftcap,
            ropeTheta: 1,
            fullRopeTheta: 1,
            partialRotaryFactor: 0,
            numLayers: p.storedLayers,
            numExperts: p.experts,
            topKExperts: p.expertsPerToken,
            tieWordEmbeddings: p.tiedEmbeddings,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: "situ",
            family: .inkling2,
            attnOutputGate: true,
            attentionScale: 1 / Double(p.headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0, linearNumVHeads: 0, linearKeyHeadDim: 0,
            linearValueHeadDim: 0, linearConvKernelSize: 0,
            quantGroupSize: p.quantGroupSize,
            variantGeometry: SSDAIVariantGeometryV1(
                scalars: scalars, lists: ["localLayers": p.localLayers]))
    }
}
