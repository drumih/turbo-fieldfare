import Foundation
import TurboFieldfareFormat

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = GTurboFormatV1.magic
    static let versionMajor = GTurboFormatV1.versionMajor
    static let versionMinor = GTurboFormatV1.versionMinor

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        let bitWidthsByQuantSlot = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        let wireArch = GTurboManifestArchV1(
            hiddenSize: arch.hiddenSize,
            ffnIntermediate: arch.intermediateSize,
            moeIntermediateSize: arch.moeIntermediateSize,
            numHeads: arch.numHeads,
            numKVHeads: arch.numKVHeads,
            numFullKVHeads: arch.numFullKVHeads,
            headDim: arch.headDim,
            fullHeadDim: arch.fullHeadDim,
            vocabSize: arch.vocabSize,
            slidingWindow: arch.slidingWindow,
            finalLogitSoftcap: arch.finalLogitSoftcap,
            ropeTheta: arch.ropeTheta,
            fullRopeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor,
            numLayers: arch.numLayers,
            numExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            attentionKEqV: arch.attentionKEqV,
            hiddenActivation: arch.hiddenActivation,
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map(Int.init))
        func slot(_ name: String) throws -> GTurboManifestQuantSlotV1 {
            guard let weightBits = bitWidthsByQuantSlot[name] else {
                throw RepackError.configurationInvalid(
                    detail: "missing manifest quant slot bit width for \(name)")
            }
            return GTurboManifestQuantSlotV1(
                weightBits: weightBits,
                scheme: plan.baseMode,
                scaleType: "BF16",
                biasType: "BF16",
                groupSize: plan.baseGroupSize)
        }
        let quant = GTurboManifestQuantV1(
            embedding: try slot("embedding"),
            attention: try slot("attention"),
            router: try slot("router"),
            sharedExpert: try slot("sharedExpert"),
            routedExpert: try slot("routedExpert"))
        var wireFiles: [String: GTurboManifestFileV1] = [:]
        wireFiles.reserveCapacity(files.count)
        for file in files {
            guard wireFiles.updateValue(
                GTurboManifestFileV1(size: file.info.size, sha256: file.info.sha256),
                forKey: file.relativePath) == nil else {
                throw RepackError.configurationInvalid(
                    detail: "duplicate manifest file entry \(file.relativePath)")
            }
        }
        return try GTurboManifestCodec.encode(GTurboManifestV1(
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            arch: wireArch,
            quant: quant,
            files: wireFiles,
            expertsPerLayer: expertsPerLayer,
            numLayers: numLayers,
            expertStride: expertStride,
            bitWidthOverridesHonored: plan.bitsOverrideCount))
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layers: [GTurboLayerV1] = []
        layers.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [GTurboExpertV1] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let physicalRank = lp.physicalRank(for: e)
                let base = UInt64(physicalRank) * lp.expertStride
                var tensors: [String: GTurboSubTensorV1] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    guard slice.dtype == GTurboFormatV1.DType.u32.rawValue
                            || slice.dtype == GTurboFormatV1.DType.bf16.rawValue else {
                        throw RepackError.configurationInvalid(
                            detail: "unsupported packed expert dtype \(slice.dtype) for \(key)")
                    }
                    let shape = try slice.logicalShape.enumerated().map { index, value in
                        guard value <= UInt64(UInt32.max) else {
                            throw RepackError.configurationInvalid(
                                detail: "packed expert shape[\(index)] exceeds UInt32")
                        }
                        return UInt32(value)
                    }
                    let previous = tensors.updateValue(GTurboSubTensorV1(
                        offset: slice.offsetInExpertBlob,
                        size: slice.sizeInExpertBlob,
                        dtype: slice.dtype == GTurboFormatV1.DType.u32.rawValue ? "U32" : "BF16",
                        shape: shape,
                        bits: slice.bitsForWeights), forKey: key)
                    guard previous == nil else {
                        throw RepackError.configurationInvalid(
                            detail: "duplicate packed expert tensor key \(key)")
                    }
                }
                experts.append(GTurboExpertV1(
                    expert: e,
                    physicalRank: nil,
                    offset: base,
                    size: lp.expertStride,
                    tensors: tensors))
            }
            layers.append(GTurboLayerV1(layer: lp.layerIndex,
                                        file: layerFile,
                                        experts: experts))
        }
        return try GTurboPackedExpertsLayoutCodec.encode(
            GTurboPackedExpertsLayoutV1(
                expertStride: expertStride,
                numLayers: arch.numLayers,
                expertsPerLayer: plan.layers.first?.expertsPerLayer ?? 0,
                layers: layers))
    }
}
