import Foundation
import TinyTitanFormat
import TinyTitanCore

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum SSDAIJSON {

    /// The magic written into every manifest this repacker produces.
    static let magic = "SSDAI"
    /// The version, single-sourced from the format module: a bump has to move
    /// the writer, the wire struct's defaults and the reader's expectations
    /// together, and two literals can drift apart.
    static let versionMajor = SSDAIFormatV1.versionMajor
    static let versionMinor = SSDAIFormatV1.versionMinor

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

    static func encodeManifest(
        plan: RepackPlan,
        modelID: String,
        sourceSnapshotHash: String,
        files: [(relativePath: String, info: FileEntry)],
        expertsPerLayer: Int,
        numLayers: Int,
        expertStride: UInt64,
        bitWidths: QuantBitWidths
    ) throws -> Data {
        let arch = plan.arch
        let archDict = archDictionary(arch)
        let quantDict = try quantObject(plan: plan, bitWidths: bitWidths)

        var filesDict: [String: Any] = [:]
        for (path, info) in files {
            filesDict[path] = ["size": info.size, "sha256": info.sha256]
        }

        let manifest: [String: Any] = [
            "magic": SSDAIJSON.magic,
            "versionMajor": SSDAIJSON.versionMajor,
            "versionMinor": SSDAIJSON.versionMinor,
            "flags": [
                "streamingPresent": plan.streamingPresent,
                "turboQuantKV": plan.turboQuantKV,
                "aneSharedExpert": plan.aneSharedExpert,
            ],
            "modelID": modelID,
            "sourceSnapshotHash": sourceSnapshotHash,
            "arch": archDict,
            "quant": quantDict,
            "files": filesDict,
            "expertsPerLayer": expertsPerLayer,
            "numLayers": numLayers,
            "expertStride": expertStride,
        ]
        return try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    /// The `arch` object, including the family extension fields.
    ///
    /// Split out of `encodeManifest` when K2's ten keys took that body past the
    /// 120-line gate. The extension blocks are the part that grows with each
    /// new family, so they live together rather than lengthening the document
    /// assembly.
    private static func archDictionary(_ arch: ArchInfo) -> [String: Any] {
        var archDict: [String: Any] = [
            "hiddenSize": arch.hiddenSize,
            "ffnIntermediate": arch.intermediateSize,
            "moeIntermediateSize": arch.moeIntermediateSize,
            "numHeads": arch.numHeads,
            "numKVHeads": arch.numKVHeads,
            "numFullKVHeads": arch.numFullKVHeads,
            "headDim": arch.headDim,
            "fullHeadDim": arch.fullHeadDim,
            "vocabSize": arch.vocabSize,
            "slidingWindow": arch.slidingWindow,
            "finalLogitSoftcap": arch.finalLogitSoftcap,
            "ropeTheta": arch.ropeTheta,
            "fullRopeTheta": arch.fullRopeTheta,
            "partialRotaryFactor": arch.partialRotaryFactor,
            "numLayers": arch.numLayers,
            "numExperts": arch.numExperts,
            "topKExperts": arch.topKExperts,
            "tieWordEmbeddings": arch.tieWordEmbeddings,
            "attentionKEqV": arch.attentionKEqV,
            "hiddenActivation": arch.hiddenActivation,
            "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) },
        ]
        // Family extension fields. Always written for the Qwen families.
        archDict["family"] = arch.family.rawValue
        archDict["attnOutputGate"] = arch.attnOutputGate
        archDict["attentionScale"] = arch.attentionScale
        archDict["embeddingScaledBySqrtHidden"] = arch.embeddingScaledBySqrtHidden
        archDict["routerScaled"] = arch.routerScaled
        archDict["ffnSandwichNorms"] = arch.ffnSandwichNorms
        archDict["sharedExpertGated"] = arch.sharedExpertGated
        archDict["ropeNeoxSubdim"] = arch.ropeNeoxSubdim
        archDict["linearNumKHeads"] = arch.linearNumKHeads
        archDict["linearNumVHeads"] = arch.linearNumVHeads
        archDict["linearKeyHeadDim"] = arch.linearKeyHeadDim
        archDict["linearValueHeadDim"] = arch.linearValueHeadDim
        archDict["linearConvKernelSize"] = arch.linearConvKernelSize
        // Extension geometry is written for every architecture that has it, so
        // manifests for the families that do not stay byte-identical. The
        // reader validates these whenever present and, from
        // `SSDAIFormatV1.extensionGeometryMandatoryFromMinor`, requires them
        // whenever the runtime contract has them -- which is what stops a
        // checkpoint with different hyper-connection / indexer / PLE geometry
        // from being run silently against the runtime's hardcoded constants.
        //
        // The gate is the geometry, not a family name. It used to read
        // `arch.family == .qwen38flash`, and the Qwen3.8-Flash-Next *draft*
        // carries the target's hyper-connections and indexer, so the one
        // installed family beyond the name check got no block at all: 16 keys
        // missing, and a reader that validates only what it is given.
        if arch.declaresExtensionGeometry {
            archDict["hcCount"] = arch.hcCount
            archDict["hcLowRank"] = arch.hcLowRank
            archDict["indexerNumHeads"] = arch.indexerNumHeads
            archDict["indexerNumKVHeads"] = arch.indexerNumKVHeads
            archDict["indexerHeadDim"] = arch.indexerHeadDim
            archDict["indexerBudget"] = arch.indexerBudget
            archDict["indexerCompressRatio"] = arch.indexerCompressRatio
            archDict["pleLayerIndices"] = arch.pleLayerIndices
            archDict["pleEmbedDim"] = arch.pleEmbedDim
            archDict["pleConvKernelSize"] = arch.pleConvKernelSize
            archDict["pleNgramSize"] = arch.pleNgramSize
            archDict["pleVocabSizeBase"] = arch.pleVocabSizeBase
            archDict["pleHeadsPerNgram"] = arch.pleHeadsPerNgram
            archDict["pleVocabDivisor"] = arch.pleVocabDivisor
            archDict["routerNormTopK"] = arch.routerNormTopK
            archDict["quantGroupSize"] = arch.quantGroupSize
        }
        if arch.family == .k2Horizon {
            // K2's layer shape: the MoVA expert set, the grouped norm, the
            // attention gate function, the router's selection bias and
            // scaling, and which layers are dense.
            archDict["movaNumExperts"] = arch.movaNumExperts
            archDict["movaTopKExperts"] = arch.movaTopKExperts
            archDict["normGroups"] = arch.normGroups
            archDict["attentionGate"] = arch.attentionGate
            archDict["routerScaling"] = arch.routerScaling
            archDict["routerHasBias"] = arch.routerHasBias
            archDict["movaRouterScaling"] = arch.movaRouterScaling
            archDict["movaRouterHasBias"] = arch.movaRouterHasBias
            archDict["movaActivation"] = arch.movaActivation
            archDict["denseLayerIndices"] = arch.denseLayerIndices
            // Not part of the K2 block until now, and the reader's fallback is
            // false for every other family: a K2 install would have normalized
            // nothing. It belongs here because K2's config states
            // `norm_topk_prob: true`; write it so the manifest says it too.
            archDict["routerNormTopK"] = arch.routerNormTopK
        }
        if let geometry = arch.variantGeometry {
            archDict["variantGeometry"] = [
                "scalars": geometry.scalars, "lists": geometry.lists,
            ]
        }
        if arch.family == .laguna {
            // Layer 0 is dense and carries no expert file. The manifest is
            // where the runtime learns which layers may legitimately have no
            // `packed_experts/layer_NN.bin`; omitting this made the loader
            // demand layer_0.bin from an install that was never going to have
            // one.
            archDict["denseLayerIndices"] = arch.denseLayerIndices
            archDict["routerNormTopK"] = arch.routerNormTopK
        }
        return archDict
    }

    /// The `quant` object: the five width *slots*, plus one entry per
    /// quantified resident tensor.
    ///
    /// The slots alone cannot describe a build. They are what the installer
    /// *derives* from the tensor data, not what the source checkpoint
    /// declared, and a "4-bit" model is not uniformly 4-bit: the Qwen 3.5
    /// 2B/4B/9B keep their embedding and several attention K/V pairs at 8
    /// bits while the slots say `attention: 4`. Both statements are true of
    /// different tensors, and only the per-tensor entries say which is which.
    ///
    /// This is the bug that was shipped and then found by comparing `.ssdai`
    /// logits against the snapshot's: a reader that trusts the slots unpacks
    /// those 8-bit tensors as 4-bit. The word count changes, the strides
    /// still divide evenly, every shape check passes, and the model answers
    /// fluently and wrongly. Writing the real width for every tensor makes
    /// the manifest say what was actually packed, so no reader has to
    /// re-derive it from the slots and none can get it wrong.
    ///
    /// Unquantized tensors (norms, scalars) carry no `quantSpec` and are
    /// deliberately absent: they are read as BF16 by `dtype` and never
    /// dequantized. A stem that lands on a slot name, or on another tensor's
    /// stem with a different width, is refused rather than skipped: the key can
    /// carry one of the two facts, and the one this object exists to protect is
    /// the tensor's real width.
    private static func quantObject(
        plan: RepackPlan,
        bitWidths: QuantBitWidths
    ) throws -> [String: Any] {
        let slots = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        func entry(_ bits: Int, slot: String? = nil) -> [String: Any] {
            [
                "weightBits": bits,
                "scheme": slot.flatMap { plan.slotSchemes[$0] } ?? plan.baseMode,
                "scaleType": "BF16", "biasType": "BF16",
                "groupSize": plan.baseGroupSize,
            ]
        }
        var dict: [String: Any] = [:]
        for (slot, bits) in slots { dict[slot] = entry(bits, slot: slot) }
        for override in try perTensorWidths(
            quantized: plan.resident.entries.compactMap { resident in
                guard let spec = resident.quantSpec else { return nil }
                return (name: resident.name, bits: spec.bits)
            },
            slotNames: Set(slots.keys))
        {
            dict[override.stem] = entry(override.bits)
        }
        // Promoted tensors: bf16 weights living in a slot whose default width
        // describes packed data. They get a name in `promoted`, not a
        // `weightBits` value, because a width is a packing and this is the
        // absence of one -- the reader that needs a kernel width resolves these
        // to 16, and the arithmetic that unpacks never sees them.
        let promoted = plan.resident.entries.compactMap { resident -> String? in
            guard resident.quantSpec == nil, resident.dtype == 1,
                Self.isSlotDescribedRole(resident.name)
            else { return nil }
            return resident.name.hasSuffix(weightSuffix)
                ? String(resident.name.dropLast(weightSuffix.count)) : resident.name
        }
        if !promoted.isEmpty { dict["promoted"] = promoted.sorted() }
        return dict
    }

    /// Each quantized resident tensor's manifest key and real width.
    ///
    /// The key is the tensor stem, so `…k_proj.weight` becomes `…k_proj`, which
    /// is what `ManifestQuant.slot(_:overrides:fallback:)` looks a tensor up by.
    /// Two collisions are refused here instead of resolved by whichever entry
    /// was written last:
    ///
    /// - a stem equal to a slot name, which would silently demote a whole slot
    ///   or lose this tensor's width; and
    /// - the same stem from two tensors at different widths, which the manifest
    ///   cannot represent.
    ///
    /// Both are the failure class the per-tensor entries exist to fix: a lost
    /// width is read back as the slot's, the strides still divide evenly, every
    /// shape check passes, and the model answers fluently and wrongly.
    static func perTensorWidths(
        quantized: [(name: String, bits: Int)],
        slotNames: Set<String>
    ) throws -> [(stem: String, bits: Int)] {
        var result: [(stem: String, bits: Int)] = []
        result.reserveCapacity(quantized.count)
        var widths: [String: Int] = [:]
        for tensor in quantized {
            let stem =
                tensor.name.hasSuffix(weightSuffix)
                ? String(tensor.name.dropLast(weightSuffix.count)) : tensor.name
            guard !slotNames.contains(stem) else {
                throw RepackError.configurationInvalid(
                    detail: "resident tensor '\(tensor.name)' packs to '\(stem)', "
                        + "which is a quant slot name; the manifest cannot carry "
                        + "both the slot and this tensor's width, so writing it "
                        + "would drop one of them")
            }
            if let previous = widths[stem] {
                guard previous == tensor.bits else {
                    throw RepackError.configurationInvalid(
                        detail: "two resident tensors pack to '\(stem)' at "
                            + "different widths (\(previous) and \(tensor.bits)); "
                            + "the manifest keys widths by stem, so one would be "
                            + "written and the other read")
                }
                continue
            }
            widths[stem] = tensor.bits
            result.append((stem: stem, bits: tensor.bits))
        }
        return result
    }

    /// Whether a resident tensor is one the five slots describe.
    ///
    /// Only those belong in `promoted` when stored at bf16: a norm is bf16 in
    /// every install and never described by a slot, so listing it would claim
    /// a slot applies to it when none does.
    private static func isSlotDescribedRole(_ name: String) -> Bool {
        guard name.hasSuffix(weightSuffix) else { return false }
        let suffixes = [
            ".embed_tokens.weight", ".lm_head.weight",
            ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
            ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight",
            ".linear_attn.out_proj.weight", ".mlp.gate.weight",
            ".mlp.shared_expert.gate_proj.weight",
            ".mlp.shared_expert.up_proj.weight",
            ".mlp.shared_expert.down_proj.weight",
        ]
        return suffixes.contains { name.hasSuffix($0) }
    }

    /// The `layout.json` element-type name of a slice's dtype code: the
    /// resident-index codes, plus the repacker-internal codes for the separate
    /// NVFP4/MXFP8 scale tensors, which have no index code of their own.
    static func layoutDTypeName(_ code: UInt8) throws -> String {
        switch code {
        case 0: return "U32"
        case 1: return "BF16"
        case 3: return SSDAIAuxiliaryDTypeV1.f32.rawValue
        case LayoutDTypeCode.f8e4m3: return SSDAIAuxiliaryDTypeV1.f8e4m3.rawValue
        case LayoutDTypeCode.ue8m0: return SSDAIAuxiliaryDTypeV1.ue8m0.rawValue
        default:
            guard let dtype = SSDAIFormatV1.DType(rawValue: code),
                let encoding = SSDAIBlockEncodingV1(residentDType: dtype)
            else {
                throw RepackError.configurationInvalid(
                    detail: "no layout element type for dtype code \(code)")
            }
            return encoding.rawValue
        }
    }

    /// Repacker-internal slice codes for element types with no resident code.
    enum LayoutDTypeCode {
        static let f8e4m3: UInt8 = 200
        static let ue8m0: UInt8 = 201
    }

    /// The suffix a resident tensor's name carries and its manifest key drops.
    private static let weightSuffix = ".weight"

    static func encodeLayout(
        plan: RepackPlan,
        expertStride: UInt64
    ) throws -> Data {
        // Every layer that carries experts must share one stride. The manifest
        // records a single value and `SSDAILayoutValidator` refuses a layout
        // whose layers disagree, so a plan like that cannot be written correctly
        // -- and the validator runs after the caller has already written the
        // packed payload, which for the 35B families is hundreds of gigabytes.
        // The two callers take the stride from the first non-empty layer, so a
        // disagreement between layers is exactly what would go unnoticed here.
        let layerStrides = Set(
            plan.layers.filter { $0.expertsPerLayer > 0 }
                .map(\.expertStride))
        // Inkling keeps each layer's experts in their source encoding, so its
        // NVFP4 and Q8_0 layers differ in stride; they are written compact
        // inside the largest one, which `SSDAILayoutValidator` checks per layer.
        let compactLayers = plan.arch.family == .inkling2
        if compactLayers, let largest = layerStrides.max(), largest != expertStride {
            throw RepackError.configurationInvalid(
                detail: "manifest stride \(expertStride) is not the largest layer stride \(largest)"
            )
        }
        guard compactLayers || layerStrides.count <= 1 else {
            throw RepackError.configurationInvalid(
                detail: "packed expert stride differs between layers: "
                    + "\(layerStrides.sorted())")
        }
        if !compactLayers, let only = layerStrides.first, only != expertStride {
            throw RepackError.configurationInvalid(
                detail: "packed expert stride \(expertStride) does not match the "
                    + "layers' \(only); the layout is written from the first "
                    + "non-empty layer, so these have to agree")
        }
        let arch = plan.arch
        var layersArr: [[String: Any]] = []
        layersArr.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let base = UInt64(e) * lp.expertStride
                var tensors: [String: Any] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales": key = slice.role + "_scales"
                    case "biases": key = slice.role + "_biases"
                    default: key = slice.role + "_" + slice.component
                    }
                    var t: [String: Any] = [
                        "offset": slice.offsetInExpertBlob,
                        "size": slice.sizeInExpertBlob,
                        "dtype": try layoutDTypeName(slice.dtype),
                        "shape": slice.logicalShape.map { Int($0) },
                    ]
                    if let bits = slice.bitsForWeights { t["bits"] = bits }
                    tensors[key] = t
                }
                let expertEntry: [String: Any] = [
                    "expert": e,
                    "offset": base,
                    "size": lp.expertStride,
                    "tensors": tensors,
                ]
                experts.append(expertEntry)
            }
            layersArr.append([
                "layer": lp.layerIndex,
                "file": layerFile,
                "expertsPerLayer": lp.expertsPerLayer,
                "experts": experts,
            ])
        }
        let obj: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": arch.numLayers,
            // Same source as `expertStride`: the first layer that actually
            // packs experts. SSDAIPackedExpertsLayoutCodec.decode requires
            // expertsPerLayer > 0 and consistent across all layers.
            "expertsPerLayer": plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer
                ?? 0,
            "layers": layersArr,
        ]
        return try JSONSerialization.data(
            withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    /// Re-writes an installed `layout.json` as a compressed one.
    ///
    /// The expert entries are deliberately left alone: their `offset`/`size`/
    /// `tensors` describe the *raw* geometry, which is what the index was built
    /// from, and the streamer ignores the offsets when a sidecar is present.
    /// Only the top-level marker changes, so a 22 MB layout can be rewritten
    /// without re-deriving 256 experts per layer.
    static func markCompressed(
        layoutData: Data, compression: SSDAIExpertCompression = .lzfse
    ) throws -> Data {
        guard var object = try JSONSerialization.jsonObject(with: layoutData) as? [String: Any]
        else {
            throw RepackError.configurationInvalid(
                detail: "packed_experts/layout.json is not a JSON object")
        }
        object["compression"] = compression.rawValue
        return try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}

extension RepackPlan {
    /// The .ssdai layout always streams routed experts from per-layer files;
    /// the remaining flags are fixed for the Qwen 3.6 baseline (no quantized
    /// KV, no ANE shared-expert fusion) and are computed here so the manifest
    /// mirrors the plan rather than a hardcoded dictionary.
    fileprivate var streamingPresent: Bool { layers.contains { $0.expertsPerLayer > 0 } }
    fileprivate var turboQuantKV: Bool { false }
    fileprivate var aneSharedExpert: Bool { false }
}
