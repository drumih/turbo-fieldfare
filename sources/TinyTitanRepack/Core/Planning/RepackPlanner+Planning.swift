import Foundation

// The planner's private helpers: name parsing, per-file planning and the
// ordering rules for resident entries.
//
// Split out of `RepackPlanner.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. `routedExpertRole`,
// `layerIndex` and `isMultimodalTensorName` widened from `private` to internal
// because the public `plan` entry point stays in RepackPlanner.swift.
extension RepackPlanner {

    static func routedExpertRole(in name: String) -> String? {
        guard name.contains(".mlp.switch_mlp.") else { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.") { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    static func layerIndex(in name: String) -> Int? {
        // matches "...layers.<N>...."
        let marker = name.hasPrefix("layers.") ? "layers." : ".layers."
        guard let r = name.range(of: marker) else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(
        meta: IndexLoader.SourceMetadata,
        arch: ArchInfo,
        shardHeaders: [Safetensors.Header],
        outputDir: String,
        passthroughFiles: [PassthroughFile] = []
    ) throws -> RepackPlan {

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        for (name, _) in registry {
            // Companion tensors (.scales/.biases) are dropped together with
            // their primary tensor below; listing them separately would make
            // `excludedMultimodalTensorNames` mention tensors that are never
            // planned on their own.
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            let b = classify(name, numLayers: arch.numLayers, family: arch.family)
            switch b {
            case .lmResident: lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(
                        detail:
                            "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .excludedMultimodal: continue
            case .excludedSidecar: continue
            case .unknown: throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering(family: arch.family))
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(
            path: residentPath,
            baseNames: lmResidentBases,
            family: arch.family,
            registry: registry, meta: meta)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let bundle = routedByLayerAndRole[layer] ?? [:]
            // Synthetic snapshots may legitimately have no routed experts.
            guard let gName = bundle["gate"], let uName = bundle["up"], let dName = bundle["down"]
            else {
                if bundle.isEmpty {
                    layerPlans.append(
                        LayerFilePlan(
                            layerIndex: layer,
                            path: (layersDir as NSString).appendingPathComponent(
                                "layer_\(String(format: "%02d", layer)).bin"),
                            expertsPerLayer: 0,
                            expertStride: 0,
                            subTensors: []))
                    continue
                }
                throw RepackError.configurationInvalid(
                    detail:
                        "layer \(layer) routed-expert bundle incomplete: \(bundle)")
            }
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let lp = try planLayerFile(
                path: path, layer: layer,
                gateName: gName, upName: uName, downName: dName,
                registry: registry, meta: meta, arch: arch)
            layerPlans.append(lp)
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(
            arch: arch,
            baseMode: meta.baseMode,
            baseGroupSize: meta.baseGroupSize,
            resident: resident,
            layers: layerPlans,
            matchedModelID: matched,
            excludedMultimodalTensorNames: excludedMultimodalNames,
            passthroughFiles: passthroughFiles)
    }

    static func isMultimodalTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") || name.hasPrefix("embed_vision.")
            || name.hasPrefix("audio_tower.")
    }

    // MARK: - Resident planning

    private static func planResidentFile(
        path: String,
        baseNames: [String],
        family: RepackModelFamily,
        registry: [String: SourceTensor],
        meta: IndexLoader.SourceMetadata
    ) throws
        -> ResidentFilePlan
    {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        let destinationNames = baseNames.map { residentDestinationName($0, family: family) }
        for n in destinationNames {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(
            SSDAIBinary.indexHeaderBytes
                + entryCount * SSDAIBinary.indexEntryBytes
                + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for (sourceName, name) in zip(baseNames, destinationNames) {
            guard let weight = registry[sourceName] else {
                throw RepackError.missingTensor(name: sourceName)
            }
            let dtype = ietnyDtype(weight.dtype)
            let isQuantizedPacked = (weight.dtype == .u32) && sourceName.hasSuffix(".weight")

            if isQuantizedPacked {
                let base = String(sourceName.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                if scales.dtype != .bf16 || biases.dtype != .bf16 {
                    throw RepackError.dtypeMismatch(
                        name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: sourceName, meta: meta)
                let logical = try logicalShape(
                    forPackedSource: weight.shape,
                    scalesShape: scales.shape)

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = scales.sizeBytes
                let bOff = sOff + sSize
                let bSize = biases.sizeBytes
                fileCursor = bOff + bSize

                entries.append(
                    ResidentEntry(
                        name: name, dtype: 0,
                        logicalShape4: try padTo4(logical),
                        fileOffset: wOff, sizeBytes: wSize,
                        scaleOffset: sOff, scaleSize: sSize,
                        biasOffset: bOff, biasSize: bSize,
                        quantSpec: spec,
                        sourceWeight: weight, sourceScales: scales, sourceBiases: biases))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(
                    ResidentEntry(
                        name: name, dtype: dtype,
                        logicalShape4: try padTo4(weight.shape),
                        fileOffset: off, sizeBytes: size,
                        scaleOffset: 0, scaleSize: 0,
                        biasOffset: 0, biasSize: 0,
                        quantSpec: nil,
                        sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(
            path: path,
            entries: entries,
            stringTable: stringTable,
            stringTableOffsets: offsets,
            indexSize: indexSize,
            residentSize: residentSize)
    }

    // MARK: - Layer planning

    private static func planLayerFile(
        path: String, layer: Int,
        gateName: String, upName: String, downName: String,
        registry: [String: SourceTensor],
        meta: IndexLoader.SourceMetadata,
        arch: ArchInfo
    ) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        guard expertCount > 0 else {
            throw RepackError.configurationInvalid(
                detail: "layer \(layer) has routed experts but numExperts is zero")
        }
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName),
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail:
                        "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)"
                )
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else {
                throw RepackError.missingScalesCompanion(name: name)
            }
            guard let b = registry[base + ".biases"] else {
                throw RepackError.missingBiasesCompanion(name: name)
            }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(
                    name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes
                || perExpertScaleSize * UInt64(expertCount) != s.sizeBytes
                || perExpertBiasSize * UInt64(expertCount) != b.sizeBytes
            {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())
            let logicalPerExpert = try logicalShape(
                forPackedSource: perExpertSourceShape,
                scalesShape: scalesLogical)

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: 0,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: 1,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: 1,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice)
            subs.append(sSlice)
            subs.append(bSlice)
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(
            layerIndex: layer, path: path,
            expertsPerLayer: expertCount,
            expertStride: expertStride,
            subTensors: subs)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d {
        case .u32: 0
        case .bf16: 1
        case .fp16: 2
        case .fp32: 3
        }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) throws -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) {
            guard v <= UInt64(UInt32.max) else {
                throw RepackError.shapeMismatch(
                    name: "shape",
                    detail: "dimension \(v) does not fit a UInt32 index")
            }
            out.append(UInt32(v))
        }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of an MLX packed quantized tensor. The scale grid is
    /// authoritative because six-bit values do not have an integral U32
    /// packing factor.
    private static func logicalShape(
        forPackedSource source: [UInt64],
        scalesShape: [UInt64]
    ) throws -> [UInt64] {
        guard !source.isEmpty else { return source }
        guard let lastScale = scalesShape.last else {
            throw RepackError.shapeMismatch(
                name: "scales",
                detail: "packed tensor has an empty scales shape")
        }
        let (scaled, overflow) = lastScale.multipliedReportingOverflow(by: 64)
        guard !overflow else {
            throw RepackError.shapeMismatch(
                name: "scales",
                detail: "scales dimension \(lastScale) overflows when scaled by 64")
        }
        var out = source
        out[out.count - 1] = scaled
        return out
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm (and, for
    /// families with an untied head, `lm_head` last).
    private static func lmResidentOrdering(family: RepackModelFamily)
        -> (String, String) -> Bool
    {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "language_model.model.norm.weight" { return (3, 0, 0, n) }
            if n == "language_model.lm_head.weight" { return (4, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot = qwenSlotRank(in: n)
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a)
            let kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order for the Qwen 3.6 family: full-attention
    /// projections/norms, then the gated-DeltaNet linear-attention bundle,
    /// then router, shared-expert gate and MLP, then the two layer norms.
    private static func qwenSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 6 }
        if n.contains(".linear_attn.in_proj_z.weight") { return 7 }
        if n.contains(".linear_attn.in_proj_a.weight") { return 8 }
        if n.contains(".linear_attn.in_proj_b.weight") { return 9 }
        if n.contains(".linear_attn.conv1d.weight") { return 10 }
        if n.hasSuffix(".linear_attn.A_log") { return 11 }
        if n.hasSuffix(".linear_attn.dt_bias") { return 12 }
        if n.contains(".linear_attn.norm.weight") { return 13 }
        if n.contains(".linear_attn.out_proj.weight") { return 14 }
        if n.contains(".mlp.gate.weight") { return 15 }
        if n.contains(".mlp.shared_expert_gate.weight") { return 16 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 17 }
        if n.contains(".mlp.shared_expert.up_proj.weight") { return 18 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 19 }
        if n.hasSuffix(".input_layernorm.weight") { return 20 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 21 }
        return 100
    }

    /// Normalize the sidecar's single decoder layer to the target tensor-name
    /// contract. MTP-only adapter tensors retain their upstream names.
    private static func residentDestinationName(
        _ source: String,
        family: RepackModelFamily
    ) -> String {
        if family == .qwen38flashMTP || family == .qwen4flashMTP {
            // Strip `mtp.` and put the decoder layer under the target's own
            // prefix, so the draft's layer body reuses the family schema
            // verbatim. The fusion projections and their norms stay at the
            // root: they exist only in the draft.
            let stripped =
                source.hasPrefix("mtp.")
                ? String(source.dropFirst("mtp.".count)) : source
            if stripped.hasPrefix("layers.") {
                return "model.language_model." + stripped
            }
            if stripped.hasPrefix("hyper_connection_mixer.") {
                return "model.language_model." + stripped
            }
            return stripped
        }
        guard family == .qwen36MTP else { return source }
        if source.hasPrefix("layers.") {
            return "language_model.model." + source
        }
        if source == "norm.weight" {
            return "language_model.model.norm.weight"
        }
        return source
    }
}
