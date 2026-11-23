import Foundation
import TinyTitanFormat

/// Plans an Inkling 2 Mini `.ssdai` install from its original ModelOpt
/// checkpoint (TT-121), keeping every tensor in the encoding the checkpoint
/// stores it in: the routed experts stay NVFP4 (MXFP8 in the last four layers),
/// everything else stays BF16 or F32. Nothing is requantized, so the install
/// is the source's bytes rearranged for streaming.
///
/// - Resident: every text-model tensor that is not a routed expert, in file
///   order and under its checkpoint name, plus the MTP head from
///   `mtp.safetensors` under an `mtp.` prefix. The vision tower is left out,
///   as every install leaves out image input.
/// - Experts: one blob per expert per layer. An NVFP4 layer holds `gate_up`
///   and `down` (packed E2M1), their E4M3 scales (one per 16 values) and the
///   per-expert F32 `amax`; an MXFP8 layer holds the E4M3 values and their
///   UE8M0 scales (one per 32). The per-layer `outer_scale_denominator` is
///   resident. NVFP4 and MXFP8 layers differ in stride, so the layers are
///   compact inside the largest one.
enum Inkling2CheckpointPlanner {
    static let mtpShard = "mtp.safetensors"
    static let mtpPrefix = "mtp."
    static let visualPrefix = "model.visual."
    static let opaqueDTypes: Set<String> = ["U8", "F8_E4M3"]
    static let residentAlignment: UInt64 = 64
    static let blobAlignment: UInt64 = 64
    /// Files copied beside the tokenizer when present: the tiktoken model the
    /// tokenizer was built from, and the configs that say how it was quantized
    /// and how it generates.
    static let tokenizerExtras: [(source: String, name: String)] = [
        ("tiktoken/tokenizer.model", "tiktoken.model"),
        ("hf_quant_config.json", "hf_quant_config.json"),
        ("generation_config.json", "generation_config.json"),
    ]

    struct Shard {
        let fileName: String
        let fileSize: UInt64
        let header: Safetensors.Header
        /// The header as the file stores it (size prefix included), which is
        /// what the source fingerprint hashes.
        let headerBytes: Data
        /// Prepended to every resident name from this shard.
        let namePrefix: String
    }

    struct Source {
        let shards: [Shard]
        let arch: ArchInfo
    }

    /// `layers.<n>.mlp.experts.{w13,w2}_weight{,_scale,_amax}` → (layer, key).
    static func expertKey(_ name: String) -> (layer: Int, key: String)? {
        let prefix = "model.llm.layers."
        guard name.hasPrefix(prefix) else { return nil }
        let parts = name.dropFirst(prefix.count).split(separator: ".")
        guard parts.count == 4, let layer = Int(parts[0]), parts[1] == "mlp",
            parts[2] == "experts"
        else { return nil }
        let field = String(parts[3])
        for (source, role) in [("w13_weight", "gate_up"), ("w2_weight", "down")] {
            switch field {
            case source: return (layer, role)
            case source + "_scale": return (layer, role + "_scale")
            case source + "_amax": return (layer, role + "_amax")
            default: continue
            }
        }
        return nil
    }

    static func plan(source: Source, outputDir: String) throws -> RepackPlan {
        var resident: [(tensor: SourceTensor, name: String)] = []
        var experts: [Int: [String: (SourceTensor, String?)]] = [:]
        var excluded: [String] = []
        for shard in source.shards {
            for tensor in shard.header.tensors.sorted(by: { $0.absoluteOffset < $1.absoluteOffset })
            {
                if shard.namePrefix.isEmpty, tensor.name.hasPrefix(visualPrefix) {
                    excluded.append(tensor.name)
                } else if shard.namePrefix.isEmpty, let key = expertKey(tensor.name) {
                    experts[key.layer, default: [:]][key.key] =
                        (tensor, shard.header.opaqueDTypeNames[tensor.name])
                } else {
                    resident.append((tensor, shard.namePrefix + tensor.name))
                }
            }
        }
        let arch = source.arch
        guard Set(experts.keys) == Set(0..<arch.numLayers) else {
            throw RepackError.configurationInvalid(
                detail: "routed experts cover layers \(experts.keys.sorted()), "
                    + "not 0..<\(arch.numLayers)")
        }
        var plan = RepackPlan(
            arch: arch, baseMode: "BF16", baseGroupSize: arch.quantGroupSize,
            resident: try planResident(resident, outputDir: outputDir),
            layers: try (0..<arch.numLayers).map { layer in
                try planLayer(
                    layer, tensors: experts[layer] ?? [:], experts: arch.numExperts,
                    outputDir: outputDir)
            },
            matchedModelID: nil, excludedMultimodalTensorNames: excluded.sorted(),
            passthroughFiles: [])
        plan.slotSchemes = ["routedExpert": SSDAIBlockEncodingV1.nvfp4.rawValue]
        return plan
    }

    private static func opaqueCopy(_ tensor: SourceTensor) -> SourceTensor {
        SourceTensor(
            name: tensor.name, shardPath: tensor.shardPath, dtype: .opaque, shape: tensor.shape,
            absoluteOffset: tensor.absoluteOffset, sizeBytes: tensor.sizeBytes)
    }

    private static func planResident(
        _ tensors: [(tensor: SourceTensor, name: String)], outputDir: String
    ) throws -> ResidentFilePlan {
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        for entry in tensors {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: entry.name.utf8)
        }
        let rawIndex = UInt64(
            SSDAIBinary.indexHeaderBytes + tensors.count * SSDAIBinary.indexEntryBytes
                + stringTable.count)
        let indexSize = Inkling2GGUFPlanner.roundUp(rawIndex, to: Layout.pageBytes)
        var cursor = indexSize
        var entries: [ResidentEntry] = []
        for (tensor, name) in tensors {
            let code: UInt8
            switch tensor.dtype {
            case .bf16: code = 1
            case .fp16: code = 2
            case .fp32: code = 3
            case .u32, .opaque:
                throw RepackError.configurationInvalid(
                    detail: "\(name): an Inkling resident tensor must be BF16, F16 or F32")
            }
            guard tensor.shape.count <= 4, tensor.shape.allSatisfy({ $0 <= UInt32.max }) else {
                throw RepackError.configurationInvalid(detail: "\(name): shape \(tensor.shape)")
            }
            cursor = Inkling2GGUFPlanner.roundUp(cursor, to: residentAlignment)
            let shape = tensor.shape.map { UInt32($0) }
            entries.append(
                ResidentEntry(
                    name: name, dtype: code,
                    logicalShape4: shape + [UInt32](repeating: 0, count: 4 - shape.count),
                    fileOffset: cursor, sizeBytes: tensor.sizeBytes,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                    quantSpec: nil, sourceWeight: opaqueCopy(tensor),
                    sourceScales: nil, sourceBiases: nil))
            cursor += tensor.sizeBytes
        }
        return ResidentFilePlan(
            path: (outputDir as NSString).appendingPathComponent("model_weights.bin"),
            entries: entries, stringTable: stringTable, stringTableOffsets: offsets,
            indexSize: indexSize, residentSize: cursor - indexSize)
    }

    /// The layer's expert encoding, from its `w13` values' source dtype:
    /// packed `U8` is NVFP4, `F8_E4M3` values are MXFP8.
    private static func encoding(
        layer: Int, tensors: [String: (SourceTensor, String?)]
    ) throws -> SSDAIBlockEncodingV1 {
        switch tensors["gate_up"]?.1 {
        case "U8": return .nvfp4
        case "F8_E4M3": return .mxfp8
        default:
            throw RepackError.configurationInvalid(
                detail: "layer \(layer): w13_weight is neither packed NVFP4 (U8) nor MXFP8 "
                    + "(F8_E4M3)")
        }
    }

    private static func planLayer(
        _ layer: Int, tensors: [String: (SourceTensor, String?)], experts: Int,
        outputDir: String
    ) throws -> LayerFilePlan {
        let encoding = try encoding(layer: layer, tensors: tensors)
        // NVFP4 scales are E4M3 and come with a per-expert F32 `amax`; MXFP8
        // scales are UE8M0 bytes, stored as `U8`, with no global scale.
        let scaleSource = encoding == .nvfp4 ? "F8_E4M3" : "U8"
        let scaleCode =
            encoding == .nvfp4
            ? SSDAIJSON.LayoutDTypeCode.f8e4m3 : SSDAIJSON.LayoutDTypeCode.ue8m0
        let expected = Set(
            ["gate_up", "down", "gate_up_scale", "down_scale"]
                + (encoding == .nvfp4 ? ["gate_up_amax", "down_amax"] : []))
        guard Set(tensors.keys) == expected else {
            throw RepackError.configurationInvalid(
                detail: "layer \(layer) (\(encoding.rawValue)) has expert tensors "
                    + "\(tensors.keys.sorted()), expected \(expected.sorted())")
        }
        var slices: [PerExpertTensorSlice] = []
        var cursor: UInt64 = 0
        func add(
            _ key: String, role: String, component: String, code: UInt8,
            shape: [UInt64], bytes: UInt64, sourceDType: String?, bits: Int?
        ) throws {
            guard let (tensor, dtypeName) = tensors[key] else { return }
            let label = "layer \(layer) \(key)"
            guard tensor.shape.first == UInt64(experts) else {
                throw RepackError.configurationInvalid(
                    detail: "\(label): \(tensor.shape) does not lead with \(experts) experts")
            }
            guard dtypeName == sourceDType, tensor.sizeBytes == bytes * UInt64(experts) else {
                throw RepackError.configurationInvalid(
                    detail: "\(label): \(dtypeName ?? "\(tensor.dtype)") \(tensor.shape) "
                        + "is not \(experts) x \(bytes) bytes of \(sourceDType ?? "F32")")
            }
            cursor = Inkling2GGUFPlanner.roundUp(cursor, to: blobAlignment)
            slices.append(
                PerExpertTensorSlice(
                    role: role, component: component, dtype: code, logicalShape: shape,
                    offsetInExpertBlob: cursor, sizeInExpertBlob: bytes,
                    sourceOffsetPerExpert: bytes, sourceTensor: opaqueCopy(tensor),
                    bitsForWeights: bits))
            cursor += bytes
        }
        var matrices: [(role: String, rows: UInt64, columns: UInt64)] = []
        for role in ["gate_up", "down"] {
            guard let (tensor, _) = tensors[role], tensor.shape.count == 3 else {
                throw RepackError.configurationInvalid(
                    detail: "layer \(layer) \(role): expected [experts, rows, columns]")
            }
            let packed = tensor.shape[2]
            matrices.append((role, tensor.shape[1], encoding == .nvfp4 ? packed * 2 : packed))
        }
        for (role, rows, columns) in matrices {
            try add(
                role, role: role, component: "weights", code: encoding.residentDType.rawValue,
                shape: [rows, columns],
                bytes: try encoding.payloadBytes(rows: rows, columns: columns, field: role),
                sourceDType: encoding == .nvfp4 ? "U8" : "F8_E4M3", bits: encoding.weightBits)
        }
        for (role, rows, columns) in matrices {
            let perRow = columns / UInt64(encoding.valuesPerBlock)
            try add(
                role + "_scale", role: role, component: "scale", code: scaleCode,
                shape: [rows, perRow],
                bytes: try encoding.scaleBytes(rows: rows, columns: columns, field: role),
                sourceDType: scaleSource, bits: nil)
        }
        for (role, _, _) in matrices {
            try add(
                role + "_amax", role: role, component: "amax", code: 3, shape: [1], bytes: 4,
                sourceDType: nil, bits: nil)
        }
        return LayerFilePlan(
            layerIndex: layer,
            path: (outputDir as NSString).appendingPathComponent(
                String(format: "packed_experts/layer_%02d.bin", layer)),
            expertsPerLayer: experts,
            expertStride: Inkling2GGUFPlanner.roundUp(cursor, to: Layout.pageBytes),
            subTensors: slices)
    }
}
