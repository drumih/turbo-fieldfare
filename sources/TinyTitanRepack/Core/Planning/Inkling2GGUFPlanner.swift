import Foundation
import TinyTitanFormat

/// Plans an Inkling 2 Mini `.ssdai` install from its GGUF (TT-121), keeping
/// every tensor in the encoding the GGUF stores it in: the routed experts stay
/// NVFP4 (Q8_0 in the last four layers), everything else stays Q8_0, F32 or
/// BF16. Nothing is requantized, so the install is the source's bytes
/// rearranged for streaming, and the plan is what the range-copy machinery
/// executes chunk by chunk.
///
/// - Resident: every non-expert tensor, in file order, each at a 64-byte
///   aligned offset in `model_weights.bin`, under its GGUF name.
/// - Experts: one blob per expert per layer — `gate_up`, `down`, and their
///   per-expert F32 scales when the layer has them — at a 16 KiB stride. NVFP4
///   and Q8_0 layers have different strides, so the layers are compact inside
///   the largest one.
/// - The GGUF header (metadata, tokenizer, chat template, tensor table) is
///   copied verbatim as `inkling2_gguf_header.bin`, so the variant builds its
///   tokenizer from the same bytes llama.cpp does.
/// - `arch.variantGeometry` carries the loop, relative-attention, controller and
///   activation constants, read from the GGUF's own metadata.
enum Inkling2GGUFPlanner {
    struct Source {
        /// The GGUF's file name, the shard id the copy plan reads.
        let fileName: String
        let header: GGUFHeaderV3
        let fileSize: UInt64
    }

    static let headerSidecarName = "inkling2_gguf_header.bin"
    static let residentAlignment: UInt64 = 64
    static let blobAlignment: UInt64 = 64

    /// How a ggml tensor type is kept: its index dtype code, and its block
    /// encoding when it has one (`nil` for plain element types).
    struct Kept {
        let code: UInt8
        let block: SSDAIBlockEncodingV1?
        let elementBytes: UInt64
        let bits: Int
    }

    static func kept(ggmlType: UInt32, tensor name: String) throws -> Kept {
        switch ggmlType {
        case 0: return Kept(code: 3, block: nil, elementBytes: 4, bits: 32)
        case 1: return Kept(code: 2, block: nil, elementBytes: 2, bits: 16)
        case 30: return Kept(code: 1, block: nil, elementBytes: 2, bits: 16)
        case 8: return Kept(code: 4, block: .ggufQ8_0, elementBytes: 0, bits: 8)
        case 40: return Kept(code: 5, block: .ggufNVFP4, elementBytes: 0, bits: 4)
        default:
            throw RepackError.configurationInvalid(
                detail:
                    "\(name): ggml type \(ggmlType) is not an encoding the Inkling install keeps")
        }
    }

    /// Bytes of a `rows x columns` matrix (ggml order: `dimensions[0]` is the
    /// row length; every later dimension multiplies into the rows).
    static func bytes(of dimensions: [UInt64], kept: Kept, tensor name: String) throws -> UInt64 {
        guard let columns = dimensions.first else {
            throw RepackError.configurationInvalid(detail: "\(name): no dimensions")
        }
        let rows = dimensions.dropFirst().reduce(UInt64(1), *)
        if let block = kept.block {
            return try block.payloadBytes(rows: rows, columns: columns, field: name)
        }
        return rows * columns * kept.elementBytes
    }

    /// `blk.<layer>.ffn_{gate_up,down}_exps.{weight,scale}` → (layer, sub-tensor key).
    static func expertKey(_ name: String) -> (layer: Int, key: String)? {
        let parts = name.split(separator: ".")
        guard parts.count == 4, parts[0] == "blk", let layer = Int(parts[1]) else { return nil }
        let role: String
        switch parts[2] {
        case "ffn_gate_up_exps": role = "gate_up"
        case "ffn_down_exps": role = "down"
        default: return nil
        }
        switch parts[3] {
        case "weight": return (layer, role)
        case "scale": return (layer, role + "_scale")
        default: return nil
        }
    }

    static func plan(source: Source, outputDir: String) throws -> RepackPlan {
        let header = source.header
        guard header.string("general.architecture") == "inkling2" else {
            throw RepackError.configurationInvalid(
                detail: "\(source.fileName) is not an Inkling 2 GGUF (architecture "
                    + "\(header.string("general.architecture") ?? "none"))")
        }
        let arch = try Inkling2GGUFArch.archInfo(header)
        var residentTensors: [GGUFHeaderV3.Tensor] = []
        var expertTensors: [Int: [String: GGUFHeaderV3.Tensor]] = [:]
        for tensor in header.tensors {
            if let key = expertKey(tensor.name) {
                expertTensors[key.layer, default: [:]][key.key] = tensor
            } else {
                residentTensors.append(tensor)
            }
        }
        let resident = try planResident(
            residentTensors.sorted { $0.absoluteOffset < $1.absoluteOffset },
            source: source, outputDir: outputDir)
        var layers: [LayerFilePlan] = []
        for layer in 0..<arch.numLayers {
            guard let tensors = expertTensors[layer] else {
                throw RepackError.configurationInvalid(
                    detail: "layer \(layer) has no routed experts")
            }
            layers.append(
                try planLayer(
                    layer, tensors: tensors, experts: arch.numExperts,
                    source: source, outputDir: outputDir))
        }
        var plan = RepackPlan(
            arch: arch, baseMode: SSDAIBlockEncodingV1.ggufQ8_0.rawValue, baseGroupSize: 32,
            resident: resident, layers: layers, matchedModelID: nil,
            excludedMultimodalTensorNames: [],
            passthroughFiles: [
                PassthroughFile(
                    sourceName: source.fileName, destinationName: headerSidecarName,
                    size: header.dataStart, required: true)
            ])
        plan.slotSchemes = ["routedExpert": SSDAIBlockEncodingV1.ggufNVFP4.rawValue]
        return plan
    }

    private static func sourceTensor(
        _ tensor: GGUFHeaderV3.Tensor, size: UInt64, source: Source
    ) throws -> SourceTensor {
        let (end, overflow) = tensor.absoluteOffset.addingReportingOverflow(size)
        guard !overflow, end <= source.fileSize else {
            throw RepackError.configurationInvalid(
                detail: "\(tensor.name) ends at \(end), past the \(source.fileSize)-byte file")
        }
        return SourceTensor(
            name: tensor.name, shardPath: source.fileName, dtype: .opaque,
            shape: tensor.dimensions.reversed(), absoluteOffset: tensor.absoluteOffset,
            sizeBytes: size)
    }

    private static func planResident(
        _ tensors: [GGUFHeaderV3.Tensor], source: Source, outputDir: String
    ) throws -> ResidentFilePlan {
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        for tensor in tensors {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: tensor.name.utf8)
        }
        let rawIndex = UInt64(
            SSDAIBinary.indexHeaderBytes + tensors.count * SSDAIBinary.indexEntryBytes
                + stringTable.count)
        let indexSize = roundUp(rawIndex, to: Layout.pageBytes)
        var cursor = indexSize
        var entries: [ResidentEntry] = []
        for tensor in tensors {
            let kept = try kept(ggmlType: tensor.ggmlType, tensor: tensor.name)
            let size = try bytes(of: tensor.dimensions, kept: kept, tensor: tensor.name)
            guard tensor.dimensions.count <= 4 else {
                throw RepackError.configurationInvalid(detail: "\(tensor.name): rank above 4")
            }
            cursor = roundUp(cursor, to: residentAlignment)
            let shape = tensor.dimensions.reversed().map { UInt32($0) }
            entries.append(
                ResidentEntry(
                    name: tensor.name, dtype: kept.code,
                    logicalShape4: shape + [UInt32](repeating: 0, count: 4 - shape.count),
                    fileOffset: cursor, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: try sourceTensor(tensor, size: size, source: source),
                    sourceScales: nil, sourceBiases: nil))
            cursor += size
        }
        return ResidentFilePlan(
            path: (outputDir as NSString).appendingPathComponent("model_weights.bin"),
            entries: entries, stringTable: stringTable, stringTableOffsets: offsets,
            indexSize: indexSize, residentSize: cursor - indexSize)
    }

    private static func planLayer(
        _ layer: Int, tensors: [String: GGUFHeaderV3.Tensor], experts: Int,
        source: Source, outputDir: String
    ) throws -> LayerFilePlan {
        var slices: [PerExpertTensorSlice] = []
        var cursor: UInt64 = 0
        // Weights first, then their scales, each 64-byte aligned in the blob.
        for key in ["gate_up", "down", "gate_up_scale", "down_scale"] {
            guard let tensor = tensors[key] else {
                if key.hasSuffix("_scale") { continue }
                throw RepackError.configurationInvalid(detail: "layer \(layer) has no \(key)")
            }
            guard tensor.dimensions.last == UInt64(experts) else {
                throw RepackError.configurationInvalid(
                    detail:
                        "\(tensor.name): \(tensor.dimensions) does not end in \(experts) experts")
            }
            let kept = try kept(ggmlType: tensor.ggmlType, tensor: tensor.name)
            // A per-expert scale is one value per expert: `[experts]` leaves a
            // one-element tensor, not an empty shape.
            let perExpertDimensions =
                tensor.dimensions.count == 1 ? [UInt64(1)] : Array(tensor.dimensions.dropLast())
            let perExpert = try bytes(of: perExpertDimensions, kept: kept, tensor: tensor.name)
            let isScale = key.hasSuffix("_scale")
            cursor = roundUp(cursor, to: blobAlignment)
            slices.append(
                PerExpertTensorSlice(
                    role: isScale ? String(key.dropLast("_scale".count)) : key,
                    component: isScale ? "scale" : "weights",
                    dtype: kept.code,
                    logicalShape: perExpertDimensions.reversed(),
                    offsetInExpertBlob: cursor, sizeInExpertBlob: perExpert,
                    sourceOffsetPerExpert: perExpert,
                    sourceTensor: try sourceTensor(
                        tensor, size: perExpert * UInt64(experts), source: source),
                    bitsForWeights: isScale ? nil : kept.bits))
            cursor += perExpert
        }
        return LayerFilePlan(
            layerIndex: layer,
            path: (outputDir as NSString).appendingPathComponent(
                String(format: "packed_experts/layer_%02d.bin", layer)),
            expertsPerLayer: experts,
            expertStride: roundUp(cursor, to: Layout.pageBytes),
            subTensors: slices)
    }

    static func roundUp(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        (value + alignment - 1) / alignment * alignment
    }
}
