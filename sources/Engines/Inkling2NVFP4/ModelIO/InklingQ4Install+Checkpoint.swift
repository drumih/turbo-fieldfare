import Foundation
import TinyTitanCore
import TinyTitanFormat

/// How one stored layer's routed experts sit in the file and in a slot (TT-125).
struct InklingNVFP4SlotPlan {
    enum Kind { case nvfp4, mxfp8 }

    struct Matrix {
        let rows: Int
        let columns: Int
        /// File offsets inside an expert's blob.
        let valuesOffset: Int
        let scalesOffset: Int
        let amaxOffset: Int?
        /// Offset of the repacked matrix inside a slot.
        let slotOffset: Int
        let slotBytes: Int
    }

    let layer: Int
    let kind: Kind
    let file: String
    /// Bytes from one expert to the next in the file, and in a slot.
    let fileStride: Int
    let slotStride: Int
    let gateUp: Matrix
    let down: Matrix
    /// NVFP4: the per-layer `outer_scale_denominator` of `gate_up` and `down`;
    /// an expert's scale is its amax divided by it.
    let gateUpDenominator: Float?
    let downDenominator: Float?
}

/// A GGUF-named role whose bytes are built at load from a checkpoint tensor.
struct InklingNVFP4Conversion {
    let source: ResidentIndexEntry
    let transform: InklingNVFP4TensorMap.Transform
}

extension InklingQ4Install {
    /// Every GGUF-named role the graph may ask for, for stored layer `layer`.
    static func graphNames(layer: Int) -> [String] {
        let suffixes =
            Array(InklingNVFP4TensorMap.layerNames.keys) + [
                "attn_rel_bias.weight", "ffn_res_query.weight", "attn_res_query.weight",
                "attn_k_conv.weight", "attn_v_conv.weight",
            ]
            + ["attn", "k", "v", "mlp"].flatMap { module in
                ["down", "forget", "gain"].map { "loop_\(module)_\($0).weight" }
            }
        return suffixes.map { "blk.\(layer).\($0)" }
    }

    /// The checkpoint's resident index seen through the graph's names: an entry
    /// per role the checkpoint can supply. Roles bound in place keep the
    /// checkpoint's bytes and dtype; converted roles are F32 entries whose bytes
    /// `InklingQ4Weights` builds from `conversions`.
    static func graphView(
        checkpoint: ResidentIndex, layers: Int
    ) throws -> (ResidentIndex, [String: InklingNVFP4Conversion]) {
        var entries: [String: ResidentIndexEntry] = [:]
        var conversions: [String: InklingNVFP4Conversion] = [:]
        let names =
            Array(InklingNVFP4TensorMap.globals.keys) + (0..<layers).flatMap(graphNames)
        for name in names {
            guard
                let source = InklingNVFP4TensorMap.source(
                    for: name, exists: { checkpoint.entries[$0] != nil }),
                let entry = checkpoint.entries[source.name]
            else { continue }
            let dims = [entry.shape.0, entry.shape.1, entry.shape.2, entry.shape.3]
                .filter { $0 != 0 }
            // The graph reads (rows, columns); a convolution kernel stored with a
            // middle axis of one ([channels][1][taps]) reads as [channels][taps].
            var shape = dims.filter { $0 != 1 || dims.count == 1 }
            switch source.transform {
            case .identity: break
            case .transpose, .transposeReversed: shape.reverse()
            }
            while shape.count < 4 { shape.append(0) }
            if source.converted {
                let count = dims.reduce(UInt64(1)) { $0 * UInt64($1) }
                entries[name] = ResidentIndexEntry(
                    name: name, dtype: 3, fileOffset: 0, sizeBytes: count * 4,
                    shape: (shape[0], shape[1], shape[2], shape[3]), scaleOffset: 0,
                    scaleSize: 0, biasOffset: 0, biasSize: 0)
                conversions[name] = InklingNVFP4Conversion(
                    source: entry, transform: source.transform)
            } else {
                entries[name] = ResidentIndexEntry(
                    name: name, dtype: entry.dtype, fileOffset: entry.fileOffset,
                    sizeBytes: entry.sizeBytes, shape: (shape[0], shape[1], shape[2], shape[3]),
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
            }
        }
        return (ResidentIndex(header: checkpoint.header, entries: entries), conversions)
    }

    /// The slot plan of every stored layer, from the checkpoint's packed layout
    /// and the per-layer scale denominators in its resident index.
    static func slotPlans(
        layout: SSDAIPackedExpertsLayoutV1, checkpoint: ResidentIndex, residentFile: URL
    ) throws -> [InklingNVFP4SlotPlan] {
        try layout.layers.sorted { $0.layer < $1.layer }.map { layer in
            guard let first = layer.experts.first,
                let gateUp = first.tensors["gate_up"], let down = first.tensors["down"],
                let gateUpScales = first.tensors["gate_up_scale"],
                let downScales = first.tensors["down_scale"]
            else { throw Error.layout("layer \(layer.layer) has no gate_up/down with scales") }
            let fileStride =
                Int(layer.experts.count > 1 ? layer.experts[1].offset - first.offset : first.size)
            let kind: InklingNVFP4SlotPlan.Kind
            switch gateUp.dtype {
            case "NVFP4": kind = .nvfp4
            case "MXFP8": kind = .mxfp8
            default: throw Error.layout("layer \(layer.layer) experts are \(gateUp.dtype)")
            }
            func bytes(_ rows: Int, _ columns: Int) -> Int {
                kind == .nvfp4
                    ? InklingNVFP4Repack.nvfp4Bytes(rows: rows, columns: columns)
                    : InklingNVFP4Repack.mxfp8Bytes(rows: rows, columns: columns)
            }
            let gRows = Int(gateUp.shape[0]), gColumns = Int(gateUp.shape[1])
            let dRows = Int(down.shape[0]), dColumns = Int(down.shape[1])
            let gateUpBytes = bytes(gRows, gColumns)
            let downBytes = bytes(dRows, dColumns)
            // Slots keep the file's 16 KiB-aligned stride when the repacked
            // matrices fit in it (NVFP4: the same bytes rearranged).
            let needed = gateUpBytes + downBytes
            let slotStride = max(fileStride, (needed + 16_383) / 16_384 * 16_384)
            let prefix = "model.llm.layers.\(layer.layer).mlp.experts."
            func denominator(_ key: String) throws -> Float? {
                guard kind == .nvfp4 else { return nil }
                guard let entry = checkpoint.entries[prefix + key], entry.sizeBytes == 4,
                    entry.dtype == 3
                else { throw Error.layout("layer \(layer.layer) has no \(key)") }
                let handle = try FileHandle(forReadingFrom: residentFile)
                defer { try? handle.close() }
                try handle.seek(toOffset: entry.fileOffset)
                guard let data = try handle.read(upToCount: 4), data.count == 4 else {
                    throw Error.layout("layer \(layer.layer) \(key) is unreadable")
                }
                return data.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
            }
            return InklingNVFP4SlotPlan(
                layer: layer.layer, kind: kind, file: layer.file, fileStride: fileStride,
                slotStride: slotStride,
                gateUp: .init(
                    rows: gRows, columns: gColumns, valuesOffset: Int(gateUp.offset),
                    scalesOffset: Int(gateUpScales.offset),
                    amaxOffset: first.tensors["gate_up_amax"].map { Int($0.offset) },
                    slotOffset: 0, slotBytes: gateUpBytes),
                down: .init(
                    rows: dRows, columns: dColumns, valuesOffset: Int(down.offset),
                    scalesOffset: Int(downScales.offset),
                    amaxOffset: first.tensors["down_amax"].map { Int($0.offset) },
                    slotOffset: gateUpBytes, slotBytes: downBytes),
                gateUpDenominator: try denominator("w13_weight_outer_scale_denominator"),
                downDenominator: try denominator("w2_weight_outer_scale_denominator"))
        }
    }
}
