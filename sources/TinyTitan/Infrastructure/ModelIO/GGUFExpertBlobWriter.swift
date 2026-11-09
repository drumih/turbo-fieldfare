import Foundation
import TinyTitanFormat

package enum GGUFExpertBlobError: Error, Equatable, CustomStringConvertible {
    case unsupportedBits(Int)
    case expertCountOutOfRange(requested: Int, available: Int)
    case shapeMismatch(name: String, rows: UInt64, columns: UInt64, elements: UInt64)
    case rowNotGroupAligned(name: String, columns: UInt64, group: Int)
    case expertStrideNotConstant(first: UInt64, expert: Int, got: UInt64)
    case writeFailed(path: String, errno: Int32)
    case fileSizeMismatch(path: String, expected: UInt64, actual: UInt64)

    package var description: String {
        switch self {
        case .unsupportedBits(let bits):
            return "GGUF expert blobs are written at 4 bits; \(bits) is not supported yet"
        case .expertCountOutOfRange(let requested, let available):
            return "asked for \(requested) experts, the layer has \(available)"
        case .shapeMismatch(let name, let rows, let columns, let elements):
            return
                "\(name) is \(rows)x\(columns) = \(rows * columns) weights, but decodes to \(elements)"
        case .rowNotGroupAligned(let name, let columns, let group):
            return "\(name) rows are \(columns) wide, not a multiple of the \(group)-wide group"
        case .expertStrideNotConstant(let first, let expert, let got):
            return
                "expert \(expert) is \(got) bytes, expert 0 is \(first); the layout needs one stride"
        case .writeFailed(let path, let code):
            return "cannot write \(path): errno \(code)"
        case .fileSizeMismatch(let path, let expected, let actual):
            return "\(path) is \(actual) bytes, expected \(expected)"
        }
    }
}

package struct GGUFExpertLayerBlob: Sendable {
    package let layer: SSDAILayerV1
    package let expertStride: UInt64
    package let fileURL: URL
}

/// Writes one layer of routed experts as a raw `.ssdai` packed-expert blob.
///
/// The blob contract is the runtime's, not a new one: per role the weights are
/// 4-bit affine packed into U32 words (`dtype: "U32"`, `shape: [rows, columns]`),
/// followed by BF16 scales and biases shaped `[rows, columns / 64]`, and the
/// sub-tensor offsets are relative to the expert's own start
/// (`Model+SchemaValidation.swift` is the reader that enforces this).
///
/// This is the writer half of the conversion route: it takes the GGUF's IQ
/// bytes, decodes them and re-quantises with the engine's own affine quantiser,
/// so the payload is exactly what the kernels already read. It writes the layer
/// file and returns the layout record; assembling `layout.json` and the manifest
/// is the caller's job, and is deliberately separate so the blob can be tested
/// without a whole install.
package enum GGUFExpertBlobWriter {
    package static let groupSize = Quantization.groupSize

    package static func writeLayer(
        layer: Int,
        from model: GGUFModel,
        expertCount: Int,
        into directory: URL,
        fileName: String,
        bits: Int = 4
    ) throws -> GGUFExpertLayerBlob {
        guard bits == 4 else { throw GGUFExpertBlobError.unsupportedBits(bits) }
        let available = try GGUFExpertLayout.expertsPerLayer(in: model)
        guard expertCount > 0, expertCount <= available else {
            throw GGUFExpertBlobError.expertCountOutOfRange(
                requested: expertCount, available: available)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(fileName)
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw GGUFExpertBlobError.writeFailed(path: url.path, errno: errno)
        }
        guard let handle = FileHandle(forWritingAtPath: url.path) else {
            throw GGUFExpertBlobError.writeFailed(path: url.path, errno: errno)
        }
        defer { try? handle.close() }
        let reader = try ShardReader(model: model)
        defer { reader.close() }

        var experts: [SSDAIExpertV1] = []
        experts.reserveCapacity(expertCount)
        var expectedSize: UInt64 = 0
        var stride: UInt64?
        for expert in 0..<expertCount {
            let blob = try serialize(
                layer: layer, expert: expert, model: model, reader: reader)
            var bytes = blob.bytes
            // `.ssdai` aligns every expert blob to alignmentBytes (16 KiB), so
            // the stride is the padded size; the sub-tensor extents stay inside
            // their unpadded range.
            let alignment = Int(SSDAIFormatV1.alignmentBytes)
            let padded = (bytes.count + alignment - 1) / alignment * alignment
            if padded != bytes.count {
                bytes.append(contentsOf: repeatElement(0, count: padded - bytes.count))
            }
            if let stride, stride != UInt64(bytes.count) {
                throw GGUFExpertBlobError.expertStrideNotConstant(
                    first: stride, expert: expert, got: UInt64(bytes.count))
            }
            stride = UInt64(bytes.count)
            expectedSize += UInt64(bytes.count)
            try write(Data(bytes), to: handle, path: url.path)
            experts.append(
                SSDAIExpertV1(
                    expert: expert,
                    physicalRank: expert,
                    offset: UInt64(expert) * UInt64(bytes.count),
                    size: UInt64(bytes.count),
                    tensors: blob.tensors))
        }

        let actual = try handle.seekToEnd()
        guard actual == expectedSize else {
            throw GGUFExpertBlobError.fileSizeMismatch(
                path: url.path, expected: expectedSize, actual: actual)
        }
        return GGUFExpertLayerBlob(
            layer: SSDAILayerV1(layer: layer, file: fileName, experts: experts),
            expertStride: stride ?? 0,
            fileURL: url)
    }

    /// Encode a layer list as `packed_experts/layout.json`, then decode it back
    /// through the format's own validator so a document the runtime would refuse
    /// cannot be written by this code.
    package static func writeLayout(
        layers: [SSDAILayerV1],
        expertStride: UInt64,
        numLayers: Int,
        expertsPerLayer: Int,
        into directory: URL
    ) throws {
        let layout = SSDAIPackedExpertsLayoutV1(
            expertStride: expertStride,
            numLayers: numLayers,
            expertsPerLayer: expertsPerLayer,
            compression: nil,
            layers: layers)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(layout)
        _ = try SSDAIPackedExpertsLayoutCodec.decode(data)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: directory.appendingPathComponent("layout.json"))
    }

    private struct Blob {
        let bytes: [UInt8]
        let tensors: [String: SSDAISubTensorV1]
    }

    private struct RoleBlob {
        var packed: [UInt8] = []
        var scales: [UInt8] = []
        var biases: [UInt8] = []
    }

    private static func serialize(
        layer: Int, expert: Int, model: GGUFModel, reader: ShardReader
    ) throws -> Blob {
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: expert, in: model)
        let roles: [(String, GGUFTensorRange, String)] = [
            ("gate", weights.gate, "ffn_gate_exps.weight"),
            ("up", weights.up, "ffn_up_exps.weight"),
            ("down", weights.down, "ffn_down_exps.weight"),
        ]
        var bytes: [UInt8] = []
        var tensors: [String: SSDAISubTensorV1] = [:]
        for (role, range, suffix) in roles {
            let name = "blk.\(layer).\(suffix)"
            guard let location = model.tensor(named: name), location.tensor.dimensions.count == 3
            else {
                throw GGUFExpertBlobError.shapeMismatch(
                    name: name, rows: 0, columns: 0, elements: 0)
            }
            // GGUF dims are [columns, rows, experts]; the layout is [rows, columns].
            let columns = location.tensor.dimensions[0]
            let rows = location.tensor.dimensions[1]
            guard columns % UInt64(groupSize) == 0 else {
                throw GGUFExpertBlobError.rowNotGroupAligned(
                    name: name, columns: columns, group: groupSize)
            }
            let data = try reader.read(range: range)
            let values = try GGUFDequant.decode(type: range.type, bytes: data)
            guard UInt64(values.count) == rows * columns else {
                throw GGUFExpertBlobError.shapeMismatch(
                    name: name, rows: rows, columns: columns, elements: UInt64(values.count))
            }
            let blob = try quantize(values: values, rows: Int(rows), columns: Int(columns))

            // Weights first, then scales, then biases, each at its required
            // alignment: U32 words for the nibbles, 2 bytes for BF16.
            let packedOffset = UInt64(bytes.count)
            bytes.append(contentsOf: blob.packed)
            align(&bytes, to: 4)
            let scalesOffset = UInt64(bytes.count)
            bytes.append(contentsOf: blob.scales)
            align(&bytes, to: 2)
            let biasesOffset = UInt64(bytes.count)
            bytes.append(contentsOf: blob.biases)
            align(&bytes, to: 2)

            let groups = columns / UInt64(groupSize)
            tensors[role] = SSDAISubTensorV1(
                offset: packedOffset,
                size: UInt64(blob.packed.count),
                dtype: "U32",
                shape: [UInt32(rows), UInt32(columns)],
                bits: 4)
            tensors["\(role)_scales"] = SSDAISubTensorV1(
                offset: scalesOffset,
                size: UInt64(blob.scales.count),
                dtype: "BF16",
                shape: [UInt32(rows), UInt32(groups)],
                bits: nil)
            tensors["\(role)_biases"] = SSDAISubTensorV1(
                offset: biasesOffset,
                size: UInt64(blob.biases.count),
                dtype: "BF16",
                shape: [UInt32(rows), UInt32(groups)],
                bits: nil)
        }
        return Blob(bytes: bytes, tensors: tensors)
    }

    private static func quantize(
        values: [Float], rows: Int, columns: Int
    ) throws -> RoleBlob {
        var blob = RoleBlob()
        blob.packed.reserveCapacity(values.count / 2)
        let groups = columns / groupSize
        blob.scales.reserveCapacity(rows * groups * 2)
        blob.biases.reserveCapacity(rows * groups * 2)
        for row in 0..<rows {
            let start = row * columns
            let quantized = Quantization.quantizeInt4Affine(
                Array(values[start..<(start + columns)]))
            blob.packed.append(contentsOf: quantized.packed)
            for value in quantized.scales { appendLittleEndian(value, to: &blob.scales) }
            for value in quantized.biases { appendLittleEndian(value, to: &blob.biases) }
        }
        return blob
    }

    private static func appendLittleEndian(_ value: UInt16, to bytes: inout [UInt8]) {
        bytes.append(UInt8(truncatingIfNeeded: value))
        bytes.append(UInt8(truncatingIfNeeded: value >> 8))
    }

    private static func align(_ bytes: inout [UInt8], to alignment: Int) {
        let remainder = bytes.count % alignment
        if remainder != 0 {
            bytes.append(contentsOf: repeatElement(0, count: alignment - remainder))
        }
    }

    private static func write(_ data: Data, to handle: FileHandle, path: String) throws {
        do {
            try handle.write(contentsOf: data)
        } catch {
            throw GGUFExpertBlobError.writeFailed(path: path, errno: errno)
        }
    }

    /// One open handle per shard, so a full 512-expert layer does not reopen the
    /// same 50 GB file four times per expert.
    private final class ShardReader {
        private let model: GGUFModel
        private var handles: [Int: FileHandle] = [:]

        init(model: GGUFModel) throws {
            self.model = model
        }

        func read(range: GGUFTensorRange) throws -> [UInt8] {
            let handle: FileHandle
            if let existing = handles[range.shardIndex] {
                handle = existing
            } else {
                let opened = try FileHandle(
                    forReadingFrom: model.shards[range.shardIndex].url)
                handles[range.shardIndex] = opened
                handle = opened
            }
            try handle.seek(toOffset: range.fileOffset)
            guard let data = try handle.read(upToCount: Int(range.byteCount)),
                data.count == Int(range.byteCount)
            else {
                throw GGUFExpertBlobError.writeFailed(
                    path: model.shards[range.shardIndex].url.path, errno: EIO)
            }
            return [UInt8](data)
        }

        func close() {
            for handle in handles.values { try? handle.close() }
            handles.removeAll()
        }
    }
}
