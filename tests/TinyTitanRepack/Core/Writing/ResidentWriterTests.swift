import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore
import TinyTitanCore

/// The resident index stores each tensor name's length in a `UInt16`, and the
/// name comes from the source manifest. That makes an over-long name input
/// rather than an invariant: it has to be a report naming the tensor, not a
/// `precondition` trap inside the binary writer.
@Suite struct ResidentWriterTests {
    private func plan(name: String) -> ResidentFilePlan {
        let source = SourceTensor(
            name: name, shardPath: "shard.safetensors",
            dtype: .bf16, shape: [1], absoluteOffset: 0,
            sizeBytes: 2)
        let entry = ResidentEntry(
            name: name, dtype: 1, logicalShape4: [1, 0, 0, 0],
            fileOffset: 0, sizeBytes: 2,
            scaleOffset: 0, scaleSize: 0,
            biasOffset: 0, biasSize: 0,
            quantSpec: nil,
            sourceWeight: source,
            sourceScales: nil,
            sourceBiases: nil)
        let table = Array(name.utf8)
        return ResidentFilePlan(
            path: "/tmp/model_weights.bin",
            entries: [entry],
            stringTable: table,
            stringTableOffsets: [0],
            indexSize: UInt64(24 + SSDAIBinary.indexEntryBytes + table.count),
            residentSize: 2)
    }

    @Test func aNameWithinTheLimitEncodes() throws {
        let data = try ResidentWriter.encodeIndex(
            plan: plan(name: "model.layers.0.mlp.up_proj.weight"))
        #expect(
            data.count == 24 + SSDAIBinary.indexEntryBytes
                + "model.layers.0.mlp.up_proj.weight".utf8.count)
    }

    @Test func anOverLongNameIsReportedNotTrapped() {
        let name = String(repeating: "x", count: Int(UInt16.max) + 1)
        #expect(throws: RepackError.self) {
            _ = try ResidentWriter.encodeIndex(plan: plan(name: name))
        }
    }

    /// The boundary itself has to encode: a name of exactly `UInt16.max` bytes
    /// is what the field can hold, and refusing it would be a new limit.
    @Test func aNameOfExactlyTheLimitEncodes() throws {
        let name = String(repeating: "x", count: Int(UInt16.max))
        let data = try ResidentWriter.encodeIndex(plan: plan(name: name))
        #expect(data.count == 24 + SSDAIBinary.indexEntryBytes + Int(UInt16.max))
    }

    /// The index is the finished output, so its ceiling is the format's, not the
    /// per-worker staging budget it used to be checked against. With the old
    /// sub-1 MB bound a real install was refused here: KAT-Coder-V2.5-Dev's
    /// index is about 28 MB, which the v1 reader (`residentIndexMaxBytes`)
    /// accepts.
    @Test func theFormatsOwnCeilingIsTheBoundaryThatEncodes() throws {
        let limit = Int(SSDAIFormatV1.residentIndexMaxBytes)
        let data = try ResidentWriter.encodeIndex(plan: plan(name: "w", indexSize: limit))
        #expect(data.count == limit)
    }

    @Test func oneBytePastTheFormatsCeilingIsRefused() {
        let limit = Int(SSDAIFormatV1.residentIndexMaxBytes)
        #expect(throws: RepackError.self) {
            _ = try ResidentWriter.encodeIndex(plan: plan(name: "w", indexSize: limit + 1))
        }
    }

    /// A real index -- many entries with real names -- encodes at its own
    /// natural size. This is the direction the KAT install failed in.
    @Test func anIndexWithManyEntriesEncodesAtItsNaturalSize() throws {
        let count = 4_000
        let names = (0..<count).map {
            "model.language_model.layers.\($0 % 40).mlp.experts.\($0).down_proj.weight"
        }
        let expected =
            24 + count * SSDAIBinary.indexEntryBytes + names.reduce(0) { $0 + $1.utf8.count }
        let data = try ResidentWriter.encodeIndex(plan: plan(names: names))
        #expect(data.count == expected)
    }

    private func plan(names: [String]) -> ResidentFilePlan {
        let sources = names.map {
            SourceTensor(
                name: $0, shardPath: "shard.safetensors", dtype: .bf16,
                shape: [1], absoluteOffset: 0, sizeBytes: 2)
        }
        var table: [UInt8] = []
        var offsets: [UInt32] = []
        var entries: [ResidentEntry] = []
        for (index, name) in names.enumerated() {
            offsets.append(UInt32(table.count))
            table.append(contentsOf: name.utf8)
            entries.append(
                ResidentEntry(
                    name: name, dtype: 1, logicalShape4: [1, 0, 0, 0],
                    fileOffset: 0, sizeBytes: 2,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: sources[index],
                    sourceScales: nil,
                    sourceBiases: nil))
        }
        let natural = 24 + names.count * SSDAIBinary.indexEntryBytes + table.count
        return ResidentFilePlan(
            path: "/tmp/model_weights.bin",
            entries: entries,
            stringTable: table,
            stringTableOffsets: offsets,
            indexSize: UInt64(natural),
            residentSize: UInt64(names.count * 2))
    }

    private func plan(name: String, indexSize: Int? = nil) -> ResidentFilePlan {
        let source = SourceTensor(
            name: name, shardPath: "shard.safetensors",
            dtype: .bf16, shape: [1], absoluteOffset: 0,
            sizeBytes: 2)
        let entry = ResidentEntry(
            name: name, dtype: 1, logicalShape4: [1, 0, 0, 0],
            fileOffset: 0, sizeBytes: 2,
            scaleOffset: 0, scaleSize: 0,
            biasOffset: 0, biasSize: 0,
            quantSpec: nil,
            sourceWeight: source,
            sourceScales: nil,
            sourceBiases: nil)
        let table = Array(name.utf8)
        return ResidentFilePlan(
            path: "/tmp/model_weights.bin",
            entries: [entry],
            stringTable: table,
            stringTableOffsets: [0],
            indexSize: UInt64(
                indexSize
                    ?? (24 + SSDAIBinary.indexEntryBytes + table.count)),
            residentSize: 2)
    }
}
