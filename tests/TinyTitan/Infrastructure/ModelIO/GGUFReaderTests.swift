import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// A minimal GGUF v3 writer, so the reader is tested against bytes rather than
/// against itself. Every value below is appended in the wire form the format
/// defines, which is what lets a wrong reader pass: the fixture must be built
/// from the specification, not from the parser's own helpers.
private struct GGUFWriter {
    var version: UInt32 = 3
    var alignment: UInt32 = 32
    var metadata: [(key: String, type: UInt32, bytes: [UInt8])] = []
    var tensors: [(name: String, dims: [UInt64], type: UInt32, offset: UInt64)] = []
    var dataSize: UInt64 = 0
    /// A metadata-only shard may end at its directory, unaligned and with no
    /// data section; every shard that carries tensors pads to the alignment.
    var padsDataSection = true

    mutating func add(_ key: String, _ type: UInt32, _ bytes: [UInt8]) {
        metadata.append((key, type, bytes))
    }

    mutating func addU8(_ key: String, _ value: UInt8) { add(key, 0, [value]) }
    mutating func addI8(_ key: String, _ value: Int8) { add(key, 1, le(UInt8(bitPattern: value))) }
    mutating func addU16(_ key: String, _ value: UInt16) { add(key, 2, le(value)) }
    mutating func addI16(_ key: String, _ value: Int16) {
        add(key, 3, le(UInt16(bitPattern: value)))
    }
    mutating func addU32(_ key: String, _ value: UInt32) { add(key, 4, le(value)) }
    mutating func addI32(_ key: String, _ value: Int32) {
        add(key, 5, le(UInt32(bitPattern: value)))
    }
    mutating func addF32(_ key: String, _ value: Float) { add(key, 6, le(value.bitPattern)) }
    mutating func addBool(_ key: String, _ value: Bool) { add(key, 7, [value ? 1 : 0]) }
    mutating func addString(_ key: String, _ value: String) { add(key, 8, stringBytes(value)) }
    mutating func addU64(_ key: String, _ value: UInt64) { add(key, 10, le(value)) }
    mutating func addI64(_ key: String, _ value: Int64) {
        add(key, 11, le(UInt64(bitPattern: value)))
    }
    mutating func addF64(_ key: String, _ value: Double) { add(key, 12, le(value.bitPattern)) }

    mutating func addStringArray(_ key: String, _ values: [String]) {
        var bytes = le(UInt32(8)) + le(UInt64(values.count))
        for value in values { bytes += stringBytes(value) }
        add(key, 9, bytes)
    }

    /// An array whose single element is itself an array of three `u32`s.
    mutating func addNestedArray(_ key: String) {
        var inner = le(UInt32(4)) + le(UInt64(3)) + le(UInt32(1)) + le(UInt32(2)) + le(UInt32(3))
        inner = le(UInt32(9)) + le(UInt64(1)) + inner
        add(key, 9, inner)
    }

    mutating func write(to url: URL) throws {
        var out = Data("GGUF".utf8)
        out.append(contentsOf: le(version))
        out.append(contentsOf: le(UInt64(tensors.count)))
        out.append(contentsOf: le(UInt64(metadata.count)))
        for entry in metadata {
            out.append(contentsOf: stringBytes(entry.key))
            out.append(contentsOf: le(entry.type))
            out.append(contentsOf: entry.bytes)
        }
        for tensor in tensors {
            out.append(contentsOf: stringBytes(tensor.name))
            out.append(contentsOf: le(UInt32(tensor.dims.count)))
            for dimension in tensor.dims { out.append(contentsOf: le(dimension)) }
            out.append(contentsOf: le(tensor.type))
            out.append(contentsOf: le(tensor.offset))
        }
        let remainder = UInt32(out.count % Int(alignment))
        let padding = remainder == 0 || !padsDataSection ? 0 : Int(alignment - remainder)
        out.append(Data(count: padding))
        out.append(Data(count: Int(dataSize)))
        try out.write(to: url)
    }

    private func stringBytes(_ value: String) -> [UInt8] {
        let utf8 = Array(value.utf8)
        return le(UInt64(utf8.count)) + utf8
    }

    private func le<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian) { Array($0) }
    }
}

@Suite struct GGUFReaderTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static func tempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-reader-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func standardTensors() -> [(String, [UInt64], UInt32, UInt64)] {
        [
            ("toy.q6k", [256, 4], GGUFType.q6K.rawValue, 0),  // 4 blocks * 210 = 840
            ("toy.q2_0", [64, 2], GGUFType.q2_0.rawValue, 864),  // 2 blocks * 18 = 36
            ("toy.f32", [10], GGUFType.f32.rawValue, 896),  // 40 bytes
        ]
    }

    /// The directory is the contract every later component reads, so this pins
    /// all thirteen metadata wire types, the block geometry of three dissimilar
    /// tensor types, alignment, and `dataOffset` in one file.
    @Test func parsesHeaderMetadataAndTensorGeometry() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var writer = GGUFWriter()
        writer.addString("general.architecture", "qwen4exp")
        writer.addU32("general.alignment", 32)
        writer.addU8("toy.u8", 200)
        writer.addI8("toy.i8", -7)
        writer.addU16("toy.u16", 65_000)
        writer.addI16("toy.i16", -12_000)
        writer.addU32("toy.u32", 4_000_000_000)
        writer.addI32("toy.i32", -2_000_000_000)
        writer.addF32("toy.f32", 1.5)
        writer.addBool("toy.bool", true)
        writer.addString("toy.string", "hello")
        writer.addStringArray("toy.tags", ["a", "b"])
        writer.addNestedArray("toy.nested")
        writer.addU64("toy.u64", 18_000_000_000)
        writer.addI64("toy.i64", -18_000_000_000)
        writer.addF64("toy.f64", 2.25)
        writer.tensors = Self.standardTensors()
        writer.dataSize = 936
        let url = directory.appendingPathComponent("toy.gguf")
        try writer.write(to: url)

        let shard = try GGUFReader.read(shardAt: url)
        #expect(shard.version == 3)
        #expect(shard.alignment == 32)
        #expect(shard.tensors.count == 3)
        #expect(shard.metadata["toy.u8"]?.integer == 200)
        #expect(shard.metadata["toy.i8"]?.integer == -7)
        #expect(shard.metadata["toy.u16"]?.integer == 65_000)
        #expect(shard.metadata["toy.i16"]?.integer == -12_000)
        #expect(shard.metadata["toy.u32"]?.integer == 4_000_000_000)
        #expect(shard.metadata["toy.i32"]?.integer == -2_000_000_000)
        #expect(shard.metadata["toy.f32"]?.double == 1.5)
        #expect(shard.metadata["toy.bool"]?.boolean == true)
        #expect(shard.metadata["toy.string"]?.string == "hello")
        #expect(shard.metadata["toy.tags"]?.strings == ["a", "b"])
        #expect(shard.metadata["toy.nested"]?.array?.count == 1)
        #expect(shard.metadata["toy.nested"]?.array?[0].array?.count == 3)
        #expect(shard.metadata["toy.u64"]?.double == 18_000_000_000)
        #expect(shard.metadata["toy.i64"]?.integer == -18_000_000_000)
        #expect(shard.metadata["toy.f64"]?.double == 2.25)

        let q6k = try #require(shard.tensor(named: "toy.q6k"))
        #expect(q6k.elementCount == 1_024)
        #expect(q6k.byteCount == 840)
        #expect(q6k.type.name == "Q6_K")
        let q2 = try #require(shard.tensor(named: "toy.q2_0"))
        #expect(q2.elementCount == 128)
        #expect(q2.byteCount == 36)
        let f32 = try #require(shard.tensor(named: "toy.f32"))
        #expect(f32.byteCount == 40)
        // Header is 24 + metadata + tensors; the data section starts aligned.
        #expect(shard.dataOffset % 32 == 0)
        #expect(shard.dataOffset + 936 == UInt64(try Data(contentsOf: url).count))
    }

    /// The metadata shard of a real split set ends at its directory, unaligned,
    /// with no data section; refusing it would refuse every Unsloth split.
    @Test func acceptsMetadataOnlyShardWithoutADataSection() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var writer = GGUFWriter()
        writer.padsDataSection = false
        writer.addString("general.architecture", "toy")
        writer.addU32("split.no", 0)
        writer.addU32("split.count", 2)
        let url = directory.appendingPathComponent("toy-00001-of-00002.gguf")
        try writer.write(to: url)
        let size = UInt64(try Data(contentsOf: url).count)
        #expect(size % 32 != 0)
        let shard = try GGUFReader.read(shardAt: url)
        #expect(shard.tensors.isEmpty)
        #expect(shard.dataOffset > size)
    }

    @Test func refusesUnknownTensorType() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var writer = GGUFWriter()
        writer.addString("general.architecture", "toy")
        writer.tensors = [("toy.bad", [32], 99, 0)]
        let url = directory.appendingPathComponent("bad.gguf")
        try writer.write(to: url)
        #expect(throws: GGUFError.unknownTensorType(name: "toy.bad", type: 99)) {
            try GGUFReader.read(shardAt: url)
        }
    }

    /// A tensor whose element count is not a multiple of its block is the defect
    /// a guessed block size would hide: it parses, and every offset after it is
    /// wrong.
    @Test func refusesTensorNotDivisibleByItsBlock() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var writer = GGUFWriter()
        writer.addString("general.architecture", "toy")
        writer.tensors = [("toy.odd", [100], GGUFType.q6K.rawValue, 0)]
        let url = directory.appendingPathComponent("odd.gguf")
        try writer.write(to: url)
        #expect(throws: GGUFError.tensorNotBlockAligned(name: "toy.odd", elements: 100, block: 256))
        {
            try GGUFReader.read(shardAt: url)
        }
    }

    @Test func refusesBadMagicAndVersionOne() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var writer = GGUFWriter()
        writer.addString("general.architecture", "toy")
        let valid = directory.appendingPathComponent("valid.gguf")
        try writer.write(to: valid)
        var bytes = try Data(contentsOf: valid)
        bytes[0] = UInt8(ascii: "X")
        let badMagic = directory.appendingPathComponent("bad-magic.gguf")
        try bytes.write(to: badMagic)
        #expect(throws: GGUFError.badMagic("XGUF")) {
            try GGUFReader.read(shardAt: badMagic)
        }

        var versionOne = try Data(contentsOf: valid)
        versionOne[4] = 1
        let old = directory.appendingPathComponent("v1.gguf")
        try versionOne.write(to: old)
        #expect(throws: GGUFError.unsupportedVersion(1)) {
            try GGUFReader.read(shardAt: old)
        }
    }

    @Test func refusesTruncatedDirectory() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var writer = GGUFWriter()
        writer.addString("general.architecture", "toy")
        writer.tensors = [("toy.q6k", [256], GGUFType.q6K.rawValue, 0)]
        writer.dataSize = 210
        let valid = directory.appendingPathComponent("full.gguf")
        try writer.write(to: valid)
        let full = try Data(contentsOf: valid)
        // Cut inside the metadata, after the 24-byte header: the directory ends
        // before its own declared count is satisfied.
        let truncated = directory.appendingPathComponent("truncated.gguf")
        try full.prefix(30).write(to: truncated)
        #expect(throws: GGUFError.truncated(offset: 24)) {
            try GGUFReader.read(shardAt: truncated)
        }
    }

    @Test func refusesTensorPastEndOfFile() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var writer = GGUFWriter()
        writer.addString("general.architecture", "toy")
        writer.tensors = [("toy.q6k", [256], GGUFType.q6K.rawValue, 0)]
        writer.dataSize = 100  // a 210-byte tensor cannot fit
        let url = directory.appendingPathComponent("short.gguf")
        try writer.write(to: url)
        let shardRead = {
            do {
                _ = try GGUFReader.read(shardAt: url)
                return false
            } catch GGUFError.tensorOutOfRange(let name, let end, let fileSize) {
                return name == "toy.q6k" && end > fileSize
            } catch {
                return false
            }
        }
        #expect(shardRead())
    }

    @Test func mergesSplitShardsByFilename() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        var first = GGUFWriter()
        first.addString("general.architecture", "toy")
        first.addU32("split.no", 0)
        first.addU32("split.count", 2)
        first.addU32("split.tensors.count", 2)
        first.tensors = [("toy.a", [256], GGUFType.q6K.rawValue, 0)]
        first.dataSize = 210
        try first.write(to: directory.appendingPathComponent("toy-00001-of-00002.gguf"))

        var second = GGUFWriter()
        second.addU32("split.no", 1)
        second.addU32("split.count", 2)
        second.tensors = [("toy.b", [64], GGUFType.q2_0.rawValue, 0)]
        second.dataSize = 18
        try second.write(to: directory.appendingPathComponent("toy-00002-of-00002.gguf"))

        let model = try GGUFModelReader.read(modelAt: directory)
        #expect(model.shards.count == 2)
        #expect(model.architecture == "toy")
        #expect(model.tensors.count == 2)
        #expect(model.tensor(named: "toy.a")?.shardIndex == 0)
        #expect(model.tensor(named: "toy.b")?.shardIndex == 1)
        #expect(model.totalTensorBytes == 228)
    }

    @Test func refusesSplitSetWithAMissingShard() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var first = GGUFWriter()
        first.addString("general.architecture", "toy")
        first.addU32("split.count", 2)
        first.tensors = [("toy.a", [256], GGUFType.q6K.rawValue, 0)]
        first.dataSize = 210
        try first.write(to: directory.appendingPathComponent("toy-00001-of-00002.gguf"))
        #expect(throws: GGUFError.self) {
            try GGUFModelReader.read(modelAt: directory)
        }
    }

    @Test func refusesAmbiguousDirectory() throws {
        let directory = try Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in ["one.gguf", "two.gguf"] {
            var writer = GGUFWriter()
            writer.addString("general.architecture", "toy")
            try writer.write(to: directory.appendingPathComponent(name))
        }
        #expect(throws: GGUFError.self) {
            try GGUFModelReader.read(modelAt: directory)
        }
    }

    /// The real install is 83 GB and must never be downloaded or copied by a
    /// test, so this reads its directories only: 1,224 tensors across four
    /// shards, with the byte counts that identified `Q2_0` before the type table
    /// was read. Gated rather than skipped, so a run without the model says so.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func readsTheInstalledUnslothCheckpoint() throws {
        let path = try #require(Self.realModelPath)
        let url = URL(fileURLWithPath: path)
        let model = try GGUFModelReader.read(modelAt: url)
        #expect(model.shards.count == 4)
        #expect(model.architecture == "qwen4exp")
        #expect(model.architectureValue("block_count")?.integer == 48)
        #expect(model.architectureValue("expert_count")?.integer == 512)
        #expect(model.architectureValue("expert_used_count")?.integer == 10)
        #expect(model.tensors.count == 1_224)
        #expect(model.totalTensorBytes == 83_615_413_760)

        var counts: [String: Int] = [:]
        for location in model.tensors.values {
            counts[location.tensor.type.name, default: 0] += 1
        }
        #expect(
            counts == [
                "F32": 557, "Q6_K": 250, "Q8_0": 248, "IQ2_S": 52, "IQ4_NL": 46,
                "IQ3_XXS": 42, "BF16": 24, "Q2_0": 3, "IQ4_XS": 2,
            ])

        let output = try #require(model.tensor(named: "output.weight"))
        #expect(output.tensor.type == .q6K)
        #expect(output.tensor.byteCount == 521_472_000)
        let down8 = try #require(model.tensor(named: "blk.8.ffn_down_exps.weight"))
        #expect(down8.tensor.type == .q2_0)
        #expect(down8.tensor.byteCount == 235_929_600)
        let ple = try #require(model.tensor(named: "per_layer_token_embd.weight"))
        #expect(ple.shardIndex == 3)
        #expect(ple.tensor.type == .iq4NL)
        #expect(ple.tensor.byteCount == 28_800_138_240)
    }

    /// Decodes one real block through the directory's offsets for each of the
    /// types the decoder implements against this file. The synthetic vectors
    /// prove the arithmetic; this proves the addressing: a shard index, a data
    /// offset and a block size that agree with 83 GB of bytes on disk.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func decodesRealBlocksAtTheirDirectoryOffsets() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let names = [
            "output.weight", "blk.8.ffn_down_exps.weight", "per_layer_token_embd.weight",
            "blk.0.ffn_gate_exps.weight", "blk.14.ffn_gate_exps.weight",
        ]
        for name in names {
            let location = try #require(model.tensor(named: name))
            let shard = model.shards[location.shardIndex]
            let blockBytes = Int(location.tensor.type.blockBytes)
            let handle = try FileHandle(forReadingFrom: shard.url)
            defer { try? handle.close() }
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            let data = try #require(try handle.read(upToCount: blockBytes))
            let values = try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](data))
            #expect(values.count == Int(location.tensor.type.blockSize))
            #expect(values.allSatisfy { $0.isFinite }, "\(name) decoded a non-finite value")
            #expect(values.contains { $0 != 0 }, "\(name) decoded an all-zero block")
        }
    }
}
