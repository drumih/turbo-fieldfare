import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The side-engine's loader and its scalar arithmetic.
///
/// The whole-model check lives in `TinyTitanBench cpu35`, because it needs a two
/// gigabyte snapshot and agreement with `tools/qwen35_reference.py` — both
/// measured, both recorded in `docs/plan-cpu-side-engine.md`. What is here is
/// everything that can be checked without one, which is the loader's contract
/// and the arithmetic between the GEMVs.
@Suite struct CPUEngineTests {

    // MARK: - safetensors

    /// Builds a real safetensors file: an eight-byte header length, a JSON
    /// header, then the payload. Written by hand rather than by a library so
    /// the test fails if the reader's idea of the format drifts.
    private func writeShard(
        _ tensors: [(
            name: String, dtype: String,
            shape: [Int], bytes: [UInt8]
        )]
    ) throws -> URL {
        var header: [String: Any] = [:]
        var payload: [UInt8] = []
        for tensor in tensors {
            header[tensor.name] = [
                "dtype": tensor.dtype, "shape": tensor.shape,
                "data_offsets": [
                    payload.count,
                    payload.count + tensor.bytes.count,
                ],
            ]
            payload.append(contentsOf: tensor.bytes)
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        // The payload has to start on the offset the header length declares;
        // safetensors pads the header with spaces to whatever alignment the
        // writer likes, and the reader must honour the declared length.
        while json.count % 8 != 0 { json.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(contentsOf: payload)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("cpu-engine-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        return url
    }

    private func bf16(_ values: [Float]) -> [UInt8] {
        var bytes: [UInt8] = []
        for value in values {
            let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
            withUnsafeBytes(of: bits.littleEndian) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    @Test func readsShapesAndOffsets() throws {
        let url = try writeShard([
            ("a", "F32", [2, 3], [UInt8](repeating: 0, count: 24)),
            ("b", "BF16", [4], bf16([1, 2, 3, 4])),
        ])
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafeTensorsFile(url: url)
        #expect(file.entries.count == 2)
        #expect(try file.entry("a").shape == [2, 3])
        #expect(try file.entry("b").dtype == "BF16")
    }

    /// BF16 is the top sixteen bits of a float32, and the reader widens by
    /// bit pattern rather than by a library — because the one it would have
    /// used cannot decode BF16 at all, which is how this reader came to
    /// exist.
    /// A shape from the file used to reach a trapping multiplication: `count`
    /// was `shape.reduce(1, *)` on every access, so this header parsed cleanly
    /// and the process aborted the first time anything asked. The shape is
    /// validated with reporting arithmetic now, so it is a thrown error at load --
    /// and the test fails rather than aborting, which is the point.
    @Test func anUnusableShapeIsRefusedRatherThanTrapping() throws {
        let url = try writeShard([("a", "F32", [Int.max, 2], [0, 0, 0, 0])])
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: SafeTensorsFile.Failure.self) {
            _ = try SafeTensorsFile(url: url)
        }
    }

    @Test func aNegativeDimensionIsRefused() throws {
        let url = try writeShard([("a", "F32", [2, -3], [0, 0, 0, 0])])
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: SafeTensorsFile.Failure.self) {
            _ = try SafeTensorsFile(url: url)
        }
    }

    /// The valid case keeps working, and `count` is the shape's product.
    @Test func aValidShapeStillCountsItsElements() throws {
        let url = try writeShard([("a", "F32", [2, 3], [UInt8](repeating: 0, count: 24))])
        defer { try? FileManager.default.removeItem(at: url) }

        let file = try SafeTensorsFile(url: url)
        #expect(try file.entry("a").count == 6)
    }

    @Test func widensBFloatByBitPattern() throws {
        let values: [Float] = [1, -2, 0.5, 1024, 0]
        let url = try writeShard([("g", "BF16", [values.count], bf16(values))])
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafeTensorsFile(url: url)
        #expect(try file.floats("g") == values, "these all round-trip exactly")
    }

    @Test func missingTensorIsAnError() throws {
        let url = try writeShard([("a", "F32", [1], [0, 0, 0, 0])])
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafeTensorsFile(url: url)
        #expect(throws: SafeTensorsFile.Failure.self) { try file.bytes("absent") }
    }

    // MARK: - a whole (tiny) snapshot

    /// A snapshot with one quantized matrix, written the way the converter
    /// writes one: `bits`-wide unsigned lanes packed low-first into UInt32,
    /// one BF16 scale and bias per group.
    private func writeSnapshot(
        rows: Int, columns: Int, bits: Int,
        group: Int = 64,
        level: (Int, Int) -> UInt32,
        scale: Float, bias: Float,
        overrides: [String: Int] = [:]
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("snapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let lanes = 32 / bits
        var words: [UInt32] = []
        for row in 0..<rows {
            for word in 0..<(columns / lanes) {
                var packed: UInt32 = 0
                for lane in 0..<lanes {
                    packed |=
                        (level(row, word * lanes + lane) & UInt32((1 << bits) - 1))
                        << (bits * lane)
                }
                words.append(packed)
            }
        }
        var weightBytes: [UInt8] = []
        for word in words {
            withUnsafeBytes(of: word.littleEndian) { weightBytes.append(contentsOf: $0) }
        }
        let groups = rows * (columns / group)
        let url = try writeShard([
            ("w.weight", "U32", [rows, columns / lanes], weightBytes),
            (
                "w.scales", "BF16", [rows, columns / group],
                bf16([Float](repeating: scale, count: groups))
            ),
            (
                "w.biases", "BF16", [rows, columns / group],
                bf16([Float](repeating: bias, count: groups))
            ),
        ])
        try FileManager.default.moveItem(
            at: url, to: directory.appendingPathComponent("model.safetensors"))

        var quantization: [String: Any] = ["bits": bits, "group_size": group, "mode": "affine"]
        for (stem, width) in overrides {
            quantization[stem] = ["bits": width, "group_size": group]
        }
        let config: [String: Any] = [
            "hidden_size": columns, "num_hidden_layers": 1, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": columns, "full_attention_interval": 4,
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": columns, "linear_value_head_dim": columns,
            "linear_conv_kernel_dim": 4, "intermediate_size": columns,
            "vocab_size": rows, "rms_norm_eps": 1e-6, "quantization": quantization,
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: [
            "weight_map": [
                "w.weight": "model.safetensors",
                "w.scales": "model.safetensors",
                "w.biases": "model.safetensors",
            ]
        ])
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// A width that is not a whole number of 64-element groups used to reach
    /// `columns / groupSize` and truncate: the guard that compares the scale
    /// count then agreed with scales sized for the truncated count, so the
    /// dequantize read fewer groups per row than the weights hold and returned
    /// plausible nonsense. It is refused now, before the division's result is
    /// trusted.
    /// AUD-142: the `config.json` read is bounded, and it is the *bound* that
    /// fires. The same directory loads under the default ceiling and is refused
    /// under a smaller one, so the refusal cannot be the architecture block
    /// complaining, and the message carries the size and the cap because "the
    /// config is invalid" would send the operator to the wrong file.
    @Test func anOversizedConfigIsRefusedByTheBoundNotByTheArchitectureParse() throws {
        let directory = try writeSnapshot(
            rows: 8, columns: 64, bits: 4,
            level: { _, _ in 1 }, scale: 1, bias: 0)
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try AffineSnapshot(directory: directory)

        let configURL = directory.appendingPathComponent("config.json")
        let size = try Data(contentsOf: configURL).count
        #expect {
            _ = try AffineSnapshot(directory: directory, maxBytes: UInt64(size - 1))
        } throws: { error in
            guard case ModelError.metadataOverBound(let name, let bytes, let cap) = error else {
                return false
            }
            return name == "config.json" && bytes == size && cap == UInt64(size - 1)
        }
    }

    /// AUD-164: the shard names come from the index document, and an install
    /// arrives copied off another machine, so a name that walks outside the
    /// snapshot directory is refused. The same fence `LocalSnapshotLoader`
    /// applies to this format on the converter's side.
    @Test func aShardNameThatEscapesTheSnapshotIsRefused() throws {
        let directory = try writeSnapshot(
            rows: 8, columns: 64, bits: 4,
            level: { _, _ in 1 }, scale: 1, bias: 0)
        defer { try? FileManager.default.removeItem(at: directory) }
        // The valid case first: without it a refusal from anywhere else in the
        // loader would read as this one.
        _ = try AffineSnapshot(directory: directory)

        let outside = directory.deletingLastPathComponent()
            .appendingPathComponent("escaped-\(UUID().uuidString).safetensors")
        try FileManager.default.copyItem(
            at: directory.appendingPathComponent("model.safetensors"), to: outside)
        defer { try? FileManager.default.removeItem(at: outside) }
        try JSONSerialization.data(withJSONObject: [
            "weight_map": [
                "w.weight": "../" + outside.lastPathComponent,
                "w.scales": "model.safetensors",
                "w.biases": "model.safetensors",
            ]
        ])
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))

        #expect {
            _ = try AffineSnapshot(directory: directory)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsafe path")
        }
    }

    /// The other half of the same boundary: a legitimate name that is a link.
    /// Opened through `SSDAIModelDirectory`, the shard must be a regular file and
    /// no component may be followed, so this is a refusal rather than a mapping
    /// of whatever the install's author pointed at.
    @Test func aSymlinkedShardIsRefusedRatherThanMapped() throws {
        let directory = try writeSnapshot(
            rows: 8, columns: 64, bits: 4,
            level: { _, _ in 1 }, scale: 1, bias: 0)
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try AffineSnapshot(directory: directory)

        let shard = directory.appendingPathComponent("model.safetensors")
        let target = directory.deletingLastPathComponent()
            .appendingPathComponent("linked-\(UUID().uuidString).safetensors")
        try FileManager.default.moveItem(at: shard, to: target)
        defer { try? FileManager.default.removeItem(at: target) }
        try FileManager.default.createSymbolicLink(at: shard, withDestinationURL: target)

        #expect(throws: ModelError.self) {
            _ = try AffineSnapshot(directory: directory)
        }
    }

    /// AUD-164 also closes the by-path opener, so the same rule holds for a
    /// caller that names the file itself: `init(url:)` will not follow a link.
    @Test func aSymlinkedURLIsRefusedByTheOpener() throws {
        let url = try writeShard([("a", "F32", [1], [0, 0, 0, 0])])
        defer { try? FileManager.default.removeItem(at: url) }
        let link = url.deletingLastPathComponent()
            .appendingPathComponent("link-\(UUID().uuidString).safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: url)
        defer { try? FileManager.default.removeItem(at: link) }

        #expect(throws: SafeTensorsFile.Failure.self) {
            _ = try SafeTensorsFile(url: link)
        }
        _ = try SafeTensorsFile(url: url)
    }

    /// AUD-165: the index document is the second metadata read in this
    /// initializer, and it was still a whole-file `Data(contentsOf:)` while the
    /// `config.json` above it went through `BoundedMetadataRead` -- the sibling
    /// AUD-142 fixed one of and left the other. The cap fires before the
    /// allocation, so the refusal names the document and both sizes.
    @Test func anOversizedIndexIsRefusedByTheBoundNotByTheParse() throws {
        let directory = try writeSnapshot(
            rows: 8, columns: 64, bits: 4,
            level: { _, _ in 1 }, scale: 1, bias: 0)
        defer { try? FileManager.default.removeItem(at: directory) }
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        // Padding rather than a bigger weight_map: the reader ignores this key, so
        // the only thing that changes is the document's size, and it is the size
        // the bound is about.
        try JSONSerialization.data(withJSONObject: [
            "weight_map": [
                "w.weight": "model.safetensors",
                "w.scales": "model.safetensors",
                "w.biases": "model.safetensors",
            ],
            "padding": String(repeating: "x", count: 4096),
        ])
        .write(to: indexURL)
        let indexSize = try Data(contentsOf: indexURL).count
        _ = try AffineSnapshot(directory: directory, maxBytes: UInt64(indexSize))

        #expect {
            _ = try AffineSnapshot(directory: directory, maxBytes: UInt64(indexSize - 1))
        } throws: { error in
            guard case ModelError.metadataOverBound(let name, let bytes, let cap) = error else {
                return false
            }
            return name == "model.safetensors.index.json"
                && bytes == indexSize && cap == UInt64(indexSize - 1)
        }
    }

    @Test func aWidthOutsideWholeGroupsIsRefused() throws {
        let directory = try writeSnapshot(
            rows: 8, columns: 32, bits: 4,
            level: { _, _ in 1 }, scale: 1, bias: 0)
        defer { try? FileManager.default.removeItem(at: directory) }
        let snapshot = try AffineSnapshot(directory: directory)

        #expect(throws: SafeTensorsFile.Failure.self) {
            _ = try snapshot.matrix("w.weight")
        }
    }

    @Test func readsAQuantizedMatrixAtBothWidths() throws {
        for bits in [4, 8] {
            let directory = try writeSnapshot(
                rows: 128, columns: 64, bits: bits,
                level: { _, _ in 1 }, scale: 1, bias: 0)
            defer { try? FileManager.default.removeItem(at: directory) }
            let snapshot = try AffineSnapshot(directory: directory)
            let matrix = try snapshot.matrix("w.weight")
            #expect(matrix.rows == 128)
            #expect(
                matrix.columns == 64,
                Comment(rawValue: "columns must be unpacked, not the stored word count"))
            #expect(matrix.bits == bits)
        }
    }

    /// The 4-bit build keeps some tensors at 8 bits, so the width is per
    /// tensor and the base is only a default. Reading it as a snapshot-wide
    /// constant is a bug this project has already had once.
    @Test func perTensorWidthOverridesTheBase() throws {
        let directory = try writeSnapshot(
            rows: 128, columns: 64, bits: 8,
            level: { _, _ in 1 }, scale: 1, bias: 0,
            overrides: ["w": 8])
        defer { try? FileManager.default.removeItem(at: directory) }
        let snapshot = try AffineSnapshot(directory: directory)
        #expect(snapshot.bits(forStem: "w") == 8)
        #expect(snapshot.bits(forStem: "elsewhere") == 8)
    }

    /// A width the CPU kernels do not implement is refused at load.
    ///
    /// On this route the width comes from `config.json` through a plain JSON
    /// cast, so unlike the `.ssdai` reader -- where
    /// `SSDAIManifestQuantV1.init(from:)` refuses it while decoding -- nothing
    /// gated it. Nothing downstream looks at it either: `dequantize` derives
    /// `lanes = 32 / bits`, so a 6-bit matrix does not fail, it unpacks five
    /// lanes per word and returns plausible nonsense. For an engine whose whole
    /// purpose is to be an independent reference for the GPU path, a wrong
    /// agreement number is worse than a refused load, and the GEMV's own
    /// `switch` ends in `preconditionFailure` -- an abort on model data.
    @Test func aSnapshotWidthNoCPUKernelImplementsIsRefusedAtLoad() throws {
        for bits in [6, 3, 16] {
            let directory = try writeSnapshot(
                rows: 64, columns: 128, bits: bits,
                level: { _, _ in 1 }, scale: 1, bias: 0)
            defer { try? FileManager.default.removeItem(at: directory) }
            do {
                _ = try AffineSnapshot(directory: directory)
                Issue.record("a \(bits)-bit snapshot must not load")
            } catch {
                let text = "\(error)"
                #expect(
                    text.contains("\(bits)-bit"),
                    "the refusal must name the width, got \(text)")
            }
        }
    }

    /// The same rule for the per-tensor block, which is where a snapshot says
    /// one tensor differs from the build's base width.
    @Test func aSnapshotOverrideAtAWidthNoKernelImplementsIsRefused() throws {
        let directory = try writeSnapshot(
            rows: 64, columns: 128, bits: 4,
            level: { _, _ in 1 }, scale: 1, bias: 0,
            overrides: ["w": 6])
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try AffineSnapshot(directory: directory)
            Issue.record("a 6-bit override must not load")
        } catch {
            let text = "\(error)"
            #expect(text.contains("w"), "the refusal must name the stem, got \(text)")
            #expect(text.contains("6-bit"), "the refusal must name the width, got \(text)")
        }
    }

    /// `y = W · x` where every level is 1 and the scale is 1: each row sums
    /// x. A wrong lane order or group stride shows up immediately.
    @Test func gemvComputesTheProduct() throws {
        for bits in [4, 8] {
            let columns = 128
            let rows = 256
            let directory = try writeSnapshot(
                rows: rows, columns: columns, bits: bits,
                level: { _, column in UInt32(column % 2) },
                scale: 1, bias: 0)
            defer { try? FileManager.default.removeItem(at: directory) }
            let snapshot = try AffineSnapshot(directory: directory)
            let matrix = try snapshot.matrix("w.weight")
            let x = (0..<columns).map { Float($0) }
            var out = [Float](repeating: 0, count: rows)
            try x.withUnsafeBufferPointer { input in
                try out.withUnsafeMutableBufferPointer { output in
                    guard let inputBase = input.baseAddress, let outputBase = output.baseAddress
                    else {
                        return
                    }
                    try CPUOps.gemv(matrix, x: inputBase, out: outputBase, threads: 4)
                }
            }
            // Levels alternate 0,1 by column, so each row sums the odd x.
            let expected = stride(from: 1, to: columns, by: 2).reduce(Float(0)) { $0 + Float($1) }
            #expect(
                out.allSatisfy { abs($0 - expected) < 0.01 },
                Comment(rawValue: "\(bits)-bit: got \(out[0]), wanted \(expected)"))
        }
    }

    /// Threading splits rows, so it cannot change the answer. The kernel's
    /// own tests assert this for INT8; this asserts it through the path the
    /// engine actually uses, including INT4.
    @Test func threadingDoesNotChangeTheResult() throws {
        let columns = 128
        let rows = 512
        for bits in [4, 8] {
            let directory = try writeSnapshot(
                rows: rows, columns: columns, bits: bits,
                level: { row, column in UInt32((row &+ column) % (1 << bits)) },
                scale: 0.01, bias: -0.5)
            defer { try? FileManager.default.removeItem(at: directory) }
            let snapshot = try AffineSnapshot(directory: directory)
            let matrix = try snapshot.matrix("w.weight")
            let x = (0..<columns).map { Float($0 % 7) * 0.25 }
            func run(_ threads: Int) throws -> [Float] {
                var out = [Float](repeating: 0, count: rows)
                try x.withUnsafeBufferPointer { input in
                    try out.withUnsafeMutableBufferPointer { output in
                        guard let inputBase = input.baseAddress,
                            let outputBase = output.baseAddress
                        else { return }
                        try CPUOps.gemv(matrix, x: inputBase, out: outputBase, threads: threads)
                    }
                }
                return out
            }
            #expect(try run(1) == run(4), Comment(rawValue: "\(bits)-bit threading must be exact"))
            #expect(try run(4) == run(8))
        }
    }

    // MARK: - the width policy

    /// A snapshot with no layers at all: enough for `CPUQwen35` to
    /// initialize, which is all the width policy needs, and it exercises the
    /// real initializer rather than a stand-in.
    private func writeEmptyModel(hidden: Int = 64) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        let url = try writeShard([
            (
                "language_model.model.norm.weight", "BF16", [hidden],
                bf16([Float](repeating: 1, count: hidden))
            )
        ])
        try FileManager.default.moveItem(
            at: url, to: directory.appendingPathComponent("model.safetensors"))
        let config: [String: Any] = [
            "hidden_size": hidden, "num_hidden_layers": 0, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": hidden, "full_attention_interval": 4,
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": hidden, "linear_value_head_dim": hidden,
            "linear_conv_kernel_dim": 4, "intermediate_size": hidden,
            "vocab_size": 8, "rms_norm_eps": 1e-6,
            "quantization": ["bits": 8, "group_size": 64, "mode": "affine"],
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: [
            "weight_map": ["language_model.model.norm.weight": "model.safetensors"]
        ])
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// Width is a scheduling decision, and the whole policy is one line:
    /// narrow while a person is waiting on the main engine, wide in the
    /// gaps. Measured, four threads costs a 35B generation 31% and one
    /// costs 3%, so this is the difference between a side-engine and a
    /// tax on the answer someone asked for.
    ///
    /// Exercised through a tiny snapshot rather than the real model, because
    /// what is being tested is the decision, not the arithmetic.
    @Test func widthFollowsContention() throws {
        let directory = try writeEmptyModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try CPUQwen35(
            snapshot: try AffineSnapshot(directory: directory),
            threads: 4)
        #expect(model.threads == 4, "without a signal the width is whatever was asked for")

        let busy = Busy()
        model.contention = { busy.value }
        model.busyThreads = 1
        model.idleThreads = 4

        busy.value = true
        model.applyWidthPolicy()
        #expect(model.threads == 1, "a person is waiting; take one core")

        busy.value = false
        model.applyWidthPolicy()
        #expect(model.threads == 4, "nothing waiting; take the performance cores")
    }

    /// A signal that never fires leaves the width exactly as configured, so
    /// a caller that does not care about contention is not surprised by it.
    @Test func noSignalLeavesTheWidthAlone() throws {
        let directory = try writeEmptyModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try CPUQwen35(
            snapshot: try AffineSnapshot(directory: directory),
            threads: 3)
        model.applyWidthPolicy()
        #expect(model.threads == 3)
    }

    /// unchecked-invariant: written and read only from the test's own thread,
    /// which is serial; the closure that reads it runs synchronously inside
    /// the same call.
    private final class Busy: @unchecked Sendable { var value = false }

    // MARK: - head sharing, the shape the 4B adds

    /// The 2B has as many delta-rule value heads as key heads and shares each
    /// KV head among four query heads; the 4B has twice as many value heads
    /// as key heads. Its whole-model check needs a real snapshot, so the
    /// mapping is pinned here by an equivalence that needs no reference: a
    /// model whose heads are shared must compute exactly what the same model
    /// computes with each shared head written out once per user.
    ///
    /// Sharing is by consecutive heads (transformers repeats in place). Each
    /// test also builds the strided alternative -- head `i` paired with
    /// `i % count` -- and requires it to differ, so the check can fail.

    /// Weights drawn from a fixed seed, so every variant of a model is built
    /// from the same draw and differs only where a test says it does.
    private struct SplitMix: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// An 8-bit matrix kept as rows of levels, scales and biases, so a row
    /// can be copied into another model bit for bit.
    private struct Rows {
        var levels: [[UInt8]] = []
        var scales: [[Float]] = []
        var biases: [[Float]] = []

        static func random(_ rows: Int, _ columns: Int, _ rng: inout SplitMix) -> Rows {
            var out = Rows()
            for _ in 0..<rows {
                out.levels.append((0..<columns).map { _ in UInt8.random(in: 0...255, using: &rng) })
                let scales = (0..<(columns / 64)).map { _ in
                    Float.random(in: 0.0005...0.002, using: &rng)
                }
                out.scales.append(scales)
                out.biases.append(scales.map { -127.5 * $0 })  // centred on zero
            }
            return out
        }

        func picking(_ rows: [Int]) -> Rows {
            Rows(
                levels: rows.map { levels[$0] }, scales: rows.map { scales[$0] },
                biases: rows.map { biases[$0] })
        }
    }

    private struct TinyModel {
        var config: [String: Any]
        var matrices: [String: Rows] = [:]
        var floats: [String: (shape: [Int], values: [Float])] = [:]
    }

    private static let tinyPrefix = "language_model.model."

    private func randomFloats(
        _ count: Int, _ range: ClosedRange<Float>,
        _ rng: inout SplitMix
    ) -> [Float] {
        (0..<count).map { _ in Float.random(in: range, using: &rng) }
    }

    /// One layer of the family at 64 wide: the embedding, the norms and the
    /// MLP around a mixer the caller adds. `interval` 1 makes layer 0 full
    /// attention, 4 makes it gated DeltaNet.
    private func tinyModel(interval: Int, _ rng: inout SplitMix) -> TinyModel {
        let p = Self.tinyPrefix
        let hidden = 64
        var model = TinyModel(config: [
            "hidden_size": hidden, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "full_attention_interval": interval,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "intermediate_size": hidden, "vocab_size": 8,
            "rms_norm_eps": 1e-6, "rope_theta": 10_000.0, "partial_rotary_factor": 0.25,
            "quantization": ["bits": 8, "group_size": 64, "mode": "affine"],
        ])
        model.matrices[p + "embed_tokens"] = Rows.random(8, hidden, &rng)
        for name in ["gate_proj", "up_proj", "down_proj"] {
            model.matrices[p + "layers.0.mlp." + name] = Rows.random(hidden, hidden, &rng)
        }
        for name in ["layers.0.input_layernorm", "layers.0.post_attention_layernorm", "norm"] {
            model.floats[p + name + ".weight"] = ([hidden], randomFloats(hidden, 0.8...1.2, &rng))
        }
        return model
    }

    private func writeTinyModel(_ model: TinyModel) throws -> URL {
        var tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])] = []
        for (stem, rows) in model.matrices.sorted(by: { $0.key < $1.key }) {
            let count = rows.levels.count
            let columns = rows.levels[0].count
            // Four 8-bit lanes per word, low first: little-endian, the word's
            // bytes are the levels in order.
            tensors.append(
                (
                    stem + ".weight", "U32", [count, columns / 4],
                    rows.levels.flatMap { $0 }
                ))
            tensors.append(
                (
                    stem + ".scales", "BF16", [count, columns / 64],
                    bf16(rows.scales.flatMap { $0 })
                ))
            tensors.append(
                (
                    stem + ".biases", "BF16", [count, columns / 64],
                    bf16(rows.biases.flatMap { $0 })
                ))
        }
        for (name, tensor) in model.floats.sorted(by: { $0.key < $1.key }) {
            tensors.append((name, "BF16", tensor.shape, bf16(tensor.values)))
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        try FileManager.default.moveItem(
            at: try writeShard(tensors), to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: model.config)
            .write(to: directory.appendingPathComponent("config.json"))
        let map = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, "model.safetensors") })
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// Logits at every position of a short sequence, so the carried state --
    /// the KV cache, the convolution tail, the delta-rule state -- is
    /// compared along with the first token.
    private func sequenceLogits(_ model: TinyModel) throws -> [Float] {
        let directory = try writeTinyModel(model)
        defer { try? FileManager.default.removeItem(at: directory) }
        let engine = try CPUQwen35(
            snapshot: try AffineSnapshot(directory: directory),
            threads: 1)
        return try [3, 1, 4, 1, 5].flatMap { try engine.step(token: $0) }
    }

    private func largestDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    /// Hv = 2·Hk, the 4B's delta rule: value heads 0 and 1 read key head 0,
    /// 2 and 3 read key head 1. Written out, that is a model with four key
    /// heads laid out [0, 0, 1, 1] -- query and key channels copied, with
    /// their convolution taps -- and nothing else changed.
    @Test func deltaRuleValueHeadsShareConsecutiveKeyHeads() throws {
        var rng = SplitMix(state: 4)
        let p = Self.tinyPrefix + "layers.0.linear_attn."
        let hidden = 64
        let dk = 16
        let keyHeads = 2
        let valueHeads = 4
        let valueWidth = valueHeads * 16
        var base = tinyModel(interval: 4, &rng)
        let qkv = Rows.random(2 * keyHeads * dk + valueWidth, hidden, &rng)
        let taps = (0..<(2 * keyHeads * dk + valueWidth)).map { _ in
            randomFloats(4, -0.6...0.6, &rng)
        }
        base.matrices[p + "in_proj_z"] = Rows.random(valueWidth, hidden, &rng)
        base.matrices[p + "out_proj"] = Rows.random(hidden, valueWidth, &rng)
        base.floats[p + "in_proj_a.weight"] = (
            [valueHeads, hidden],
            randomFloats(valueHeads * hidden, -0.1...0.1, &rng)
        )
        base.floats[p + "in_proj_b.weight"] = (
            [valueHeads, hidden],
            randomFloats(valueHeads * hidden, -0.1...0.1, &rng)
        )
        base.floats[p + "A_log"] = ([valueHeads], randomFloats(valueHeads, -1...0.5, &rng))
        base.floats[p + "dt_bias"] = ([valueHeads], randomFloats(valueHeads, -0.5...0.5, &rng))
        base.floats[p + "norm.weight"] = ([16], randomFloats(16, 0.5...1.5, &rng))

        /// The model with key heads laid out as `layout`, each entry naming
        /// one of the two drawn key heads.
        func variant(_ layout: [Int]) -> TinyModel {
            var model = base
            let query = layout.flatMap { Array(($0 * dk)..<($0 * dk + dk)) }
            let key = layout.flatMap {
                Array((keyHeads * dk + $0 * dk)..<(keyHeads * dk + $0 * dk + dk))
            }
            let channels =
                query + key + Array((2 * keyHeads * dk)..<(2 * keyHeads * dk + valueWidth))
            model.matrices[p + "in_proj_qkv"] = qkv.picking(channels)
            model.floats[p + "conv1d.weight"] = (
                [channels.count, 1, 4], channels.flatMap { taps[$0] }
            )
            model.config["linear_num_key_heads"] = layout.count
            return model
        }
        let shared = try sequenceLogits(variant([0, 1]))
        let consecutive = try sequenceLogits(variant([0, 0, 1, 1]))
        let strided = try sequenceLogits(variant([0, 1, 0, 1]))
        #expect(
            largestDifference(shared, consecutive) < 1e-4,
            "32 value heads over 16 key heads must read key head h / 2")
        #expect(
            largestDifference(shared, strided) > 1e-3,
            "the strided pairing must be distinguishable, or this proves nothing")
    }

    /// Sixteen query heads over four KV heads, the 4B's attention, is the
    /// same claim at a group of four; here a group of two, over the same
    /// code path. Written out, KV heads [0, 0, 1, 1], one per query head.
    @Test func queryHeadsShareConsecutiveKeyValueHeads() throws {
        var rng = SplitMix(state: 16)
        let p = Self.tinyPrefix + "layers.0.self_attn."
        let hidden = 64
        let dim = 16
        var base = tinyModel(interval: 1, &rng)
        base.matrices[p + "q_proj"] = Rows.random(4 * 2 * dim, hidden, &rng)
        base.matrices[p + "o_proj"] = Rows.random(hidden, 4 * dim, &rng)
        let keys = Rows.random(2 * dim, hidden, &rng)
        let values = Rows.random(2 * dim, hidden, &rng)
        base.floats[p + "q_norm.weight"] = ([dim], randomFloats(dim, 0.5...1.5, &rng))
        base.floats[p + "k_norm.weight"] = ([dim], randomFloats(dim, 0.5...1.5, &rng))

        func variant(_ layout: [Int]) -> TinyModel {
            var model = base
            let rows = layout.flatMap { Array(($0 * dim)..<($0 * dim + dim)) }
            model.matrices[p + "k_proj"] = keys.picking(rows)
            model.matrices[p + "v_proj"] = values.picking(rows)
            model.config["num_key_value_heads"] = layout.count
            return model
        }
        let shared = try sequenceLogits(variant([0, 1]))
        let consecutive = try sequenceLogits(variant([0, 0, 1, 1]))
        let strided = try sequenceLogits(variant([0, 1, 0, 1]))
        #expect(
            largestDifference(shared, consecutive) < 1e-4,
            "query head h must read KV head h / (heads / kvHeads)")
        #expect(
            largestDifference(shared, strided) > 1e-3,
            "the strided pairing must be distinguishable, or this proves nothing")
    }

    /// The 2B and 4B tie the output to the embedding; the 9B does not, and
    /// ships its own `lm_head` under `language_model.lm_head.weight` — note,
    /// without the `.model.` the embedding prefix carries. An engine that
    /// read the embedding there anyway would produce fluent nonsense rather
    /// than fail, so the head is made a *distinct* matrix: the untied logits
    /// must differ from the tied ones, and must move when only the head
    /// moves, which an ignored head cannot do.
    @Test func anUntiedModelReadsItsOwnHead() throws {
        var rng = SplitMix(state: 21)
        let hidden = 64
        let dim = 16
        let p = Self.tinyPrefix + "layers.0.self_attn."
        // Full attention, so the mixer is the plain q/k/v the other attention
        // tests use; the head is orthogonal to which mixer runs. `q_proj` is
        // hidden + 4*dim because it carries the fused output gate.
        var base = tinyModel(interval: 1, &rng)
        base.matrices[p + "q_proj"] = Rows.random(hidden + 4 * dim, hidden, &rng)
        base.matrices[p + "o_proj"] = Rows.random(hidden, 4 * dim, &rng)
        base.matrices[p + "k_proj"] = Rows.random(2 * dim, hidden, &rng)
        base.matrices[p + "v_proj"] = Rows.random(2 * dim, hidden, &rng)
        base.floats[p + "q_norm.weight"] = ([dim], randomFloats(dim, 0.5...1.5, &rng))
        base.floats[p + "k_norm.weight"] = ([dim], randomFloats(dim, 0.5...1.5, &rng))

        func logits(tied: Bool, head: Rows?) throws -> [Float] {
            var model = base
            model.config["tie_word_embeddings"] = tied
            if let head { model.matrices["language_model.lm_head"] = head }
            let directory = try writeTinyModel(model)
            defer { try? FileManager.default.removeItem(at: directory) }
            let engine = try CPUQwen35(
                snapshot: try AffineSnapshot(directory: directory),
                threads: 1)
            return try [3, 1, 4, 1, 5].flatMap { try engine.step(token: $0) }
        }

        let tiedLogits = try logits(tied: true, head: nil)
        // Untied, head M: the engine must read M, not the embedding.
        let headM = Rows.random(8, hidden, &rng)
        let headA = try logits(tied: false, head: headM)
        #expect(
            largestDifference(headA, tiedLogits) > 1e-3,
            "the untied head must be read, not the embedding")
        // The same model with a different head must move the logits: the only
        // way an ignored head passes both checks is by coincidence.
        let headB = Rows.random(8, hidden, &rng)
        let headBLogits = try logits(tied: false, head: headB)
        #expect(
            largestDifference(headBLogits, headA) > 1e-3,
            "changing the head must change the logits")
    }

    /// A head ratio that is not whole would floor into a wrong mapping;
    /// the engine refuses the snapshot instead of running it.
    @Test func aFractionalHeadRatioIsRefused() throws {
        var rng = SplitMix(state: 3)
        var model = tinyModel(interval: 4, &rng)
        model.config["linear_num_key_heads"] = 3
        let directory = try writeTinyModel(model)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: SafeTensorsFile.Failure.self) {
            try CPUQwen35(snapshot: try AffineSnapshot(directory: directory))
        }
    }

    // MARK: - the arithmetic between the GEMVs

    @Test func rmsNormScalesToUnitRootMeanSquare() {
        var values: [Float] = [3, 4, 0, 0]
        CPUOps.rmsNorm(&values, gamma: [1, 1, 1, 1], epsilon: 0)
        let mean = values.reduce(0) { $0 + $1 * $1 } / 4
        #expect(abs(mean - 1) < 1e-5)
    }

    /// A per-head norm takes each head's own statistics. Sharing one across
    /// heads is a plausible-looking bug that changes every attention output.
    @Test func perSliceNormUsesEachSlicesOwnStatistics() {
        var values: [Float] = [1, 1, 100, 100]
        CPUOps.rmsNormPerSlice(&values, width: 2, gamma: [1, 1], epsilon: 0)
        #expect(abs(values[0] - 1) < 1e-4)
        #expect(abs(values[2] - 1) < 1e-4, "the large slice normalizes to the same place")
    }

    @Test func l2NormalizeMakesUnitSlices() {
        var values: [Float] = [3, 4, 6, 8]
        CPUOps.l2NormalizePerSlice(&values, width: 2, epsilon: 0)
        #expect(abs((values[0] * values[0] + values[1] * values[1]) - 1) < 1e-5)
        #expect(abs((values[2] * values[2] + values[3] * values[3]) - 1) < 1e-5)
    }

    @Test func softmaxSumsToOneAndIsShiftInvariant() {
        var a: [Float] = [1, 2, 3]
        var b: [Float] = [101, 102, 103]
        CPUOps.softmaxInPlace(&a)
        CPUOps.softmaxInPlace(&b)
        #expect(abs(a.reduce(0, +) - 1) < 1e-6)
        for index in a.indices { #expect(abs(a[index] - b[index]) < 1e-6) }
    }

    /// Position zero is the identity, whatever the constants are — which is
    /// exactly why `rope_theta` and the partial fraction cannot be checked
    /// there, and why sequence parity is a separate gate.
    @Test func ropeAtPositionZeroIsTheIdentity() {
        let original = (0..<8).map { Float($0) }
        var values = original
        CPUOps.applyRoPE(&values, headDim: 8, rotaryDim: 4, position: 0, theta: 10_000)
        #expect(values == original)
    }

    /// Partial rotary: only the first `rotaryDim` dimensions move. Rotating
    /// the whole head is the most plausible way to get a model that runs and
    /// is quietly wrong.
    @Test func ropeLeavesTheUnrotatedTailAlone() {
        var values = (0..<8).map { Float($0 + 1) }
        CPUOps.applyRoPE(&values, headDim: 8, rotaryDim: 4, position: 3, theta: 10_000)
        #expect(values[4...] == ArraySlice((4..<8).map { Float($0 + 1) }))
        #expect(values[0] != 1, "the rotated prefix must actually move")
    }

    /// The pairs are `(i, i + rotaryDim/2)` — a half-split inside the
    /// rotated prefix, not adjacent elements. A rotation preserves the
    /// length of each pair, which is what pins the pairing.
    @Test func ropePairsAcrossTheHalfOfTheRotatedPrefix() {
        var values: [Float] = [1, 0, 0, 0, 9, 9, 9, 9]
        CPUOps.applyRoPE(&values, headDim: 8, rotaryDim: 4, position: 5, theta: 10_000)
        let paired = values[0] * values[0] + values[2] * values[2]
        #expect(abs(paired - 1) < 1e-5, "index 0 rotates against index 2, not index 1")
    }

    @Test func activationsMatchTheirDefinitions() {
        #expect(abs(CPUOps.sigmoid(0) - 0.5) < 1e-6)
        #expect(abs(CPUOps.silu(0)) < 1e-6)
        #expect(abs(CPUOps.silu(1) - 1 / (1 + expf(-1))) < 1e-6)
        // softplus must not overflow where a naive log(1 + e^x) would.
        #expect(abs(CPUOps.softplus(100) - 100) < 1e-3)
        #expect(CPUOps.softplus(-100) >= 0)
        #expect(abs(CPUOps.softplus(0) - logf(2)) < 1e-6)
    }
}

/// The CPU serving path: which architectures it will take, how it samples,
/// and the ceilings it enforces rather than promises.
@Suite struct CPUServingTests {

    /// A snapshot outside the family is refused by name. A forward pass on
    /// the wrong layer shape does not crash — it produces fluent nonsense,
    /// which is the same failure as a bad converter with a longer feedback
    /// loop.
    @Test func onlyKnownArchitecturesAreServed() {
        #expect(CPUModelFamily.resolve(modelType: "qwen3_5_dense") == .qwen35Dense)
        #expect(CPUModelFamily.resolve(modelType: "qwen3_5_text") == .qwen35Dense)
        #expect(CPUModelFamily.resolve(modelType: "llama") == nil)
        #expect(CPUModelFamily.resolve(modelType: nil) == nil)

        let refusal = CPUModelFamily.refusal(modelType: "llama")
        #expect(refusal.contains("llama"), "say which one was refused")
        #expect(refusal.contains("qwen3_5_dense"), "and which are served")
    }

    /// Greedy is the default and has to be exactly greedy: the memory work
    /// compares runs against each other, which only means anything if the
    /// same prompt gives the same answer.
    @Test func greedyPicksTheMaximumEveryTime() {
        let sampler = CPUSampler()
        #expect(sampler.isGreedy)
        let logits: [Float] = [0.1, 5.0, -2, 4.9, 0]
        let generator = sampler.makeGenerator()
        for _ in 0..<20 {
            #expect(sampler.pick(logits, using: generator) == 1)
        }
    }

    /// Top-k of one is greedy however hot the temperature, which is the
    /// property that makes the two knobs composable.
    @Test func topKOfOneIsGreedy() {
        let sampler = CPUSampler(temperature: 2, topP: 1, topK: 1, seed: 7)
        let logits: [Float] = [0.1, 5.0, -2, 4.9, 0]
        let generator = sampler.makeGenerator()
        for _ in 0..<20 {
            #expect(sampler.pick(logits, using: generator) == 1)
        }
    }

    /// Sampling stays inside the distribution it was given: nothing outside
    /// the top-k may ever be chosen, however the random draw falls.
    @Test func samplingNeverEscapesTopK() {
        let sampler = CPUSampler(temperature: 1.5, topP: 1, topK: 2, seed: 99)
        let logits: [Float] = [0.1, 5.0, -2, 4.9, 0]
        let generator = sampler.makeGenerator()
        var seen = Set<Int>()
        for _ in 0..<200 { seen.insert(sampler.pick(logits, using: generator)) }
        #expect(seen.isSubset(of: [1, 3]), "chose from outside the top two: \(seen)")
    }

    /// A seeded sampler repeats itself, so a run can be reproduced.
    @Test func aSeededSamplerIsReproducible() {
        let logits: [Float] = [1, 2, 3, 2, 1]
        func draw() -> [Int] {
            let sampler = CPUSampler(temperature: 1, topP: 1, topK: 0, seed: 42)
            let generator = sampler.makeGenerator()
            return (0..<30).map { _ in sampler.pick(logits, using: generator) }
        }
        #expect(draw() == draw())
    }

    /// Top-p trims by mass. With almost all of it on one token, a small p
    /// leaves only that token.
    @Test func topPKeepsOnlyTheMass() {
        let sampler = CPUSampler(temperature: 0.5, topP: 0.5, topK: 0, seed: 3)
        let logits: [Float] = [0, 10, 0, 0]
        let generator = sampler.makeGenerator()
        for _ in 0..<20 { #expect(sampler.pick(logits, using: generator) == 1) }
    }

    /// AUD-176. A presence penalty takes a token the history already contains
    /// out of contention, and it does it while greedy: the GPU sampler edits
    /// the logits before the argmax, so a penalised greedy run is not the plain
    /// argmax either.
    @Test func presencePenaltyTakesASeenTokenOutOfContention() {
        let logits: [Float] = [1, 5, 4]
        let generator = CPUSampler(presencePenalty: 10).makeGenerator()
        #expect(CPUSampler(presencePenalty: 10).pick(logits, history: [], using: generator) == 1)
        #expect(CPUSampler(presencePenalty: 10).pick(logits, history: [1], using: generator) == 2)
    }

    /// AUD-176. The repetition penalty scales by the logit's **sign**, matching
    /// `Sampler.applyPenaltiesInPlace`: a positive logit is divided and a
    /// negative one is multiplied, so both move away from being chosen. Testing
    /// only the positive case would let an implementation that always divided
    /// pass, and that one *raises* a negative logit toward the top.
    @Test func repetitionPenaltyScalesBySignNotByMagnitude() {
        let logits: [Float] = [3, 2.5]
        let negativeLogits: [Float] = [-2, -2.5]
        let generator = CPUSampler(repetitionPenalty: 2).makeGenerator()
        // 3 / 2 = 1.5, under the 2.5 that was second.
        #expect(CPUSampler(repetitionPenalty: 2).pick(logits, history: [0], using: generator) == 1)
        // -2 * 2 = -4, under the -2.5 that was second.
        #expect(
            CPUSampler(repetitionPenalty: 2)
                .pick(negativeLogits, history: [0], using: generator) == 1)
        #expect(
            CPUSampler(repetitionPenalty: 2)
                .pick(negativeLogits, history: [], using: generator) == 0)
    }

    /// Both penalties at their defaults must change nothing, or this fix would
    /// have traded one wrong answer for a slower right one: every request that
    /// does not ask for a penalty keeps the distribution it had.
    @Test func penaltiesAreNeutralAtTheirDefaults() {
        let logits: [Float] = [0.1, 5.0, -2, 4.9, 0]
        let history: [Int32] = [1, 2, 3, 4]
        let sampler = CPUSampler(
            temperature: 0, topP: 1, topK: 0,
            presencePenalty: 0, repetitionPenalty: 1, seed: 5)
        let generator = sampler.makeGenerator()
        for _ in 0..<20 {
            #expect(sampler.pick(logits, history: history, using: generator) == 1)
        }
    }

    /// The history is model output, and an id outside the row is a bad id, not
    /// a crash: this engine takes a snapshot a converter wrote, so a shape
    /// mismatch has to cost a token, not the process.
    @Test func aHistoryIdPastTheRowIsSkippedRatherThanFatal() {
        let logits: [Float] = [1, 5, 4]
        let sampler = CPUSampler(presencePenalty: 2, repetitionPenalty: 1.5)
        let generator = sampler.makeGenerator()
        #expect(sampler.pick(logits, history: [999, -3], using: generator) == 1)
    }

}
