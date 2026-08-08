import Testing
import Foundation
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore
import TurboFieldfareValidationSupport

/// Stage-D capstone: repack a synthetic official-layout Kimi K3 checkpoint
/// (safetensors + index + config on disk) into a `.gturbo` v2 bundle through
/// the real K3 planner/driver, load that bundle with `K3Engine`, and check
/// prefill logits against the `K3ForwardReference` CPU oracle — the oracle
/// tensors are read back out of the repacked bundle itself, so the engine and
/// the oracle see the exact same quantized values.
///
/// Also pins the K3 trunk quantizer to the house CPU references
/// (`Quantization.quantizeInt4Affine` / `quantizeInt8Affine`) bit-for-bit.
@Suite(.serialized)
struct K3RepackEndToEndTests {

    // MARK: - Quantizer bit parity with the house reference

    @Test func trunkQuantizerMatchesHouseReferenceBitForBit() throws {
        var rng = SplitMix64(seed: 0xB17C_A170)
        func row(_ count: Int, _ lo: Float, _ hi: Float) -> [Float] {
            (0..<count).map { _ in rng.uniform(lo, hi) }
        }
        var rows: [[Float]] = [
            row(64, -0.06, 0.06),
            row(128, -0.5, 0.5),
            row(256, -1, 1),
            [Float](repeating: 2.5, count: 64),          // constant group
            [Float](repeating: 0, count: 128),           // all zero
            row(64, -1e-4, 1e-4),                        // tiny magnitudes
            (-60..<68).map { Float($0) },                // exact integers, 2 groups
        ]
        rows.append(row(192, -30, 30))
        for values in rows {
            let q4 = K3TrunkQuantizer.quantizeRowAffine(values, bits: 4)
            let h4 = Quantization.quantizeInt4Affine(values)
            #expect(q4.packed == h4.packed)
            #expect(q4.scales == h4.scales)
            #expect(q4.biases == h4.biases)
            let q8 = K3TrunkQuantizer.quantizeRowAffine(values, bits: 8)
            let h8 = Quantization.quantizeInt8Affine(values)
            #expect(q8.packed == h8.packed)
            #expect(q8.scales == h8.scales)
            #expect(q8.biases == h8.biases)
        }
    }

    // MARK: - (e) repack -> K3Engine.load -> prefill logits vs oracle

    @Test func repackedBundlePrefillLogitsMatchOracle() async throws {
        for trunkQuant in K3TrunkQuant.allCases {
            try await Self.verifyRepackedBundlePrefill(trunkQuant: trunkQuant)
        }
    }

    static func verifyRepackedBundlePrefill(trunkQuant: K3TrunkQuant) async throws {
        let config = K3ForwardRunnerTests.tinyConfig()
        let snapshotDir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent(
                "turbofieldfare-k3e2e-\(trunkQuant.rawValue)-src-\(UUID().uuidString)")
        let output = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent(
                "turbofieldfare-k3e2e-\(trunkQuant.rawValue)-\(UUID().uuidString).gturbo")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: output)
            try? FileManager.default.removeItem(atPath: output + ".partial")
            try? FileManager.default.removeItem(atPath: output + ".resume.json")
            try? FileManager.default.removeItem(atPath: output + ".install.lock")
        }
        _ = try K3SyntheticCheckpoint.build(at: snapshotDir)

        let result = try await K3RemoteStreamingRepacker(
            options: K3RemoteStreamingRepackOptions(
                repoID: "local/kimi-k3-synthetic",
                revision: "synthetic",
                outputDir: output,
                requireKnownSource: false,
                trunkQuant: trunkQuant,
                minFreeReserveBytes: 0,
                overwrite: true,
                rangeRetryAttempts: 0,
                retryBaseDelayNs: 0),
            source: K3LocalSnapshotSource(directory: snapshotDir)
        ).run()
        #expect(!result.dryRun)

        // The bundle loads through the real runtime: manifest arch/quant
        // contracts, resident schema, and the canonical MXFP4 expert layout
        // are all validated by K3Model.load.
        let bundleURL = URL(fileURLWithPath: output)
        let engine = try K3Engine.load(bundleURL: bundleURL,
                                       maxContext: 64,
                                       expecting: config,
                                       prefetchPolicy: .predict)

        // Oracle tensors: read back out of the bundle (affine rows through
        // the house dequant references; bf16/fp32 verbatim), so oracle and
        // engine compute with identical values.
        let oracle = try Self.makeOracle(bundleURL: bundleURL, config: config)

        for prompt in K3ForwardRunnerTests.prompts {
            var oracleRef = oracle
            var oracleLogits: [Float] = []
            for token in prompt { oracleLogits = oracleRef.forward(token: token) }

            let stats = try engine.generate(promptTokens: prompt,
                                            config: GenerationConfig(temperature: 0),
                                            maxNew: 1)
            #expect(stats.newTokens == 1)
            let gpuLogits = engine.lastLogits()
            let rel = RelError.compute(actual: gpuLogits, reference: oracleLogits)
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "prompt \(prompt.prefix(3))…: logits rel=\(rel)")
            #expect(K3ForwardRunnerTests.argmax(gpuLogits)
                        == K3ForwardRunnerTests.argmax(oracleLogits),
                    "prompt \(prompt.prefix(3))…: argmax mismatch")
        }
    }

    // MARK: - Oracle construction from the repacked bundle

    static func makeOracle(bundleURL: URL, config: K3ArchConfig) throws
        -> K3ForwardReference {
        let manifestData = try Data(contentsOf:
            bundleURL.appendingPathComponent("manifest.json"))
        let manifest = try GTurboManifestCodecV2.decode(manifestData)
        let residentData = try Data(contentsOf:
            bundleURL.appendingPathComponent("model_weights.bin"))
        var tensors: [String: [Float]] = [:]
        var shapes: [String: (rows: Int, columns: Int)] = [:]

        try residentData.withUnsafeBytes { raw in
            let header = try GTurboResidentIndexCodecV2.decodeHeader(raw)
            let entries = try GTurboResidentIndexCodecV2.decodeRegion(raw, header: header)
            let byName = Dictionary(entries.map { ($0.name, $0) },
                                    uniquingKeysWith: { _, _ in
                                        fatalError("duplicate resident entry")
                                    })
            func bytes(_ offset: UInt64, _ size: UInt64) -> [UInt8] {
                Array(raw[Int(offset)..<Int(offset + size)])
            }
            func u16s(_ offset: UInt64, _ count: Int) -> [UInt16] {
                (0..<count).map { i in
                    UInt16(raw[Int(offset) + 2 * i])
                        | (UInt16(raw[Int(offset) + 2 * i + 1]) << 8)
                }
            }
            func f32s(_ offset: UInt64, _ count: Int) -> [Float] {
                (0..<count).map { i in
                    var bits: UInt32 = 0
                    for j in 0..<4 {
                        bits |= UInt32(raw[Int(offset) + 4 * i + j]) << UInt32(8 * j)
                    }
                    return Float(bitPattern: bits)
                }
            }

            for item in K3Model.schemaEntries(config: config, quant: manifest.quant) {
                guard let entry = byName[item.name] else {
                    Issue.record("repacked bundle is missing \(item.name)")
                    continue
                }
                switch item.kind {
                case .affine(let rows, let columns, let bits):
                    let packedAll = bytes(entry.fileOffset, entry.sizeBytes)
                    let scalesAll = u16s(entry.scaleOffset, Int(entry.scaleSize) / 2)
                    let biasesAll = u16s(entry.biasOffset, Int(entry.biasSize) / 2)
                    let rowPacked = columns * bits / 8
                    let rowGroups = columns / Quantization.groupSize
                    var values: [Float] = []
                    values.reserveCapacity(rows * columns)
                    for row in 0..<rows {
                        let packed = Array(
                            packedAll[(row * rowPacked)..<((row + 1) * rowPacked)])
                        let scales = Array(
                            scalesAll[(row * rowGroups)..<((row + 1) * rowGroups)])
                        let biases = Array(
                            biasesAll[(row * rowGroups)..<((row + 1) * rowGroups)])
                        if bits == 4 {
                            values.append(contentsOf: Quantization.dequantizeInt4Affine(
                                Quantization.Int4AffineRow(packed: packed,
                                                           scales: scales,
                                                           biases: biases),
                                n: columns))
                        } else {
                            precondition(bits == 8)
                            values.append(contentsOf: Quantization.dequantizeInt8Affine(
                                Quantization.Int8AffineRow(packed: packed,
                                                           scales: scales,
                                                           biases: biases),
                                n: columns))
                        }
                    }
                    tensors[item.name] = values
                    shapes[item.name] = (rows, columns)
                case .bf16Vector(let count):
                    tensors[item.name] = u16s(entry.fileOffset, count).map {
                        Quantization.bf16ToFloat($0)
                    }
                    shapes[item.name] = (count, 0)
                case .fp32Vector(let count):
                    tensors[item.name] = f32s(entry.fileOffset, count)
                    shapes[item.name] = (count, 0)
                case .fp32Matrix(let rows, let columns):
                    tensors[item.name] = f32s(entry.fileOffset, rows * columns)
                    shapes[item.name] = (rows, columns)
                }
            }
        }

        // Expert blobs: the repacked bytes are bit-identical to the source
        // MXFP4 payload (asserted by the repack suite), so the oracle can
        // read them straight from the bundle's layer files.
        let dLatent = UInt32(config.moeLatentBottleneckSize)
        let inter = UInt32(config.moeExpertIntermediateSize)
        let blobSize = Int(K3ExpertSubtensorOffsets.canonicalBlobSize(
            dLatent: dLatent, intermediate: inter))
        var expertBlobs: [Int: [[UInt8]]] = [:]
        for layer in config.moeLayers0.sorted() {
            let path = bundleURL
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(String(format: "layer_%02d.bin", layer))
            let data = try Data(contentsOf: path)
            var blobs: [[UInt8]] = []
            for expert in 0..<config.moeNumExperts {
                let start = expert * Int(config.expertStride)
                blobs.append(Array(data[start..<(start + blobSize)]))
            }
            expertBlobs[layer] = blobs
        }
        return K3ForwardReference(config: config,
                                  tensors: tensors,
                                  shapes: shapes,
                                  expertBlobs: expertBlobs)
    }
}
