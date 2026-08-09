import Testing
import Foundation
import Darwin
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareCLICore
import TurboFieldfareValidationSupport

/// Synthetic end-to-end for the Stage-C2 stack: a TINY K3 v2 bundle
/// fabricated in a temp dir (same writer pattern as `K3ModelLoaderTests`),
/// loaded through `K3Engine`, and checked against the independent fp32 CPU
/// oracle `K3ForwardReference` — logits after prefill, greedy generation
/// token-for-token, seeded sampling against `K3SamplerReference`, reset
/// determinism, and the first-token expert-streaming miss counts.
///
/// Tiny dims: hidden 256, 5 layers (0 KDA+dense, 1 KDA, 2 KDA, 3 MLA,
/// 4 KDA), KDA 4 heads x 32, MLA latent 64 / rope 8 / nope 16 / v 16 /
/// heads 4, AttnRes block size 2 (boundaries at layers 0, 2, 4), MoE 16
/// experts top-4, latent 128, vocab 512, maxContext 64.
///
/// Three dims deviate from the stage plan, forced by the quant contracts the
/// C1 schema pins: qLoraRank 64 (not 32 — the q_b GEMV's columns must be a
/// multiple of the affine group size 64), expert intermediate 64 (not 48 —
/// MXFP4 needs a multiple of 32 along every packed axis), and shared-expert
/// intermediate 64 (not 48 — the shared down-projection's columns must be a
/// multiple of 64).
@Suite struct K3ForwardRunnerTests {

    @Test func externalExpertStripesMatchCanonicalBundle() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let context = try MetalContext()
        let model = try K3Model.load(
            bundleURL: fixture.url,
            device: context.device,
            expecting: fixture.config)
        var roots: [URL] = []
        for shard in 0..<2 {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("k3-expert-shard-\(shard)-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            roots.append(root)
            var layers: [K3ExpertShardDescriptor.Layer] = []
            for layer in fixture.config.moeLayers0.sorted() {
                let experts = stride(
                    from: shard,
                    to: fixture.config.expertsPerLayer,
                    by: 2).map { $0 }
                let file = String(format: "layer_%02d.bin", layer)
                let url = root.appendingPathComponent(file)
                var bytes: [UInt8] = []
                for expert in experts {
                    let blob = fixture.expertBlobs[layer]![expert]
                    bytes += blob
                    bytes += [UInt8](
                        repeating: 0,
                        count: Int(fixture.config.expertStride) - blob.count)
                }
                try Data(bytes).write(to: url)
                layers.append(.init(
                    layer: layer,
                    file: file,
                    experts: experts,
                    size: UInt64(bytes.count),
                    sha256: try Sha256Verifier.hashFile(at: url)))
            }
            let descriptor = K3ExpertShardDescriptor(
                modelID: model.modelID,
                sourceSnapshotHash: model.sourceSnapshotHash,
                expertStride: fixture.config.expertStride,
                expertsPerLayer: fixture.config.expertsPerLayer,
                shardIndex: shard,
                shardCount: 2,
                layers: layers)
            try JSONEncoder().encode(descriptor).write(
                to: root.appendingPathComponent("expert-shard.json"))
        }
        defer { for root in roots { try? FileManager.default.removeItem(at: root) } }

        let prompt: [Int32] = [7, 11, 19]
        let canonical = try K3Engine.load(
            bundleURL: fixture.url,
            maxContext: 64,
            expecting: fixture.config,
            prefetchPolicy: .off)
        _ = try canonical.generate(
            promptTokens: prompt,
            config: GenerationConfig(temperature: 0),
            maxNew: 1,
            prefillMode: .chunked(chunkTokens: 32))

        let striped = try K3Engine.load(
            bundleURL: fixture.url,
            maxContext: 64,
            expecting: fixture.config,
            prefetchPolicy: .off,
            expertShardRoots: roots)
        _ = try striped.generate(
            promptTokens: prompt,
            config: GenerationConfig(temperature: 0),
            maxNew: 1,
            prefillMode: .chunked(chunkTokens: 32))

        #expect(RelError.maxAbsDiff(
            canonical.lastLogits(), striped.lastLogits()) == 0)
    }

    @Test func directResidentBanksMatchCanonicalDecodeExactly() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let prompt: [Int32] = [7, 11, 19]
        let generation = GenerationConfig(temperature: 0)

        let canonical = try K3Engine.load(
            bundleURL: fixture.url,
            maxContext: 64,
            expecting: fixture.config,
            prefetchPolicy: .off)
        var canonicalTokens: [Int32] = []
        _ = try canonical.generate(
            promptTokens: prompt,
            config: generation,
            maxNew: 3,
            prefillMode: .chunked(chunkTokens: 32),
            onToken: { canonicalTokens.append($0) })

        let residentBytes = UInt64(
            fixture.config.moeLayers0.count * fixture.config.moeTopKExperts)
            * fixture.config.expertStride
        let resident = try K3Engine.load(
            bundleURL: fixture.url,
            maxContext: 64,
            expecting: fixture.config,
            prefetchPolicy: .off,
            residentExpertCacheBytes: residentBytes)
        var residentTokens: [Int32] = []
        let stats = try resident.generate(
            promptTokens: prompt,
            config: generation,
            maxNew: 3,
            prefillMode: .chunked(chunkTokens: 32),
            onToken: { residentTokens.append($0) })

        #expect(residentTokens == canonicalTokens)
        #expect(RelError.maxAbsDiff(
            canonical.lastLogits(), resident.lastLogits()) == 0)
        #expect(stats.expertStreaming.residentCacheCapacity
            == fixture.config.moeLayers0.count * fixture.config.moeTopKExperts)
        #expect(stats.expertStreaming.residentCacheCopyBytes == 0)
        #expect(stats.expertStreaming.residentCachePopulateBytes == 0)
    }

    @Test func cliJSONLBatchLoadsOnceAndEmitsOneResultPerJob() async throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let tokenizerSource = try #require(Bundle.module.url(
            forResource: "tiktoken.model",
            withExtension: nil,
            subdirectory: "Fixtures/k3"))
        let tokenizerDirectory = fixture.url.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(
            at: tokenizerDirectory, withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: tokenizerSource,
            to: tokenizerDirectory.appendingPathComponent("tiktoken.model"))

        let temporary = FileManager.default.temporaryDirectory
        let batchURL = temporary.appendingPathComponent(
            "k3-batch-\(UUID().uuidString).jsonl")
        let outputURL = temporary.appendingPathComponent(
            "k3-batch-out-\(UUID().uuidString).jsonl")
        let errorURL = temporary.appendingPathComponent(
            "k3-batch-err-\(UUID().uuidString).txt")
        defer {
            try? FileManager.default.removeItem(at: batchURL)
            try? FileManager.default.removeItem(at: outputURL)
            try? FileManager.default.removeItem(at: errorURL)
        }
        try Data(("{\"id\":\"a\",\"prompt\":\"!\",\"max_new\":1}\n"
            + "{\"id\":\"b\",\"prompt\":\"!\",\"max_new\":1}\n").utf8)
            .write(to: batchURL)
        FileManager.default.createFile(atPath: outputURL.path, contents: Data())
        FileManager.default.createFile(atPath: errorURL.path, contents: Data())
        let output = try FileHandle(forWritingTo: outputURL)
        let errors = try FileHandle(forWritingTo: errorURL)
        defer {
            try? output.close()
            try? errors.close()
        }

        let args = Args(
            model: fixture.url.path,
            batchFile: batchURL.path,
            maxNew: 1,
            maxNewExplicit: true,
            maxContext: 64,
            temperature: 0,
            temperatureExplicit: true,
            expertIOWorkers: "1")
        let result = await runK3(
            args: args,
            stdout: output,
            stderr: errors,
            expecting: fixture.config)
        try errors.synchronize()
        let errorText = try String(contentsOf: errorURL, encoding: .utf8)
        #expect(result.exitCode == 0, "\(errorText)")
        try output.synchronize()
        let lines = try String(contentsOf: outputURL, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
        #expect(lines.count == 2)
        for (index, line) in lines.enumerated() {
            let object = try #require(try JSONSerialization.jsonObject(
                with: Data(line.utf8)) as? [String: Any])
            #expect(object["index"] as? Int == index)
            #expect(object["error"] == nil)
            #expect(object["text"] is String)
            #expect(object["new_tokens"] as? Int == 1)
        }
    }

    @Test func realWeightActivationProbeMatchesIndependentReference() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        let result = try engine.activationDiagnostics(token: 7)
        #expect(result.embedding.elements == fixture.config.hiddenSize)
        #expect(result.layer0InputNorm.elements == fixture.config.hiddenSize)
        #expect(result.passed, "\(result.summaryLine)")
    }

    @Test func chunkedDiagnosticCapturesRouterAndHead() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        let prompt: [Int32] = [7, 11, 19]
        _ = try engine.activationDiagnostics(token: prompt[0])
        _ = try engine.generate(promptTokens: prompt,
                                config: GenerationConfig(temperature: 0),
                                maxNew: 1,
                                prefillMode: .chunked(chunkTokens: 32))
        let router = try #require(engine.routerActivationDiagnostics())
        #expect(router.passed, "\(router.summaryLine)")
        let logits = engine.lastLogits()
        let top = logits.enumerated().max { $0.element < $1.element }!.offset
        let head = try #require(try engine.headActivationDiagnostics(tokenIDs: [top]))
        #expect(head.passed, "\(head.summaryLine)")
    }

    /// Pins the first production-size AttnRes boundary. The official model
    /// appends the stream entering layer 12, not the result of combining that
    /// stream with the layer-0 block. Check the independent CPU lifecycle and
    /// both Metal orchestration paths against that exact vector.
    @Test func layer12BoundaryAppendsPreAttnResInput() throws {
        let config = Self.layer12BoundaryConfig()
        let fixture = try Self.makeFixture(config: config)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let token: Int32 = 17

        var oracle = Self.makeOracle(fixture)
        _ = oracle.forward(token: token)
        let incoming = try #require(oracle.lastAttnResBoundaryInputs[12])
        #expect(oracle.lastAttnResBlocks.count == 2)
        #expect(oracle.lastAttnResBlocks[1] == incoming)

        // Prove the fixture distinguishes the old bug: the pre-attn combined
        // value is materially different from the incoming value that must be
        // stored in block slot 1.
        let lp = "language_model.model.layers.12"
        let norm = try #require(
            fixture.oracleTensors["\(lp).self_attention_res_norm.weight"])
        let proj = try #require(
            fixture.oracleTensors["\(lp).self_attention_res_proj.weight"])
        let scoreVector = zip(norm, proj).map(*)
        let oldWrongValue = K3AttnResReference.apply(
            blocks: [oracle.lastAttnResBlocks[0]], prefix: incoming,
            scoreVector: scoreVector, eps: Float(config.rmsNormEpsilon))
            .map { Float(Float16($0)) }
        #expect(RelError.maxAbsDiff(oldWrongValue, incoming) > 1e-3)

        let context = try MetalContext()
        let model = try K3Model.load(bundleURL: fixture.url, device: context.device,
                                     expecting: config)
        let state = try K3State(device: context.device, config: config,
                                maxContext: 32)
        let streaming = try K3ExpertStreaming(model: model)
        let runner = try K3ForwardRunner(model: model, context: context,
                                         state: state, streaming: streaming)

        try runner.produce(token: token, position: 0, emitHead: true)
        #expect(state.attnResBlockCount == 2)
        let serialPtr = state.attnResBlocks.contents().bindMemory(
            to: Float16.self, capacity: K3AttnRes.maxBlocks * config.hiddenSize)
        let serialBlock = (0..<config.hiddenSize).map {
            Float(serialPtr[config.hiddenSize + $0])
        }
        let serialRel = RelError.compute(actual: serialBlock, reference: incoming)
        #expect(serialRel < 2e-2, "layer 12 serial block rel=\(serialRel)")

        state.reset()
        streaming.resetPrediction()
        let prefiller = try K3ChunkedPrefiller(
            model: model, context: context, state: state,
            shared: runner.prefillShared, streaming: streaming,
            chunkTokens: 32, forceFallback: true)
        prefiller.enableActivationTrace()
        try prefiller.prefill(tokens: [token])
        let chunkedBlocks = try #require(prefiller.lastDiagnosticAttnResBlocks)
        #expect(chunkedBlocks.count == 2)
        let chunkedRel = RelError.compute(actual: chunkedBlocks[1], reference: incoming)
        #expect(chunkedRel < 2e-2, "layer 12 chunked block rel=\(chunkedRel)")
    }

    static let seed: UInt64 = 0xC2E2

    static func tinyConfig() -> K3ArchConfig {
        K3ArchConfig(
            hiddenSize: 256, vocabSize: 512, numLayers: 5,
            denseMLPIntermediateSize: 256,
            rmsNormEpsilon: 1e-5, tieWordEmbeddings: false,
            hiddenActivation: "situ_glu", bosTokenID: 500, eosTokenID: 501,
            denseLayers: [1], kdaLayers: [1, 2, 3, 5], fullAttnLayers: [4],
            kdaNumHeads: 4, kdaHeadDim: 32, kdaConvWidth: 4,
            kdaDecayLowRankSize: 64, kdaDecayProjectionSize: 128,
            kdaGateLowerBound: -5.0, kdaFullRankOutputGate: true,
            mlaNumHeads: 4, mlaQLoraRank: 64, mlaKVLoraRank: 64,
            mlaQKNopeHeadDim: 16, mlaQKRopeHeadDim: 8, mlaVHeadDim: 16,
            mlaOutputGate: true,
            attnResBlockSize: 2,
            moeNumExperts: 16, moeTopKExperts: 4,
            moeLatentBottleneckSize: 128, moeExpertIntermediateSize: 64,
            moeNumSharedExperts: 1, moeSharedExpertIntermediateSize: 64,
            situGLUGateBeta: 4.0, situGLUUpBeta: 25.0,
            routerRenormalize: true, routerCorrectionBias: true,
            expertsPerLayer: 16, expertStride: 16_384)
    }

    /// Same small tensor geometry with the production AttnRes block size and
    /// enough depth to exercise the first non-trivial boundary at layer 12.
    static func layer12BoundaryConfig() -> K3ArchConfig {
        K3ArchConfig(
            hiddenSize: 256, vocabSize: 512, numLayers: 13,
            denseMLPIntermediateSize: 256,
            rmsNormEpsilon: 1e-5, tieWordEmbeddings: false,
            hiddenActivation: "situ_glu", bosTokenID: 500, eosTokenID: 501,
            denseLayers: [1],
            kdaLayers: [1, 2, 3, 5, 6, 7, 9, 10, 11, 13],
            fullAttnLayers: [4, 8, 12],
            kdaNumHeads: 4, kdaHeadDim: 32, kdaConvWidth: 4,
            kdaDecayLowRankSize: 64, kdaDecayProjectionSize: 128,
            kdaGateLowerBound: -5.0, kdaFullRankOutputGate: true,
            mlaNumHeads: 4, mlaQLoraRank: 64, mlaKVLoraRank: 64,
            mlaQKNopeHeadDim: 16, mlaQKRopeHeadDim: 8, mlaVHeadDim: 16,
            mlaOutputGate: true,
            attnResBlockSize: 12,
            moeNumExperts: 16, moeTopKExperts: 4,
            moeLatentBottleneckSize: 128, moeExpertIntermediateSize: 64,
            moeNumSharedExperts: 1, moeSharedExpertIntermediateSize: 64,
            situGLUGateBeta: 4.0, situGLUUpBeta: 25.0,
            routerRenormalize: true, routerCorrectionBias: true,
            expertsPerLayer: 16, expertStride: 16_384)
    }

    struct Fixture {
        let url: URL
        let config: K3ArchConfig
        /// Exact fp32 values the GPU pipeline computes with (dequantized
        /// affine rows, exact bf16 values, raw fp32), keyed by checkpoint
        /// tensor name.
        var oracleTensors: [String: [Float]]
        var oracleShapes: [String: (rows: Int, columns: Int)]
        /// MoE layer0 -> per-expert packed MXFP4 blob (expert-stride-padded).
        var expertBlobs: [Int: [[UInt8]]]
    }

    // MARK: - Weight synthesis

    /// Per-tensor magnitude policy: plausible, activation-friendly ranges,
    /// deterministic per tensor name via SeedTree.
    static func masterValues(name: String, count: Int, rng: inout SplitMix64)
        -> [Float] {
        func uniform(_ lo: Float, _ hi: Float) -> [Float] {
            (0..<count).map { _ in rng.uniform(lo, hi) }
        }
        if name.hasSuffix("embed_tokens.weight") { return uniform(-0.5, 0.5) }
        if name.hasSuffix("lm_head.weight") { return uniform(-0.2, 0.2) }
        if name.hasSuffix("_res_proj.weight") { return uniform(-0.15, 0.15) }
        if name.hasSuffix("_res_norm.weight")
            || name.hasSuffix("routed_expert_norm.weight") {
            return uniform(0.7, 1.3)
        }
        if name.hasSuffix("layernorm.weight") || name.hasSuffix("norm.weight") {
            return uniform(0.7, 1.3)
        }
        if name.hasSuffix("o_norm.weight") { return uniform(0.5, 1.5) }
        if name.hasSuffix("A_log") { return uniform(0.0, 2.8) }
        if name.hasSuffix("dt_bias") { return uniform(-0.3, 0.3) }
        if name.hasSuffix("conv1d.weight") { return uniform(-0.25, 0.25) }
        if name.hasSuffix("conv1d.bias") {
            // The decode conv is bias-free (verified C reference contract);
            // the tensors exist in the schema but are intentionally unused.
            return [Float](repeating: 0, count: count)
        }
        if name.hasSuffix("block_sparse_moe.gate.weight") {
            return uniform(-0.05, 0.05)
        }
        if name.hasSuffix("e_score_correction_bias") {
            return uniform(-0.25, 0.25)
        }
        // Every remaining tensor is an int4 projection matrix.
        return uniform(-0.06, 0.06)
    }

    struct Payload {
        let name: String
        let dtype: UInt8
        let shape: [UInt32]
        var weights: [UInt8]
        var scales: [UInt8]
        var biases: [UInt8]
        let oracle: [Float]
        let oracleShape: (rows: Int, columns: Int)
    }

    static func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
        for value in values {
            bytes.append(UInt8(truncatingIfNeeded: value))
            bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        }
    }

    static func appendF32(_ values: [Float], to bytes: inout [UInt8]) {
        for var value in values {
            withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
        }
    }

    /// Quantize one schema entry per the K3 quant profile, keeping the exact
    /// fp32 mirror (dequantized / bf16-exact / raw) for the oracle.
    static func synthesize(entry: K3Model.SchemaEntry,
                           tree: SeedTree) -> Payload {
        var rng = tree.key(entry.name)
        switch entry.kind {
        case .affine(let rows, let columns, let bits):
            let master = masterValues(name: entry.name, count: rows * columns,
                                      rng: &rng)
            var weights: [UInt8] = []
            var scales: [UInt8] = []
            var biases: [UInt8] = []
            var oracle = [Float]()
            oracle.reserveCapacity(rows * columns)
            for row in 0..<rows {
                let slice = Array(master[(row * columns)..<((row + 1) * columns)])
                if bits == 4 {
                    let q = Quantization.quantizeInt4Affine(slice)
                    weights.append(contentsOf: q.packed)
                    appendU16(q.scales, to: &scales)
                    appendU16(q.biases, to: &biases)
                    oracle.append(contentsOf: Quantization.dequantizeInt4Affine(
                        q, n: columns))
                } else {
                    precondition(bits == 8)
                    let q = Quantization.quantizeInt8Affine(slice)
                    weights.append(contentsOf: q.packed)
                    appendU16(q.scales, to: &scales)
                    appendU16(q.biases, to: &biases)
                    oracle.append(contentsOf: Quantization.dequantizeInt8Affine(
                        q, n: columns))
                }
            }
            return Payload(name: entry.name,
                           dtype: GTurboFormatV1.DType.u32.rawValue,
                           shape: [UInt32(rows), UInt32(columns), 0, 0],
                           weights: weights, scales: scales, biases: biases,
                           oracle: oracle, oracleShape: (rows, columns))
        case .bf16Vector(let count):
            let master = masterValues(name: entry.name, count: count, rng: &rng)
            let exact = master.map { Quantization.bf16ToFloat(Quantization.bf16Bits($0)) }
            var weights: [UInt8] = []
            appendU16(exact.map { Quantization.bf16Bits($0) }, to: &weights)
            return Payload(name: entry.name,
                           dtype: GTurboFormatV1.DType.bf16.rawValue,
                           shape: [UInt32(count), 0, 0, 0],
                           weights: weights, scales: [], biases: [],
                           oracle: exact, oracleShape: (count, 0))
        case .fp32Vector(let count):
            let master = masterValues(name: entry.name, count: count, rng: &rng)
            var weights: [UInt8] = []
            appendF32(master, to: &weights)
            return Payload(name: entry.name,
                           dtype: GTurboFormatV1.DType.fp32.rawValue,
                           shape: [UInt32(count), 0, 0, 0],
                           weights: weights, scales: [], biases: [],
                           oracle: master, oracleShape: (count, 0))
        case .fp32Matrix(let rows, let columns):
            let master = masterValues(name: entry.name, count: rows * columns,
                                      rng: &rng)
            var weights: [UInt8] = []
            appendF32(master, to: &weights)
            return Payload(name: entry.name,
                           dtype: GTurboFormatV1.DType.fp32.rawValue,
                           shape: [UInt32(rows), UInt32(columns), 0, 0],
                           weights: weights, scales: [], biases: [],
                           oracle: master, oracleShape: (rows, columns))
        }
    }

    /// One MXFP4 expert blob: random E2M1 nibbles with small E8M0 scales
    /// (plausible magnitudes). The oracle dequantizes the same bytes, so the
    /// expert weights carry zero quantization drift by construction.
    static func synthesizeExpertBlob(layer: Int, expert: Int,
                                     config: K3ArchConfig,
                                     tree: SeedTree) -> [UInt8] {
        let dLatent = UInt32(config.moeLatentBottleneckSize)
        let inter = UInt32(config.moeExpertIntermediateSize)
        let offsets = K3ExpertSubtensorOffsets.canonical(dLatent: dLatent,
                                                         intermediate: inter)
        let blobSize = Int(K3ExpertSubtensorOffsets.canonicalBlobSize(
            dLatent: dLatent, intermediate: inter))
        var rng = tree.key("expert-\(layer)-\(expert)")
        var blob = [UInt8](repeating: 0, count: blobSize)
        let packedRegions = [offsets.w1PackedOff, offsets.w2PackedOff,
                             offsets.w3PackedOff]
        let scaleRegions = [offsets.w1ScalesOff, offsets.w2ScalesOff,
                            offsets.w3ScalesOff]
        let packedBytes = Int(dLatent) * Int(inter) / 2
        let scaleBytes = Int(dLatent) * Int(inter) / 32
        for region in packedRegions {
            for i in 0..<packedBytes {
                blob[Int(region) + i] = UInt8(truncatingIfNeeded: rng.next())
            }
        }
        for region in scaleRegions {
            for i in 0..<scaleBytes {
                // 2^-9 ... 2^-4 group scales; 255 (zero-group guard) avoided.
                blob[Int(region) + i] = UInt8(118 + rng.next() % 6)
            }
        }
        return blob
    }

    // MARK: - Bundle fabrication

    static func makeFixture(config: K3ArchConfig = tinyConfig()) throws -> Fixture {
        let tree = SeedTree(seed)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("k3-forward-toy-\(UUID().uuidString)")
        let expertsDir = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: expertsDir,
                                                withIntermediateDirectories: true)

        // 1. Resident payload: every schema tensor, in schema order.
        let schema = K3Model.schemaEntries(config: config)
        let payloads = schema.map { synthesize(entry: $0, tree: tree) }

        let names = payloads.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboFormatV1.residentHeaderBytes
        let entryBytes = GTurboFormatV1.residentEntryBytes
        let stringTableBase = headerBytes + names.count * entryBytes
        var nameOffsets: [UInt32] = []
        var nameCursor = 0
        for name in names {
            nameOffsets.append(UInt32(stringTableBase + nameCursor))
            nameCursor += name.utf8.count
        }
        let alignment = GTurboFormatV2.alignmentBytes
        let rawIndexBytes = UInt64(stringTableBase + stringTable.count)
        let indexSize = ((rawIndexBytes + alignment - 1) / alignment) * alignment

        var cursor = indexSize
        var entries: [GTurboResidentIndexEntryV2] = []
        var regions: [(offset: UInt64, bytes: [UInt8])] = []
        for payload in payloads {
            cursor = ((cursor + 15) / 16) * 16
            let weightOffset = cursor
            cursor += UInt64(payload.weights.count)
            let scaleOffset = payload.scales.isEmpty ? 0 : cursor
            cursor += UInt64(payload.scales.count)
            let biasOffset = payload.biases.isEmpty ? 0 : cursor
            cursor += UInt64(payload.biases.count)
            regions.append((weightOffset, payload.weights))
            if !payload.scales.isEmpty { regions.append((scaleOffset, payload.scales)) }
            if !payload.biases.isEmpty { regions.append((biasOffset, payload.biases)) }
            entries.append(GTurboResidentIndexEntryV2(
                name: payload.name, dtype: payload.dtype,
                fileOffset: weightOffset, sizeBytes: UInt64(payload.weights.count),
                shape: payload.shape,
                scaleOffset: scaleOffset, scaleSize: UInt64(payload.scales.count),
                biasOffset: biasOffset, biasSize: UInt64(payload.biases.count)))
        }
        let residentSize = cursor - indexSize

        var fileBuf = [UInt8](repeating: 0, count: Int(cursor))
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboResidentIndexCodecV2.writeHeader(
                into: base,
                header: GTurboResidentIndexHeaderV2(
                    indexSize: indexSize, residentSize: residentSize,
                    entryCount: UInt64(entries.count)))
            for (index, entry) in entries.enumerated() {
                GTurboResidentIndexCodecV2.writeEntry(
                    into: base.advanced(by: headerBytes + index * entryBytes),
                    entry: entry, nameOffset: nameOffsets[index])
            }
            _ = stringTable.withUnsafeBytes { table in
                memcpy(base.advanced(by: stringTableBase), table.baseAddress!,
                       stringTable.count)
            }
            for region in regions {
                _ = region.bytes.withUnsafeBytes { bytes in
                    memcpy(base.advanced(by: Int(region.offset)), bytes.baseAddress!,
                           bytes.count)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 2. packed_experts: MoE layers 1...4, 16 experts each at the
        //    canonical 6-subtensor MXFP4 layout, stride-padded blobs.
        let dLatent = UInt32(config.moeLatentBottleneckSize)
        let inter = UInt32(config.moeExpertIntermediateSize)
        let offsets = K3ExpertSubtensorOffsets.canonical(dLatent: dLatent,
                                                         intermediate: inter)
        let stride = config.expertStride
        let moeLayers = config.moeLayers0.sorted()
        var expertBlobs: [Int: [[UInt8]]] = [:]
        var layerFiles: [String] = []
        for layer in moeLayers {
            let basename = String(format: "layer_%02d.bin", layer)
            layerFiles.append(basename)
            var bytes = [UInt8](repeating: 0, count: Int(stride) * config.moeNumExperts)
            var blobs: [[UInt8]] = []
            for expert in 0..<config.moeNumExperts {
                let blob = synthesizeExpertBlob(layer: layer, expert: expert,
                                                config: config, tree: tree)
                blobs.append(blob)
                bytes.replaceSubrange(
                    (expert * Int(stride))..<(expert * Int(stride) + blob.count),
                    with: blob)
            }
            expertBlobs[layer] = blobs
            try Data(bytes).write(to: expertsDir.appendingPathComponent(basename))
        }

        let subtensors: [String: GTurboSubTensorV2] = [
            "w1_packed": GTurboSubTensorV2(
                offset: UInt64(offsets.w1PackedOff),
                size: UInt64(inter) * UInt64(dLatent) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType, shape: [inter, dLatent],
                bits: GTurboFormatV2.mxfp4PackedBits),
            "w1_scales": GTurboSubTensorV2(
                offset: UInt64(offsets.w1ScalesOff),
                size: UInt64(inter) * UInt64(dLatent) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType, shape: [inter, dLatent / 32],
                bits: GTurboFormatV2.mxfp4ScaleBits),
            "w2_packed": GTurboSubTensorV2(
                offset: UInt64(offsets.w2PackedOff),
                size: UInt64(dLatent) * UInt64(inter) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType, shape: [dLatent, inter],
                bits: GTurboFormatV2.mxfp4PackedBits),
            "w2_scales": GTurboSubTensorV2(
                offset: UInt64(offsets.w2ScalesOff),
                size: UInt64(dLatent) * UInt64(inter) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType, shape: [dLatent, inter / 32],
                bits: GTurboFormatV2.mxfp4ScaleBits),
            "w3_packed": GTurboSubTensorV2(
                offset: UInt64(offsets.w3PackedOff),
                size: UInt64(inter) * UInt64(dLatent) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType, shape: [inter, dLatent],
                bits: GTurboFormatV2.mxfp4PackedBits),
            "w3_scales": GTurboSubTensorV2(
                offset: UInt64(offsets.w3ScalesOff),
                size: UInt64(inter) * UInt64(dLatent) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType, shape: [inter, dLatent / 32],
                bits: GTurboFormatV2.mxfp4ScaleBits),
        ]
        let layout = GTurboPackedExpertsLayoutV2(
            expertStride: stride,
            numLayers: config.numLayers,
            expertsPerLayer: config.expertsPerLayer,
            layers: moeLayers.map { layerID in
                GTurboLayerV2(
                    layer: layerID,
                    file: String(format: "layer_%02d.bin", layerID),
                    experts: (0..<config.moeNumExperts).map { expert in
                        GTurboExpertV2(
                            expert: expert, physicalRank: nil,
                            offset: UInt64(expert) * stride, size: stride,
                            tensors: subtensors)
                    })
            })
        let layoutData = try GTurboPackedExpertsLayoutCodecV2.encode(layout)
        try layoutData.write(to: expertsDir.appendingPathComponent("layout.json"))

        // 3. manifest.json with real sizes + hashes.
        var files: [String: GTurboManifestFileV2] = [
            "model_weights.bin": GTurboManifestFileV2(
                size: UInt64(fileBuf.count), sha256: weightsSha),
            "packed_experts/layout.json": GTurboManifestFileV2(
                size: UInt64(layoutData.count),
                sha256: Sha256Verifier.hashData(layoutData)),
        ]
        for basename in layerFiles {
            let sha = try Sha256Verifier.hashFile(
                at: expertsDir.appendingPathComponent(basename))
            files["packed_experts/\(basename)"] = GTurboManifestFileV2(
                size: stride * UInt64(config.moeNumExperts), sha256: sha)
        }
        let manifest = GTurboManifestV2(
            flags: KimiK3FormatProfile.flags,
            modelID: "fixture/kimi-k3-forward-tiny",
            sourceSnapshotHash: nil,
            arch: GTurboManifestArchV2(
                hiddenSize: config.hiddenSize, vocabSize: config.vocabSize,
                numLayers: config.numLayers,
                denseMLPIntermediateSize: config.denseMLPIntermediateSize,
                rmsNormEpsilon: config.rmsNormEpsilon,
                tieWordEmbeddings: config.tieWordEmbeddings,
                hiddenActivation: config.hiddenActivation,
                bosTokenID: config.bosTokenID, eosTokenID: config.eosTokenID,
                denseLayers: config.denseLayers, kdaLayers: config.kdaLayers,
                fullAttnLayers: config.fullAttnLayers,
                kda: GTurboManifestKDAV2(
                    numHeads: config.kdaNumHeads, headDim: config.kdaHeadDim,
                    convWidth: config.kdaConvWidth,
                    decayLowRankSize: config.kdaDecayLowRankSize,
                    decayProjectionSize: config.kdaDecayProjectionSize,
                    gateLowerBound: config.kdaGateLowerBound,
                    fullRankOutputGate: config.kdaFullRankOutputGate),
                mla: GTurboManifestMLAV2(
                    numHeads: config.mlaNumHeads, qLoraRank: config.mlaQLoraRank,
                    kvLoraRank: config.mlaKVLoraRank,
                    qkNopeHeadDim: config.mlaQKNopeHeadDim,
                    qkRopeHeadDim: config.mlaQKRopeHeadDim,
                    vHeadDim: config.mlaVHeadDim,
                    outputGate: config.mlaOutputGate),
                attnRes: GTurboManifestAttnResV2(blockSize: config.attnResBlockSize),
                moe: GTurboManifestMoEV2(
                    numExperts: config.moeNumExperts,
                    topKExperts: config.moeTopKExperts,
                    latentBottleneckSize: config.moeLatentBottleneckSize,
                    expertIntermediateSize: config.moeExpertIntermediateSize,
                    numSharedExperts: config.moeNumSharedExperts,
                    sharedExpertIntermediateSize: config.moeSharedExpertIntermediateSize,
                    situGLUGateBeta: config.situGLUGateBeta,
                    situGLUUpBeta: config.situGLUUpBeta,
                    routerRenormalize: config.routerRenormalize,
                    routerCorrectionBias: config.routerCorrectionBias)),
            quant: KimiK3FormatProfile.quant,
            files: files,
            expertsPerLayer: config.expertsPerLayer,
            numLayers: config.numLayers,
            expertStride: config.expertStride)
        try GTurboManifestCodecV2.encode(manifest)
            .write(to: dir.appendingPathComponent("manifest.json"))

        var oracleTensors: [String: [Float]] = [:]
        var oracleShapes: [String: (rows: Int, columns: Int)] = [:]
        for payload in payloads {
            oracleTensors[payload.name] = payload.oracle
            oracleShapes[payload.name] = payload.oracleShape
        }
        return Fixture(url: dir, config: config,
                       oracleTensors: oracleTensors,
                       oracleShapes: oracleShapes,
                       expertBlobs: expertBlobs)
    }

    // MARK: - Helpers

    static func makeEngine(_ fixture: Fixture) throws -> K3Engine {
        try K3Engine.load(bundleURL: fixture.url,
                          maxContext: 64,
                          expecting: fixture.config,
                          prefetchPolicy: .predict)
    }

    static func makeOracle(_ fixture: Fixture) -> K3ForwardReference {
        K3ForwardReference(config: fixture.config,
                           tensors: fixture.oracleTensors,
                           shapes: fixture.oracleShapes,
                           expertBlobs: fixture.expertBlobs)
    }

    static func argmax(_ logits: [Float]) -> Int32 {
        var best = 0
        for i in 1..<logits.count where logits[i] > logits[best] { best = i }
        return Int32(best)
    }

    /// The deterministic prompts (token ids in [0, 512)).
    static let prompts: [[Int32]] = [
        [3, 45, 190, 256, 11, 400, 77, 201],
        [500, 12, 33, 90, 128, 256, 384, 7],
        [42, 42, 100, 300, 210, 55, 499, 1],
    ]

    // MARK: - (a) logits after prefill match the CPU oracle

    @Test func prefillLogitsMatchOracle() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)

        for prompt in Self.prompts {
            var oracle = Self.makeOracle(fixture)
            var oracleLogits: [Float] = []
            for token in prompt { oracleLogits = oracle.forward(token: token) }

            let stats = try engine.generate(
                promptTokens: prompt,
                config: GenerationConfig(temperature: 0),
                maxNew: 1)
            #expect(stats.newTokens == 1)

            let gpuLogits = engine.lastLogits()
            // Measured rel is ~7e-4...1.3e-3 on this fixture (logit range
            // ~11), an order of magnitude inside the fp16-chained bar.
            let rel = RelError.compute(actual: gpuLogits, reference: oracleLogits)
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "prompt \(prompt.prefix(3))…: logits rel=\(rel)")
            #expect(Self.argmax(gpuLogits) == Self.argmax(oracleLogits),
                    "prompt \(prompt.prefix(3))…: argmax mismatch")
        }
    }

    // MARK: - (b) greedy generation matches oracle-greedy token-for-token

    @Test func greedyGenerationMatchesOracleTokenForToken() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)

        for prompt in Self.prompts {
            var emitted: [Int32] = []
            let stats = try engine.generate(
                promptTokens: prompt,
                config: GenerationConfig(temperature: 0),
                maxNew: 6,
                onToken: { emitted.append($0) })
            #expect(stats.newTokens == emitted.count
                    || stats.newTokens == emitted.count + 1)  // eos not emitted

            // Oracle greedy replay.
            var oracle = Self.makeOracle(fixture)
            var expected: [Int32] = []
            var logits: [Float] = []
            for token in prompt { logits = oracle.forward(token: token) }
            for _ in 0..<6 {
                let next = Self.argmax(logits)
                if next == Int32(fixture.config.eosTokenID) { break }
                expected.append(next)
                if expected.count >= 6 { break }
                logits = oracle.forward(token: next)
            }
            #expect(emitted == expected,
                    "prompt \(prompt.prefix(3))…: greedy \(emitted) != \(expected)")
        }
    }

    // MARK: - (c) seeded sampling matches the sampler reference

    @Test func seededSamplingMatchesReference() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)

        let prompt = Self.prompts[0]
        let config = GenerationConfig(temperature: 0.8, topK: 32, topP: 0.9,
                                      seed: 0x5EED)
        var emitted: [Int32] = []
        _ = try engine.generate(promptTokens: prompt, config: config, maxNew: 6,
                                onToken: { emitted.append($0) })

        // Reference: oracle forward + K3SamplerReference with the same
        // per-position seeds the GPU sampler derives.
        var oracle = Self.makeOracle(fixture)
        var expected: [Int32] = []
        var logits: [Float] = []
        for token in prompt { logits = oracle.forward(token: token) }
        var position: UInt32 = 0
        for _ in 0..<6 {
            let seed = Sampler.seedFor(config: config, position: Int(position))
            let next = K3SamplerReference.sample(
                logits: logits, temperature: config.temperature,
                topK: config.topK ?? 0, topP: config.topP ?? 1.0,
                seed: seed, position: position)
            position += 1
            if next == UInt32(fixture.config.eosTokenID) { break }
            expected.append(Int32(next))
            if expected.count >= 6 { break }
            logits = oracle.forward(token: Int32(next))
        }
        #expect(emitted == expected,
                "sampled \(emitted) != reference \(expected)")
    }

    // MARK: - (d) reset() reproduces identical logits

    @Test func resetReproducesIdenticalLogits() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        let prompt = Self.prompts[1]

        _ = try engine.generate(promptTokens: prompt,
                                config: GenerationConfig(temperature: 0),
                                maxNew: 1)
        let first = engine.lastLogits()
        engine.reset()
        _ = try engine.generate(promptTokens: prompt,
                                config: GenerationConfig(temperature: 0),
                                maxNew: 1)
        let second = engine.lastLogits()
        #expect(first == second, "reset + re-run must be bit-identical")
    }

    // MARK: - (e) first-token expert streaming miss counts

    @Test func firstTokenExpertReadsAreAllMisses() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        engine.resetStatistics()

        // One prompt token => one produce: every MoE layer demand-reads its
        // full top-4 (cold banks), and the wrap-around prediction for the
        // first MoE layer is issued during the last layer's I/O window.
        _ = try engine.generate(promptTokens: [17],
                                config: GenerationConfig(temperature: 0),
                                maxNew: 1)
        let stats = engine.expertStreamingStats()
        let moeLayers = fixture.config.moeLayers0.count
        let topK = fixture.config.moeTopKExperts
        #expect(stats.demandMisses == UInt64(moeLayers * topK),
                "first token misses \(stats.demandMisses) != \(moeLayers * topK)")
        #expect(stats.demandHits == 0)
        #expect(stats.prefetchesIssued == UInt64(topK),
                "wrap-around prefetch \(stats.prefetchesIssued) != \(topK)")
        #expect(stats.prefetchSkippedCold == UInt64(moeLayers - 1))
    }

    // MARK: - engine-level guards

    @Test func contextOverflowThrows() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        #expect {
            try engine.generate(promptTokens: Self.prompts[0],
                                config: GenerationConfig(temperature: 0),
                                maxNew: 100)
        } throws: { error in
            if case GeneratorError.contextOverflow = error { return true }
            return false
        }
    }
}
