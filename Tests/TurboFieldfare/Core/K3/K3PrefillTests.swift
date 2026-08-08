import Testing
import Foundation
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
import TurboFieldfareValidationSupport

/// Stage-E1 chunked prefill vs serial replay on the tiny K3 bundle (the
/// `K3ForwardRunnerTests` fabrication is reused directly: same fixture, same
/// config, same oracle helpers).
///
/// Parity targets: the batched kernels mirror the decode kernels' math with
/// an added token dimension, so chunked prefill must land within fp16
/// rounding of serial replay — logits max abs diff <= 2e-2 for the portable
/// fallback GEMM (measured ~8e-3). The NAX tensor-op path rounds dequantized
/// weights into fp16 tiles (the hardware contract — the house Gemma prefill
/// does the same), which roughly doubles the spread: bar 5e-2 there
/// (measured ~3.5e-2 worst-element on logits of range ~11, ~3e-3 relative).
/// State slabs sit at ~2e-3 max-norm relative. Greedy continuation is
/// token-for-token identical on both paths.
@Suite struct K3PrefillTests {

    /// Fallback (factored-fp32 dequant) logits parity bar.
    static let fallbackLogitsBar: Float = 2e-2
    /// NAX (fp16 weight tiles) logits parity bar.
    static let tensorOpsLogitsBar: Float = 5e-2

    typealias Fixture = K3ForwardRunnerTests.Fixture

    static func makeEngine(_ fixture: Fixture, forceFallback: Bool = false) throws -> K3Engine {
        try K3Engine.load(bundleURL: fixture.url,
                          maxContext: 64,
                          expecting: fixture.config,
                          prefetchPolicy: .predict,
                          prefillForceFallback: forceFallback)
    }

    /// Prompts of the required lengths (1, 3, 7, 31, 64), deterministic ids.
    static func prompt(_ length: Int) -> [Int32] {
        var rng = SeedTree(0xE1).key("prefill-prompt-\(length)")
        return (0..<length).map { _ in Int32(rng.next() % 512) }
    }

    // MARK: - (1) logits parity: chunked vs serialReplay

    @Test func chunkedLogitsMatchSerialReplay() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)

        var worst: Float = 0
        for length in [1, 3, 7, 31, 64] {
            for chunk in [32, 64] {
                let prompt = Self.prompt(length)
                _ = try engine.generate(promptTokens: prompt,
                                        config: GenerationConfig(temperature: 0),
                                        maxNew: 1, prefillMode: .serialReplay)
                let serial = engine.lastLogits()
                _ = try engine.generate(promptTokens: prompt,
                                        config: GenerationConfig(temperature: 0),
                                        maxNew: 1,
                                        prefillMode: .chunked(chunkTokens: chunk))
                let chunked = engine.lastLogits()
                let maxAbs = RelError.maxAbsDiff(serial, chunked)
                worst = max(worst, maxAbs)
                #expect(maxAbs <= Self.tensorOpsLogitsBar,
                        "len \(length) chunk \(chunk): maxAbs=\(maxAbs)")
            }
        }
        print("K3Prefill logits parity worst maxAbs = \(worst)")
    }

    // MARK: - (2) state parity after prefill (manual stack, direct slab reads)

    /// Build the runtime stack by hand so the state slabs can be read
    /// between a serial and a chunked run of the same prompt.
    private struct ManualStack {
        let context: MetalContext
        let model: K3Model
        let state: K3State
        let streaming: K3ExpertStreaming
        let runner: K3ForwardRunner
    }

    private static func makeManualStack(_ fixture: Fixture) throws -> ManualStack {
        let context = try MetalContext()
        let model = try K3Model.load(bundleURL: fixture.url, device: context.device,
                                     expecting: fixture.config)
        let state = try K3State(device: context.device, config: fixture.config,
                                maxContext: 64)
        let streaming = try K3ExpertStreaming(model: model)
        let runner = try K3ForwardRunner(model: model, context: context,
                                         state: state, streaming: streaming)
        return ManualStack(context: context, model: model, state: state,
                           streaming: streaming, runner: runner)
    }

    private static func readFP32(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float] {
        let base = buffer.contents().advanced(by: offset)
            .bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    private static func readFP16(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float] {
        let base = buffer.contents().advanced(by: offset)
        return (0..<count).map {
            Float(base.load(fromByteOffset: $0 * MemoryLayout<Float16>.stride,
                            as: Float16.self))
        }
    }

    @Test func chunkedStateMatchesSerialReplay() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let stack = try Self.makeManualStack(fixture)
        let config = fixture.config

        for (length, chunk) in [(7, 32), (31, 32), (64, 64)] {
            let prompt = Self.prompt(length)

            // Serial replay.
            stack.state.reset()
            for (index, token) in prompt.enumerated() {
                try stack.runner.produce(token: token, position: index,
                                         emitHead: index == prompt.count - 1)
            }
            var serialKDA: [[Float]] = []
            var serialConv: [[Float]] = []
            var serialMLA: [[Float]] = []
            for layer in 0..<config.numLayers where config.isKDA(layer0: layer) {
                let s = stack.state.kdaRecurrentState(layer0: layer)
                serialKDA.append(Self.readFP32(s.buffer, offset: s.offset,
                                               count: config.kdaStateElementsPerLayer))
                let cv = stack.state.kdaConvState(layer0: layer)
                serialConv.append(Self.readFP32(cv.buffer, offset: cv.offset,
                                                count: config.kdaConvStateElementsPerLayer))
            }
            for layer in 0..<config.numLayers where config.isMLA(layer0: layer) {
                let m = stack.state.mlaCache(layer0: layer)
                serialMLA.append(Self.readFP16(m.buffer, offset: m.offset,
                                               count: length * config.mlaCacheRowElements))
            }
            let serialLogits = Self.readFP32(stack.runner.logitsBuffer, offset: 0,
                                             count: config.vocabSize)

            // Chunked.
            stack.state.reset()
            stack.streaming.resetPrediction()
            let prefiller = try K3ChunkedPrefiller(
                model: stack.model, context: stack.context, state: stack.state,
                shared: stack.runner.prefillShared, streaming: stack.streaming,
                chunkTokens: chunk)
            try prefiller.prefill(tokens: prompt)

            var worstState: Float = 0
            for (index, layer) in config.kdaLayers0.sorted().enumerated() {
                let s = stack.state.kdaRecurrentState(layer0: layer)
                let chunked = Self.readFP32(s.buffer, offset: s.offset,
                                            count: config.kdaStateElementsPerLayer)
                // Max-norm relative: measured <= 2.3e-3 on this fixture (the
                // fp16 pipeline noise compounds through the recurrence; a
                // per-element floor metric over-reads near-zero entries).
                let diff = RelError.compute(actual: chunked,
                                            reference: serialKDA[index])
                worstState = max(worstState, diff)
                #expect(diff < 1e-2,
                        "len \(length) chunk \(chunk) KDA layer \(layer): rel=\(diff)")
                let cv = stack.state.kdaConvState(layer0: layer)
                let chunkedConv = Self.readFP32(cv.buffer, offset: cv.offset,
                                                count: config.kdaConvStateElementsPerLayer)
                let convDiff = RelError.maxAbsDiff(chunkedConv, serialConv[index])
                #expect(convDiff <= 4e-3,
                        "len \(length) chunk \(chunk) conv layer \(layer): abs=\(convDiff)")
            }
            for (index, layer) in config.mlaLayers0.sorted().enumerated() {
                let m = stack.state.mlaCache(layer0: layer)
                let chunked = Self.readFP16(m.buffer, offset: m.offset,
                                            count: length * config.mlaCacheRowElements)
                // fp16 cache rows: measured max-norm relative <= 2e-3.
                let diff = RelError.compute(actual: chunked,
                                            reference: serialMLA[index])
                #expect(diff < 1e-2,
                        "len \(length) chunk \(chunk) MLA layer \(layer): rel=\(diff)")
            }
            let chunkedLogits = Self.readFP32(stack.runner.logitsBuffer, offset: 0,
                                              count: config.vocabSize)
            let logitDiff = RelError.maxAbsDiff(chunkedLogits, serialLogits)
            #expect(logitDiff <= Self.tensorOpsLogitsBar,
                    "len \(length) chunk \(chunk): logits abs=\(logitDiff)")
            print("K3Prefill state parity len \(length) chunk \(chunk): "
                    + "worstKDA=\(worstState) logits=\(logitDiff)")
        }
    }

    // MARK: - (3) greedy continuation after chunked prefill

    @Test func greedyContinuationMatchesAfterChunkedPrefill() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)

        for length in [8, 31] {
            let prompt = Self.prompt(length)
            var serialTokens: [Int32] = []
            _ = try engine.generate(promptTokens: prompt,
                                    config: GenerationConfig(temperature: 0),
                                    maxNew: 6, prefillMode: .serialReplay,
                                    onToken: { serialTokens.append($0) })
            var chunkedTokens: [Int32] = []
            _ = try engine.generate(promptTokens: prompt,
                                    config: GenerationConfig(temperature: 0),
                                    maxNew: 6,
                                    prefillMode: .chunked(chunkTokens: 32),
                                    onToken: { chunkedTokens.append($0) })
            #expect(chunkedTokens == serialTokens,
                    "len \(length): chunked \(chunkedTokens) != serial \(serialTokens)")
        }
    }

    // MARK: - (4) forced non-NAX fallback parity

    @Test func fallbackPrefillMatchesSerialReplay() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture, forceFallback: true)

        var worst: Float = 0
        for length in [1, 7, 31, 64] {
            let prompt = Self.prompt(length)
            _ = try engine.generate(promptTokens: prompt,
                                    config: GenerationConfig(temperature: 0),
                                    maxNew: 1, prefillMode: .serialReplay)
            let serial = engine.lastLogits()
            _ = try engine.generate(promptTokens: prompt,
                                    config: GenerationConfig(temperature: 0),
                                    maxNew: 1,
                                    prefillMode: .chunked(chunkTokens: 64))
            let chunked = engine.lastLogits()
            let maxAbs = RelError.maxAbsDiff(serial, chunked)
            worst = max(worst, maxAbs)
            #expect(maxAbs <= Self.fallbackLogitsBar,
                    "len \(length): fallback maxAbs=\(maxAbs)")
        }
        print("K3Prefill fallback parity worst maxAbs = \(worst)")
    }

    // MARK: - (5) NAX vs fallback GEMM parity (direct, same chunked path)

    @Test func tensorOpsVsFallbackLogitsParity() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let naxEngine = try Self.makeEngine(fixture)
        let simdEngine = try Self.makeEngine(fixture, forceFallback: true)

        let prompt = Self.prompt(64)
        _ = try naxEngine.generate(promptTokens: prompt,
                                   config: GenerationConfig(temperature: 0),
                                   maxNew: 1, prefillMode: .chunked(chunkTokens: 64))
        let nax = naxEngine.lastLogits()
        _ = try simdEngine.generate(promptTokens: prompt,
                                    config: GenerationConfig(temperature: 0),
                                    maxNew: 1,
                                    prefillMode: .chunked(chunkTokens: 64))
        let simd = simdEngine.lastLogits()
        // The tensor op rounds dequantized weights to fp16 before the
        // matmul; the simd GEMM keeps the factored fp32 form. fp16 weight
        // rounding dominates the difference: ~1e-3 on this fixture's logits.
        let maxAbs = RelError.maxAbsDiff(nax, simd)
        #expect(maxAbs <= Self.tensorOpsLogitsBar, "NAX vs simd maxAbs=\(maxAbs)")
        // On NAX silicon the two paths MUST differ (fp16 weight tiles vs
        // factored fp32): a zero diff means the tensor path silently failed
        // to engage and both engines ran the fallback.
        let probe = try K3PrefillGEMM(context: MetalContext())
        if probe.usingTensorOps {
            #expect(maxAbs > 0, "tensor ops compiled but produced fallback-identical logits")
        }
        print("K3Prefill NAX-vs-fallback maxAbs = \(maxAbs) "
                + "(usingTensorOps=\(probe.usingTensorOps))")
    }

    // MARK: - (6) stats + smoke timings (report only)

    @Test func chunkStatsAndSmokeTimings() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        let prompt = Self.prompt(64)

        let serial = try engine.generate(promptTokens: prompt,
                                         config: GenerationConfig(temperature: 0),
                                         maxNew: 1, prefillMode: .serialReplay)
        var chunkedMillis: [Double] = []
        for chunk in [32, 64] {
            let stats = try engine.generate(
                promptTokens: prompt,
                config: GenerationConfig(temperature: 0),
                maxNew: 1, prefillMode: .chunked(chunkTokens: chunk))
            #expect(stats.prefillChunkMillis.count == 64 / chunk,
                    "chunk \(chunk): \(stats.prefillChunkMillis.count) chunks")
            chunkedMillis.append(stats.prefillSeconds * 1000)
        }
        // Report only: at 256-wide the tiny model is launch-bound; the real
        // NAX win is at 7168-wide shapes.
        print(String(format: "K3Prefill smoke (64 tok, tiny): serial %.2fms "
                        + "chunked32 %.2fms chunked64 %.2fms",
                     serial.prefillSeconds * 1000,
                     chunkedMillis[0], chunkedMillis[1]))
    }

    // MARK: - (7) exact prefix snapshot / restore

    @Test func prefixSnapshotResumeMatchesFullSerialReplay() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        let prompt = Self.prompt(8)
        let sampling = GenerationConfig(temperature: 0)

        let seed = try engine.generate(
            promptTokens: prompt,
            config: sampling,
            maxNew: 3,
            prefillMode: .serialReplay)
        #expect(seed.position == seed.stateBackedTokenIDs.count)
        #expect(seed.uncommittedBoundaryTokenIDs.count == 1)
        let snapshot = try engine.capturePrefixSnapshot(
            tokenIDs: seed.stateBackedTokenIDs)
        #expect(snapshot.tokenIDs == seed.stateBackedTokenIDs)
        #expect(snapshot.memoryBytes > 0)

        let continuedPrompt = seed.stateBackedTokenIDs + [17, 23]
        var cachedOutput: [Int32] = []
        let cached = try engine.generate(
            promptTokens: continuedPrompt,
            config: sampling,
            maxNew: 4,
            prefillMode: .serialReplay,
            prefixSnapshot: snapshot,
            onToken: { cachedOutput.append($0) })
        let cachedLogits = engine.lastLogits()

        var replayOutput: [Int32] = []
        let replayed = try engine.generate(
            promptTokens: continuedPrompt,
            config: sampling,
            maxNew: 4,
            prefillMode: .serialReplay,
            onToken: { replayOutput.append($0) })
        let replayLogits = engine.lastLogits()

        #expect(cached.cachedPromptTokens == snapshot.tokenIDs.count)
        #expect(cached.computedPrefillTokens == 2)
        #expect(replayed.cachedPromptTokens == 0)
        #expect(replayed.computedPrefillTokens == continuedPrompt.count)
        #expect(cachedOutput == replayOutput)
        #expect(cached.reason == replayed.reason)
        #expect(cached.stateBackedTokenIDs == replayed.stateBackedTokenIDs)
        #expect(cached.uncommittedBoundaryTokenIDs
                == replayed.uncommittedBoundaryTokenIDs)
        #expect(RelError.maxAbsDiff(cachedLogits, replayLogits) == 0)
    }

    @Test func prefixSnapshotRejectsChangedPrefix() throws {
        let fixture = try K3ForwardRunnerTests.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let engine = try Self.makeEngine(fixture)
        let prompt = Self.prompt(8)
        let seed = try engine.generate(
            promptTokens: prompt,
            config: GenerationConfig(temperature: 0),
            maxNew: 1)
        let snapshot = try engine.capturePrefixSnapshot(
            tokenIDs: seed.stateBackedTokenIDs)
        var changed = seed.stateBackedTokenIDs
        changed[0] &+= 1
        #expect(throws: GeneratorError.self) {
            try engine.generate(
                promptTokens: changed,
                config: GenerationConfig(temperature: 0),
                maxNew: 1,
                prefixSnapshot: snapshot)
        }
    }
}
