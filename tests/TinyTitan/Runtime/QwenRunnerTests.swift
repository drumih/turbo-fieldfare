import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
@testable import TinyTitanCore

@testable import TinyTitan

/// Qwen 3.6 runtime integration: runner construction against the qwen toy
/// fixture (no auxiliary sandwich/scale tensors present — init must not touch
/// them), decode and chunked-prefill smoke over the hybrid linear/full layer
/// graph, and the KV-cache + GDN-state reset interplay.
@Suite struct QwenRunnerTests {

    private func makeRunner(
        weightBits: Int = 4,
        slots: Int = 1,
        forceLogitsHead: Bool = false,
        maxContext: Int = 64,
        hostKVWindowTokens: Int? = nil,
        qsa: Bool = false
    ) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: weightBits, qsa: qsa)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir,
            device: ctx.device,
            expecting: qsa ? .qwenToyQSA() : .qwenToy())
        // A 4-bit lm_head defaults to the fused greedy head, which never writes
        // the logits buffer; a logits comparison must force the logits head, as
        // the server does.
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: forceLogitsHead,
            hostKVWindowTokens: hostKVWindowTokens)
        let runner = try RealForwardRunner(
            model: model,
            context: ctx,
            maxContext: maxContext,
            slots: slots,
            runtimeConfiguration: runtime)
        return (dir, ctx, runner)
    }

    @Test(arguments: [8])
    func decodeAndChunkedPrefillSupportHigherBitCheckpoints(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(!runner.usesFusedGreedyHead)
        let logits = try makeLogits(ctx, vocab: 1024)
        let tokens: [Int32] = [1, 2]
        var progress: [Int] = []
        let result = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { progress.append($0) })
        #expect(result == PrefillResult(newPosition: 2, seed: .logitsWritten))
        #expect(progress == [2])
        try await runner.produce(token: 3, position: 2, into: logits)
        #expect(runner.continuationPosition == 3)
        let values = Fp16Buffer.read(logits, count: 1024)
        #expect(values.allSatisfy { $0.isFinite })
    }

    @Test(arguments: [8])
    func higherBitPrefillThenDecodeMatchesPureDecode(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = Fp16Buffer.read(logits, count: 1024)

        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        let actual = Fp16Buffer.read(logits, count: 1024)

        for (lhs, rhs) in zip(actual, reference) {
            #expect(abs(Float(lhs) - Float(rhs)) < 0.05)
        }
    }

    @Test(arguments: [4, 8])
    func higherBitMultiTokenPrefillMatchesPureDecode(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits, forceLogitsHead: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)
        let tokens: [Int32] = [11, 7, 5, 3, 9]

        for (position, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        let reference = Fp16Buffer.read(logits, count: 1024)

        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        let actual = Fp16Buffer.read(logits, count: 1024)

        for (lhs, rhs) in zip(actual, reference) {
            #expect(abs(Float(lhs) - Float(rhs)) < 0.05)
        }
    }

    /// Two sequences decoding through one runner, interleaved token by token in
    /// their own KV and GDN slots, must produce exactly what each sequence
    /// produces alone. If any slot offset were dropped — the KV region base, the
    /// GDN state offset, the conv tail — the two would share state and this
    /// comparison would drift rather than fail loudly.
    @Test(arguments: [8])
    func batchedDecodeKeepsSlotsIndependent(bits: Int) async throws {
        let aTokens: [Int32] = [11, 7, 5]
        let bTokens: [Int32] = [3, 9, 4]
        let vocab = 1024

        /// One sequence alone in slot 0, returning the logits after each token.
        func alone(_ tokens: [Int32]) async throws -> [[Float]] {
            let (dir, ctx, runner) = try makeRunner(weightBits: bits)
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try makeLogits(ctx, vocab: vocab)
            var out: [[Float]] = []
            for (position, token) in tokens.enumerated() {
                try await runner.produce(token: token, position: position, into: logits)
                out.append(Fp16Buffer.read(logits, count: vocab))
            }
            return out
        }

        let referenceA = try await alone(aTokens)
        let referenceB = try await alone(bTokens)

        let (dir, ctx, runner) = try makeRunner(
            weightBits: bits, slots: 2, hostKVWindowTokens: 64)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.slots == 2)
        #expect(runner.kv?.streamsLayer(3) == true)
        let logits = try makeLogits(ctx, vocab: vocab)
        var gotA: [[Float]] = []
        var gotB: [[Float]] = []
        for position in 0..<max(aTokens.count, bTokens.count) {
            if position < aTokens.count {
                try await runner.produce(
                    token: aTokens[position], position: position,
                    slot: 0, into: logits)
                gotA.append(Fp16Buffer.read(logits, count: vocab))
            }
            if position < bTokens.count {
                try await runner.produce(
                    token: bTokens[position], position: position,
                    slot: 1, into: logits)
                gotB.append(Fp16Buffer.read(logits, count: vocab))
            }
        }

        for (slot, pair) in [("0", (referenceA, gotA)), ("1", (referenceB, gotB))] {
            #expect(pair.0.count == pair.1.count)
            for (step, (want, got)) in zip(pair.0, pair.1).enumerated() {
                for (i, (w, g)) in zip(want, got).enumerated() {
                    #expect(
                        abs(w - g) <= 1e-3,
                        "slot \(slot) step \(step) logit \(i): \(g) vs \(w)")
                }
            }
        }
    }

    /// `produceBatch` shares the LM head across rows: each row's final norm
    /// goes to a batch buffer and two rows run through the two-row kernel.
    /// The logits must still match what each row produces alone.
    @Test(arguments: [4, 8])
    func produceBatchMatchesPerRowLogits(bits: Int) async throws {
        let aTokens: [Int32] = [11, 7, 5]
        let bTokens: [Int32] = [3, 9, 4]
        let vocab = 1024

        func alone(_ tokens: [Int32]) async throws -> [[Float]] {
            let (dir, ctx, runner) = try makeRunner(
                weightBits: bits, forceLogitsHead: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try makeLogits(ctx, vocab: vocab)
            var out: [[Float]] = []
            for (position, token) in tokens.enumerated() {
                try await runner.produce(
                    token: token, position: position, into: logits)
                out.append(Fp16Buffer.read(logits, count: vocab))
            }
            return out
        }

        let referenceA = try await alone(aTokens)
        let referenceB = try await alone(bTokens)

        let (dir, ctx, runner) = try makeRunner(
            weightBits: bits, slots: 2, forceLogitsHead: true,
            hostKVWindowTokens: 64)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.kv?.streamsLayer(3) == true)
        let logitsA = try makeLogits(ctx, vocab: vocab)
        let logitsB = try makeLogits(ctx, vocab: vocab)
        var gotA: [[Float]] = []
        var gotB: [[Float]] = []
        for position in 0..<max(aTokens.count, bTokens.count) {
            var rows: [(token: Int32, position: Int, slot: Int)] = []
            var buffers: [MTLBuffer] = []
            if position < aTokens.count {
                rows.append((aTokens[position], position, 0))
                buffers.append(logitsA)
            }
            if position < bTokens.count {
                rows.append((bTokens[position], position, 1))
                buffers.append(logitsB)
            }
            try await runner.produceBatch(rows, logits: buffers)
            if position < aTokens.count {
                gotA.append(Fp16Buffer.read(logitsA, count: vocab))
            }
            if position < bTokens.count {
                gotB.append(Fp16Buffer.read(logitsB, count: vocab))
            }
        }
        for (name, pair) in [("A", (referenceA, gotA)), ("B", (referenceB, gotB))] {
            #expect(pair.0.count == pair.1.count)
            for (step, (want, got)) in zip(pair.0, pair.1).enumerated() {
                for (i, (w, g)) in zip(want, got).enumerated() {
                    #expect(
                        abs(w - g) <= 1e-3,
                        "row \(name) step \(step) logit \(i): \(g) vs \(w)")
                }
            }
        }
    }

    /// Width 4 goes through the four-row head (one weight read for all four
    /// rows instead of two two-row reads), with each row in its own slot. The
    /// logits must still match what each row produces alone, and the counter
    /// proves the four-row kernel is the one that ran.
    @Test(arguments: [4, 8])
    func produceBatchAtWidthFourMatchesPerRowLogits(bits: Int) async throws {
        let batches: [[Int32]] = [
            [11, 7, 5], [3, 9, 4], [13, 2, 8], [6, 1, 12],
        ]
        let vocab = 1024

        func alone(_ tokens: [Int32]) async throws -> [[Float]] {
            let (dir, ctx, runner) = try makeRunner(
                weightBits: bits, forceLogitsHead: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try makeLogits(ctx, vocab: vocab)
            var out: [[Float]] = []
            for (position, token) in tokens.enumerated() {
                try await runner.produce(
                    token: token, position: position, into: logits)
                out.append(Fp16Buffer.read(logits, count: vocab))
            }
            return out
        }

        var references: [[[Float]]] = []
        for tokens in batches {
            references.append(try await alone(tokens))
        }

        let (dir, ctx, runner) = try makeRunner(
            weightBits: bits, slots: 4, forceLogitsHead: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let buffers = try batches.map { _ in
            try makeLogits(ctx, vocab: vocab)
        }
        var got: [[[Float]]] = Array(repeating: [], count: batches.count)
        for position in 0..<batches[0].count {
            let rows = batches.enumerated().map { slot, tokens in
                (token: tokens[position], position: position, slot: slot)
            }
            try await runner.produceBatch(rows, logits: buffers)
            for slot in batches.indices {
                got[slot].append(Fp16Buffer.read(buffers[slot], count: vocab))
            }
        }
        // The 8-bit checkpoint has an affine head, which uses the four-row
        // kernel; the 4-bit one keeps two-row pairs (see `encodeHeadFourRows`).
        #expect(
            runner.headFourRowLaunches == (bits == 8 ? batches[0].count : 0))
        for slot in batches.indices {
            for (step, pair) in zip(references[slot], got[slot]).enumerated() {
                for (i, want) in pair.0.enumerated() {
                    let value = pair.1[i]
                    #expect(
                        abs(want - value) <= 1e-3,
                        "row \(slot) step \(step) logit \(i): \(value) vs \(want)")
                }
            }
        }
    }

    /// Chunked prefill must fill the slot it is told to, for every slot. Before
    /// this, only slot 0 could take the chunked path (other slots fell back to
    /// decode-as-prefill, which does not match numerically on 4-bit), and the
    /// chunked attention read the K/V buffers at offset 0 regardless of slot.
    @Test(arguments: [4, 8])
    func chunkedPrefillFillsTheRequestedSlot(bits: Int) async throws {
        let tokens: [Int32] = [11, 7, 5, 3, 9]
        let vocab = 1024
        let soloMaxContext = 65536

        let (soloDir, soloCtx, solo) = try makeRunner(
            weightBits: bits,
            forceLogitsHead: true,
            maxContext: soloMaxContext)
        defer { try? FileManager.default.removeItem(at: soloDir) }
        let soloLogits = try makeLogits(soloCtx, vocab: vocab)
        _ = try await solo.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: soloLogits, onProgress: { _ in })
        try await solo.produce(token: 2, position: tokens.count, into: soloLogits)
        let expected = Fp16Buffer.read(soloLogits, count: vocab)

        for slot in 0..<4 {
            let (dir, ctx, runner) = try makeRunner(
                weightBits: bits, slots: 4,
                forceLogitsHead: true,
                maxContext: soloMaxContext)
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try makeLogits(ctx, vocab: vocab)
            _ = try await runner.prefillChunked(
                tokens: tokens[...], startPosition: 0, slot: slot, outputMode: .logits,
                config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
            try await runner.produce(
                token: 2, position: tokens.count, slot: slot,
                into: logits)
            let actual = Fp16Buffer.read(logits, count: vocab)
            for (i, (want, got)) in zip(expected, actual).enumerated() {
                #expect(
                    abs(want - got) <= 1e-3,
                    "slot \(slot) prefill+decode logit \(i): \(got) vs solo \(want)")
            }
        }
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard
            let buf = ctx.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// Two sequences driven **concurrently** through one runner must still
    /// produce each sequence's solo logits. They share the runner's scratch, so
    /// this only holds because `ForwardStepGate` serializes the forward step;
    /// without it the two would interleave inside a token and corrupt both.
    @Test(arguments: [8])
    func concurrentSlotsSerializeThroughTheStepGate(bits: Int) async throws {
        let aTokens: [Int32] = [11, 7, 5]
        let bTokens: [Int32] = [3, 9, 4]
        let vocab = 1024

        func alone(_ tokens: [Int32]) async throws -> [[Float]] {
            let (dir, ctx, runner) = try makeRunner(weightBits: bits)
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try makeLogits(ctx, vocab: vocab)
            var out: [[Float]] = []
            for (position, token) in tokens.enumerated() {
                try await runner.produce(token: token, position: position, into: logits)
                out.append(Fp16Buffer.read(logits, count: vocab))
            }
            return out
        }

        let referenceA = try await alone(aTokens)
        let referenceB = try await alone(bTokens)

        let (dir, ctx, runner) = try makeRunner(weightBits: bits, slots: 2)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Each task makes its own logits buffer: an MTLBuffer is not Sendable,
        // so one built outside and captured by a `sending` closure is rejected.
        let (gotA, gotB) = try await withThrowingTaskGroup(
            of: (Int, [[Float]]).self
        ) { group in
            group.addTask {
                guard
                    let logits = ctx.device.makeBuffer(
                        length: vocab * MemoryLayout<Float16>.stride,
                        options: .storageModeShared)
                else {
                    throw ModelError.residentBufferWrapFailed
                }
                var rows: [[Float]] = []
                for (position, token) in aTokens.enumerated() {
                    try await runner.produce(
                        token: token, position: position,
                        slot: 0, into: logits)
                    rows.append(Fp16Buffer.read(logits, count: vocab))
                }
                return (0, rows)
            }
            group.addTask {
                guard
                    let logits = ctx.device.makeBuffer(
                        length: vocab * MemoryLayout<Float16>.stride,
                        options: .storageModeShared)
                else {
                    throw ModelError.residentBufferWrapFailed
                }
                var rows: [[Float]] = []
                for (position, token) in bTokens.enumerated() {
                    try await runner.produce(
                        token: token, position: position,
                        slot: 1, into: logits)
                    rows.append(Fp16Buffer.read(logits, count: vocab))
                }
                return (1, rows)
            }
            var a: [[Float]] = []
            var b: [[Float]] = []
            for try await (slot, rows) in group {
                if slot == 0 { a = rows } else { b = rows }
            }
            return (a, b)
        }

        for (slot, pair) in [("0", (referenceA, gotA)), ("1", (referenceB, gotB))] {
            #expect(pair.0.count == pair.1.count)
            for (step, (want, got)) in zip(pair.0, pair.1).enumerated() {
                for (i, (w, g)) in zip(want, got).enumerated() {
                    #expect(
                        abs(w - g) <= 1e-3,
                        "slot \(slot) step \(step) logit \(i): \(g) vs \(w)")
                }
            }
        }
    }

    /// (a) Runner init with the qwen arch: GDN kernels, state manager, and
    /// the int8 router/gate buffers must construct without touching any
    /// auxiliary sandwich/scale tensors (the fixture does not contain them —
    /// any access would throw `tensorNotFound`).
    @Test func runnerInit_qwenToy_doesNotTouchAuxTensors() throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.maxContext == 64)
        #expect(runner.usesFusedGreedyHead)
    }

    /// Decode smoke over the hybrid layer graph: two linear (GDN) layers and
    /// two gated full-attention layers, 8-of-8 routing, gated shared expert,
    /// untied greedy head. Verifies tokens are produced, the cursor advances,
    /// and reset() rewinds both the KV cache and the GDN state.
    @Test func decodeSmoke_hybridLayerGraph() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 1, position: 0, into: logits)
        let first = runner.lastGreedyToken
        #expect(first < 1024)
        try await runner.produce(token: Int32(first), position: 1, into: logits)
        #expect(runner.continuationPosition == 2)

        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 1, position: 0, into: logits)
        // Same input from the empty state must reproduce the same argmax —
        // this fails if reset() leaves stale GDN state or conv tail behind.
        #expect(runner.lastGreedyToken == first)
    }

    /// Chunked prefill smoke: one chunk through the qwen prefill path
    /// (batched GDN projections + conv tail carry, packed q_proj split,
    /// sub-dim RoPE, no V norm, affine router scales, residual-add tail),
    /// then a decode continuation on top of the prefilled state.
    @Test func prefillChunkedSmoke_thenDecodeContinuation() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        let tokens: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let result = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        #expect(result.newPosition == 8)
        if case .greedyToken(let token) = result.seed {
            #expect(token < 1024)
        } else {
            Issue.record("expected a greedy seed token from the fused head")
        }

        try await runner.produce(token: 9, position: 8, into: logits)
        #expect(runner.lastGreedyToken < 1024)
        #expect(runner.continuationPosition == 9)
    }

    /// Prefill/decode consistency: prefilling [t0] then decoding t1 must give
    /// the same argmax as decoding t0, t1 step by step from a fresh state —
    /// the two paths share state (KV + GDN recurrent state + conv tail).
    @Test func prefillThenDecode_matchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        // Pure decode reference.
        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = runner.lastGreedyToken

        // Prefill the first token, then decode the second.
        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        #expect(runner.lastGreedyToken == reference)
    }

    @Test(arguments: [4, 8])
    func inferenceStateSnapshotRestoresKVAndGDNExactly(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let snapshot = try runner.captureInferenceState()
        #expect(snapshot.descriptor.position == 2)
        #expect(!snapshot.descriptor.kvSegmentLengths.isEmpty)
        #expect(!snapshot.descriptor.gdnSegmentLengths.isEmpty)
        #expect(snapshot.payload.count == snapshot.descriptor.payloadBytes)

        try await runner.produce(token: 5, position: 2, into: logits)
        let reference = Fp16Buffer.read(logits, count: 1024)

        runner.reset()
        try runner.restoreInferenceState(snapshot)
        #expect(runner.continuationPosition == 2)
        try await runner.produce(token: 5, position: 2, into: logits)
        let restored = Fp16Buffer.read(logits, count: 1024)

        #expect(restored == reference)
    }

    @Test func inferenceStateSnapshotHonorsSizeLimitWithoutMutatingRunner() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)
        try await runner.produce(token: 1, position: 0, into: logits)

        #expect(throws: InferenceStateSnapshotError.self) {
            try runner.captureInferenceState(maximumBytes: 1)
        }
        #expect(runner.continuationPosition == 1)
    }

    @Test func corruptInferenceStateSnapshotFailsClosedAndResets() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)
        try await runner.produce(token: 1, position: 0, into: logits)
        let snapshot = try runner.captureInferenceState()
        let corrupt = InferenceStateSnapshot(
            descriptor: snapshot.descriptor,
            payload: snapshot.payload.dropLast())

        #expect(throws: InferenceStateSnapshotError.self) {
            try runner.restoreInferenceState(corrupt)
        }
        #expect(runner.continuationPosition == 0)
    }

    /// (b) KV manager + GDN state manager interplay under the qwen mask:
    /// linear layers carry no per-token KV storage, full layers are linear
    /// append-only, and reset() returns both to the empty-context state.
    @Test func kvManagerAndGdnState_resetInterplay() throws {
        let cfg = ArchConfig.qwenToy()
        let ctx = try MetalContext()
        let kv = try KVCacheManager(
            device: ctx.device,
            config: cfg,
            maxContext: 32,
            fp16RingEnabled: true,
            slidingWindow: cfg.slidingWindow,
            maxPrefillChunkTokens: 32)
        #expect(kv.layerKind(0) == .linear)
        #expect(kv.layerKind(1) == .full)
        #expect(kv.layerKind(2) == .linear)
        #expect(kv.layerKind(3) == .full)
        // Linear layers: no KV rows; full layers: 2 heads * 32 dim * 2 B.
        #expect(kv.capacity(layer: 0) == 0)
        #expect(kv.stride(layer: 0) == 0)
        #expect(kv.capacity(layer: 1) == 32)
        #expect(kv.stride(layer: 1) == 2 * 32 * 2)
        // No SWA layers => the FP16 ring never engages anywhere.
        for L in 0..<cfg.numLayers {
            #expect(kv.ringCapacity(layer: L) == 0)
        }

        let gdnState = try GDNStateManager(device: ctx.device, config: cfg)
        #expect(gdnState.isLinear(layer: 0))
        #expect(!gdnState.isLinear(layer: 1))
        let la = cfg.linearAttention
        #expect(
            gdnState.stateBytesPerLayer
                == la.numVHeads * la.valueHeadDim * la.keyHeadDim * 4)
        #expect(
            gdnState.convTailBytesPerLayer
                == (la.convKernelSize - 1) * la.qkvDim * 2)

        // Dirty the recurrent state and the KV cursor, then reset both.
        let state = gdnState.stateBuffer(layer: 0)
        state.contents().assumingMemoryBound(to: Float.self)[0] = 42
        let tail = gdnState.convTailBuffer(layer: 2)
        tail.contents().assumingMemoryBound(to: UInt16.self)[0] = 0x3C00
        kv.advance(by: 4)
        #expect(kv.position == 4)

        kv.reset()
        gdnState.reset()
        #expect(kv.position == 0)
        #expect(state.contents().assumingMemoryBound(to: Float.self)[0] == 0)
        #expect(tail.contents().assumingMemoryBound(to: UInt16.self)[0] == 0)
    }

    /// Prefill scratch layout: the qwen shape must size the packed q_proj /
    /// split-gate / GDN buffers; the MTP sidecar shape differs (no linear
    /// layers, no GDN buffers, but still gated attention output).
    @Test func prefillScratchLayout_qwenAndMtpSizes() {
        let qwen = PrefillChunkScratchLayout(config: .qwenToy(), chunkTokens: 32)
        // max(packed q_proj 2*4*32, gdn qkvDim 256) = 256.
        #expect(qwen.qProjElementsPerToken == 256)
        #expect(qwen.attnGateElementsPerToken == 4 * 32)
        #expect(qwen.gdnQKVDim == 256)
        #expect(qwen.gdnValueDim == 128)
        #expect(qwen.gdnVHeads == 4)
        #expect(qwen.sharedScalarGateElements == 1)
        #expect(qwen.qElements == 32 * 256)
        #expect(qwen.attentionOutputElements == 32 * 4 * 32)

        let mtp = PrefillChunkScratchLayout(config: .qwen36MTP, chunkTokens: 128)
        // Gate-packed q_proj with no linear layers: 2x the per-head rows.
        #expect(mtp.qProjElementsPerToken == 2 * mtp.maxQElementsPerToken)
        #expect(mtp.attnGateElementsPerToken == mtp.maxQElementsPerToken)
        #expect(mtp.gdnQKVDim == 0)
        #expect(mtp.gdnValueDim == 0)
        #expect(mtp.sharedScalarGateElements == 1)
        #expect(mtp.qElements == 128 * 2 * mtp.maxQElementsPerToken)
    }

    /// The host-KV tier is a storage change, not an arithmetic one: with a
    /// window that holds the whole toy context, decode and chunked prefill must
    /// produce the contiguous cache's logits.
    @Test(arguments: [4, 8])
    func hostKVWindowMatchesContiguousLogits(bits: Int) async throws {
        let vocab = 1024
        let tokens: [Int32] = [11, 7, 5, 3, 9]

        func run(window: Int?) async throws -> [Float] {
            let (dir, ctx, runner) = try makeRunner(
                weightBits: bits, forceLogitsHead: true,
                maxContext: 64, hostKVWindowTokens: window)
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try makeLogits(ctx, vocab: vocab)
            _ = try await runner.prefillChunked(
                tokens: tokens[...], startPosition: 0, outputMode: .logits,
                config: .production(chunkTokens: 32), into: logits,
                onProgress: { _ in })
            try await runner.produce(token: 13, position: tokens.count, into: logits)
            return Fp16Buffer.read(logits, count: vocab)
        }

        let contiguous = try await run(window: nil)
        let streamed = try await run(window: 64)
        for i in 0..<vocab {
            #expect(
                abs(contiguous[i] - streamed[i]) < 0.05,
                "logit \(i): \(streamed[i]) vs \(contiguous[i])")
        }
    }

    /// Host-KV streaming past the window, with the real QSA path engaged: the
    /// tree prefill chunks a streamed runner to one token (window 8 / selection
    /// width 5), while the contiguous run keeps its 32-token chunks. The
    /// selections must agree, and the streamed cache must decode the same.
    @Test func hostKVStreamsQSADecodeAndPrefill() async throws {
        let vocab = 1024
        let tokens: [Int32] = [11, 7, 5, 3, 9, 4, 2, 8, 6, 1]

        func run(window: Int?) async throws -> [Float] {
            let (dir, ctx, runner) = try makeRunner(
                weightBits: 8, forceLogitsHead: true, maxContext: 32,
                hostKVWindowTokens: window, qsa: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try makeLogits(ctx, vocab: vocab)
            _ = try await runner.prefillChunked(
                tokens: tokens[...], startPosition: 0, outputMode: .logits,
                config: .production(chunkTokens: 32), into: logits,
                onProgress: { _ in })
            try await runner.produce(token: 13, position: tokens.count, into: logits)
            if window != nil {
                #expect(runner.kv?.streamsLayer(3) == true)
                #expect(
                    (runner.kv?.windowCopyBytes ?? 0) > 0,
                    "the streamed run must gather a selection")
                #expect(
                    (runner.kv?.windowEvictions ?? 0) > 0,
                    "the streamed run must cross its window")
            }
            return Fp16Buffer.read(logits, count: vocab)
        }

        let contiguous = try await run(window: nil)
        let streamed = try await run(window: 8)
        for i in 0..<vocab {
            #expect(
                abs(contiguous[i] - streamed[i]) < 0.05,
                "logit \(i): \(streamed[i]) vs \(contiguous[i])")
        }
    }

}
