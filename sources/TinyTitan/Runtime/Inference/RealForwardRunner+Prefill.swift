import Foundation
import Metal
import TinyTitanCore

/// Chunked prefill: the public entry points, the chunk loop that drives them,
/// and the scratch/validation setup they need.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes. The stage code moved to its own
/// files on 2026-09-28 — `+PrefillLayer.swift` (the chunk and per-layer
/// executors), `+PrefillProjection.swift` (views, projections, final head),
/// `+PrefillKV.swift` (cache writes) and `+PrefillMoE.swift` (the routed
/// stage) — leaving the loop that calls them here.
extension RealForwardRunner {
    /// Prefill-to-decode transition: forget prefill's LFU counts so decode's
    /// working set can take the slots. Opt-in (TINYTITAN_EXPERT_USE_RESET=1):
    /// measured decode-only on a 3.7k-token prompt, the first 64 decode
    /// tokens hit 68% / 82% / 87% (tokens 0-16 / 16-32 / 32-64) with and
    /// without it, identical to three decimals. The cache is not cold after
    /// prefill -- the "52%" that suggested it was the runner's cumulative
    /// statistic with prefill's tile misses mixed in -- and the leftovers
    /// decode reuses outweigh the ones it has to evict.
    var expertUseResetEnabled: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_USE_RESET"] == "1"
    }
    func resetExpertUseCountsAfterPrefill() {
        guard expertUseResetEnabled else { return }
        model.resetRoutedExpertUseCounts()
    }

    /// The one-token-at-a-time prefill a hyper-connection family started
    /// on, kept as the oracle the batched path is checked against.
    ///
    /// It runs the verified decode path per token, so it produces the KV
    /// state and logits the batched path must reproduce. It is also
    /// unusably slow -- every token pays a full pass over the routed
    /// experts, where a chunk amortizes them -- so it is not the default.
    private func prefillSequentialHyperConnection(
        tokens: ArraySlice<Int32>, startPosition: Int, slot: Int,
        outputMode: PrefillOutputMode, into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        var position = startPosition
        for (offset, token) in tokens.enumerated() {
            try Task.checkCancellation()
            try await produceToken(
                token: token,
                position: position,
                slot: slot,
                into: logits,
                emitHead: offset == tokens.count - 1,
                outputMode: outputMode)
            position += 1
            onProgress(offset + 1)
        }
        return PrefillResult(newPosition: position, seed: .logitsWritten)
    }

    /// Intended to release the slot-cache wiring for prefill, which streams
    /// experts in bulk and, on the ANE path, has to leave Core ML room for
    /// its arenas.
    ///
    /// In practice this is a no-op and has always been: the cache is not
    /// wired until the first decode token, so `slotsPinned` is already
    /// false when prefill asks. Measured with `TINYTITAN_WIRE_TRACE=1` over a
    /// full ANE-prefill request: 40 `mlock` calls at the handover, zero
    /// `munlock` calls anywhere. The shipped behaviour is "wire once, at
    /// the handover", not the release/re-apply cycle `703f35a`'s message
    /// describes.
    ///
    /// Kept because it is correct for any future path that does wire
    /// earlier, and because removing it would silently change that path's
    /// behaviour. It is not load-bearing today.
    ///
    /// Skipped when the profile keeps the cache wired: pinning at allocation
    /// and then releasing here is self-defeating, and cost one wrong
    /// conclusion already. For a row that does not wire it, this call is what
    /// makes the cache pageable through prefill (TT-008).
    private func releasePrefillCacheWiring() {
        if !profile.keepExpertCacheWired {
            model.setExpertCachePinned(false)
        }
    }

    private func validateChunkedPrefill(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        config: PrefillRuntimeConfig,
        slot: Int = 0
    ) throws {
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position(slot: slot) ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition) for slot \(slot)"
            )
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)"
            )
        }
    }

    public func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await prefillChunked(
            tokens: tokens, startPosition: startPosition, slot: 0,
            outputMode: outputMode, config: config, into: logits,
            onProgress: onProgress)
    }

    /// Slot-aware chunked prefill: the chunk lands in `slot`'s KV and GDN
    /// regions, so every sequence takes the same (golden) prefill path instead
    /// of the numerically different decode-as-prefill fallback.
    public func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        // Prefill shares the runner's scratch with decode, so a batched slot
        // must not run a chunk while another slot is decoding. One gate covers
        // both; a prefill holds it for its whole burst.
        try await forwardStepGate.acquire()
        do {
            let result = try await runPrefillChunked(
                tokens: tokens, startPosition: startPosition, slot: slot,
                outputMode: outputMode, config: config, into: logits,
                onProgress: onProgress)
            await forwardStepGate.release()
            return result
        } catch {
            await forwardStepGate.release()
            throw error
        }
    }

    func runPrefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int = 0,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        defer { resetExpertUseCountsAfterPrefill() }
        // The chunked path does not go through `produceToken`, so it needs
        // the sparse-attention gate of its own. The chunk's last query sees
        // the most keys and decides the whole chunk.
        try requireQSADensePrefill(visibleKeys: startPosition + tokens.count)
        // The one-token-at-a-time prefill a hyper-connection family started
        // on, kept as the oracle the batched path is checked against.
        //
        // It runs the verified decode path per token, so it produces the KV
        // state and logits the batched path must reproduce. It is also
        // unusably slow -- every token pays a full pass over the routed
        // experts, where a chunk amortizes them -- so it is not the default.
        if cfg.hyperConnections.enabled && Self.sequentialHyperConnectionPrefill {
            return try await prefillSequentialHyperConnection(
                tokens: tokens, startPosition: startPosition, slot: slot,
                outputMode: outputMode, into: logits, onProgress: onProgress)
        }
        releasePrefillCacheWiring()
        try validateChunkedPrefill(
            tokens: tokens, startPosition: startPosition,
            config: config, slot: slot)
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let effectiveConfig = effectivePrefillConfig(config)
        let scratch = try ensurePrefillScratch(config: effectiveConfig)
        let spans = PrefillChunkPlanner.spans(
            tokenCount: tokens.count,
            startPosition: startPosition,
            config: effectiveConfig)
        do {
            for (spanIndex, span) in spans.enumerated() {
                try Task.checkCancellation()
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                try await executePrefillChunk(
                    tokens: tokens[lower..<upper],
                    startPosition: span.startPosition,
                    slot: slot,
                    outputMode: outputMode,
                    logits: logits,
                    scratch: scratch,
                    config: effectiveConfig,
                    writeFinalHead: spanIndex == spans.count - 1)
                try Task.checkCancellation()
                onProgress(span.completedCount)
            }
        } catch {
            // Any failure — cancellation, a GPU command-buffer error, an I/O
            // error mid-routed-fetch — may have written partial KV rows and
            // left the chunk state dirty. Reset so the next request does not
            // trip `chunkedRunnerDirty` on a stale in-flight chunk.
            reset()
            throw error
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(
                newPosition: startPosition + tokens.count,
                seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(
            newPosition: startPosition + tokens.count,
            seed: .logitsWritten)
    }

    /// Host-KV streaming bounds a chunk by the selection union that must fit
    /// the window. `config.chunkTokens` is the caller's ceiling; the KV tier
    /// only ever lowers it, and the lowered value is used everywhere the chunk
    /// size matters (scratch, planner, dispatch).
    func effectivePrefillConfig(
        _ config: PrefillRuntimeConfig
    ) -> PrefillRuntimeConfig {
        guard let kv, kv.streamsKVFromHost else { return config }
        let capped = kv.streamingPrefillChunkTokens(
            configured: config.chunkTokens)
        return capped == config.chunkTokens
            ? config : .hostKVChunked(chunkTokens: capped)
    }

    func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

}
