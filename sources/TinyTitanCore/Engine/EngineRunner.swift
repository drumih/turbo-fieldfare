import Metal

/// The runner-facing half of the TT-097 engine contract: everything the
/// shared server session reads off a loaded engine, in one nominal protocol
/// the 6.x runtime and every `EngineVariant` engine conform to.
///
/// Members are copied verbatim from the 6.x `RealForwardRunner` declarations
/// the session actually reads (`grep -hoE 'runner\.[a-zA-Z0-9]+'
/// sources/TinyTitanLib/ServerModelSession*.swift`); no engine gains or loses
/// behavior by conforming. The 33 per-counter reads collapse behind
/// `statistics`/`resetStatistics()` (`EngineStatistics`); the fused-head,
/// wide-verify and MTP couplings stay off the contract because the session
/// does not read them — a conforming engine's loop, not this protocol, owns
/// them.
public protocol EngineRunner:
    ChunkedPrefillRunner, ContinuableLogitProducer, ContextWindowReporting
{
    /// The diagnostic aggregate: every counter the session reads, forwarded
    /// from the engine's existing counters with no recomputation.
    var statistics: EngineStatistics { get }
    /// Clear the diagnostic counters through the engine's own resets.
    func resetStatistics()

    /// Decode one row per slot into the corresponding logits buffer.
    func produceBatch(
        _ rows: [(token: Int32, position: Int, slot: Int)],
        logits: [MTLBuffer]
    ) async throws

    /// Reset one slot's KV and recurrent state, leaving other slots live.
    func reset(slot: Int)

    /// Whether a snapshot from this runner can be restored.
    var supportsInferenceStateRestore: Bool { get }

    /// Capture the KV + recurrent-state checkpoint, capped at `maximumBytes`.
    func captureInferenceState(maximumBytes: Int?) throws -> InferenceStateSnapshot
    /// Restore a checkpoint captured by `captureInferenceState`, with the
    /// indexer/PLE auxiliary payload when the engine needs one.
    func restoreInferenceState(
        _ snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
    ) throws
    /// Capture the auxiliary (indexer/PLE/n-gram-context) payload, or nil for
    /// an engine without those subsystems.
    func captureAuxiliaryState() throws -> InferenceAuxiliaryState?

    /// Expert I/O attributable to decode alone, against the prefill handover
    /// baseline; nil before that baseline exists.
    func decodeExpertIO() -> (hits: UInt64, misses: UInt64, bytes: UInt64)?
    /// Expert-cache statistics as of this call.
    func expertStreamingStatistics() -> ExpertStreamingStatistics

    /// Prefetch-ring summary line, when a ring is configured.
    var prefetchRingSummary: String? { get }
    /// The one-line tensor-ops summary the loading path logs to stderr.
    var prefillTensorOpsLogLine: String { get }

    /// Aggregated per-role GPU milliseconds for the current generation.
    func kernelGPUTimingSummary() -> [(role: String, millis: Double, count: Int)]
    /// Merged GPU busy/span pair for the current generation.
    func kernelGPUOccupancy() -> (busyMillis: Double, spanMillis: Double)
    /// Per-transition GPU idle attribution for the current generation.
    func kernelGPUGaps() -> [(transition: String, millis: Double, count: Int)]
    /// Drop the recorded per-command-buffer GPU timings.
    func resetKernelGPUTimings()
}
