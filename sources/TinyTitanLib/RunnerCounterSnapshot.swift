// The runner's lifetime stage counters, snapshotted at the start of a request.
//
// Moved out of `ServerCoordinator.swift` (2026-10-02, phase A1 of
// `docs/plan-embedded-library.md`) as a pure declaration move. The orchestrator
// in this target builds the snapshot and diffs it to report this request's
// per-stage deltas; the declaration had been split into the coordinator's file
// only because the two shared `ServerInference.swift` before it was split.
import TinyTitanCore

struct RunnerCounterSnapshot {
    let cb1: UInt64
    let io: UInt64
    let cb2: UInt64
    let head: UInt64
    let headFused: UInt64
    let rdadvise: UInt64
    let rdadviseCalls: UInt64
    let rdadviseBytes: UInt64
    let wait: UInt64
    let body: UInt64
    let prefetchIssued: UInt64
    let prefetchAdopted: UInt64
    let preamble: UInt64
    let preambleRelease: UInt64
    let preamblePin: UInt64
    let preambleReserve: UInt64
    let embed: UInt64
    let gather: UInt64
    let loopSample: UInt64
    let loopProgress: UInt64
    let loopOther: UInt64
    let missIo: UInt64
    let exposedIo: UInt64
    let hitFixupLayers: UInt64
    let routerReadback: UInt64
    let cachePlan: UInt64
    let ioQueue: UInt64
    let ioCompletionToFixup: UInt64
    let ioHostWaits: UInt64
    let ioHostWaitsAvoided: UInt64
    let gpuClassifiedHits: UInt64
    let gpuClassifiedMisses: UInt64
    let gpuAllHitLayers: UInt64
    let expertStreaming: ExpertStreamingStatistics
}
