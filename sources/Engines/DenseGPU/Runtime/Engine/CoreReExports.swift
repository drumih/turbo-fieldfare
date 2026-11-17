import TinyTitanCore

// The engine-contract family this runtime used to own. TT-097 (Phase 1
// Task 3) promoted the definitions into TinyTitanCore — byte-identical code
// motion; the shipped GPU variant carried byte-identical copies of the same
// files and now re-exports the same Core names — so one nominal protocol and
// value-type family is what 6.x and every variant engine conform to, and one
// shared server session can be typed against it.
//
// These re-exports keep every existing reference in this module (and in the
// modules that `import TinyTitan`) resolving unchanged. New code should
// import TinyTitanCore directly. The `package`-level planning helpers
// (`PrefillChunkCommitState`, `PrefillChunkSpan`, `PrefillChunkPlanner`,
// `AuxiliaryCoding`) are deliberately not re-exported: files that use them
// import TinyTitanCore.

public typealias LogitProducer = TinyTitanCore.LogitProducer
public typealias ContinuableLogitProducer = TinyTitanCore.ContinuableLogitProducer
public typealias ContextWindowReporting = TinyTitanCore.ContextWindowReporting
public typealias ChunkedPrefillRunner = TinyTitanCore.ChunkedPrefillRunner
public typealias PrefillOutputMode = TinyTitanCore.PrefillOutputMode
public typealias PrefillSeed = TinyTitanCore.PrefillSeed
public typealias PrefillResult = TinyTitanCore.PrefillResult
public typealias PrefillError = TinyTitanCore.PrefillError
public typealias PrefillRuntimeConfig = TinyTitanCore.PrefillRuntimeConfig
public typealias InferenceAuxiliaryState = TinyTitanCore.InferenceAuxiliaryState
public typealias InferenceAuxiliaryStateError = TinyTitanCore.InferenceAuxiliaryStateError
public typealias InferenceStateSnapshot = TinyTitanCore.InferenceStateSnapshot
public typealias InferenceStateSnapshotDescriptor =
    TinyTitanCore.InferenceStateSnapshotDescriptor
public typealias InferenceStateSnapshotError = TinyTitanCore.InferenceStateSnapshotError
public typealias ExpertStreamingStatistics = TinyTitanCore.ExpertStreamingStatistics
