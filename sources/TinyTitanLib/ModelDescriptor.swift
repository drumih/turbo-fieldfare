// What was loaded, in the vocabulary an embedder speaks.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). It is kit-native on purpose: no engine type crosses this boundary, so a
// consumer never has to name a `Model` or a `Manifest` to describe what it is
// holding.
//
// The counts are read from the install's own manifest: `weightBytes` is the
// sum of its declared file sizes, and `expertCacheBytes` is the routed-expert
// cache the load actually placed (slots x per-layer expert stride x layers).
/// One loaded model, described without the engine's types.
public struct ModelDescriptor: Sendable, Equatable {
    /// The advertised model id, quantization suffix included.
    public let id: String
    /// The manifest's family name, e.g. `qwen36`.
    public let family: String
    /// The context window the session was loaded with.
    public let contextWindow: Int
    /// Bytes of the install's files, as its manifest declares them.
    public let weightBytes: UInt64
    /// Bytes the resident routed-expert cache holds for this load.
    public let expertCacheBytes: UInt64

    /// Package-visible so the CPU engine target can describe its load without
    /// an engine type crossing the facade.
    package init(
        id: String,
        family: String,
        contextWindow: Int,
        weightBytes: UInt64,
        expertCacheBytes: UInt64
    ) {
        self.id = id
        self.family = family
        self.contextWindow = contextWindow
        self.weightBytes = weightBytes
        self.expertCacheBytes = expertCacheBytes
    }
}
