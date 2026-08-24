import Foundation

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    func unload() async
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
    /// Resident bytes, which include the mapped weights the footprint omits.
    /// Defaulted so a reporter that cannot answer simply does not.
    var currentInferenceResidentBytes: UInt64? { get }
    /// Bytes of image tower the inference process holds mapped, or nil when it
    /// has no vision runtime. The only figure that separates the two image
    /// residency policies: both charge the process the same few MB.
    var currentInferenceTowerBytes: UInt64? { get }
}

extension AppInferenceMemoryReporting {
    public var currentInferenceResidentBytes: UInt64? { nil }
    public var currentInferenceTowerBytes: UInt64? { nil }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}
