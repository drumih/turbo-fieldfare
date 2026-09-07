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
    /// Synchronously releases this client's descriptors and any service it
    /// launched. External services are disconnected, never shut down.
    func shutdownForTermination()
    /// Ends the current conversation and opens `epoch` as the only lineage the
    /// inference side will accept turns for. The model stays loaded.
    ///
    /// Required rather than defaulted: a client that silently did nothing here
    /// would keep appending a new chat's turns onto the previous chat's KV, and
    /// nothing downstream could detect it.
    func resetConversation(epoch: UUID) async throws
    /// Replays a stored conversation's token IDs into the KV, opens `epoch` as
    /// its lineage, and returns the token count the cache then holds.
    ///
    /// The count comes back from the inference side rather than being assumed
    /// from the lineage: the gauge and the image budget both read it, and a
    /// figure the app made up would be wrong in exactly the case that matters —
    /// a replay that did not put back what the record said.
    ///
    /// `onPrefillProgress` receives the same events a live turn reports, so a
    /// reopen shows progress instead of a frozen window for minutes.
    func restoreConversation(
        _ lineage: AppConversationLineage,
        epoch: UUID,
        options: AppRuntimeOptions,
        maxContextTokens: Int,
        onPrefillProgress: @escaping @Sendable (Int, Int) -> Void
    ) async throws -> Int
}

extension AppModelLifecycleClient {
    public func shutdownForTermination() {}
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
