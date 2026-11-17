import Foundation

/// Storage precision for the autoregressive attention key/value cache.
/// Quantized modes use affine groups of 64 values and keep their scale and
/// bias alongside each token row; model weights are unaffected.
public enum KVCachePrecision: Int, Codable, CaseIterable, Sendable {
    case int4 = 4
    case int8 = 8
    case fp16 = 16

    public var label: String { "\(rawValue)-bit" }
    public var isQuantized: Bool { self != .fp16 }
}

public enum RuntimeRoPEScalingMode: String, Codable, CaseIterable, Sendable {
    case none
    case yarn
}

/// The model-free inputs a session factory carries between its model, runner,
/// and cache construction stages. Engine-specific configuration remains inside
/// the factory; this value does not expose a concrete runtime module.
public struct SessionBuildOptions: Sendable {
    public let context: MetalContext?
    public let maxContext: Int
    public let slots: Int
    public let promptCacheMaximumEntries: Int
    public let promptCacheMemoryLimitBytes: Int
    public let promptCacheDiskDirectory: URL?
    public let promptCacheDiskLimitBytes: Int
    public let prefillChunkTokens: Int?
    public let kvCachePrecision: KVCachePrecision
    public let hostKVWindowTokens: Int?
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let thinkingMode: ModelThinkingMode
    public let reasoningEffort: ModelReasoningEffort?
    public let expertCacheSlots: Int?
    public let expertCacheBudgetBytes: Int?
    public let mtpModelDirectory: URL?
    public let mtpMemoryMiB: Int

    public init(
        context: MetalContext?,
        maxContext: Int,
        slots: Int,
        promptCacheMaximumEntries: Int,
        promptCacheMemoryLimitBytes: Int,
        promptCacheDiskDirectory: URL?,
        promptCacheDiskLimitBytes: Int,
        prefillChunkTokens: Int?,
        kvCachePrecision: KVCachePrecision,
        hostKVWindowTokens: Int?,
        ropeScalingMode: RuntimeRoPEScalingMode,
        thinkingMode: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort?,
        expertCacheSlots: Int?,
        expertCacheBudgetBytes: Int?,
        mtpModelDirectory: URL?,
        mtpMemoryMiB: Int
    ) {
        self.context = context
        self.maxContext = maxContext
        self.slots = slots
        self.promptCacheMaximumEntries = promptCacheMaximumEntries
        self.promptCacheMemoryLimitBytes = promptCacheMemoryLimitBytes
        self.promptCacheDiskDirectory = promptCacheDiskDirectory
        self.promptCacheDiskLimitBytes = promptCacheDiskLimitBytes
        self.prefillChunkTokens = prefillChunkTokens
        self.kvCachePrecision = kvCachePrecision
        self.hostKVWindowTokens = hostKVWindowTokens
        self.ropeScalingMode = ropeScalingMode
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
        self.expertCacheSlots = expertCacheSlots
        self.expertCacheBudgetBytes = expertCacheBudgetBytes
        self.mtpModelDirectory = mtpModelDirectory
        self.mtpMemoryMiB = mtpMemoryMiB
    }
}
