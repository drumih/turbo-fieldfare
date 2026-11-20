import Foundation
import Metal

/// Constructs an engine without exposing its concrete module to the session.
/// The server selects a factory; the session consumes only the Core contracts.
public enum EngineFactoryError: Error, CustomStringConvertible {
    case unsupported(String)
    public var description: String {
        switch self {
        case .unsupported(let detail): detail
        }
    }
}

public struct EngineFactory: Sendable {
    public let prepare: @Sendable (SessionBuildOptions) -> Void
    public let makeContext: @Sendable (MTLDevice?) throws -> MetalContext
    public let prepareLoad: @Sendable (SessionBuildOptions) throws -> EngineLoadPlan
    public let metadata: @Sendable (URL) throws -> EngineModelMetadata
    public let validateReasoning:
        @Sendable (EngineModelMetadata, ModelThinkingMode, ModelReasoningEffort?) throws -> Void
    public let resolveArchitecture: @Sendable (URL) throws -> ArchConfig
    public let preview: @Sendable (URL, Int?) throws -> EngineSessionPreview
    public let readManifest: @Sendable (URL, ArchConfig) throws -> Manifest
    public let buildModel: @Sendable (URL, ArchConfig, MetalContext?) throws -> any EngineModel
    /// Context a factory wants when the caller left the native default in place
    /// and did not ask for scaling. The dense GPU factory uses 1M so a public
    /// install does not run out of context; thresholded RoPE keeps short text on
    /// the native path, so this is capacity, not a numerics change.
    public let defaultMaxContextTokens: Int?
    /// The same load with the session's resolved expert-cache slot count, for a
    /// variant whose model owns its streaming configuration. `nil` keeps the
    /// old closure for engines whose profile is read inside the model.
    public let buildModelWithSlots:
        (@Sendable (URL, ArchConfig, MetalContext?, Int) throws -> any EngineModel)?
    public let buildRunner:
        @Sendable (any EngineModel, SessionBuildOptions) throws -> any EngineRunner
    public let mtpDecoder: @Sendable () -> (any EngineMTPDecoder)?
    public let makeScratch: @Sendable (MetalContext, Int, Float) throws -> RawCompletionScratch
    public let runRawCompletion: @Sendable (RawCompletionRequest) async throws -> RawDecodeResult
    public let makeWarmup: @Sendable (any EngineModel, Int?) -> EngineExpertWarmup?
    public let perSlotBytes: @Sendable (any EngineModel, SessionBuildOptions, EngineLoadPlan) -> Int

    public init(
        prepare: @escaping @Sendable (SessionBuildOptions) -> Void = { _ in },
        makeContext: @escaping @Sendable (MTLDevice?) throws -> MetalContext = {
            if let device = $0 { return try MetalContext(device: device) }
            return try MetalContext()
        },
        prepareLoad: @escaping @Sendable (SessionBuildOptions) throws -> EngineLoadPlan = { _ in
            throw EngineFactoryError.unsupported("the selected factory has no load planner")
        },
        metadata: @escaping @Sendable (URL) throws -> EngineModelMetadata = { _ in
            throw EngineFactoryError.unsupported("the selected factory has no metadata reader")
        },
        validateReasoning:
            @escaping @Sendable (
                EngineModelMetadata, ModelThinkingMode, ModelReasoningEffort?
            ) throws -> Void = { _, _, _ in },
        resolveArchitecture: @escaping @Sendable (URL) throws -> ArchConfig = { _ in
            throw EngineFactoryError.unsupported(
                "the selected factory has no architecture resolver")
        },
        preview: @escaping @Sendable (URL, Int?) throws -> EngineSessionPreview = { _, _ in
            throw EngineFactoryError.unsupported("the selected factory has no session preview")
        },
        readManifest: @escaping @Sendable (URL, ArchConfig) throws -> Manifest = { directory, _ in
            try SSDAIManifestReader.read(directoryURL: directory)
        },
        defaultMaxContextTokens: Int? = nil,
        buildModel: @escaping @Sendable (URL, ArchConfig, MetalContext?) throws -> any EngineModel,
        buildModelWithSlots:
            (@Sendable (URL, ArchConfig, MetalContext?, Int) throws -> any EngineModel)? = nil,
        buildRunner:
            @escaping @Sendable (any EngineModel, SessionBuildOptions) throws -> any EngineRunner,
        mtpDecoder: @escaping @Sendable () -> (any EngineMTPDecoder)? = { nil },
        makeScratch: @escaping @Sendable (MetalContext, Int, Float) throws -> RawCompletionScratch =
            { _, _, _ in RawCompletionScratch() },
        runRawCompletion:
            @escaping @Sendable (RawCompletionRequest) async throws -> RawDecodeResult =
            { _ in
                throw GeneratorError.invalidGenerationConfig(
                    "the selected engine factory does not provide raw completion")
            },
        makeWarmup: @escaping @Sendable (any EngineModel, Int?) -> EngineExpertWarmup? = { _, _ in
            nil
        },
        perSlotBytes:
            @escaping @Sendable (any EngineModel, SessionBuildOptions, EngineLoadPlan) -> Int = {
                _, _, _ in 0
            }
    ) {
        self.prepare = prepare
        self.makeContext = makeContext
        self.prepareLoad = prepareLoad
        self.metadata = metadata
        self.validateReasoning = validateReasoning
        self.resolveArchitecture = resolveArchitecture
        self.preview = preview
        self.readManifest = readManifest
        self.buildModel = buildModel
        self.defaultMaxContextTokens = defaultMaxContextTokens
        self.buildModelWithSlots = buildModelWithSlots
        self.buildRunner = buildRunner
        self.mtpDecoder = mtpDecoder
        self.makeScratch = makeScratch
        self.runRawCompletion = runRawCompletion
        self.makeWarmup = makeWarmup
        self.perSlotBytes = perSlotBytes
    }
}
