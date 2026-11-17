import Foundation

/// Constructs an engine without exposing its concrete module to the session.
/// The server selects a factory; the session consumes only the Core contracts.
public struct EngineFactory: Sendable {
    public let buildModel: @Sendable (URL, ArchConfig, MetalContext?) throws -> any EngineModel
    public let buildRunner:
        @Sendable (any EngineModel, SessionBuildOptions) throws -> any EngineRunner

    public init(
        buildModel: @escaping @Sendable (URL, ArchConfig, MetalContext?) throws -> any EngineModel,
        buildRunner:
            @escaping @Sendable (any EngineModel, SessionBuildOptions) throws -> any EngineRunner
    ) {
        self.buildModel = buildModel
        self.buildRunner = buildRunner
    }
}
