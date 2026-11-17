/// Builds one actor-confined CPU engine without exposing its concrete module.
/// The closure captures the engine's own snapshot representation; Core does not
/// need to know how a variant stores or maps its tensors.
public struct CPUEngineFactory: Sendable {
    public let build: @Sendable () throws -> any CPUEngineRunner

    public init(build: @escaping @Sendable () throws -> any CPUEngineRunner) {
        self.build = build
    }
}
