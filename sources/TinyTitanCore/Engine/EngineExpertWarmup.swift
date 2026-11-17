/// A running engine-owned warm-up. The session owns its lifecycle; the factory
/// owns the work and the opaque completion value.
public struct EngineExpertWarmup: Sendable {
    public let run: @Sendable () async -> (any Sendable)?

    public init(run: @escaping @Sendable () async -> (any Sendable)?) {
        self.run = run
    }
}
