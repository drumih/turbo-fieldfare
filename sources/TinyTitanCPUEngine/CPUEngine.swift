import Foundation
import TinyTitanCore
import TinyTitanLib

/// The dense CPU engine, behind the same `Session` facade the GPU `Engine`
/// publishes.
///
/// It loads the CPU backend the server also serves, and hands sessions the
/// same public `Session`/`ChatMessage`/`GenerationOptions`/`GenerationEvent`
/// surface. The CPU backend is its own generation path (logits per token, no
/// Metal), so there is no `Engine`-style device or Metal context here; the
/// caller chooses this type, not a flag inside `Engine`.
public actor CPUEngine {
    /// The context ceiling the CPU backend enforces, public so a caller can
    /// name the same default without reaching into the backend type.
    public static let contextCeiling = 32_768

    /// What was loaded. Fixed at init and readable without awaiting.
    public nonisolated let descriptor: ModelDescriptor
    /// The sampling row the install declares, resolved for the thinking mode
    /// the backend was loaded at.
    public nonisolated let samplingDefaults: SamplingDefaults

    private var backend: CPUModelBackend?
    /// The strong half of the session link; `Session` holds it weakly, so
    /// `unload()` releases the engine and live sessions report `engineShutDown`.
    private var box: BackendBox?
    private let reasoningProfile: ServerReasoningProfile

    public init(
        snapshotDirectory: URL,
        factory: CPUEngineFactory? = nil,
        maximumContext: Int = CPUEngine.contextCeiling,
        thinkingMode: ModelThinkingMode = .off,
        routedExpertWeightBits: Int? = nil
    ) async throws {
        let backend = try await CPUModelBackend(
            snapshotDirectory: snapshotDirectory,
            factory: factory,
            maximumContext: maximumContext,
            thinkingMode: thinkingMode,
            routedExpertWeightBits: routedExpertWeightBits)
        self.backend = backend
        self.box = BackendBox(backend)
        self.descriptor = backend.descriptor
        self.samplingDefaults = SamplingDefaults(backend.samplingDefaults)
        self.reasoningProfile = ServerReasoningProfile(
            family: backend.descriptor.family,
            thinkingMode: thinkingMode,
            reasoningEffort: nil,
            reasoningControl: .binaryThinking)
    }

    /// A new conversation over the loaded model. Cheap and non-throwing; a
    /// session made after `unload()` is inert and throws on use.
    public func session(system: String? = nil) -> Session {
        Session(
            system: system,
            box: box,
            descriptor: descriptor,
            reasoningProfile: reasoningProfile)
    }

    /// Releases the resident engine. Live sessions then report `engineShutDown`.
    public func unload() async {
        backend = nil
        box = nil
    }

    /// There is no expert warm-up on a dense CPU engine; the method exists so a
    /// front end can treat the two engines alike.
    public func awaitExpertWarmup() async {}
}
