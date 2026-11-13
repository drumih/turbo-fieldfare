// The engine an embedder holds: one install, one Metal device, one resident
// model.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). It is a thin, honest layer over the moved orchestrator: loading is
// `ServerModelSession.load`, which is the same path the server takes, and the
// descriptor is read from the install's manifest. No second implementation of
// anything lives here.
//
// Phase A1 gaps, all reported rather than papered over:
//
//   * The loading path builds its own `MetalContext`, which takes the system
//     default device, so the caller's device is honoured only when it is that
//     one; anything else is refused with `metalUnavailable`.
//   * `ServerModelSession.load` derives its streaming mode from the expert
//     cache slot count and resolves its integrity policy from the install, so
//     `EngineConfiguration` cannot expose those two knobs yet.
import Foundation
import Metal
import TinyTitan

/// One loaded install and the device it runs on.
///
/// The engine owns the resident weights, the expert cache and the Metal
/// pipelines; a `Session` owns one conversation's state. Serialising through
/// the actor means a caller cannot touch the engine while it loads or unloads.
public actor Engine {
    /// What was loaded. Readable without awaiting the actor: it is fixed at
    /// init and survives `unload()`.
    public nonisolated let descriptor: ModelDescriptor

    /// The sampling row this install declares, resolved for the thinking mode
    /// the engine was loaded at. A caller that leaves a `GenerationOptions`
    /// field at its house value can fill it from here instead of reaching into
    /// the runtime.
    public nonisolated let samplingDefaults: SamplingDefaults

    private var modelSession: ServerModelSession?
    private let reasoningProfile: ServerReasoningProfile

    /// Loads `directory` on `device`.
    ///
    /// Throws `TinyTitanError` for the failures the facade can classify
    /// (`modelNotFound`, `notAnInstall`, `unsupportedFamily`,
    /// `metalUnavailable`, `cancelled`). A failure the engine reports with an
    /// internal error type is rethrown as it is, rather than flattened into a
    /// case that would claim more than the facade knows.
    public init(
        directory: URL,
        device: MTLDevice,
        configuration: EngineConfiguration = .init()
    ) async throws {
        // Before anything that can log: the loader's first line is the RAM
        // profile, and an embedder that installed a sink should not miss it.
        ServerLog.useSink(configuration.logSink)
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw TinyTitanError.modelNotFound(directory)
        }
        let family: ModelFamily
        do {
            family = try ManifestReader.peekFamily(directoryURL: directory)
        } catch {
            throw TinyTitanError.notAnInstall(directory)
        }
        let arch = try Self.resolveArch(family: family, directory: directory)
        let manifest = try Self.readManifest(directory: directory, arch: arch)
        let loaded: ServerModelSession
        do {
            loaded = try await ServerModelSession.load(
                modelDirectory: directory,
                maxContext: configuration.contextWindow,
                slots: configuration.maxConcurrentGenerations,
                prefillChunkTokens: configuration.prefillChunkTokens,
                kvCachePrecision: configuration.cachePrecision.engineValue,
                ropeScalingMode: configuration.ropeScaling.engineValue,
                thinkingMode: configuration.thinkingMode.engineValue,
                reasoningEffort: configuration.reasoningEffort?.engineValue,
                integrityPolicy: configuration.integrityPolicy.engineValue,
                expertCacheSlots: configuration.expertCacheSlots,
                expertCacheBudgetBytes: configuration.expertCacheBudgetBytes,
                rdadvisePolicy: configuration.readAhead?.engineValue,
                forceLogitsHead: configuration.forceLogitsHead,
                device: device)
        } catch is ServerInferenceError {
            // The loader's only typed refusal is "this build cannot run it".
            throw TinyTitanError.unsupportedFamily(family: family.rawValue)
        } catch let error as GFTokenizerError {
            switch error {
            case .modelDirectoryNotFound:
                throw TinyTitanError.modelNotFound(directory)
            case .missingToolTemplate, .missingSpecialToken, .invalidChatTemplate:
                throw TinyTitanError.notAnInstall(directory)
            case .unsupportedForDialect:
                throw error
            }
        } catch is CancellationError {
            throw TinyTitanError.cancelled
        } catch let error as MetalError {
            // The context could not be built on the caller's device. That is the
            // one case the facade has language for; a shader or pipeline failure
            // is not, and is rethrown as it is.
            switch error {
            case .noDevice, .noQueue:
                throw TinyTitanError.metalUnavailable(reason: error.description)
            case .missingShaderResource, .missingFunction, .libraryCompileFailed,
                .commandEncoderFailed, .bufferAllocationFailed, .invalidState:
                throw error
            }
        } catch let error as ModelError {
            guard let classified = Self.classify(error, directory: directory, family: family)
            else { throw error }
            throw classified
        }
        self.descriptor = Self.describe(
            session: loaded, manifest: manifest)
        self.reasoningProfile = ServerReasoningProfile(
            family: loaded.modelFamily,
            thinkingMode: loaded.loadedReasoning.thinkingMode,
            reasoningEffort: loaded.loadedReasoning.effort)
        self.samplingDefaults = Self.effectiveSampling(
            profile: loaded.samplingDefaults,
            family: loaded.modelFamily,
            thinking: loaded.loadedReasoning.thinkingMode.isEnabled)
        self.modelSession = loaded
    }

    /// A new conversation over the loaded model, with an optional system
    /// prompt prepended to every request.
    ///
    /// Cheap and non-throwing: the model is already resident, so the session
    /// only carries the system prompt and its own generation state. A session
    /// made after `unload()` is inert and throws `engineShutDown` on use.
    public func session(system: String? = nil) -> Session {
        Session(
            system: system,
            modelSession: modelSession,
            descriptor: descriptor,
            reasoningProfile: reasoningProfile)
    }

    /// Drops the engine's reference to the model.
    ///
    /// Sessions hold the model weakly, so once the last in-flight generation
    /// has drained the resident buffers and the expert cache go away with it;
    /// any session still alive then reports `engineShutDown`.
    public func unload() async {
        modelSession = nil
    }

    /// Wait for the startup expert warm-up to finish instead of letting the
    /// first request stop it. For a one-shot caller (the CLI) whose only
    /// request would otherwise arrive before any layer is warm. `package`: the
    /// warm-up is not part of the facade's promise.
    package func awaitExpertWarmup() async {
        await modelSession?.awaitExpertWarmup()
    }

    /// The sampling row an install declares, resolved for `thinkingMode`.
    ///
    /// Read from `manifest.json` alone — no weights, no Metal, no load — so a
    /// caller that must decide something before the engine exists (whether a
    /// pure-greedy plan may take the fused head) can still ask. An unreadable
    /// install falls back to the house row, exactly as the CLI did.
    static func effectiveSampling(
        profile: GenerationDefaults.Sampling,
        family: ModelFamily?,
        thinking: Bool
    ) -> SamplingDefaults {
        // Qwen3.8 publishes a row per mode and the profile carries only the
        // thinking one, so the mode picks between them here — the same choice
        // the server makes during validation.
        if let family,
            family == .qwen38flash || family == .qwen38flashMTP
                || family == .qwen4flash || family == .qwen4flashMTP
        {
            return SamplingDefaults(
                GenerationDefaults.forFamily(family, thinking: thinking))
        }
        return SamplingDefaults(profile)
    }

    /// The smallest prefill chunk the engine accepts that covers `prompt`, or
    /// the largest allowed chunk when none does.
    ///
    /// `EngineConfiguration.prefillChunkTokens` is settled at load, so a front
    /// end whose flag means "size the chunk to the prompt" has to know the
    /// prompt's length before the engine exists. This answers that with the
    /// install's tokenizer alone — no weights, no Metal — so the flag keeps its
    /// meaning without the front end owning a tokenizer.
    public static func prefillChunk(
        covering prompt: Prompt,
        directory: URL,
        thinkingMode: ThinkingMode = .off,
        reasoningEffort: ReasoningEffort? = nil
    ) async throws -> Int {
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: directory,
            thinkingMode: thinkingMode.engineValue,
            reasoningEffort: reasoningEffort?.engineValue)
        let tokenCount: Int
        switch prompt {
        case .raw(let text):
            tokenCount = tokenizer.encode(text, addBOS: true).count
        case .messages(let messages):
            let rendered = try tokenizer.applyChatTemplate(
                messages.map {
                    GFTokenizer.Message(
                        role: $0.role.tokenizerRole, content: $0.content)
                })
            tokenCount = tokenizer.encode(rendered, addBOS: false).count
        }
        return RuntimeConfiguration.allowedPrefillChunkTokens
            .first { $0 >= tokenCount }
            ?? PrefillRuntimeConfig.maxChunkTokens
    }

    /// Turn the loader's own failure into the facade's, or `nil` to rethrow it.
    ///
    /// The taxonomy a caller acts on is three-way: a wrong format is the
    /// caller's to fix, a corrupt install is a re-download, and an architecture
    /// this build cannot run is a support question. Failures outside that
    /// taxonomy — a Metal command buffer that errored, a runtime invariant —
    /// say something about the machine rather than about the install, and are
    /// deliberately **not** flattened into a case that would claim more than the
    /// facade knows.
    ///
    /// `package` rather than private so the library's own tests can pin the
    /// mapping without a model on disk, and exhaustive rather than defaulted so
    /// a new loader failure has to be classified rather than inherited.
    package static func classify(
        _ error: ModelError,
        directory: URL,
        family: ModelFamily
    ) -> TinyTitanError? {
        switch error {
        case .notASSDAIDirectory, .unsupportedVersion, .unknownFlag:
            return .unsupportedFormat(detail: error.description)
        case .checksumMismatch(let file):
            return .integrityFailure(path: file, detail: error.description)
        case .trustedReceiptInvalid(let detail), .indexCorrupt(let detail):
            return .integrityFailure(path: directory.path, detail: detail)
        case .metadataOverBound:
            // A document over its bound is a corrupt or planted file, not a
            // support question: the install is the caller's to replace.
            return .integrityFailure(path: directory.path, detail: error.description)
        case .archMismatch, .unsupportedArchitecture:
            return .unsupportedFamily(family: family.rawValue)
        case .partialInstall, .missingFile, .tensorNotFound, .tensorSizeMismatch,
            .expertStrideNotPageAligned:
            return .notAnInstall(directory)
        case .residentBufferWrapFailed, .posixFailed, .expertCacheUnplaceable,
            .commandBufferFailed, .internalInconsistency:
            return nil
        }
    }

    private static func resolveArch(family: ModelFamily, directory: URL) throws -> ArchConfig {
        do {
            return try ArchConfig.resolved(forFamily: family, directoryURL: directory)
        } catch let error as ArchResolutionError {
            switch error {
            case .unsupportedFamily(let name):
                throw TinyTitanError.unsupportedFamily(family: name)
            case .manifestOmitsField:
                throw TinyTitanError.notAnInstall(directory)
            case .manifestDeclaresInvalidValue(let family, let field, let value):
                throw TinyTitanError.unsupportedFamily(
                    family: "\(family) (\(field)=\(value))")
            }
        } catch {
            throw TinyTitanError.notAnInstall(directory)
        }
    }

    private static func readManifest(directory: URL, arch: ArchConfig) throws -> Manifest {
        do {
            return try ManifestReader.load(directoryURL: directory, expecting: arch)
        } catch {
            throw TinyTitanError.notAnInstall(directory)
        }
    }

    private static func describe(
        session: ServerModelSession,
        manifest: Manifest
    ) -> ModelDescriptor {
        ModelDescriptor(
            id: session.defaultModelID,
            family: session.modelFamily.rawValue,
            contextWindow: session.maximumContext,
            weightBytes: manifest.files.values.reduce(UInt64(0)) { $0 + $1.size },
            expertCacheBytes: UInt64(session.expertCacheSlots) * manifest.expertStride
                * UInt64(manifest.arch.numLayers))
    }
}
