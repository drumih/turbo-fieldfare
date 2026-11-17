import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanLegacyEngine
@testable import TinyTitanLib
@testable import TinyTitanCore

/// K1 end-to-end: two real `ServerModelSession`s share only the rendered
/// system prefix. The second session restores it, prefills only the user turn
/// behind it, and produces output byte-identical to a control that prefilled
/// the whole prompt.
///
/// The session is built from its own initializer rather than
/// `ServerModelSession.load`: the loader resolves the *production* Qwen3.6
/// geometry for the family, and the toy fixture is deliberately tiny. Every
/// path under test -- prompt rendering, the interior-checkpoint capture hook,
/// cache resolution, the shared-system restore and the decode loop -- is the
/// production one.
@Suite struct SharedSystemPrefixSessionTests {
    private let systemA = "You are a terse assistant. \(UUID().uuidString)"
    private let systemB = "You are a verbose assistant. \(UUID().uuidString)"

    @Test func aNewSessionReusesTheSystemPrefixAndOnlyTheSuffix() async throws {
        let harness = try await Harness()
        defer { harness.remove() }
        let completionA = try await harness.generate(
            system: systemA, user: "first question")
        let controlA = try await harness.controlGenerate(
            system: systemA, user: "first question")
        harness.expectIdentical(completionA, controlA)

        let systemEnd = try harness.systemEnd(
            system: systemA, user: "a different question")
        #expect(await harness.waitForCheckpoint(system: systemA))
        let loaded = try await harness.sharedStore.load(
            harness.key(system: systemA, systemEnd: systemEnd),
            domain: harness.domain)
        #expect(loaded.metadata.kind == .sharedSystem)
        #expect(loaded.metadata.position == systemEnd)
        #expect(
            loaded.metadata.prefixTokenIDs
                == Array(
                    harness.promptIDs(
                        system: systemA, user: "a different question"
                    ).prefix(systemEnd)))

        let completionB = try await harness.generate(
            system: systemA, user: "a different question")
        let controlB = try await harness.controlGenerate(
            system: systemA, user: "a different question")
        // Only the system segment was reused; the user turn was prefilled.
        #expect(completionB.usage.promptTokensDetails.cachedTokens == systemEnd)
        let fullPrompt = harness.promptIDs(
            system: systemA, user: "a different question")
        #expect(completionB.usage.promptTokens == fullPrompt.count)
        #expect(
            completionB.usage.promptTokens
                - completionB.usage.promptTokensDetails.cachedTokens
                == fullPrompt.count - systemEnd)
        harness.expectIdentical(completionB, controlB)
    }

    @Test func aDifferentSystemPromptMissesAndMatchesTheControl() async throws {
        let harness = try await Harness()
        defer { harness.remove() }
        _ = try await harness.generate(system: systemA, user: "first")
        #expect(await harness.waitForCheckpoint(system: systemA))

        let otherSystem = try await harness.generate(
            system: systemB, user: "first")
        #expect(otherSystem.usage.promptTokensDetails.cachedTokens == 0)
        let control = try await harness.controlGenerate(
            system: systemB, user: "first")
        harness.expectIdentical(otherSystem, control)
    }

    @Test func aDifferentRuntimeDomainMissesAndMatchesTheControl() async throws {
        let harness = try await Harness()
        defer { harness.remove() }
        _ = try await harness.generate(system: systemA, user: "first")
        #expect(await harness.waitForCheckpoint(system: systemA))

        // Same storage, same rendered tokens, different maximum context: the
        // domain hash differs, so the content address differs.
        let other = try harness.session(
            maxContext: 512,
            domain: harness.domain(at: 512),
            store: harness.sharedStore,
            stateStore: harness.stateStore)
        let completion = try await other.generate(
            harness.request(system: systemA, user: "first")
        ) { _ in }
        #expect(completion.usage.promptTokensDetails.cachedTokens == 0)
        let control = try await harness.controlGenerate(
            system: systemA, user: "first", maxContext: 512)
        harness.expectIdentical(completion, control)
    }

    @Test func theSystemPrefixSurvivesARestartWithTheDiskTier() async throws {
        let harness = try await Harness(disk: true)
        defer { harness.remove() }
        _ = try await harness.generate(system: systemA, user: "first")
        #expect(await harness.waitForDurableCheckpoint(system: systemA))

        let systemEnd = try harness.systemEnd(
            system: systemA, user: "second question")
        // A new process: a new actor over the same directory with no memory
        // index. The record must be found on disk, not in the live store.
        let restarted = try PromptCheckpointStore(
            configuration: harness.storageConfiguration)
        let session = try harness.session(
            store: restarted, stateStore: harness.stateStore)
        let completion = try await session.generate(
            harness.request(system: systemA, user: "second question")
        ) { _ in }
        #expect(
            completion.usage.promptTokensDetails.cachedTokens == systemEnd)
        let control = try await harness.controlGenerate(
            system: systemA, user: "second question")
        harness.expectIdentical(completion, control)
    }

    @Test func qwen38AuxiliaryStateIsCarriedOrTheRestoreMisses() async throws {
        let harness = try await Harness(qsa: true)
        defer { harness.remove() }
        _ = try await harness.generate(system: systemA, user: "first")
        #expect(await harness.waitForCheckpoint(system: systemA))
        let systemEnd = try harness.systemEnd(system: systemA, user: "first")
        let key = harness.key(system: systemA, systemEnd: systemEnd)
        let captured = try await harness.sharedStore.load(
            key, domain: harness.domain)
        // A QSA model's checkpoint carries the indexer payload.
        #expect(captured.auxiliary != nil)
        #expect(captured.auxiliary?.position == systemEnd)

        // A record without the auxiliary payload must be a miss: the runner
        // refuses it and resets, never continues from the wrong indexer
        // state.
        try await harness.sharedStore.save(
            captured.metadata,
            snapshot: captured.snapshot,
            auxiliary: nil)
        // A different user turn makes this a different conversation, so the
        // same-conversation interior path cannot answer it: only the shared
        // system prefix is in play.
        let withoutAux = try await harness.generate(
            system: systemA, user: "second question")
        #expect(withoutAux.usage.promptTokensDetails.cachedTokens == 0)
        let control = try await harness.controlGenerate(
            system: systemA, user: "second question")
        harness.expectIdentical(withoutAux, control)

        // With the payload restored, the same shared entry is a hit.
        try await harness.sharedStore.save(
            captured.metadata,
            snapshot: captured.snapshot,
            auxiliary: captured.auxiliary)
        let withAux = try await harness.generate(
            system: systemA, user: "third question")
        #expect(withAux.usage.promptTokensDetails.cachedTokens == systemEnd)
        let auxControl = try await harness.controlGenerate(
            system: systemA, user: "third question")
        harness.expectIdentical(withAux, auxControl)
    }
}

// MARK: - Harness

/// One install, one Metal context, one tokenizer, one shared checkpoint store,
/// and the sessions built on top of them.
private struct Harness {
    let directory: URL
    let fixture: URL
    let context: MetalContext
    let tokenizer: GFTokenizer
    let modelArch: ArchConfig
    let sharedStore: PromptCheckpointStore
    let stateStore: ServerPromptStateStore
    let domain: ServerPromptCacheDomain
    let maxContext: Int
    let memoryLimit = 8 * 1_048_576

    let diskDirectory: URL?

    init(qsa: Bool = false, disk: Bool = false) async throws {
        let directory = try ToyInstall.make(qsa: qsa)
        let fixture = try ChatMLTemplateTests.fixtureFolder()
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load(from: fixture)
        let diskDirectory: URL? =
            disk
            ? FileManager.default.temporaryDirectory.appendingPathComponent(
                "shared-system-disk-\(UUID().uuidString)")
            : nil
        let stateStore = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 8 * 1_048_576,
                diskDirectory: nil,
                diskLimitBytes: 0))
        let domain = ServerPromptCacheDomain(
            modelID: qsa ? "qwen-toy-qsa" : "qwen-toy",
            sourceSnapshotHash: "snapshot",
            runtimeProfileHash: "profile",
            maximumContext: 256,
            kvStorage: "int8",
            fp16RingEnabled: false,
            templateSHA256: "template")
        self.directory = directory
        self.fixture = fixture
        self.context = context
        self.tokenizer = tokenizer
        self.modelArch = qsa ? .qwenToyQSA() : .qwenToy()
        self.stateStore = stateStore
        self.domain = domain
        self.maxContext = 256
        self.diskDirectory = diskDirectory
        // The production path obtains this same actor through
        // `PromptCheckpointStore.sharedInstance`; holding it here makes the
        // sessions below genuinely share one store.
        self.sharedStore = try PromptCheckpointStore.sharedInstance(
            configuration: Self.storage(
                memoryLimit: memoryLimit, diskDirectory: diskDirectory))
    }

    static func storage(
        memoryLimit: Int, diskDirectory: URL?
    ) -> ServerPromptCacheStorageConfiguration {
        ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: memoryLimit,
            diskDirectory: diskDirectory,
            diskLimitBytes: 64 * 1_048_576)
    }

    var storageConfiguration: ServerPromptCacheStorageConfiguration {
        Self.storage(memoryLimit: memoryLimit, diskDirectory: diskDirectory)
    }

    func remove() {
        try? FileManager.default.removeItem(at: directory)
        if let diskDirectory {
            try? FileManager.default.removeItem(at: diskDirectory)
        }
    }

    func domain(at maximumContext: Int) -> ServerPromptCacheDomain {
        ServerPromptCacheDomain(
            modelID: domain.modelID,
            sourceSnapshotHash: domain.sourceSnapshotHash,
            runtimeProfileHash: domain.runtimeProfileHash,
            maximumContext: maximumContext,
            kvStorage: domain.kvStorage,
            fp16RingEnabled: domain.fp16RingEnabled,
            templateSHA256: domain.templateSHA256)
    }

    func session(
        maxContext: Int? = nil,
        domain: ServerPromptCacheDomain? = nil,
        store: PromptCheckpointStore,
        stateStore: ServerPromptStateStore?
    ) throws -> ServerModelSession {
        let contextWindow = maxContext ?? self.maxContext
        let model = try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: modelArch)
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: true, kvCachePrecision: .int8)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: contextWindow,
            slots: 1,
            runtimeConfiguration: runtime)
        // The injected factory closures are the production 6.x pair, exactly
        // as the ExpertWarmupTests fixture supplies them; the scratch comes
        // from the factory so it is the Core-typed buffer the session holds.
        let factory = EngineFactories.sixXGPU()
        let scratch = try factory.makeScratch(
            context,
            model.config.vocabSize,
            Float(model.config.finalLogitSoftcap))
        return ServerModelSession(
            context: context,
            model: model,
            metadata: EngineModelMetadata(
                modelID: model.modelID, family: model.config.modelFamily.rawValue,
                weightBits: model.routedExpertWeightBits, sampling: .house,
                reasoningControl: .binaryThinking),
            defaultModelID: model.modelID,
            tokenizer: tokenizer,
            tokenizerFolder: fixture,
            loadedReasoning: RequestReasoning(thinkingMode: .off, effort: nil),
            runner: runner,
            mtpDecoder: nil,
            runCompletion: factory.runRawCompletion,
            makeWarmup: factory.makeWarmup,
            scratches: [scratch],
            prefillConfig: .production(chunkTokens: 64),
            expertCacheSlots: 0,
            slots: 1,
            maxContext: contextWindow,
            promptCacheMode: .multiPrefix,
            promptCacheDomain: domain ?? self.domain(at: contextWindow),
            promptCache: ServerPromptCache(maximumEntries: 2),
            promptStateStore: stateStore,
            promptCheckpointStore: store,
            concisePrompt: nil)
    }

    func request(
        system: String, user: String, maxNewTokens: Int = 8
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: Self.messages(system: system, user: user),
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(
                maxNewTokens: maxNewTokens, temperature: 0),
            maximumCompletionTokens: maxNewTokens)
    }

    static func messages(
        system: String, user: String
    ) -> [GFTokenizer.Message] {
        [
            GFTokenizer.Message(role: .system, content: system),
            GFTokenizer.Message(role: .user, content: user),
        ]
    }

    func promptIDs(system: String, user: String) -> [Int32] {
        let messages = Self.messages(system: system, user: user)
        guard let rendered = try? tokenizer.applyChatTemplate(messages)
        else { return [] }
        return tokenizer.encode(rendered, addBOS: false)
    }

    func systemEnd(system: String, user: String) throws -> Int {
        let messages = Self.messages(system: system, user: user)
        return try #require(
            PromptCheckpointPlanner.systemEndPosition(
                promptIDs: promptIDs(system: system, user: user),
                messages: messages,
                imStartID: tokenizer.imStartID))
    }

    func key(system: String, systemEnd: Int) -> PromptCheckpointKey {
        PromptCheckpointKey.make(
            domain: domain,
            prefixTokenIDs: Array(
                promptIDs(system: system, user: "first").prefix(systemEnd)))
    }

    /// One fresh session on the shared store. Each call is a new session, so
    /// the second call is the "new session" whose request must restore.
    func generate(
        system: String, user: String
    ) async throws -> ServerCompletion {
        let session = try session(
            store: sharedStore, stateStore: stateStore)
        return try await session.generate(
            request(system: system, user: user)
        ) { _ in }
    }

    /// The full-prefill control: a session with its own empty store.
    func controlGenerate(
        system: String, user: String, maxContext: Int? = nil
    ) async throws -> ServerCompletion {
        let controlDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "shared-system-control-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: controlDirectory) }
        let controlStore = try PromptCheckpointStore.sharedInstance(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: memoryLimit,
                diskDirectory: controlDirectory,
                diskLimitBytes: 64 * 1_048_576))
        let controlState = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: memoryLimit,
                diskDirectory: nil,
                diskLimitBytes: 0))
        let session = try session(
            maxContext: maxContext,
            store: controlStore,
            stateStore: controlState)
        return try await session.generate(
            request(system: system, user: user)
        ) { _ in }
    }

    /// Wait for the durable disk copy, not just the RAM record: the capture
    /// writes it from a detached task.
    func waitForDurableCheckpoint(system: String) async -> Bool {
        guard let diskDirectory,
            let end = try? systemEnd(system: system, user: "first")
        else { return false }
        let record = diskDirectory.appendingPathComponent(
            key(system: system, systemEnd: end).contentHash,
            isDirectory: true)
        let metadata = record.appendingPathComponent(
            PromptCheckpointStore.metadataName)
        for _ in 0..<400 {
            if FileManager.default.fileExists(atPath: metadata.path) {
                return true
            }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return false
    }

    func waitForCheckpoint(system: String) async -> Bool {
        guard let end = try? systemEnd(system: system, user: "first")
        else { return false }
        let key = key(system: system, systemEnd: end)
        for _ in 0..<400 {
            if await sharedStore.contains(key) { return true }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return false
    }

    func expectIdentical(
        _ first: ServerCompletion,
        _ second: ServerCompletion
    ) {
        #expect(
            first.content == second.content, "content differs")
        #expect(first.reasoning == second.reasoning, "reasoning differs")
        #expect(
            first.finishReason == second.finishReason,
            "finish reason differs")
        #expect(
            first.usage.completionTokens == second.usage.completionTokens,
            "generated token count differs")
    }
}

// MARK: - Install

/// A tiny runnable install: `QwenToySynthetic` weights plus the ChatML
/// tokenizer fixture as the `tokenizer/` sidecar a model session expects.
enum ToyInstall {
    static func make(qsa: Bool = false) throws -> URL {
        let directory = try QwenToySynthetic.write(
            weightBits: 4, qsa: qsa)
        let tokenizer = directory.appendingPathComponent(
            "tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(
            at: tokenizer, withIntermediateDirectories: true)
        let fixture = try ChatMLTemplateTests.fixtureFolder()
        for name in [
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        ] {
            try FileManager.default.copyItem(
                at: fixture.appendingPathComponent(name),
                to: tokenizer.appendingPathComponent(name))
        }
        return directory
    }
}
