import Foundation
import Testing
import TinyTitan
import TinyTitanMemory

@testable import TinyTitanLib
@testable import TinyTitanServerCore
import TinyTitanCore

/// The resident side-engine's home in the server: which install it loads, and
/// how the engine's vocabulary becomes memory's.
///
/// No model is loaded here — a fake `SideEngineModel` answers the adapter, and
/// resolution is a filesystem question — so these run with the unit tests.
@Suite struct SideEngineServiceTests {

    // MARK: - which install

    private func temporaryDirectory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "side-engine-\(name)-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func anExplicitOffMeansNoEngine() {
        #expect(
            ServerSideEngineFactory.resolve(
                environment: ["TINYTITAN_SIDE_ENGINE": "0"], modelsDirectory: "/models") == nil)
        #expect(
            ServerSideEngineFactory.resolve(
                environment: ["TINYTITAN_SIDE_ENGINE": "off"], modelsDirectory: "/models") == nil)
    }

    @Test func theDefaultInstallIsThe4BUnderTheModelsDirectory() throws {
        let models = try temporaryDirectory("models")
        defer { try? FileManager.default.removeItem(at: models) }
        let install = models.appendingPathComponent("qwen3.5_4B_4Bit", isDirectory: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)

        let resolved = ServerSideEngineFactory.resolve(
            environment: [:],
            modelsDirectory: models.path)
        #expect(resolved == install.path)
        #expect(ServerSideEngineFactory.defaultInstall == "qwen3.5_4B_4Bit")
    }

    @Test func aNamedInstallResolvesUnderTheModelsDirectory() throws {
        let models = try temporaryDirectory("models")
        defer { try? FileManager.default.removeItem(at: models) }
        let install = models.appendingPathComponent("qwen3.5_9B_4Bit", isDirectory: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)

        let resolved = ServerSideEngineFactory.resolve(
            environment: ["TINYTITAN_SIDE_ENGINE": "qwen3.5_9B_4Bit"],
            modelsDirectory: models.path)
        #expect(resolved == install.path)
    }

    @Test func aMissingInstallResolvesToNothing() throws {
        let models = try temporaryDirectory("models")
        defer { try? FileManager.default.removeItem(at: models) }
        #expect(
            ServerSideEngineFactory.resolve(
                environment: [:],
                modelsDirectory: models.path) == nil)
        #expect(
            ServerSideEngineFactory.resolve(
                environment: [:],
                modelsDirectory: nil) == nil)
        #expect(
            ServerSideEngineFactory.resolve(
                environment: ["TINYTITAN_SIDE_ENGINE": "qwen3.5_9B_4Bit"],
                modelsDirectory: models.path) == nil)
    }

    @Test func anExplicitPathIsUsedAsGiven() throws {
        let directory = try temporaryDirectory("install")
        defer { try? FileManager.default.removeItem(at: directory) }
        let resolved = ServerSideEngineFactory.resolve(
            environment: ["TINYTITAN_SIDE_ENGINE": directory.path],
            modelsDirectory: nil)
        #expect(resolved == directory.path)
    }

    // MARK: - the adapter

    @Test func aYesAndANoBecomeTrueAndFalse() async {
        let engine = SideEngine { FakeSideEngineModel(answer: "YES") }
        let adapter = SideEngineMemoryAdapter(engine: engine)
        let answer = await adapter.duplicates(
            MemoryFact(
                key: "characters/marcus/eyes",
                value: "grey"),
            MemoryFact(
                key: "characters/marcus/eye_colour",
                value: "grey"))
        #expect(answer == true)
    }

    @Test func aCompletionOutsideTheAnswerSetIsNoDecision() async {
        let engine = SideEngine { FakeSideEngineModel(answer: "Maybe") }
        let adapter = SideEngineMemoryAdapter(engine: engine)
        let answer = await adapter.contradicts(
            MemoryFact(key: "state/inn", value: "standing"),
            MemoryFact(key: "state/inn", value: "burned"))
        #expect(answer == nil)
    }

    /// T7 through the adapter: the two legal answers map, and only those. The
    /// caller that consumes it is `MemoryRetrievalHinter`, never a write.
    @Test func retrievalAnswersMapThroughAsYesAndNo() async {
        let yes = SideEngineMemoryAdapter(engine: SideEngine { FakeSideEngineModel(answer: "YES") })
        let no = SideEngineMemoryAdapter(engine: SideEngine { FakeSideEngineModel(answer: "NO") })
        let unknown = SideEngineMemoryAdapter(
            engine: SideEngine { FakeSideEngineModel(answer: "perhaps") })
        let fact = MemoryFact(key: "rules/ferry", value: "runs only on Sundays")

        #expect(await yes.couldAnswer("How often does the boat cross?", fact) == true)
        #expect(await no.couldAnswer("How often does the boat cross?", fact) == false)
        #expect(await unknown.couldAnswer("How often does the boat cross?", fact) == nil)
    }

    @Test func aModelThatWillNotLoadIsNoDecision() async {
        let engine = SideEngine { throw SideEngineError.missingTokenizer("/nowhere") }
        let adapter = SideEngineMemoryAdapter(engine: engine)
        let answer = await adapter.duplicates(
            MemoryFact(key: "rules/ferry", value: "runs only on Sundays"),
            MemoryFact(key: "rules/ferry_schedule", value: "only Sundays"))
        #expect(answer == nil)
    }

    @Test func aConflictAndAnUpdateMapThrough() async {
        let engine = SideEngine { FakeSideEngineModel(answer: "CONFLICT") }
        let adapter = SideEngineMemoryAdapter(engine: engine)
        let answer = await adapter.supersedes(
            MemoryFact(
                key: "characters/marcus/eyes",
                value: "grey"),
            MemoryFact(
                key: "characters/marcus/eyes",
                value: "hazel"),
            rule: "eye colour is fixed.")
        #expect(answer == .conflict)
    }

    @Test func withoutARuleThereIsNoSupersessionAnswer() async {
        let engine = SideEngine { FakeSideEngineModel(answer: "CONFLICT") }
        let adapter = SideEngineMemoryAdapter(engine: engine)
        let answer = await adapter.supersedes(
            MemoryFact(
                key: "characters/marcus/eyes",
                value: "grey"),
            MemoryFact(
                key: "characters/marcus/eyes",
                value: "hazel"),
            rule: nil)
        #expect(answer == nil)
    }

    @Test func shutdownReleasesTheWeightsAndStopsAnswering() async {
        let engine = SideEngine { FakeSideEngineModel(answer: "NO") }
        let adapter = SideEngineMemoryAdapter(engine: engine)
        let before = await adapter.duplicates(
            MemoryFact(key: "a/one", value: "1"),
            MemoryFact(key: "a/two", value: "1"))
        #expect(before == false)

        await adapter.shutdown()
        let loaded = await engine.isLoaded
        #expect(loaded == false)
        let after = await adapter.duplicates(
            MemoryFact(key: "a/one", value: "1"),
            MemoryFact(key: "a/two", value: "1"))
        #expect(after == nil)
    }

    /// The whole path with a real install: resolve, load, judge, shut down.
    ///
    /// A model run, and **release-only**: in a debug build the CPU engine is
    /// unoptimized and one judgement takes tens of minutes, so there is no
    /// point running it there. `TINYTITAN_SIDE_ENGINE_E2E` names the install,
    /// defaulting to the shipped 4B (`=qwen3.5_9B_4Bit` for the 9B); run it
    /// with `swift test -c release --no-parallel --filter
    /// theRealInstallAnswersThroughTheFactoryAndTheAdapter`. Both installs
    /// decide the two pairs below, and `benchmark/side_engine_wired_cases.py`
    /// has the wider set.
    @Test(.enabled(if: sideEngineEndToEndEnabled()))
    func theRealInstallAnswersThroughTheFactoryAndTheAdapter() async throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TinyTitanServer
            .deletingLastPathComponent()  // tests
            .deletingLastPathComponent()  // <root>
        let modelsDirectory = root.appendingPathComponent("models")
        let requested = ProcessInfo.processInfo.environment["TINYTITAN_SIDE_ENGINE_E2E"]
        let name =
            (requested?.isEmpty == false ? requested : nil)
            ?? ServerSideEngineFactory.defaultInstall

        let side = try #require(
            ServerSideEngineFactory.make(
                environment: [ServerSideEngineFactory.environmentKey: name],
                modelsDirectory: modelsDirectory.path,
                isClientGenerating: nil),
            "no \(name) install under \(modelsDirectory.path)")
        let adapter = SideEngineMemoryAdapter(engine: side)

        // Stored first, incoming second. On the 4B the same pair reversed
        // answers NO, which is why this order is asserted against the real
        // model here as well as in the unit tests.
        let duplicate = await adapter.duplicates(
            MemoryFact(key: "characters/marcus/eyes", value: "grey"),
            MemoryFact(key: "characters/marcus/eye_colour", value: "grey"))
        #expect(duplicate == true)

        let different = await adapter.duplicates(
            MemoryFact(key: "characters/marcus/eyes", value: "grey"),
            MemoryFact(key: "characters/ines/eyes", value: "green"))
        #expect(different == false)

        // T2, on one case each way. Both installs reject this narration line
        // and keep this standing fact; the 9B rejects only 3 of the benchmark's
        // 10 narration lines against the 4B's 9, which is why the default is
        // the 4B.
        let narration = await adapter.isDurable(
            MemoryFact(key: "session/note2", value: "I will write the next ten chapters now."))
        #expect(narration == false)

        let standing = await adapter.isDurable(
            MemoryFact(key: "characters/marcus/eyes", value: "grey"))
        #expect(standing == true)

        // T4 with and without the rule it needs. The conflict is the
        // benchmark's own case, and the update is a state change.
        let conflict = await adapter.supersedes(
            MemoryFact(key: "characters/marcus/eyes", value: "grey"),
            MemoryFact(key: "characters/marcus/eyes", value: "hazel"),
            rule: "eye colour is fixed and must never change.")
        #expect(conflict == .conflict)

        let update = await adapter.supersedes(
            MemoryFact(key: "state/inn", value: "standing"),
            MemoryFact(key: "state/inn", value: "burned to the ground"),
            rule: "eye colour is fixed and must never change.")
        #expect(update == .update)

        await adapter.shutdown()
    }
}

/// A debug build runs the CPU engine unoptimized: one judgement there takes
/// tens of minutes, so the model-run test is release-only.
private func sideEngineEndToEndEnabled() -> Bool {
    ProcessInfo.processInfo.environment["TINYTITAN_SIDE_ENGINE_E2E"] != nil
        && !_isDebugAssertConfiguration()
}

/// Answers every question with one scripted string.
///
/// unchecked-invariant: `answer` is set once at construction and only read
/// afterwards.
private final class FakeSideEngineModel: SideEngineModel, @unchecked Sendable {
    private let answer: String

    init(answer: String) {
        self.answer = answer
    }

    var contention: (@Sendable () -> Bool)?
    var busyThreads = 1
    var idleThreads = 4
    var endOfSequence = 0
    var threads: Int { 1 }

    func reset() {}
    func encode(system: String, user: String) throws -> [Int] { [1] }
    func decode(_ tokens: [Int]) -> String { answer }
    func generate(prompt: [Int], maximumTokens: Int, stopping: Set<Int>) throws -> [Int] { [0] }
}
