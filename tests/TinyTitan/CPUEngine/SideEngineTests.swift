import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The resident 2B helper, without a 2B.
///
/// The model itself is checked by `benchmark/side_engine_tasks.py` over a real
/// snapshot, which is a model run. What is here is everything a fake model can
/// settle: the parser, the prompts' wording against the benchmark that
/// measured them, and the actor's contract — lazy load, one load, a clean
/// state per job, a shutdown that keeps its meaning. The actor drives a
/// protocol rather than `CPUQwen35` for exactly this reason.
@Suite struct SideEngineTests {

    // MARK: - the parser

    @Test func aFirstWordIsAnAnswerAndNothingElseIs() {
        #expect(SideEngineAnswer(firstWordOf: "YES", allowed: [.yes, .no]) == .yes)
        #expect(SideEngineAnswer(firstWordOf: "yes", allowed: [.yes, .no]) == .yes)
        #expect(SideEngineAnswer(firstWordOf: "No.", allowed: [.yes, .no]) == .no)
        #expect(SideEngineAnswer(firstWordOf: "1. YES", allowed: [.yes, .no]) == .yes)
        // A flourish after the decision is still the decision.
        #expect(
            SideEngineAnswer(
                firstWordOf: "YES, the person wrote it",
                allowed: [.yes, .no]) == .yes)
        // A word that merely starts with the answer is not the answer.
        #expect(SideEngineAnswer(firstWordOf: "NOTHING", allowed: [.yes, .no]) == nil)
        #expect(SideEngineAnswer(firstWordOf: "YESTERDAY", allowed: [.yes, .no]) == nil)
        #expect(SideEngineAnswer(firstWordOf: "NONE", allowed: [.yes, .no]) == nil)
        #expect(SideEngineAnswer(firstWordOf: "", allowed: [.yes, .no]) == nil)
        #expect(SideEngineAnswer(firstWordOf: "   ", allowed: [.yes, .no]) == nil)
    }

    @Test func anAnswerOutsideTheTasksOwnSetIsNoDecision() {
        // YES is not an answer to T4, and CONFLICT is not an answer to T1.
        #expect(
            SideEngineAnswer(
                firstWordOf: "YES",
                allowed: SideEngineTask.supersession.answers) == nil)
        #expect(
            SideEngineAnswer(
                firstWordOf: "CONFLICT",
                allowed: SideEngineTask.durability.answers) == nil)
        #expect(
            SideEngineAnswer(
                firstWordOf: "UPDATE.",
                allowed: SideEngineTask.supersession.answers) == .update)
    }

    @Test func everyTaskIsOneQuestionWithAClosedSet() {
        #expect(SideEngineTask.supersession.answers == [.update, .conflict])
        for task in SideEngineTask.allCases where task != .supersession {
            #expect(task.answers == [.yes, .no])
        }
    }

    @Test func aJudgementKnowsWhichTaskItIs() {
        #expect(SideEngineJudgement.durability(key: "k", value: "v").task == .durability)
        #expect(
            SideEngineJudgement.retrieval(question: "q", key: "k", value: "v").task
                == .retrieval)
        #expect(
            SideEngineJudgement.supersession(key: "k", earlier: "a", now: "b").task
                == .supersession)
    }

    @Test func aRuleIsOnlyShownWhenTheCallerHasOne() {
        let without = SideEngineJudgement.supersession(
            key: "state/inn",
            earlier: "standing", now: "burned")
        #expect(without.userPrompt.hasPrefix("EARLIER:"))
        #expect(!without.userPrompt.contains("RULE:"))

        let with = SideEngineJudgement.supersession(
            key: "characters/marcus/eyes",
            earlier: "grey", now: "hazel",
            rule: "eye colour is fixed.")
        #expect(with.userPrompt.hasPrefix("RULE: eye colour is fixed.\nEARLIER:"))
    }

    @Test func anErrorSaysWhatItRead() {
        let answer = SideEngineError.unparsableAnswer(task: .durability, completion: "Dunno")
        #expect(answer.description.contains("T2"))
        #expect(answer.description.contains("Dunno"))
        #expect(answer.description.contains("YES/NO"))
        #expect(
            SideEngineError.missingTokenizer("/tmp/x").description
                == "no tokenizer in /tmp/x")
        #expect(SideEngineError.shutDown.description == "the side-engine has been shut down")
    }

    // MARK: - the prompts are the measured ones

    /// One judgement per task, with the exact user prompt it must produce.
    /// These are the shapes `benchmark/side_engine_tasks.py` asks; the user
    /// side is pinned here because the benchmark builds it with f-strings.
    static let samples: [(judgement: SideEngineJudgement, user: String)] = [
        (
            .clauseAttribution(
                personWrote: "I keep the diary in the drawer.",
                address: "diary/location", clause: "in the drawer"),
            "WHAT THE PERSON WROTE:\nI keep the diary in the drawer.\n\n"
                + "STATEMENT: diary/location = in the drawer\nDid the person state this?"
        ),
        (
            .durability(key: "rules/ferry", value: "runs only on Sundays"),
            "FACT: rules/ferry = runs only on Sundays\nKeep it?"
        ),
        (
            .contradiction(
                aKey: "characters/marcus/eyes", aValue: "grey",
                bKey: "characters/marcus/eyes", bValue: "hazel"),
            "A: characters/marcus/eyes = grey\nB: characters/marcus/eyes = hazel\n"
                + "Do A and B disagree?"
        ),
        (
            .supersession(
                key: "characters/marcus/eyes", earlier: "grey", now: "hazel",
                rule: "eye colour is fixed and must never change."),
            "RULE: eye colour is fixed and must never change.\n"
                + "EARLIER: characters/marcus/eyes = grey\nNOW: characters/marcus/eyes = hazel\n"
                + "Which is it?"
        ),
        (
            .duplication(
                aKey: "setting/town", aValue: "Ashgrove",
                bKey: "setting/place", bValue: "Ashgrove"),
            "A: setting/town = Ashgrove\nB: setting/place = Ashgrove\nSame fact?"
        ),
        (
            .replyCheck(
                key: "rules/ferry", value: "runs only on Sundays",
                reply: "Take the Tuesday ferry."),
            "KNOWN: rules/ferry = runs only on Sundays\nREPLY: Take the Tuesday ferry.\n"
                + "Does the reply contradict what is known?"
        ),
        (
            .retrieval(
                question: "When does the ferry run?",
                key: "rules/ferry", value: "runs only on Sundays"),
            "QUESTION: When does the ferry run?\nFACT: rules/ferry = runs only on Sundays\n"
                + "Could this fact answer it?"
        ),
    ]

    /// The fixed words of each user template, which the benchmark builds
    /// around its values. A rename on either side trips this.
    static let userAnchors = [
        "WHAT THE PERSON WROTE:", "STATEMENT: ", "Did the person state this?",
        "FACT: ", "Keep it?",
        "A: ", "B: ", "Do A and B disagree?", "Same fact?",
        "EARLIER: ", "NOW: ", "Which is it?",
        "KNOWN: ", "REPLY: ", "Does the reply contradict what is known?",
        "QUESTION: ", "Could this fact answer it?",
    ]

    @Test func theShippedPromptsAreTheBenchmarksOwn() throws {
        let source = try String(contentsOf: benchmarkScript(), encoding: .utf8)
        let systems = Self.pythonSystems(in: source)
        #expect(systems.count == SideEngineTask.allCases.count)
        for sample in Self.samples {
            let task = sample.judgement.task.rawValue
            #expect(
                sample.judgement.systemPrompt == systems[task],
                "\(task)'s system prompt has drifted from benchmark/side_engine_tasks.py")
            #expect(
                sample.judgement.userPrompt == sample.user,
                "\(task)'s user prompt changed shape")
        }
    }

    @Test func everyUserTemplateIsStillInTheBenchmark() throws {
        let source = try String(contentsOf: benchmarkScript(), encoding: .utf8)
        for anchor in Self.userAnchors {
            #expect(source.contains(anchor), "the benchmark no longer asks \"\(anchor)\"")
        }
    }

    private func benchmarkScript() throws -> URL {
        // <root>/tests/TinyTitan/CPUEngine/<this file>
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // CPUEngine
            .deletingLastPathComponent()  // TinyTitan
            .deletingLastPathComponent()  // tests
            .deletingLastPathComponent()  // <root>
        let url = root.appendingPathComponent("benchmark/side_engine_tasks.py")
        try #require(
            FileManager.default.fileExists(atPath: url.path),
            "benchmark/side_engine_tasks.py is the source of these prompts")
        return url
    }

    /// Reads `SYSTEMS` out of the benchmark by joining each entry's adjacent
    /// string literals and appending `ONE_WORD` where the entry writes
    /// `+ ONE_WORD`. A Python parser would be the honest tool; this covers the
    /// one shape the block is written in, and a key that fails to appear makes
    /// the count assertion above fail rather than quietly pass.
    private static func pythonSystems(in source: String) -> [String: String] {
        let oneWord = pythonStringConstant(named: "ONE_WORD", in: source)
        var systems: [String: String] = [:]
        var key: String?
        var pieces: [String] = []
        var usesOneWord = false
        var started = false
        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            let text = String(line)
            if !started {
                if text.contains("SYSTEMS = {") { started = true }
                continue
            }
            if key == nil {
                guard let open = text.range(of: "\": (") else {
                    if text.trimmingCharacters(in: .whitespaces) == "}" { break }
                    continue
                }
                key = text[..<open.lowerBound].split(separator: "\"").last.map(String.init)
                pieces = fragments(in: String(text[open.upperBound...]))
            } else {
                pieces.append(contentsOf: fragments(in: text))
            }
            if text.contains("ONE_WORD") { usesOneWord = true }
            if text.hasSuffix("),") {
                if let key {
                    systems[key] = pieces.joined() + (usesOneWord ? (oneWord ?? "") : "")
                }
                key = nil
                pieces = []
                usesOneWord = false
            }
        }
        return systems
    }

    /// One `NAME = "..."` constant. The constants this reads have no escapes.
    private static func pythonStringConstant(named name: String, in source: String) -> String? {
        guard let open = source.range(of: "\(name) = \"") else { return nil }
        let rest = source[open.upperBound...]
        guard let close = rest.firstIndex(of: "\"") else { return nil }
        return String(rest[..<close])
    }

    /// Every `"..."` literal on one line, in order. The system prompts hold no
    /// escapes and no embedded quotes, which is what makes this enough.
    private static func fragments(in line: String) -> [String] {
        var out: [String] = []
        var current: String?
        for character in line {
            if character == "\"" {
                if let open = current {
                    out.append(open)
                    current = nil
                } else {
                    current = ""
                }
            } else if current != nil {
                current?.append(character)
            }
        }
        return out
    }

    // MARK: - the resident service

    @Test func theWeightsAreNotMappedUntilTheFirstJudgement() async throws {
        let loads = Counter()
        let engine = SideEngine {
            loads.increment()
            return FakeSideEngineModel()
        }
        let before = await engine.isLoaded
        #expect(before == false)
        #expect(loads.value == 0)

        let answer = try await engine.judge(.durability(key: "a", value: "b"))
        #expect(answer == .yes)
        let after = await engine.isLoaded
        #expect(after)
        #expect(loads.value == 1)
    }

    /// An actor is reentrant at every suspension, so the second `judge` to
    /// arrive while the first is still loading must not map a second copy.
    @Test func theLoaderRunsOnceWhenTwoJudgementsRaceForIt() async throws {
        let loads = Counter()
        let engine = SideEngine {
            loads.increment()
            // Long enough that the second judgement is inside `resident()`
            // while the first is still waiting on the load.
            try? await Task.sleep(for: .milliseconds(60))
            return FakeSideEngineModel()
        }
        async let first = engine.judge(.durability(key: "k", value: "v"))
        async let second = engine.judge(.durability(key: "k", value: "v"))
        let firstAnswer = try await first
        let secondAnswer = try await second

        #expect(firstAnswer == .yes)
        #expect(secondAnswer == .yes)
        #expect(loads.value == 1)
    }

    @Test func everyJudgementStartsFromACleanState() async throws {
        let model = FakeSideEngineModel()
        let engine = SideEngine { model }
        let judgement = SideEngineJudgement.contradiction(
            aKey: "a", aValue: "1",
            bKey: "b", bValue: "2")
        _ = try await engine.judge(judgement)
        _ = try await engine.judge(judgement)

        #expect(model.resets == 2)
        #expect(model.generations == 2)
        let encodes = model.encodes
        #expect(encodes.count == 2)
        #expect(encodes.first?.system == judgement.systemPrompt)
        #expect(encodes.first?.user == judgement.userPrompt)
    }

    @Test func onlyTheTasksOwnWordsComeBackAsAnAnswer() async throws {
        let model = FakeSideEngineModel()
        model.completions = ["CONFLICT", "yes, a fixed attribute", "NONE"]
        let engine = SideEngine { model }

        let supersession = try await engine.judge(
            .supersession(key: "state/inn", earlier: "standing", now: "burned"))
        #expect(supersession == .conflict)

        let durable = try await engine.judge(.durability(key: "k", value: "v"))
        #expect(durable == .yes)

        // The third completion is no answer at all, and the caller is told
        // what was read rather than handed a guess.
        do {
            _ = try await engine.judge(.durability(key: "k", value: "v"))
            Issue.record("a completion outside the answer set must throw")
        } catch let SideEngineError.unparsableAnswer(task, completion) {
            #expect(task == .durability)
            #expect(completion == "NONE")
        }
    }

    @Test func aJudgementDoesNotOverlapAnother() async throws {
        let model = FakeSideEngineModel()
        model.generationDelay = 0.02
        let engine = SideEngine { model }
        // Load first, so the race below is about `generate`, not the loader.
        _ = try await engine.judge(.durability(key: "k", value: "v"))
        model.clearEvents()

        async let first = engine.judge(.durability(key: "k", value: "v"))
        async let second = engine.judge(.durability(key: "k", value: "v"))
        _ = try await first
        _ = try await second

        #expect(model.overlaps == 0)
        #expect(model.events == ["enter", "leave", "enter", "leave"])
    }

    @Test func theWidthFollowsWhetherAClientIsGenerating() async throws {
        let generating = Flag()
        let model = FakeSideEngineModel()
        model.busyThreads = 1
        model.idleThreads = 4
        let engine = SideEngine(isClientGenerating: { generating.value }, load: { model })

        _ = try await engine.judge(.durability(key: "k", value: "v"))
        #expect(model.contention != nil)
        let idle = await engine.currentThreads
        #expect(idle == 4)

        generating.value = true
        let busy = await engine.currentThreads
        #expect(busy == 1)
    }

    @Test func shutdownDropsTheWeightsAndRefusesLaterJudgements() async throws {
        let engine = SideEngine { FakeSideEngineModel() }
        _ = try await engine.judge(.durability(key: "k", value: "v"))
        await engine.shutdown()

        let loaded = await engine.isLoaded
        #expect(loaded == false)
        await #expect(throws: SideEngineError.self) {
            _ = try await engine.judge(.durability(key: "k", value: "v"))
        }
    }

    /// A shutdown that lands while the weights are loading must not be
    /// outrun by the load it was too late to cancel.
    @Test func aShutdownDuringTheLoadLeavesNoModelBehind() async throws {
        let started = Latch()
        let gate = Gate()
        let engine = SideEngine {
            await started.reach()
            await gate.wait()
            return FakeSideEngineModel()
        }
        async let judging = engine.judge(.durability(key: "k", value: "v"))
        await started.wait()

        await engine.shutdown()
        await gate.open()

        // The judgement was waiting on the load when it was shut down, so it
        // must fail rather than resolve with a model nobody asked for.
        do {
            _ = try await judging
            Issue.record("a judgement waiting on a load must not resolve after shutdown")
        } catch SideEngineError.shutDown {
            // expected
        } catch {
            Issue.record("expected shutDown, got \(error)")
        }
        let loaded = await engine.isLoaded
        #expect(loaded == false)
    }
}

// MARK: - fakes

/// The engine behind the protocol, with the four calls recorded.
private final class FakeSideEngineModel: SideEngineModel, @unchecked Sendable {
    private let lock = NSLock()
    private var _resets = 0
    private var _encodes: [(system: String, user: String)] = []
    private var _generations = 0
    private var _events: [String] = []
    private var _insideGenerate = false
    private var _overlaps = 0
    private var completionIndex = 0

    /// One completion per `judge`, in order; the last repeats.
    var completions: [String] = ["YES"]
    /// Held for this long inside `generate`, to widen the overlap window.
    var generationDelay: TimeInterval = 0

    var contention: (@Sendable () -> Bool)?
    var busyThreads = 1
    var idleThreads = 4
    var endOfSequence = 0

    var resets: Int { lock.withLock { _resets } }
    var encodes: [(system: String, user: String)] { lock.withLock { _encodes } }
    var generations: Int { lock.withLock { _generations } }
    var events: [String] { lock.withLock { _events } }
    var overlaps: Int { lock.withLock { _overlaps } }
    var threads: Int { (contention?() ?? false) ? busyThreads : idleThreads }

    func clearEvents() { lock.withLock { _events.removeAll() } }

    func reset() { lock.withLock { _resets += 1 } }

    func encode(system: String, user: String) throws -> [Int] {
        lock.withLock { _encodes.append((system, user)) }
        return [1, 2, 3]
    }

    func decode(_ tokens: [Int]) -> String {
        lock.withLock {
            let completion = completions[min(completionIndex, completions.count - 1)]
            completionIndex += 1
            return completion
        }
    }

    func generate(
        prompt: [Int], maximumTokens: Int,
        stopping: Set<Int>
    ) throws -> [Int] {
        lock.withLock {
            _generations += 1
            _events.append("enter")
            if _insideGenerate { _overlaps += 1 }
            _insideGenerate = true
        }
        if generationDelay > 0 { Thread.sleep(forTimeInterval: generationDelay) }
        lock.withLock {
            _insideGenerate = false
            _events.append("leave")
        }
        return [0]
    }
}

/// A counter a `@Sendable` load closure can bump.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = 0
    var value: Int { lock.withLock { _value } }
    func increment() { lock.withLock { _value += 1 } }
}

/// A boolean the scheduler closure can read.
private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = false
    var value: Bool {
        get { lock.withLock { _value } }
        set { lock.withLock { _value = newValue } }
    }
}

/// One-shot: `wait` returns once `reach` has been called.
private actor Latch {
    private var reached = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func reach() {
        reached = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }

    func wait() async {
        if reached { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

/// A gate the test opens when it wants the held load to finish.
private actor Gate {
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func open() {
        opened = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }

    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}
