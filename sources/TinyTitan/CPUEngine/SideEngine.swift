import TinyTitanCore
import Foundation

/// The seven single-decision tasks the side-engine is asked, and the prompts
/// that ask them.
///
/// `docs/side-engine-tasks.md` is the argument: a small model fails at
/// composition — "return the parts of this fact the person stated" was 0 of 12
/// — and succeeds at one decision at a time (T1, "did the person state this
/// clause?", measured 92%). Every task here is therefore **one question with a
/// closed answer set**: YES/NO, or UPDATE/CONFLICT. The prompts are the
/// measured ones from `benchmark/side_engine_tasks.py`; this file is the
/// shipped copy of them, and `SideEngineTests` pins the wording so the two
/// cannot drift apart silently.
///
/// **Which model, and for which tasks.** The engine is model-agnostic — any
/// dense Qwen3.5 snapshot fits — but the tasks are not decided equally at every
/// size. Measured over the same cases (`docs/side-engine-tasks.md`):
/// contradiction is good from the smallest up; duplication and retrieval need a
/// 4B; durability needs a 4B and is *worse* on the 9B; the reply check needs a
/// 9B; and supersession needs the stored rule supplied with the question, where
/// both a 4B and a 9B are exact. The 2B is not the verification instrument.
public enum SideEngineTask: String, Sendable, CaseIterable {
    case clauseAttribution = "T1"
    case durability = "T2"
    case contradiction = "T3"
    case supersession = "T4"
    case duplication = "T5"
    case replyCheck = "T6"
    case retrieval = "T7"

    /// The only words this task may answer with. Anything else — including a
    /// fluent paragraph that contains the right word — is a failure, because a
    /// small model's failure mode is the lazy branch, not silence.
    public var answers: [SideEngineAnswer] {
        switch self {
        case .supersession: return [.update, .conflict]
        default: return [.yes, .no]
        }
    }
}

/// One legal answer.
public enum SideEngineAnswer: String, Sendable, CaseIterable, Equatable {
    case yes = "YES"
    case no = "NO"
    case update = "UPDATE"
    case conflict = "CONFLICT"

    /// The first word of a completion, if it is one of `allowed`.
    ///
    /// The first *word* rather than the whole string: the tasks are asked for
    /// one word and a model that adds a full stop or a short flourish after it
    /// has still decided. A completion whose first word is not in the set is
    /// `nil` — the caller sees "no decision", never a guess.
    public init?(firstWordOf completion: String, allowed: [SideEngineAnswer]) {
        var word = ""
        for scalar in completion.unicodeScalars {
            if CharacterSet.letters.contains(scalar) {
                word.unicodeScalars.append(scalar)
            } else if !word.isEmpty {
                break
            }
        }
        guard let match = SideEngineAnswer(rawValue: word.uppercased()),
            allowed.contains(match)
        else { return nil }
        self = match
    }
}

/// One question, with everything the prompt needs.
public enum SideEngineJudgement: Sendable, Equatable {
    case clauseAttribution(personWrote: String, address: String, clause: String)
    case durability(key: String, value: String)
    case contradiction(aKey: String, aValue: String, bKey: String, bValue: String)
    /// The rule is data, not part of the question: an eye colour changing is
    /// only a conflict if something already says it never may, and no model can
    /// know that from the two statements. A caller with a stored rule supplies
    /// it, and the prompt shows it as a `RULE:` line; without one the CONFLICT
    /// half is a guess.
    case supersession(
        key: String, earlier: String, now: String,
        rule: String? = nil)
    case duplication(aKey: String, aValue: String, bKey: String, bValue: String)
    case replyCheck(key: String, value: String, reply: String)
    case retrieval(question: String, key: String, value: String)

    /// Every prompt ends with this: the prohibition is what stops a 2B from
    /// explaining, which reads as an answer to a client and as noise to a
    /// parser.
    static let oneWord = "Do not explain. Do not quote. Answer with one word and nothing else."

    public var task: SideEngineTask {
        switch self {
        case .clauseAttribution: return .clauseAttribution
        case .durability: return .durability
        case .contradiction: return .contradiction
        case .supersession: return .supersession
        case .duplication: return .duplication
        case .replyCheck: return .replyCheck
        case .retrieval: return .retrieval
        }
    }

    public var systemPrompt: String {
        switch task {
        case .clauseAttribution:
            return "You decide whether one statement came from the person or not. "
                + "Answer with exactly one word: YES or NO. YES means the person "
                + "wrote it or clearly implied it. NO means it does not appear in "
                + "what they wrote, however true it might be. " + Self.oneWord
        case .durability:
            return "You decide whether one fact is worth keeping after this session "
                + "ends. Answer with exactly one word: YES or NO. YES only for a "
                + "standing fact a later session needs: a decision and its reason, a "
                + "fixed attribute, a rule, a constraint, a preference, or the state "
                + "of the work right now. NO for anything that reports what happened "
                + "in this session instead of stating how things are: story text, "
                + "narration, chapter content, a summary of what was written, a "
                + "remark about the writing, a plan to write, an offer, or an "
                + "acknowledgement. A fact that would only make sense to someone who "
                + "read this session is NO. " + Self.oneWord
        case .contradiction:
            return "You decide whether two statements disagree. Answer with exactly "
                + "one word: YES or NO. YES means both cannot be true at once. NO "
                + "means they can both be true, including when they are about "
                + "different things, or when one simply says more than the other. "
                + "Different wording for the same thing is NO. " + Self.oneWord
        case .supersession:
            return "Something has changed about one fact. You decide which kind of "
                + "change it is. Answer with exactly one word: UPDATE or CONFLICT. "
                + "UPDATE: the earlier value was true before and the newer one is "
                + "true now, so both can be true in turn -- the world moved on. "
                + "CONFLICT: a rule says this value never changes, or the two are "
                + "about the same moment, so both cannot be true and one is wrong. A "
                + "change of state -- a place burned, a person found, a service "
                + "stopped -- is UPDATE. A change to something a rule fixes -- an "
                + "eye colour under a rule that it never changes -- is CONFLICT. "
                + Self.oneWord
        case .duplication:
            return "You decide whether two facts say the same thing. Answer with "
                + "exactly one word: YES or NO. YES means a reader learns nothing "
                + "from the second that the first did not already tell them. NO "
                + "means the second adds something, or is about something else. "
                + Self.oneWord
        case .replyCheck:
            return "You check one reply against one thing that is known. Answer with "
                + "exactly one word: YES or NO. YES means the reply says something "
                + "that cannot be true if the known fact is true. NO means it "
                + "agrees, or does not touch on it at all. Silence is not a "
                + "contradiction. " + Self.oneWord
        case .retrieval:
            return "You decide whether one stored fact could answer one question. "
                + "Answer with exactly one word: YES or NO. YES means the fact "
                + "contains the answer, or part of it. NO means it does not, even "
                + "if it is about the same subject. " + Self.oneWord
        }
    }

    /// The whole fact goes in, key included: the key carries the claim as often
    /// as the value does, and withholding it halved accuracy when measured.
    public var userPrompt: String {
        switch self {
        case .clauseAttribution(let personWrote, let address, let clause):
            return "WHAT THE PERSON WROTE:\n\(personWrote)\n\n"
                + "STATEMENT: \(address) = \(clause)\n"
                + "Did the person state this?"
        case .durability(let key, let value):
            return "FACT: \(key) = \(value)\nKeep it?"
        case .contradiction(let aKey, let aValue, let bKey, let bValue):
            return "A: \(aKey) = \(aValue)\nB: \(bKey) = \(bValue)\n"
                + "Do A and B disagree?"
        case .supersession(let key, let earlier, let now, let rule):
            let prefix = rule.map { "RULE: \($0)\n" } ?? ""
            return prefix + "EARLIER: \(key) = \(earlier)\nNOW: \(key) = \(now)\n"
                + "Which is it?"
        case .duplication(let aKey, let aValue, let bKey, let bValue):
            return "A: \(aKey) = \(aValue)\nB: \(bKey) = \(bValue)\nSame fact?"
        case .replyCheck(let key, let value, let reply):
            return "KNOWN: \(key) = \(value)\nREPLY: \(reply)\n"
                + "Does the reply contradict what is known?"
        case .retrieval(let question, let key, let value):
            return "QUESTION: \(question)\nFACT: \(key) = \(value)\n"
                + "Could this fact answer it?"
        }
    }
}

/// What the resident service drives.
///
/// A protocol rather than `CPUQwen35` directly so the service — the scheduler,
/// the serialisation, the answer parsing — is tested without a model, which the
/// test rules require and which a 1.9 GB load would otherwise make impossible.
public protocol SideEngineModel: Sendable {
    /// Forget the KV state between jobs. The resident service exists so that
    /// the *weights* stay mapped while each job starts clean.
    func reset()
    /// Render this model's own chat template and tokenize it. The template is
    /// what makes an instruction-tuned model answer rather than continue.
    func encode(system: String, user: String) throws -> [Int]
    func decode(_ tokens: [Int]) -> String
    func generate(prompt: [Int], maximumTokens: Int, stopping: Set<Int>) throws -> [Int]
    var endOfSequence: Int { get }
    /// Whether someone is waiting on the main engine, re-read before every
    /// token; `busyThreads`/`idleThreads` are what it chooses between.
    var contention: (@Sendable () -> Bool)? { get set }
    var busyThreads: Int { get set }
    var idleThreads: Int { get set }
    /// The width the next token will use, for diagnostics.
    var threads: Int { get }
}

/// Errors a caller has to be able to tell apart.
public enum SideEngineError: Error, CustomStringConvertible {
    case missingTokenizer(String)
    case unparsableAnswer(task: SideEngineTask, completion: String)
    case shutDown

    public var description: String {
        switch self {
        case .missingTokenizer(let path):
            "no tokenizer in \(path)"
        case .unparsableAnswer(let task, let completion):
            "\(task.rawValue) answered \(completion.debugDescription), which is not one of "
                + task.answers.map(\.rawValue).joined(separator: "/")
        case .shutDown:
            "the side-engine has been shut down"
        }
    }
}

/// The resident helper: one side-engine model, mapped once, answering one
/// decision at a time.
///
/// **Residency.** The loader runs on the first `judge` and never again, so a
/// process that never needs a memory judgement never maps the weights.
///
/// **One at a time.** The actor serialises everything, which the engine
/// requires (one KV state, one position) and which the caller wants: a
/// judgement that overlapped another would interleave two prompts into one
/// state.
///
/// **The scheduler.** `isClientGenerating` is the read of "is someone waiting
/// on the GPU right now" — the server's `ServerCoordinator.generating`, whose
/// own comment says the side-engine reads it. The engine re-reads it before
/// every token and takes `busyThreads` (one: costs a concurrent 35B generation
/// 3%, measured) or `idleThreads` (the performance cores: four costs 31% while
/// someone waits, and is the whole win when nobody is).
public actor SideEngine {
    public typealias Load = @Sendable () async throws -> SideEngineModel

    private let load: Load
    private let isClientGenerating: (@Sendable () -> Bool)?
    private var model: SideEngineModel?
    /// The first load, held as a task rather than as a flag. `judge` suspends
    /// while the model loads, and an actor is reentrant at every suspension,
    /// so without this a second `judge` arriving mid-load would see `model ==
    /// nil` and map a second copy of 1.9 GB of weights.
    private var loading: Task<SideEngineModel, Error>?
    private var stopped = false

    /// - Parameters:
    ///   - isClientGenerating: read before every token; nil leaves the width
    ///     alone, which is what a benchmark wants.
    ///   - load: constructs the model on first use. Async because a tokenizer
    ///     loads asynchronously.
    public init(
        isClientGenerating: (@Sendable () -> Bool)? = nil,
        load: @escaping Load
    ) {
        self.isClientGenerating = isClientGenerating
        self.load = load
    }

    public var isLoaded: Bool { model != nil }

    /// The width the resident model will use for its next token.
    public var currentThreads: Int? { model?.threads }

    public func judge(
        _ judgement: SideEngineJudgement,
        maximumTokens: Int = 8
    ) async throws -> SideEngineAnswer {
        guard !stopped else { throw SideEngineError.shutDown }
        let model = try await resident()
        model.reset()
        let prompt = try model.encode(
            system: judgement.systemPrompt,
            user: judgement.userPrompt)
        let produced = try model.generate(
            prompt: prompt,
            maximumTokens: maximumTokens,
            stopping: [model.endOfSequence])
        let completion = model.decode(produced)
        guard
            let answer = SideEngineAnswer(
                firstWordOf: completion,
                allowed: judgement.task.answers)
        else {
            throw SideEngineError.unparsableAnswer(
                task: judgement.task,
                completion: completion)
        }
        return answer
    }

    public func shutdown() {
        loading?.cancel()
        loading = nil
        model = nil
        stopped = true
    }

    private func resident() async throws -> SideEngineModel {
        if let model { return model }
        let task: Task<SideEngineModel, Error>
        if let loading {
            task = loading
        } else {
            let work = load
            task = Task { try await work() }
            loading = task
        }
        do {
            var model = try await task.value
            loading = nil
            // A `shutdown` that arrived while the weights were loading keeps
            // its meaning: the model is dropped instead of installed.
            guard !stopped else { throw SideEngineError.shutDown }
            model.contention = isClientGenerating
            self.model = model
            return model
        } catch {
            loading = nil
            throw error
        }
    }
}

/// `CPUQwen35` plus the tokenizer from the same directory.
///
/// This wrapper adds no model assumption to the engine, so the snapshot
/// directory is the whole choice of model. For the shipped prompts the 4B is
/// the verified floor and the 9B decides more (`docs/side-engine-tasks.md`);
/// the 2B does not.
///
/// unchecked-invariant: every member is touched only from the `SideEngine`
/// actor that owns this wrapper, which serialises `reset`, `encode`, and
/// `generate`. The underlying engine is a class with one KV state, so that
/// serialisation is the correctness argument, not a convenience.
public final class CPUQwen35SideEngineModel: SideEngineModel, @unchecked Sendable {
    private let engine: CPUQwen35
    private let tokenizer: GFTokenizer

    public init(snapshotDirectory: URL, threads: Int? = nil) async throws {
        // A shipped `.ssdai` install and a flat HF/affine snapshot are both
        // reached, the same rule the batch command uses.
        let snapshot: AffineSnapshot
        if FileManager.default.fileExists(
            atPath: snapshotDirectory.appendingPathComponent("manifest.json").path)
        {
            snapshot = try AffineSnapshot(ssdai: snapshotDirectory)
        } else {
            snapshot = try AffineSnapshot(directory: snapshotDirectory)
        }
        guard
            let folder = GFTokenizer.resolvedTokenizerFolder(
                forModelDirectory: snapshotDirectory)
        else {
            throw SideEngineError.missingTokenizer(snapshotDirectory.path)
        }
        self.engine = try CPUQwen35(snapshot: snapshot, threads: threads)
        self.tokenizer = try await GFTokenizer.load(from: folder)
    }

    public func reset() { engine.reset() }

    public func encode(system: String, user: String) throws -> [Int] {
        let rendered = try tokenizer.applyChatTemplate([
            GFTokenizer.Message(role: .system, content: system),
            GFTokenizer.Message(role: .user, content: user),
        ])
        return tokenizer.encode(rendered, addBOS: false).map(Int.init)
    }

    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens.map(Int32.init))
    }

    public func generate(
        prompt: [Int], maximumTokens: Int,
        stopping: Set<Int>
    ) throws -> [Int] {
        try engine.generate(
            prompt: prompt, maximumTokens: maximumTokens,
            stopping: stopping)
    }

    public var endOfSequence: Int { Int(tokenizer.eosID) }

    public var contention: (@Sendable () -> Bool)? {
        get { engine.contention }
        set { engine.contention = newValue }
    }

    public var busyThreads: Int {
        get { engine.busyThreads }
        set { engine.busyThreads = newValue }
    }

    public var idleThreads: Int {
        get { engine.idleThreads }
        set { engine.idleThreads = newValue }
    }

    public var threads: Int { engine.threads }
}
