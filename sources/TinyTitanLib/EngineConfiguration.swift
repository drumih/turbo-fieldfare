import TinyTitanCore
// The configuration vocabulary an embedder passes in: `EngineConfiguration`
// and the kit's own enum names that keep an engine type out of a public
// signature, plus the sampling row an install declares.
//
// Split out of `Engine.swift` (2026-10-06) under the 500-line-per-file rule
// as pure code motion; nothing widened, because every declaration here was
// already `public` or module-internal.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). A value in this file is a promise: `public` here is deliberate, and
// the `engineValue` bridging each enum carries is the only place the
// runtime's vocabulary is allowed to show.
import Foundation

/// How an engine loads and runs its model.
public struct EngineConfiguration: Sendable {
    /// The context window the session is loaded at.
    public var contextWindow: Int
    /// Bytes the routed-expert cache may use (`nil` = the install's own
    /// profile, which the loader derives from the manifest).
    public var expertCacheBudgetBytes: Int?
    /// KV-cache width.
    public var cachePrecision: CachePrecision
    /// How many generations the session may run at once. The loader clamps it
    /// to what the per-slot stores can hold beside the expert cache.
    public var maxConcurrentGenerations: Int
    /// Prefill chunk in tokens (`nil` = the install's own profile row, then the
    /// loader's family default). A larger chunk amortizes routed-expert reads
    /// over more tokens; a smaller one uses less GPU scratch.
    public var prefillChunkTokens: Int?
    /// Routed-expert cache slots per layer (`nil` = derive from
    /// `expertCacheBudgetBytes` or the install's tuned budget). The two knobs
    /// are alternatives, exactly as the loader's precedence states: an explicit
    /// count wins over a budget target.
    public var expertCacheSlots: Int?
    /// Context scaling. `.yarn` needs a context window from
    /// `RuntimeConfiguration.supportedYaRNContextTokens`.
    public var ropeScaling: RoPEScaling
    /// The reasoning switch the model is loaded at. It changes the rendered
    /// chat template, so it is a load-time decision, not a request one.
    public var thinkingMode: ThinkingMode
    /// Reasoning-effort level for a family whose template defines levels
    /// (`nil` = the template's own default).
    public var reasoningEffort: ReasoningEffort?
    /// Expert read-ahead advice. `nil` keeps `TINYTITAN_RDADVISE_POLICY` (or
    /// the loader's default); a value is an explicit override.
    public var readAhead: ReadAheadAdvice?
    /// Force the logits head instead of letting a pure-greedy generation use
    /// the fused greedy head. The fused head is faster but only exists for a
    /// 4-bit lm_head/attention install whose family has no hyper-connections;
    /// a caller that needs the logits buffer (diagnostics, or matching a run
    /// made against it) sets this. Sampling always forces the logits head.
    public var forceLogitsHead: Bool
    /// How much of the install to re-read before running it.
    ///
    /// There is no streaming-mode knob beside this one on purpose: the runtime's
    /// `ExpertStreamingMode` has a single case whose only parameter is the slot
    /// count, and `expertCacheSlots` and `expertCacheBudgetBytes` above are how
    /// a caller chooses it. A second name for the same decision would be a knob
    /// that lies about being a choice.
    public var integrityPolicy: InstallIntegrity
    /// Where this library's diagnostics go.
    ///
    /// `nil` — the default — writes them to stderr, which is what the server
    /// wants and what every line did before this knob existed. An embedder that
    /// owns its output passes its own sink, and `{ _ in }` silences the library
    /// entirely (which is what the CLI's `--quiet` does).
    ///
    /// The destination is **process-wide**: the orchestrator logs statically
    /// from deep inside the engine, so the most recently created `Engine` sets
    /// it for the process. Two engines with different sinks do not each get
    /// their own; the second one wins.
    public var logSink: (@Sendable (String) -> Void)?

    public init(
        contextWindow: Int = 262_144,
        expertCacheBudgetBytes: Int? = nil,
        cachePrecision: CachePrecision = .eightBit,
        maxConcurrentGenerations: Int = 1,
        prefillChunkTokens: Int? = nil,
        expertCacheSlots: Int? = nil,
        ropeScaling: RoPEScaling = .none,
        thinkingMode: ThinkingMode = .off,
        reasoningEffort: ReasoningEffort? = nil,
        readAhead: ReadAheadAdvice? = nil,
        forceLogitsHead: Bool = true,
        integrityPolicy: InstallIntegrity = .automatic,
        logSink: (@Sendable (String) -> Void)? = nil
    ) {
        self.contextWindow = contextWindow
        self.expertCacheBudgetBytes = expertCacheBudgetBytes
        self.cachePrecision = cachePrecision
        self.maxConcurrentGenerations = maxConcurrentGenerations
        self.prefillChunkTokens = prefillChunkTokens
        self.expertCacheSlots = expertCacheSlots
        self.ropeScaling = ropeScaling
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
        self.readAhead = readAhead
        self.forceLogitsHead = forceLogitsHead
        self.integrityPolicy = integrityPolicy
        self.logSink = logSink
    }
}

/// How much of an install the engine re-reads before it runs it.
///
/// The trade is startup time against trust: a 125B install is a quarter of a
/// terabyte, and hashing all of it is minutes, which is what the installer's
/// receipt exists to avoid.
public enum InstallIntegrity: Sendable, Equatable {
    /// Today's rule, and the default: trust the installer's receipt when the
    /// directory carries one — the check is then against the manifest and the
    /// install's path — and hash the payload when it does not.
    case automatic
    /// Hash the payload regardless of any receipt. Slow on a large install, and
    /// the point of the knob: a caller who does not trust the directory.
    case verifyEveryFile
    /// Trust the receipt's recorded digests. Fast, and strict: an install
    /// without a valid receipt is refused rather than quietly re-hashed, which
    /// would mask a moved directory or a tampered receipt and defeat the point.
    case trustInstallerReceipt

    var engineValue: ModelIntegrityPolicy? {
        switch self {
        case .automatic: nil  // the loader resolves it from the directory
        case .verifyEveryFile: .fullSha256
        case .trustInstallerReceipt: .sizeCheckTrustedReceipt
        }
    }
}

/// KV-cache precision in the kit's own vocabulary, so no engine enum appears
/// in a public signature.
public enum CachePrecision: Sendable, Equatable {
    case fourBit
    case eightBit
    case sixteenBit

    var engineValue: KVCachePrecision {
        switch self {
        case .fourBit: .int4
        case .eightBit: .int8
        case .sixteenBit: .fp16
        }
    }
}

/// Context scaling, in the kit's own vocabulary.
public enum RoPEScaling: Sendable, Equatable {
    case none
    case yarn

    var engineValue: RuntimeRoPEScalingMode {
        switch self {
        case .none: .none
        case .yarn: .yarn
        }
    }
}

/// The reasoning switch a model is loaded at.
public enum ThinkingMode: Sendable, Equatable {
    case off
    case on

    var engineValue: ModelThinkingMode { self == .on ? .on : .off }
}

/// A reasoning-effort level, for the families whose chat template defines one.
public enum ReasoningEffort: String, Sendable, Equatable, CaseIterable {
    case low
    case medium
    case xhigh

    var engineValue: ModelReasoningEffort {
        switch self {
        case .low: .low
        case .medium: .medium
        case .xhigh: .xhigh
        }
    }
}

/// Expert read-ahead advice, in the kit's own vocabulary.
public enum ReadAheadAdvice: String, Sendable, Equatable, CaseIterable {
    case off
    case `default`
    case bounded
    case adaptive

    var engineValue: RDAdvicePolicyMode {
        switch self {
        case .off: .off
        case .default: .default
        case .bounded: .bounded
        case .adaptive: .adaptive
        }
    }
}

/// One model's sampling row, in the kit's own vocabulary.
///
/// The facade has no engine enum in a public signature, so a caller that wants
/// to fill what it did not name asks the engine for this instead of reaching
/// into `GenerationDefaults` or `ModelProfile`.
public struct SamplingDefaults: Sendable, Equatable {
    public var temperature: Double
    public var topK: Int
    public var topP: Double
    /// OpenAI presence penalty: subtracted once per distinct id already in the
    /// history. Zero is neutral.
    public var presencePenalty: Double

    public init(
        temperature: Double,
        topK: Int,
        topP: Double,
        presencePenalty: Double = 0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.presencePenalty = presencePenalty
    }

    package init(_ sampling: SamplingProfile) {
        self.init(
            temperature: Double(sampling.temperature),
            topK: sampling.topK,
            topP: Double(sampling.topP),
            presencePenalty: Double(sampling.presencePenalty))
    }

    /// The row an install declares, read from `manifest.json` alone — no
    /// weights, no Metal, no load.
    ///
    /// A front end that has to fill `GenerationOptions` before the engine
    /// exists (the CLI, whose head selection depends on whether its plan is
    /// pure greedy) asks this; the value equals `Engine.samplingDefaults`
    /// afterwards. An unreadable install falls back to the house row, which is
    /// what the pre-facade CLI did.
    public static func forInstall(
        at directory: URL,
        thinkingMode: ThinkingMode = .off,
        factory: EngineFactory
    ) -> SamplingDefaults {
        let metadata = try? factory.metadata(directory)
        return SamplingDefaults(
            (metadata?.sampling ?? .house).effective(thinking: thinkingMode == .on))
    }
}
