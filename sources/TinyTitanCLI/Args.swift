import TinyTitanCore
import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanVariants

public enum PrefillChunkChoice: Equatable, Sendable {
    case fixed(Int)
    case auto
}

public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    /// Whether the caller named these, as opposed to inheriting the house
    /// values. The family's own defaults are applied in `Run` once the
    /// manifest is read; an explicit flag always wins over them.
    public var temperatureWasSet: Bool = false
    public var topKWasSet: Bool = false
    public var topPWasSet: Bool = false
    public var repetitionPenalty: Float
    /// OpenAI presence penalty: subtracted once per distinct id already in the
    /// history. Zero unless the family's row carries one (Qwen3.8 instruct 1.5).
    public var presencePenalty: Float
    public var presencePenaltyWasSet: Bool = false
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var concise: Bool
    public var thinkingMode: ModelThinkingMode
    public var reasoningEffort: ModelReasoningEffort?
    /// Routed-expert slots, or nil to derive from the family's measured
    /// tuning and the payload's expert stride. A flat default cannot be
    /// right for both families: 8 GiB is 128 slots at the 35B stride and 96
    /// at Qwen3.8-Flash-Next's.
    public var expertCacheSlots: Int?
    /// A target for the whole process, or nil to leave the install's profile in
    /// charge. The loader turns it into a slot count by subtracting the
    /// resident weights and the runtime reserve, and prints the `ram target`
    /// line; an explicit `expertCacheSlots` wins over it.
    public var ramBudgetBytes: Int?
    public var rdadvise: String
    public var prefillChunk: PrefillChunkChoice?
    /// An explicitly requested engine. Nil keeps the automatic routing: the
    /// 7.0 variant where one ships, the 6.x factory otherwise.
    public var engine: EngineChoice?
    public var kvCachePrecision: KVCachePrecision
    public var ropeScalingMode: RuntimeRoPEScalingMode
    /// A measured calibration profile to apply where no explicit flag wins.
    public var calibrationProfile: String?
    /// One line naming what the profile applied, for the CLI's stderr.
    public var calibrationLogLine: String?

    public init(
        model: String,
        prompt: String? = nil,
        messagesFile: String? = nil,
        maxNew: Int = 1_024,
        maxContext: Int = 4096,
        temperature: Float = GenerationDefaults.temperature,
        topK: Int? = GenerationDefaults.topK,
        topP: Float? = GenerationDefaults.topP,
        temperatureWasSet: Bool = false,
        topKWasSet: Bool = false,
        topPWasSet: Bool = false,
        repetitionPenalty: Float = 1.0,
        presencePenalty: Float = GenerationDefaults.presencePenalty,
        presencePenaltyWasSet: Bool = false,
        seed: UInt64? = nil,
        stops: [String] = [],
        quiet: Bool = false,
        concise: Bool = false,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil,
        expertCacheSlots: Int? = nil,
        ramBudgetBytes: Int? = nil,
        rdadvise: String = "default",
        prefillChunk: PrefillChunkChoice? = nil,
        engine: EngineChoice? = nil,
        kvCachePrecision: KVCachePrecision = .int8,
        ropeScalingMode: RuntimeRoPEScalingMode = .none,
        calibrationProfile: String? = nil,
        calibrationLogLine: String? = nil
    ) {
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.temperatureWasSet = temperatureWasSet
        self.topKWasSet = topKWasSet
        self.topPWasSet = topPWasSet
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.presencePenaltyWasSet = presencePenaltyWasSet
        self.expertCacheSlots = expertCacheSlots
        self.ramBudgetBytes = ramBudgetBytes
        self.rdadvise = rdadvise
        self.prefillChunk = prefillChunk
        self.engine = engine
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
        self.calibrationProfile = calibrationProfile
        self.calibrationLogLine = calibrationLogLine
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.concise = concise
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing: return "one of --prompt or --messages-file is required"
        }
    }
}

extension Args {
    /// lint:allow-long same shape as ServerArguments.parse: a flag table
    /// where the exhaustive switch is the point.
    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var maxNew = 1_024
        // Matches the server and the published benchmark settings. The
        // allocation is lazy -- a server started at 262,144 loads with the
        // same resident footprint as one started at 4,096 -- so defaulting
        // low only surprised people whose prompt was longer than 4k.
        var maxContext = RuntimeConfiguration.nativeMaximumContextTokens
        var maxContextWasSet = false
        var temperatureWasSet = false
        var topKWasSet = false
        var topPWasSet = false
        var temperature: Float = GenerationDefaults.temperature
        var topK: Int? = GenerationDefaults.topK
        var topP: Float? = GenerationDefaults.topP
        var repetitionPenalty: Float = 1.0
        var presencePenalty: Float = GenerationDefaults.presencePenalty
        var presencePenaltyWasSet = false
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        var concise = false
        var thinkingMode: ModelThinkingMode = .off
        var reasoningEffort: ModelReasoningEffort?
        var expertCacheSlots: Int?
        var ramBudgetBytes: Int?
        var rdadvise = "default"
        var prefillChunk: PrefillChunkChoice?
        var engine: EngineChoice?
        var kvCachePrecision: KVCachePrecision = .int8
        var ropeScalingMode: RuntimeRoPEScalingMode = .none
        var calibrationProfile: String?
        var calibrationLogLine: String?

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--concise":
                concise = true
                index += 1
            case "--thinking":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = ModelThinkingMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                thinkingMode = parsed
            case "--reasoning-effort":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = ModelReasoningEffort(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                reasoningEffort = parsed
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                    (1...RuntimeConfiguration.maximumContextTokens).contains(parsed)
                else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                    (1...RuntimeConfiguration.maximumContextTokens).contains(parsed)
                else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
                maxContextWasSet = true
            case "--rope-scaling":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeRoPEScalingMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                ropeScalingMode = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0, parsed <= 2 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperatureWasSet = true
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
                topKWasSet = true
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
                topPWasSet = true
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                // At least 1, matching AppGenerationRequest: a penalty below
                // 1 multiplies the repeated logit instead of dividing it, so
                // it rewards repetition -- the opposite of the flag. This CLI
                // is the scripted verification path, so it must not accept a
                // sampling configuration every other front end rejects.
                guard let parsed = Float(value), parsed.isFinite, parsed >= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--presence-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                // OpenAI's own bounds. The sampler subtracts the value once per
                // distinct id already in the history; Qwen3.8's instruct row is
                // the one published row that uses a non-zero value (1.5).
                guard let parsed = Float(value), parsed.isFinite,
                    parsed >= -2, parsed <= 2
                else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                presencePenalty = parsed
                presencePenaltyWasSet = true
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--expert-cache-slots":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                    RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed)
                else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCacheSlots = parsed
            case "--ram-budget":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeConfiguration.parseBudgetBytes(value),
                    parsed >= RuntimeConfiguration.minimumProcessTargetBytes
                else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                ramBudgetBytes = parsed
            case "--rdadvise":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["off", "default", "bounded", "adaptive"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                rdadvise = value
            case "--calibration-profile":
                calibrationProfile = try takeValue(argv, &index, flag: flag)
            case "--prefill-chunk":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    prefillChunk = .auto
                } else if let parsed = Int(value),
                    RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed)
                {
                    prefillChunk = .fixed(parsed)
                } else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--engine":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = EngineChoice(rawValue: value.lowercased()) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                engine = parsed
            case "--kv-bits":
                let value = try takeValue(argv, &index, flag: flag)
                guard let bits = Int(value),
                    let parsed = KVCachePrecision(rawValue: bits)
                else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                kvCachePrecision = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if prompt == nil && messagesFile == nil { throw ArgsError.modeMissing }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        if ropeScalingMode == .yarn {
            if !maxContextWasSet {
                maxContext = RuntimeConfiguration.defaultYaRNContextTokens
            }
            guard RuntimeConfiguration.supportedYaRNContextTokens.contains(maxContext) else {
                throw ArgsError.invalidValue(flag: "--max-context", value: String(maxContext))
            }
        } else if maxContext > RuntimeConfiguration.nativeMaximumContextTokens {
            throw ArgsError.invalidValue(flag: "--max-context", value: String(maxContext))
        }
        // Family support is checked in Run against the installed manifest;
        // the effort/thinking combination is a pure argument error here.
        if let effort = reasoningEffort, !thinkingMode.isEnabled {
            throw ArgsError.invalidValue(
                flag: "--reasoning-effort",
                value: "\(effort.rawValue) requires --thinking on")
        }
        if let path = calibrationProfile {
            do {
                let profile = try CalibrationProfile.load(
                    fileAt: URL(fileURLWithPath: path))
                let explicitChunk: Int? = {
                    if case .fixed(let chunk)? = prefillChunk { return chunk }
                    return nil
                }()
                let identity = try ManifestReader.peekIdentity(
                    directoryURL: URL(fileURLWithPath: model))
                let application = try profile.application(
                    for: CalibrationProfile.currentMachine(),
                    contextTokens: maxContext,
                    modelID: identity.modelID,
                    family: identity.family.rawValue,
                    weightBits: identity.weightBits,
                    explicitPrefillChunkTokens: explicitChunk,
                    explicitExpertCacheSlots: expertCacheSlots)
                if prefillChunk == nil, let chunk = application.prefillChunkTokens {
                    prefillChunk = .fixed(chunk)
                }
                if expertCacheSlots == nil {
                    expertCacheSlots = application.expertCacheSlots
                }
                calibrationLogLine = application.logLine
            } catch {
                throw ArgsError.invalidValue(
                    flag: "--calibration-profile", value: "\(error)")
            }
        }
        // The target stays a target here: the loader owns the residency
        // arithmetic (resident weights, runtime reserve, the slot ladder) and
        // the `ram target` line that records what it chose. Resolving it in the
        // parser would duplicate that derivation and make the loader print a
        // cache budget for a flag that named a process ceiling.
        return Args(
            model: model,
            prompt: prompt,
            messagesFile: messagesFile,
            maxNew: maxNew,
            maxContext: maxContext,
            temperature: temperature,
            topK: topK,
            topP: topP,
            temperatureWasSet: temperatureWasSet,
            topKWasSet: topKWasSet,
            topPWasSet: topPWasSet,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            presencePenaltyWasSet: presencePenaltyWasSet,
            seed: seed,
            stops: stops,
            quiet: quiet,
            concise: concise,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort,
            expertCacheSlots: expertCacheSlots,
            ramBudgetBytes: ramBudgetBytes,
            rdadvise: rdadvise,
            prefillChunk: prefillChunk,
            engine: engine,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode,
            calibrationProfile: calibrationProfile,
            calibrationLogLine: calibrationLogLine)
    }

    private static func takeValue(
        _ argv: [String],
        _ index: inout Int,
        flag: String
    ) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
