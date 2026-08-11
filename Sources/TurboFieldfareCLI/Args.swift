import TurboFieldfare

public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var forceJSON: Bool
    public var quiet: Bool
    public var expertCacheSlots: Int
    public var expertCachePolicy: RuntimeExpertCachePolicy
    public var prefillPolicy: RuntimePrefillPolicy
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: RDAdvicePolicyMode

    public init(model: String,
                prompt: String? = nil,
                messagesFile: String? = nil,
                maxNew: Int = 1_024,
                maxContext: Int = 4096,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                forceJSON: Bool = false,
                quiet: Bool = false,
                expertCacheSlots: Int = RuntimeConfiguration.production.expertCacheSlots,
                expertCachePolicy: RuntimeExpertCachePolicy = RuntimeConfiguration.production.expertCachePolicy,
                prefillPolicy: RuntimePrefillPolicy = RuntimeConfiguration.production.prefillPolicy,
                prefillChunkTokens: Int = RuntimeConfiguration.production.prefillChunkTokens,
                rdadvisePolicy: RDAdvicePolicyMode = RuntimeConfiguration.production.rdadvisePolicy) {
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stops = stops
        self.forceJSON = forceJSON
        self.quiet = quiet
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillPolicy = prefillPolicy
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
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
    public static let usage = """
    TurboFieldfareCLI — Gemma 4 26B-A4B text generation

    usage: TurboFieldfareCLI --model <dir> (--prompt <string> | --messages-file <path>) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.
      --prompt <string>         Raw-completion prompt.
      --messages-file <path>    JSON chat messages with role and content fields.

    options:
      --max-new <int>            Generated-token limit (default 1024).
      --max-context <int>        Context limit in tokens (default 4096).
      --temperature <float>      Sampling temperature (default 0.2; 0 = greedy).
      --top-k <int>              Top-k truncation, 1...256 (default 64; 0 = off).
      --top-p <float>            Nucleus truncation (default 0.95).
      --repetition-penalty <f>   Repetition penalty (default 1.0).
      --seed <uint64>            Deterministic sampling seed (default off).
      --stop <string>            Stop substring (repeatable).
      --force-json               Constrain decoding to grammar-valid JSON;
                                 EOS is forced when the root value closes.
                                 Incompatible with --stop
      --quiet                    Suppress the timing footer.
      --expert-cache-slots <n>   Expert-cache slots: 8, 16, 24, or 32 (default 16).
      --expert-cache-policy <s>  Expert-cache policy: lfu or lru (default lfu).
      --prefill on|off           Enable or disable chunked prompt prefill (default on).
                                 Chunked prefill requires 16 or more cache slots.
      --prefill-chunk-tokens <n> Prefill chunk size: 32, 64, or 128 (default 128).
      --rdadvise <s>             Read-advice policy: off, default, bounded, or adaptive (default off).
      --help                     Show this message.
    """

    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool) throws -> RuntimeConfiguration {
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots", value: "\(expertCacheSlots)")
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ArgsError.invalidValue(
                flag: "--prefill-chunk-tokens", value: "\(prefillChunkTokens)")
        }
        let chunkedPrefillSupported = expertCacheSlots >=
            RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        guard prefillPolicy == .off || chunkedPrefillSupported else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots",
                value: "\(expertCacheSlots) requires --prefill off")
        }
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            rdadvisePolicy: rdadvisePolicy,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var maxNew = 1_024
        var maxContext = 4096
        var temperature: Float = 0.2
        var topK: Int? = 64
        var topP: Float? = 0.95
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var forceJSON = false
        var quiet = false
        let runtimeDefaults = RuntimeConfiguration.production
        var expertCacheSlots = runtimeDefaults.expertCacheSlots
        var expertCachePolicy = runtimeDefaults.expertCachePolicy
        var prefillPolicy = runtimeDefaults.prefillPolicy
        var prefillChunkTokens = runtimeDefaults.prefillChunkTokens
        var rdadvisePolicy = runtimeDefaults.rdadvisePolicy

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            case "--force-json":
                forceJSON = true
                index += 1
            case "--expert-cache-slots":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCachePolicy = parsed
            case "--prefill":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--prefill-chunk-tokens":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                prefillChunkTokens = parsed
            case "--rdadvise":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RDAdvicePolicyMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                rdadvisePolicy = parsed
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if prompt == nil && messagesFile == nil { throw ArgsError.modeMissing }
        if forceJSON && !stops.isEmpty {
            throw ArgsError.mutuallyExclusive("--force-json", "--stop")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        let arguments = Args(model: model,
                             prompt: prompt,
                             messagesFile: messagesFile,
                             maxNew: maxNew,
                             maxContext: maxContext,
                             temperature: temperature,
                             topK: topK,
                             topP: topP,
                             repetitionPenalty: repetitionPenalty,
                             seed: seed,
                             stops: stops,
                             forceJSON: forceJSON,
                             quiet: quiet,
                             expertCacheSlots: expertCacheSlots,
                             expertCachePolicy: expertCachePolicy,
                             prefillPolicy: prefillPolicy,
                             prefillChunkTokens: prefillChunkTokens,
                             rdadvisePolicy: rdadvisePolicy)
        _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        return arguments
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
