public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var maxNew: Int
    /// Tracks CLI intent so K3 can use its own operational default without
    /// changing Gemma's established public default.
    public var maxNewExplicit: Bool
    public var maxContext: Int
    public var temperature: Float
    public var temperatureExplicit: Bool
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    // K3 (.gturbo v2) options; ignored on the Gemma (v1) path.
    public var reasoningEffort: String?
    public var noThinking: Bool
    public var prefill: String
    public var prefillChunk: Int
    public var expertPredict: Bool
    /// `auto` or a fixed bounded worker count (`1...32`).
    public var expertIOWorkers: String
    /// Page-aligned subreads per expert. The K3 default is one whole-expert read.
    public var expertIOSplits: Int
    /// `auto`, `buffered`, or Darwin `F_NOCACHE` expert reads.
    public var expertIOCache: String
    /// Full hashes by default; trusted-install is an explicit startup tradeoff.
    public var modelVerification: String
    /// Validate a real first-token activation before K3 generation.
    public var k3ActivationDiagnostics: Bool
    public var verbose: Bool

    public init(model: String,
                prompt: String? = nil,
                messagesFile: String? = nil,
                maxNew: Int = 1_024,
                maxNewExplicit: Bool = false,
                maxContext: Int = 4096,
                temperature: Float = 0.2,
                temperatureExplicit: Bool = false,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                reasoningEffort: String? = nil,
                noThinking: Bool = false,
                prefill: String = "chunked",
                prefillChunk: Int = 32,
                expertPredict: Bool = true,
                expertIOWorkers: String = "auto",
                expertIOSplits: Int = 1,
                expertIOCache: String = "auto",
                modelVerification: String = "full-sha256",
                k3ActivationDiagnostics: Bool = false,
                verbose: Bool = false) {
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.maxNew = maxNew
        self.maxNewExplicit = maxNewExplicit
        self.maxContext = maxContext
        self.temperature = temperature
        self.temperatureExplicit = temperatureExplicit
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.reasoningEffort = reasoningEffort
        self.noThinking = noThinking
        self.prefill = prefill
        self.prefillChunk = prefillChunk
        self.expertPredict = expertPredict
        self.expertIOWorkers = expertIOWorkers
        self.expertIOSplits = expertIOSplits
        self.expertIOCache = expertIOCache
        self.modelVerification = modelVerification
        self.k3ActivationDiagnostics = k3ActivationDiagnostics
        self.verbose = verbose
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
      --max-new <int>           Generated-token limit (default 1024).
      --max-context <int>       Context limit in tokens (default 4096).
      --temperature <float>     Sampling temperature (default 0.2; 0 = greedy).
      --top-k <int>             Top-k truncation, 1...256 (default 64; 0 = off).
      --top-p <float>           Nucleus truncation (default 0.95).
      --repetition-penalty <f>  Repetition penalty (default 1.0).
      --seed <uint64>           Deterministic sampling seed (default off).
      --stop <string>           Stop substring (repeatable).
      --quiet                   Suppress the timing footer.
      --help                    Show this message.

    K3 (.gturbo v2 bundle) options:
      --reasoning-effort <e>    Thinking effort: low, high, or max (default off).
      --no-thinking             Drop the assistant think channel.
      --prefill <mode>          Prompt prefill: serial or chunked (default chunked).
      --prefill-chunk <int>     Chunked-prefill chunk tokens: 32, 64, 128, or 256
                                (K3 default 32).
      --expert-predict <on|off> Predictive expert prefetch (default on).
      --expert-io-workers <n>   Bounded pread workers: auto or 1...32
                                (default auto; measures 1, 2, and 4).
      --expert-io-splits <n>    Page-aligned reads per expert: 1, 2, 4, or 8
                                (K3 default 1 whole-expert read).
      --expert-io-cache <mode>  auto, buffered, or uncached (F_NOCACHE)
                                (default auto; production K3 uses uncached).
      --model-verification <m>  full-sha256 or trusted-install
                                (default full-sha256).
      --k3-activation-diagnostics
                                Compare real embedding and layer-0 input-norm
                                activations against a scalar CPU reference.
      --verbose                 Extra expert-streaming stats after the footer.
    """

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var maxNew = 1_024
        var maxNewExplicit = false
        var maxContext = 4096
        var temperature: Float = 0.2
        var temperatureExplicit = false
        var topK: Int? = 64
        var topP: Float? = 0.95
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        var reasoningEffort: String?
        var noThinking = false
        var prefill = "chunked"
        var prefillChunk = 32
        var expertPredict = true
        var expertIOWorkers = "auto"
        var expertIOSplits = 1
        var expertIOCache = "auto"
        var modelVerification = "full-sha256"
        var k3ActivationDiagnostics = false
        var verbose = false

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--no-thinking":
                noThinking = true
                index += 1
            case "--verbose":
                verbose = true
                index += 1
            case "--k3-activation-diagnostics":
                k3ActivationDiagnostics = true
                index += 1
            case "--reasoning-effort":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["low", "high", "max"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                reasoningEffort = value
            case "--prefill":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["serial", "chunked"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                prefill = value
            case "--prefill-chunk":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), [32, 64, 128, 256].contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                prefillChunk = parsed
            case "--expert-predict":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = ["on": true, "off": false][value] else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertPredict = parsed
            case "--expert-io-workers":
                let value = try takeValue(argv, &index, flag: flag)
                if value != "auto" {
                    guard let parsed = Int(value), (1...32).contains(parsed) else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                }
                expertIOWorkers = value
            case "--expert-io-splits":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), [1, 2, 4, 8].contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertIOSplits = parsed
            case "--expert-io-cache":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["auto", "buffered", "uncached"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertIOCache = value
            case "--model-verification":
                let value = try takeValue(argv, &index, flag: flag)
                guard ["full-sha256", "trusted-install"].contains(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                modelVerification = value
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
                maxNewExplicit = true
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
                temperatureExplicit = true
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
        return Args(model: model,
                    prompt: prompt,
                    messagesFile: messagesFile,
                    maxNew: maxNew,
                    maxNewExplicit: maxNewExplicit,
                    maxContext: maxContext,
                    temperature: temperature,
                    temperatureExplicit: temperatureExplicit,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    seed: seed,
                    stops: stops,
                    quiet: quiet,
                    reasoningEffort: reasoningEffort,
                    noThinking: noThinking,
                    prefill: prefill,
                    prefillChunk: prefillChunk,
                    expertPredict: expertPredict,
                    expertIOWorkers: expertIOWorkers,
                    expertIOSplits: expertIOSplits,
                    expertIOCache: expertIOCache,
                    modelVerification: modelVerification,
                    k3ActivationDiagnostics: k3ActivationDiagnostics,
                    verbose: verbose)
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
