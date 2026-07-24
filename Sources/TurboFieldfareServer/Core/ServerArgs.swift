import Foundation

/// Hand-rolled flag parsing, matching the style of the CLI and repack tools.
public struct ServerArgs: Equatable, Sendable {
    public var model: String
    public var host: String
    public var port: Int
    public var maxContext: Int
    public var modelID: String
    public var quiet: Bool

    public init(model: String,
                host: String = "127.0.0.1",
                port: Int = 1234,
                maxContext: Int = 4096,
                modelID: String = "gemma-4-26b-a4b-it-4bit",
                quiet: Bool = false) {
        self.model = model
        self.host = host
        self.port = port
        self.maxContext = maxContext
        self.modelID = modelID
        self.quiet = quiet
    }
}

public enum ServerArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        }
    }
}

extension ServerArgs {
    public static let usage = """
    TurboFieldfareServer — OpenAI-compatible HTTP API for Gemma 4 26B-A4B

    usage: TurboFieldfareServer --model <dir> [options]

    required:
      --model <dir>             Path to a .gturbo model directory.

    options:
      --host <string>           Bind address (default 127.0.0.1).
      --port <int>              Bind port (default 1234).
      --max-context <int>       Context limit in tokens (default 4096).
      --model-id <string>       Model id reported by /v1/models
                                (default gemma-4-26b-a4b-it-4bit).
      --quiet                   Suppress the startup banner.
      --help                    Show this message.
    """

    public static func parse(_ argv: [String]) throws -> ServerArgs {
        var parsed = ServerArgs(model: "")
        var model: String?

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ServerArgsError.helpRequested
            case "--quiet":
                parsed.quiet = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--host":
                parsed.host = try takeValue(argv, &index, flag: flag)
            case "--port":
                let value = try takeValue(argv, &index, flag: flag)
                guard let port = Int(value), (1...65535).contains(port) else {
                    throw ServerArgsError.invalidValue(flag: flag, value: value)
                }
                parsed.port = port
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let maxContext = Int(value), maxContext > 0 else {
                    throw ServerArgsError.invalidValue(flag: flag, value: value)
                }
                parsed.maxContext = maxContext
            case "--model-id":
                parsed.modelID = try takeValue(argv, &index, flag: flag)
            default:
                throw ServerArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ServerArgsError.requiredMissing("--model") }
        parsed.model = model
        return parsed
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ServerArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
