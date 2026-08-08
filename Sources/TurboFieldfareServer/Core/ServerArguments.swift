import Foundation

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let modelID: String
    /// True when `--model-id` was passed explicitly (K3 bundles default the
    /// API id to the manifest's model id instead of the Gemma default).
    public let modelIDExplicit: Bool
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    /// `full-sha256` by default; completed bundles may explicitly trust the
    /// install receipt to avoid repeated multi-terabyte K3 layer hashes.
    public let modelVerification: String

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --port <1...65535>     Loopback port (default 8080).
      --model-id <id>        API model identifier (default gemma-4-26b-a4b-it;
                             K3 v2 bundles default to the manifest model id).
      --max-context <tokens> 4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --model-verification <full-sha256|trusted-install>
                             Bundle verification policy (default full-sha256).
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var modelID = "gemma-4-26b-a4b-it"
        var modelIDExplicit = false
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var modelVerification = "full-sha256"
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelID = value
                modelIDExplicit = true
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--model-verification":
                guard ["full-sha256", "trusted-install"].contains(value) else {
                    throw ServerArgumentError.invalid(
                        "--model-verification must be full-sha256 or trusted-install")
                }
                modelVerification = value
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               modelID: modelID,
                               modelIDExplicit: modelIDExplicit,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               modelVerification: modelVerification)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
