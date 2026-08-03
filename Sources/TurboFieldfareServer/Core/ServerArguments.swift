import Foundation

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let host: String
    public let allowRemote: Bool
    public let modelID: String
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode

    /// Hosts that are always safe because they never leave the local machine.
    /// Any other host requires `--allow-remote` to start.
    static let loopbackHosts: Set<String> = ["127.0.0.1", "::1", "localhost"]

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --port <1...65535>     Listening port (default 8080).
      --host <addr>          Bind address (default 127.0.0.1). Non-loopback hosts
                             require --allow-remote.
      --allow-remote         Permit binding to a non-loopback host. The server has
                             no authentication or TLS, so only enable this on a
                             trusted isolated network.
      --model-id <id>        API model identifier (default gemma-4-26b-a4b-it).
      --max-context <tokens> 4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var host = "127.0.0.1"
        var allowRemote = false
        var modelID = "gemma-4-26b-a4b-it"
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            switch flag {
            case "--allow-remote":
                allowRemote = true
                index += 1
                continue
            default:
                break
            }
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
            case "--host":
                let trimmed = value.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else {
                    throw ServerArgumentError.invalid("--host must not be empty")
                }
                host = trimmed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelID = value
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
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        let isLoopback = Self.loopbackHosts.contains(host.lowercased())
        guard isLoopback || allowRemote else {
            throw ServerArgumentError.invalid("""
                --host \(host) is non-loopback. The server has no authentication or TLS; \
                binding a non-loopback host exposes the model to the network. \
                To permit this, pass --allow-remote and use only a trusted isolated network.
                """)
        }
        return ServerArguments(model: model,
                               port: port,
                               host: host,
                               allowRemote: allowRemote,
                               modelID: modelID,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode)
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
