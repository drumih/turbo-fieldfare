import Foundation
import Metal
import TurboFieldfare

private struct MessageJSON: Decodable {
    let role: String
    let content: String
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    if args.chat {
        return await runChat(args: args, stdout: stdout, stderr: stderr)
    }
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let promptIds: [Int32]
        if let rawPrompt = args.prompt {
            promptIds = tokenizer.encode(rawPrompt, addBOS: true)
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            let messages = try rows.map { row -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: row.role) else {
                    throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return GFTokenizer.Message(role: role, content: row.content)
            }
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            return errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        let runtime = RuntimeConfiguration(
            forceLogitsHead: !config.isPureGreedy)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}

private func runChat(args: Args,
                     stdout: FileHandle,
                     stderr: FileHandle) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let runtime = RuntimeConfiguration(forceLogitsHead: true)
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)

        var history: [GFTokenizer.Message] = []
        if let systemPrompt = args.systemPrompt {
            history.append(GFTokenizer.Message(role: .system, content: systemPrompt))
        }

        stderr.write(Data("TurboFieldfare interactive chat. Type your message and press Enter.\n".utf8))
        stderr.write(Data("Commands: /clear  /history  /quit\n\n".utf8))

        while true {
            stderr.write(Data("you> ".utf8))
            guard let input = readLine(strippingNewline: true) else { break }

            let trimmed = input.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            switch trimmed {
            case "/quit", "/exit", "/q":
                break
            case "/clear":
                history.removeAll()
                if let systemPrompt = args.systemPrompt {
                    history.append(GFTokenizer.Message(role: .system, content: systemPrompt))
                }
                stderr.write(Data("conversation cleared.\n\n".utf8))
                continue
            case "/history":
                for msg in history {
                    stderr.write(Data("[\(msg.role.rawValue)] \(msg.content ?? "")\n".utf8))
                }
                stderr.write(Data("\n".utf8))
                continue
            default:
                break
            }
            if trimmed == "/quit" || trimmed == "/exit" || trimmed == "/q" { break }

            history.append(GFTokenizer.Message(role: .user, content: trimmed))

            let rendered = try tokenizer.applyChatTemplate(history)
            let promptIds = tokenizer.encode(rendered, addBOS: false)

            guard promptIds.count < args.maxContext else {
                stderr.write(Data("warning: context overflow (\(promptIds.count) tokens), trimming oldest messages.\n".utf8))
                while history.count > 1 && promptIds.count >= args.maxContext {
                    if history.first?.role == .system {
                        if history.count > 2 {
                            history.remove(at: 1)
                        } else {
                            break
                        }
                    } else if !history.isEmpty {
                        history.removeFirst()
                    } else {
                        break
                    }
                }
                let retryRendered = try tokenizer.applyChatTemplate(history)
                let retryIds = tokenizer.encode(retryRendered, addBOS: false)
                guard retryIds.count < args.maxContext else {
                    stderr.write(Data("error: cannot fit prompt in context even after trimming\n".utf8))
                    _ = history.popLast()
                    continue
                }
                try await generateTurn(
                    args: args, tokenizer: tokenizer, runner: runner,
                    context: context, scratch: scratch, runtime: runtime,
                    history: &history, promptIds: retryIds,
                    stdout: stdout, stderr: stderr)
                continue
            }

            try await generateTurn(
                args: args, tokenizer: tokenizer, runner: runner,
                context: context, scratch: scratch, runtime: runtime,
                history: &history, promptIds: promptIds,
                stdout: stdout, stderr: stderr)
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func generateTurn(args: Args,
                          tokenizer: GFTokenizer,
                          runner: RealForwardRunner,
                          context: MetalContext,
                          scratch: RawCompletionScratch,
                          runtime: RuntimeConfiguration,
                          history: inout [GFTokenizer.Message],
                          promptIds: [Int32],
                          stdout: FileHandle,
                          stderr: FileHandle) async throws {
    let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
    let config = GenerationConfig(
        maxNewTokens: effectiveMaxNew,
        temperature: args.temperature,
        topK: args.topK,
        topP: args.topP,
        repetitionPenalty: args.repetitionPenalty,
        seed: args.seed,
        stopStrings: args.stops,
        extraStopTokens: [])

    var content = ""
    let stats = try await runRawCompletion(
        producer: runner,
        tokenizer: tokenizer,
        promptIds: promptIds,
        config: config,
        context: context,
        scratch: scratch,
        prefillConfig: runtime.prefillConfig) { progress in
            switch progress {
            case .prefill:
                break
            case .token(_, _, let delta):
                if !delta.isEmpty {
                    stdout.write(Data(delta.utf8))
                    content += delta
                }
            case .tail(let tail):
                stdout.write(Data(tail.utf8))
                content += tail
            }
        }

    stdout.write(Data("\n".utf8))

    if !content.isEmpty {
        history.append(GFTokenizer.Message(role: .assistant, content: content))
    }

    if !args.quiet {
        let tokensPerSecond = stats.decodeSeconds > 0
            ? Double(stats.newTokens) / stats.decodeSeconds
            : 0
        let footer = "[prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
        stderr.write(Data(footer.utf8))
    }
}
