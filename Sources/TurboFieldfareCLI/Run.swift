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
    // Utility mode: export the app conversation history without a model.
    if let outputPath = args.exportConversations {
        do {
            let source = conversationsFileURL()
            guard FileManager.default.fileExists(atPath: source.path) else {
                return errored(stderr,
                               "no conversation history at \(source.path); run the app once first",
                               2)
            }
            let data = try Data(contentsOf: source)
            guard JSONSerialization.isValidJSONObject(
                try JSONSerialization.jsonObject(with: data)) else {
                return errored(stderr, "conversation history is not valid JSON", 2)
            }
            try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
            stdout.write(Data("exported conversation history to \(outputPath)\n".utf8))
            return RunResult(exitCode: 0)
        } catch {
            return errored(stderr, "export failed: \(error)", 1)
        }
    }
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: RuntimeConfiguration().expertCacheSlots),
            expertCachePolicy: RuntimeConfiguration().modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: RuntimeConfiguration())
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)

        if args.cognitiveMode {
            let basePrompt: String
            if let messagesFile = args.messagesFile {
                let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                    options: [.mappedIfSafe])
                let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
                let messages = try rows.map { row in
                    guard let role = GFTokenizer.Role(rawValue: row.role) else {
                        throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                    }
                    return GFTokenizer.Message(role: role, content: row.content)
                }
                basePrompt = try tokenizer.applyChatTemplate(messages)
            } else {
                guard let raw = args.prompt else {
                    return errored(stderr, "missing prompt for cognitive mode", 2)
                }
                basePrompt = raw
            }

            var engine = CognitiveCycleEngine(userPrompt: basePrompt)
            var totalNew = 0
            var finalSeconds = 0.0

            while let step = engine.nextStep() {
                try Task.checkCancellation()
                let ids = tokenizer.encode(step.prompt, addBOS: false)
                stderr.write(Data("\n[cognitive pass: \(step.kind.header)]\n".utf8))

                let config = GenerationConfig(
                    maxNewTokens: min(args.maxNew, args.maxContext),
                    temperature: args.temperature,
                    topK: args.topK,
                    topP: args.topP,
                    repetitionPenalty: args.repetitionPenalty,
                    seed: args.seed,
                    stopStrings: args.stops,
                    extraStopTokens: [])
                let runtime = RuntimeConfiguration(
                    forceLogitsHead: !config.isPureGreedy)

                var passText = ""
                let stats = try await runRawCompletion(
                    producer: runner,
                    tokenizer: tokenizer,
                    promptIds: ids,
                    config: config,
                    context: context,
                    scratch: scratch,
                    prefillConfig: runtime.prefillConfig) { progress in
                        switch progress {
                        case .prefill:
                            break
                        case .token(_, _, let delta):
                            passText += delta
                            if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                        case .tail(let tail):
                            passText += tail
                            stdout.write(Data(tail.utf8))
                        }
                    }
                engine.record(output: passText)
                totalNew += stats.newTokens
                finalSeconds += stats.decodeSeconds
            }

            if !args.quiet {
                let tokPerSec = finalSeconds > 0 ? Double(totalNew) / finalSeconds : 0
                let footer = "\n[cognitive passes=4 new=\(totalNew)tok decode=\(String(format: "%.2f", finalSeconds))s tok/s=\(String(format: "%.3f", tokPerSec))]\n"
                stderr.write(Data(footer.utf8))
            }
            return RunResult(exitCode: 0)
        }

        // Standard single-pass generation.
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

/// Location of the app's conversation history file.
private func conversationsFileURL() -> URL {
    let applicationSupport = FileManager.default.urls(
        for: .applicationSupportDirectory, in: .userDomainMask).first
        ?? URL(fileURLWithPath: NSHomeDirectory())
    return applicationSupport
        .appendingPathComponent("TurboFieldfare", isDirectory: true)
        .appendingPathComponent("conversations.json")
}
