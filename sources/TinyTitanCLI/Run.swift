// The CLI driver, built on the library.
//
// Phase A2 of `docs/plan-embedded-library.md`: the CLI is a front end of
// `TinyTitanLib`, like the server. It builds one `Engine`, one `Session`, and
// drives a single generation; it owns no tokenizer, no forward runner and no
// decode loop. The one flag that cannot be answered from `Args` alone —
// `--prefill-chunk auto`, which is sized to the prompt — is answered by an
// `Engine` helper that loads only the tokenizer.
import Foundation
import Metal
import TinyTitan
import TinyTitanLib
import TinyTitanCore

private struct MessageJSON: Decodable {
    let role: String
    let content: String?

    enum CodingKeys: String, CodingKey { case role, content }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        if let value = try container.decodeIfPresent(JSONValue.self, forKey: .content) {
            content = Self.text(value)
        } else {
            content = nil
        }
    }

    private static func text(_ value: JSONValue) -> String? {
        switch value {
        case .string(let s):
            return s
        case .null:
            return nil
        case .array(let parts):
            var out = ""
            for part in parts {
                guard case .object(let dict) = part,
                    case .string(let type)? = dict["type"], type == "input_text",
                    case .string(let text)? = dict["text"]
                else { continue }
                out += text
            }
            return out
        default:
            return nil
        }
    }
}

/// A malformed `--messages-file`, reported with the tokenizer loader's wording
/// so the CLI's error text is unchanged even though the tokenizer is no longer
/// this target's business.
private struct MessageFileError: Error, CustomStringConvertible {
    let description: String
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

/// lint:allow-long the CLI driver: resolve the sampling plan, choose the head,
/// load the engine, run one generation, print the timing footer. It is the
/// top-level script for a one-shot tool, and its steps have no other caller.
public func run(
    args: Args,
    stdout: FileHandle = .standardOutput,
    stderr: FileHandle = .standardError
) async -> RunResult {
    do {
        if !args.quiet, let line = args.calibrationLogLine {
            stderr.write(Data((line + "\n").utf8))
        }
        let modelURL = URL(fileURLWithPath: args.model)
        if let nemotron = await NemotronRun.runIfNemotron(
            args: args, stdout: stdout, stderr: stderr)
        {
            return nemotron
        }
        let thinkingMode: ThinkingMode = args.thinkingMode == .on ? .on : .off
        let reasoningEffort = args.reasoningEffort.flatMap {
            ReasoningEffort(rawValue: $0.rawValue)
        }
        // The family's own row, read from the manifest before the engine
        // exists: which head the load may use depends on whether this plan is
        // pure greedy. Anything the caller named on the command line wins;
        // this only fills what they left alone.
        let declared = SamplingDefaults.forInstall(
            at: modelURL, thinkingMode: thinkingMode)
        let options = GenerationOptions(
            maxTokens: min(args.maxNew, args.maxContext),
            temperature: args.temperatureWasSet
                ? Double(args.temperature) : declared.temperature,
            topP: args.topPWasSet ? Double(args.topP ?? 0) : declared.topP,
            topK: args.topKWasSet ? (args.topK ?? 0) : declared.topK,
            repetitionPenalty: Double(args.repetitionPenalty),
            presencePenalty: args.presencePenaltyWasSet
                ? Double(args.presencePenalty) : declared.presencePenalty,
            seed: args.seed,
            stop: args.stops)
        // The head selection is the one load-time setting the sampling plan
        // decides: a pure-greedy request may use the fused greedy head, and
        // anything else needs real logits. The pre-facade CLI set exactly this.
        let isPureGreedy =
            options.temperature == 0
            && options.presencePenalty == 0
            && options.repetitionPenalty == 1
        let prompt = try buildPrompt(args: args)
        // K2-Horizon has no runner graph; the CLI drives the probe runtime
        // instead. Branch before the runner-specific chunk resolution.
        if (try? ManifestReader.peekFamily(directoryURL: modelURL)) == .k2Horizon {
            return await runK2Probe(
                args: args, modelURL: modelURL, prompt: prompt,
                options: options, thinkingMode: thinkingMode,
                reasoningEffort: reasoningEffort,
                stdout: stdout, stderr: stderr)
        }
        // A directory of .gguf shards is not an install and has no manifest;
        // it runs in place on the as-is decoder. Branched before the
        // tokenizer-only prefill-chunk resolution, which expects an install.
        if isGGUFModelDirectory(modelURL) {
            return await runGGUFAsIs(
                args: args, modelURL: modelURL, prompt: prompt,
                options: options, thinkingMode: thinkingMode,
                reasoningEffort: reasoningEffort,
                stdout: stdout, stderr: stderr)
        }
        let prefillChunkTokens = try await resolvePrefillChunk(
            args: args, prompt: prompt, modelURL: modelURL,
            thinkingMode: thinkingMode, reasoningEffort: reasoningEffort)
        // `--quiet` means quiet: the library's load and generation lines go to
        // the same stderr this command writes its footer on, and a user who
        // asked for no reporting does not want them either. Written as a
        // statement rather than a ternary: the closure-and-`nil` conditional
        // makes the type checker give up rather than pick.
        let logSink: (@Sendable (String) -> Void)?
        if args.quiet {
            logSink = { _ in }
        } else {
            logSink = nil
        }
        let configuration = EngineConfiguration(
            contextWindow: args.maxContext,
            expertCacheBudgetBytes: args.ramBudgetBytes,
            cachePrecision: cachePrecision(args.kvCachePrecision),
            prefillChunkTokens: prefillChunkTokens,
            expertCacheSlots: args.expertCacheSlots,
            ropeScaling: args.ropeScalingMode == .yarn ? .yarn : .none,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort,
            readAhead: readAheadAdvice(args.rdadvise),
            forceLogitsHead: !isPureGreedy,
            logSink: logSink)
        guard let device = MTLCreateSystemDefaultDevice() else {
            return errored(stderr, "no Metal device", 1)
        }
        let engine = try await Engine(
            directory: modelURL, device: device, configuration: configuration)
        // The one request would arrive before any layer is warm and stop the
        // warm-up at once, so the CLI lets it finish (it logs its own line;
        // TINYTITAN_WARM_EXPERTS=0 skips it).
        await engine.awaitExpertWarmup()
        let session = await engine.session()
        // The engine's resolved row and the pre-load peek agree; asserting it
        // here would change no output, so the plan above is what runs.
        let summary = try await session.respond(to: prompt, options: options) { event in
            switch event {
            case .token(let text):
                if !text.isEmpty { stdout.write(Data(text.utf8)) }
            case .reasoning:
                // The answer goes to stdout; the model's thoughts do not. This
                // is what the CLI did before it moved onto the facade, and the
                // saved baselines pin it.
                break
            case .toolCall:
                // The CLI offers the model no tools, so this cannot arrive; if
                // the facade ever lets it ask for one, this is where it would
                // be reported rather than silently dropped.
                break
            case .promptProcessed, .finished:
                break
            }
        }
        if !args.quiet {
            let tokensPerSecond =
                summary.decodeSeconds > 0
                ? Double(summary.completionTokens) / summary.decodeSeconds
                : 0
            let footer =
                "\n[stop=\(summary.decodeStopReason.rawValue) prefill=\(summary.promptTokens)tok/\(String(format: "%.2f", summary.prefillSeconds))s new=\(summary.completionTokens)tok decode=\(String(format: "%.2f", summary.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch let error as TinyTitanError {
        switch error {
        case .contextWindowExceeded(let prompt, let window):
            // The prompt count is known for a raw completion and not for a
            // rendered chat prompt; report the window either way.
            let counted = prompt > 0 ? "prompt \(prompt) " : "prompt "
            return errored(
                stderr, "context overflow: \(counted)reaches maxContext \(window)", 2)
        case .cancelled:
            stdout.write(Data("\n".utf8))
            return RunResult(exitCode: 130)
        default:
            return errored(stderr, "\(error)", 1)
        }
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

/// One of `--prompt` (raw, or templated when `--concise` asks for it) or
/// `--messages-file` (always templated), with concise mode folded in.
private func buildPrompt(args: Args) throws -> Prompt {
    let concise = args.concise ? ConcisePrompt.standard : nil
    if let rawPrompt = args.prompt {
        // Concise mode turns the raw prompt into a one-turn conversation, the
        // same shape the pre-facade CLI rendered through the chat template.
        if let concise {
            return .messages(
                applyingConcise(
                    concise, to: [ChatMessage(role: .user, content: rawPrompt)]))
        }
        return .raw(rawPrompt)
    }
    guard let messagesFile = args.messagesFile else {
        throw MessageFileError(description: "one of --prompt or --messages-file is required")
    }
    // lint:allow-unbounded-read --messages-file names this path, so its size is
    // whatever the operator pointed at: a file outside every trust boundary, read
    // `.mappedIfSafe` so a large transcript is mapped rather than copied -- the
    // same two reasons the resident-payload read carries, on an input this process
    // was told to open.
    let data = try Data(
        contentsOf: URL(fileURLWithPath: messagesFile),
        options: [.mappedIfSafe])
    let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
    var messages = try rows.map { row -> ChatMessage in
        guard let role = chatRole(row.role) else {
            throw MessageFileError(
                description: "invalid chat messages: unsupported role \(row.role)")
        }
        return ChatMessage(role: role, content: row.content ?? "")
    }
    if let concise {
        messages = applyingConcise(concise, to: messages)
    }
    return .messages(messages)
}

/// The tokenizer's five roles. A role outside them is reported, as before.
///
/// `developer` is folded to `system` deliberately: the template renders the
/// two identically (`Role.templateRole`), but the facade treats a developer
/// turn as a reason to render through the *tool* template, which the pre-facade
/// CLI — which always rendered the plain chat template — never did. Folding
/// keeps the bytes the old path produced.
private func chatRole(_ raw: String) -> ChatMessage.Role? {
    switch raw {
    case "system", "developer": .system
    case "user": .user
    case "assistant": .assistant
    case "tool": .tool
    default: nil
    }
}

/// `ConcisePrompt.appendingSystemPrompt`'s rule over the kit's message type:
/// fold into the first system/developer turn, or open one.
private func applyingConcise(
    _ prompt: String,
    to messages: [ChatMessage]
) -> [ChatMessage] {
    guard
        let index = messages.firstIndex(where: {
            $0.role == .system || $0.role == .developer
        })
    else {
        return [ChatMessage(role: .system, content: prompt)] + messages
    }
    var result = messages
    result[index] = ChatMessage(
        role: .system, content: result[index].content + "\n\n" + prompt)
    return result
}

/// `--prefill-chunk`: an explicit size, or the smallest allowed chunk that
/// covers the prompt. `nil` leaves the install's own profile row in charge.
private func resolvePrefillChunk(
    args: Args,
    prompt: Prompt,
    modelURL: URL,
    thinkingMode: ThinkingMode,
    reasoningEffort: ReasoningEffort?
) async throws -> Int? {
    switch args.prefillChunk {
    case .fixed(let tokens):
        return tokens
    case .auto:
        return try await Engine.prefillChunk(
            covering: prompt, directory: modelURL,
            thinkingMode: thinkingMode, reasoningEffort: reasoningEffort)
    case nil:
        return nil
    }
}

private func cachePrecision(_ precision: KVCachePrecision) -> CachePrecision {
    switch precision {
    case .int4: .fourBit
    case .int8: .eightBit
    case .fp16: .sixteenBit
    }
}

/// `--rdadvise` spells a case `ReadAheadAdvice` already has; the CLI's parser
/// has validated the string, so this only crosses the vocabulary boundary.
private func readAheadAdvice(_ raw: String) -> ReadAheadAdvice? {
    ReadAheadAdvice(rawValue: raw)
}

func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
