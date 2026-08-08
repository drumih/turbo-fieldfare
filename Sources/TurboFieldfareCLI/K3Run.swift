import Foundation
import Metal
import TurboFieldfare

/// One `--messages-file` row for the K3 chat path. Mirrors the OpenAI chat
/// shape: plain string content, optional name, assistant `reasoning_content`
/// and `tool_calls` (arguments as a JSON object string), and tool
/// `tool_call_id`.
private struct K3MessageJSON: Decodable {
    struct FunctionCall: Decodable {
        let name: String
        let arguments: String
    }
    struct ToolCall: Decodable {
        let id: String?
        let function: FunctionCall
    }

    let role: String
    let content: String?
    let name: String?
    let reasoningContent: String?
    let toolCalls: [ToolCall]?
    let toolCallID: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

/// Kimi K3 (.gturbo v2) CLI run path. Same footer shape as the Gemma path via
/// `K3GenerateStats.footerLine`; `--verbose` adds the expert-streaming
/// counters. Generation stops at the model eos (`<|end_of_msg|>`, 163586) or
/// the token budget; the K3 engine does not evaluate `--stop` substrings, so
/// passing any is a clear usage error here.
public func runK3(args: Args,
                  stdout: FileHandle = .standardOutput,
                  stderr: FileHandle = .standardError) async -> RunResult {
    do {
        guard args.stops.isEmpty else {
            return k3Errored(stderr, "--stop substrings are not supported for K3 bundles", 2)
        }
        let bundleURL = URL(fileURLWithPath: args.model)
        let tokenizer = try K3Tokenizer(vocabURL: bundleURL
            .appendingPathComponent("tokenizer", isDirectory: true)
            .appendingPathComponent("tiktoken.model"))

        let promptIDs: [Int32]
        if let rawPrompt = args.prompt {
            // Raw completion: ordinary BPE, specials never match.
            promptIDs = try tokenizer.encode(rawPrompt, allowSpecial: false).map(Int32.init)
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([K3MessageJSON].self, from: data)
            let messages = try rows.map(k3Message)
            let options = K3ChatOptions(
                thinking: !args.noThinking,
                thinkingEffort: args.reasoningEffort.flatMap(K3ThinkingEffort.init(rawValue:)))
            promptIDs = try K3ChatRenderer.renderToIDs(
                messages: messages, options: options, tokenizer: tokenizer).map(Int32.init)
        } else {
            return k3Errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
        guard !promptIDs.isEmpty else { return k3Errored(stderr, "empty prompt", 2) }
        guard promptIDs.count < args.maxContext else {
            return k3Errored(
                stderr,
                "context overflow: prompt \(promptIDs.count) reaches maxContext \(args.maxContext)",
                2)
        }
        // The generic CLI keeps Gemma's historical defaults. K3 is an
        // SSD-streamed batch runtime, so an omitted budget is deliberately
        // bounded and its upstream generation profile is greedy.
        let requestedMaxNew = args.maxNewExplicit ? args.maxNew : 64
        let effectiveTemperature = args.temperatureExplicit ? args.temperature : 0
        let effectiveMaxNew = min(requestedMaxNew, args.maxContext - promptIDs.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: effectiveTemperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed)
        let prefillMode: K3PrefillMode = args.prefill == "serial"
            ? .serialReplay
            : .chunked(chunkTokens: args.prefillChunk)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return k3Errored(stderr, "no Metal device", 1)
        }
        let ioWorkers: K3ExpertIOWorkers = args.expertIOWorkers == "auto"
            ? .adaptive
            : .fixed(Int(args.expertIOWorkers)!)
        let integrityPolicy: ModelIntegrityPolicy = args.modelVerification == "trusted-install"
            ? .sizeCheckTrustedReceipt
            : .fullSha256
        let ioCachePolicy = K3ExpertIOCachePolicy(rawValue: args.expertIOCache)!
        let engine = try K3Engine.load(
            bundleURL: bundleURL,
            maxContext: args.maxContext,
            prefetchPolicy: args.expertPredict ? .predict : .off,
            ioSplits: args.expertIOSplits,
            ioWorkers: ioWorkers,
            ioCachePolicy: ioCachePolicy,
            integrityPolicy: integrityPolicy)

        if args.k3ActivationDiagnostics {
            let diagnostics = try engine.activationDiagnostics(token: promptIDs[0])
            stderr.write(Data("\(diagnostics.summaryLine)\n".utf8))
            guard diagnostics.passed else {
                return k3Errored(stderr,
                                 "K3 real-weight activation diagnostic failed; "
                                     + "generation was not started",
                                 1)
            }
        }

        var detokenizer = K3Detokenizer(tokenizer: tokenizer)
        let stats = try engine.generate(
            promptTokens: promptIDs,
            config: config,
            maxNew: effectiveMaxNew,
            prefillMode: prefillMode) { token in
                let delta = detokenizer.push(token)
                if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
            }
        let tail = detokenizer.flush()
        if !tail.isEmpty { stdout.write(Data(tail.utf8)) }

        if args.k3ActivationDiagnostics {
            let top = k3TopLogits(engine.lastLogits(), count: 8)
            let sampled = stats.uncommittedBoundaryTokenIDs.last.map(Int.init)
            let sampledText = sampled.map { tokenizer.decode([$0]) } ?? ""
            let entries = top.map { item in
                let value = String(format: "%.5f", item.value)
                return "\(item.id):\(value):"
                    + String(reflecting: tokenizer.decode([item.id]))
            }.joined(separator: ", ")
            let sampledID = sampled.map(String.init) ?? "none"
            let logitLine = "[k3-logits sampled=\(sampledID) "
                + "sampledText=\(String(reflecting: sampledText)) top=[\(entries)]]\n"
            stderr.write(Data(logitLine.utf8))

            guard let routerDiagnostics = engine.routerActivationDiagnostics() else {
                return k3Errored(stderr, "K3 router activation was not captured", 1)
            }
            stderr.write(Data("\(routerDiagnostics.summaryLine)\n".utf8))
            guard routerDiagnostics.passed else {
                return k3Errored(stderr, "K3 router activation diagnostic failed", 1)
            }
            let headRows = top.map(\.id) + (sampled.map { [$0] } ?? [])
            guard let headDiagnostics = try engine.headActivationDiagnostics(
                tokenIDs: headRows) else {
                return k3Errored(stderr, "K3 head activation was not captured", 1)
            }
            stderr.write(Data("\(headDiagnostics.summaryLine)\n".utf8))
            guard headDiagnostics.passed else {
                return k3Errored(stderr, "K3 lm-head activation diagnostic failed", 1)
            }
        }

        if !args.quiet {
            stderr.write(Data("\n\(stats.footerLine)\n".utf8))
            if args.verbose {
                let experts = stats.expertStreaming
                let mode = args.prefill == "serial"
                    ? "serial"
                    : "chunked(\(args.prefillChunk))"
                let ioTune = experts.ioTuningComplete ? "done" : "sampling"
                let samplingMode = effectiveTemperature == 0
                    ? "greedy"
                    : "temperature(\(effectiveTemperature))"
                let line = "[prefill=\(mode)"
                    + " sampling=\(samplingMode)"
                    + " demand=\(experts.demandHits)/\(experts.demandTotal)hit"
                    + " prefetch=\(experts.prefetchesIssued)"
                    + " skippedBusy=\(experts.prefetchSkippedBankBusy)"
                    + " skippedCold=\(experts.prefetchSkippedCold)"
                    + " read=\(experts.bytesRead / (1024 * 1024))MB"
                    + " ioWorkers=\(experts.ioWorkerLimit)"
                    + " ioSplits=\(args.expertIOSplits)"
                    + " ioCache=\(experts.ioCacheMode)"
                    + " ioPeak=\(experts.peakConcurrentReads)"
                    + " ioTune=\(ioTune)"
                    + " ioBatches=\(experts.ioBatches)"
                    + " verification=\(args.modelVerification)"
                    + " position=\(stats.position)]\n"
                stderr.write(Data(line.utf8))
            }
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return k3Errored(stderr, "\(error)", 1)
    }
}

private func k3TopLogits(_ logits: [Float], count: Int) -> [(id: Int, value: Float)] {
    logits.enumerated()
        .filter { $0.element.isFinite }
        .sorted {
            if $0.element != $1.element { return $0.element > $1.element }
            return $0.offset < $1.offset
        }
        .prefix(count)
        .map { (id: $0.offset, value: $0.element) }
}

private func k3Message(_ row: K3MessageJSON) throws -> K3ChatMessage {
    let role: K3ChatRole
    switch row.role {
    case "system", "developer":
        role = .system
    case "user":
        role = .user
    case "assistant":
        role = .assistant
    case "tool":
        role = .tool
    default:
        throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
    }
    let toolCalls = (row.toolCalls ?? []).map { call in
        K3ToolCall(id: call.id,
                   name: call.function.name,
                   arguments: .jsonString(call.function.arguments))
    }
    return K3ChatMessage(role: role,
                         content: row.content.map { .text($0) },
                         name: row.name,
                         reasoningContent: row.reasoningContent,
                         toolCalls: toolCalls,
                         toolCallID: row.toolCallID)
}

private func k3Errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
