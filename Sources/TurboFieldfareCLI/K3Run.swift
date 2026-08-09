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

/// One JSONL job for `--batch-file`. Per-job generation overrides are kept
/// intentionally small; all jobs share one model, I/O scheduler, and exact
/// direct-resident expert banks while decode state is reset between rows.
private struct K3BatchJobJSON: Decodable {
    let id: String?
    let prompt: String?
    let messages: [K3MessageJSON]?
    let maxNew: Int?
    let temperature: Float?

    enum CodingKeys: String, CodingKey {
        case id, prompt, messages, temperature
        case maxNew = "max_new"
    }
}

private struct K3BatchResultJSON: Encodable {
    let index: Int
    let id: String?
    let text: String?
    let error: String?
    let stop: String?
    let prefillTokens: Int?
    let newTokens: Int?
    let timeToFirstTokenSeconds: Double?
    let decodeTokensPerSecond: Double?
    let expertBytesRead: UInt64?
    let residentCacheHits: UInt64?

    enum CodingKeys: String, CodingKey {
        case index, id, text, error, stop
        case prefillTokens = "prefill_tokens"
        case newTokens = "new_tokens"
        case timeToFirstTokenSeconds = "ttft_seconds"
        case decodeTokensPerSecond = "decode_tokens_per_second"
        case expertBytesRead = "expert_bytes_read"
        case residentCacheHits = "resident_cache_hits"
    }
}

/// Kimi K3 (.gturbo v2) CLI run path. Same footer shape as the Gemma path via
/// `K3GenerateStats.footerLine`; `--verbose` adds the expert-streaming
/// counters. Generation stops at the model eos (`<|end_of_msg|>`, 163586) or
/// the token budget; the K3 engine does not evaluate `--stop` substrings, so
/// passing any is a clear usage error here.
public func runK3(args: Args,
                  stdout: FileHandle = .standardOutput,
                  stderr: FileHandle = .standardError,
                  expecting: K3ArchConfig = .kimiK3) async -> RunResult {
    do {
        guard args.stops.isEmpty else {
            return k3Errored(stderr, "--stop substrings are not supported for K3 bundles", 2)
        }
        let bundleURL = URL(fileURLWithPath: args.model)
        let tokenizer = try K3Tokenizer(vocabURL: bundleURL
            .appendingPathComponent("tokenizer", isDirectory: true)
            .appendingPathComponent("tiktoken.model"))

        if let batchFile = args.batchFile {
            return try runK3Batch(args: args,
                                  batchFile: batchFile,
                                  bundleURL: bundleURL,
                                  tokenizer: tokenizer,
                                  stdout: stdout,
                                  stderr: stderr,
                                  expecting: expecting)
        }

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
        let prefetchPolicy: K3ExpertPrefetchPolicy = !args.expertPredict
            ? .off
            : (args.expertPredictSelective ? .selective : .predict)
        let engine = try K3Engine.load(
            bundleURL: bundleURL,
            maxContext: args.maxContext,
            expecting: expecting,
            prefetchPolicy: prefetchPolicy,
            ioSplits: args.expertIOSplits,
            ioWorkers: ioWorkers,
            ioCachePolicy: ioCachePolicy,
            residentExpertCacheBytes: UInt64(args.expertCacheGiB) << 30,
            expertShardRoots: args.expertShardRoots.map {
                URL(fileURLWithPath: $0).standardizedFileURL
            },
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
                let residentCopyMB = (experts.residentCacheCopyBytes
                    + experts.residentCachePopulateBytes) / (1024 * 1024)
                let line = "[prefill=\(mode)"
                    + " sampling=\(samplingMode)"
                    + " demand=\(experts.demandHits)/\(experts.demandTotal)hit"
                    + " ramCache=\(experts.residentCacheHits)/"
                    + "\(experts.residentCacheHits + experts.residentCacheMisses)hit"
                    + " entries=\(experts.residentCacheEntries)/"
                    + "\(experts.residentCacheCapacity)"
                    + " ramCopy=\(residentCopyMB)MB"
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

/// Sequential K3 batch runner. This is deliberately not advertised as a
/// multi-sequence Metal batch: it amortizes model startup and lets queued jobs
/// share the exact direct-resident expert banks, while each job retains
/// independent KDA/MLA state and identical single-sequence logits.
private func runK3Batch(args: Args,
                        batchFile: String,
                        bundleURL: URL,
                        tokenizer: K3Tokenizer,
                        stdout: FileHandle,
                        stderr: FileHandle,
                        expecting: K3ArchConfig) throws -> RunResult {
    let data = try Data(contentsOf: URL(fileURLWithPath: batchFile),
                        options: [.mappedIfSafe])
    guard let source = String(data: data, encoding: .utf8) else {
        return k3Errored(stderr, "--batch-file must be UTF-8 JSONL", 2)
    }
    let rows = source.split(whereSeparator: \.isNewline)
    guard !rows.isEmpty else {
        return k3Errored(stderr, "--batch-file contains no jobs", 2)
    }
    guard MTLCreateSystemDefaultDevice() != nil else {
        return k3Errored(stderr, "no Metal device", 1)
    }

    let ioWorkers: K3ExpertIOWorkers = args.expertIOWorkers == "auto"
        ? .adaptive
        : .fixed(Int(args.expertIOWorkers)!)
    let integrityPolicy: ModelIntegrityPolicy = args.modelVerification == "trusted-install"
        ? .sizeCheckTrustedReceipt
        : .fullSha256
    let prefetchPolicy: K3ExpertPrefetchPolicy = !args.expertPredict
        ? .off
        : (args.expertPredictSelective ? .selective : .predict)
    let engine = try K3Engine.load(
        bundleURL: bundleURL,
        maxContext: args.maxContext,
        expecting: expecting,
        prefetchPolicy: prefetchPolicy,
        ioSplits: args.expertIOSplits,
        ioWorkers: ioWorkers,
        ioCachePolicy: K3ExpertIOCachePolicy(rawValue: args.expertIOCache)!,
        residentExpertCacheBytes: UInt64(args.expertCacheGiB) << 30,
        expertShardRoots: args.expertShardRoots.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        integrityPolicy: integrityPolicy)
    let prefillMode: K3PrefillMode = args.prefill == "serial"
        ? .serialReplay
        : .chunked(chunkTokens: args.prefillChunk)

    if args.k3ActivationDiagnostics {
        let firstJob = try JSONDecoder().decode(
            K3BatchJobJSON.self, from: Data(rows[0].utf8))
        let probeIDs = try k3BatchPromptIDs(firstJob, args: args, tokenizer: tokenizer)
        guard let token = probeIDs.first else {
            return k3Errored(stderr, "first batch job has an empty prompt", 2)
        }
        let diagnostics = try engine.activationDiagnostics(token: token)
        stderr.write(Data("\(diagnostics.summaryLine)\n".utf8))
        guard diagnostics.passed else {
            return k3Errored(stderr,
                             "K3 real-weight activation diagnostic failed; "
                                + "batch generation was not started",
                             1)
        }
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var failures = 0
    var succeeded = 0
    var totalRead: UInt64 = 0
    var totalCacheHits: UInt64 = 0

    for (index, row) in rows.enumerated() {
        var jobID: String?
        do {
            try Task.checkCancellation()
            let job = try JSONDecoder().decode(
                K3BatchJobJSON.self, from: Data(row.utf8))
            jobID = job.id
            let promptIDs = try k3BatchPromptIDs(job, args: args, tokenizer: tokenizer)
            guard !promptIDs.isEmpty else {
                throw GFTokenizerError.invalidChatTemplate("empty prompt")
            }
            guard promptIDs.count < args.maxContext else {
                throw GFTokenizerError.invalidChatTemplate(
                    "context overflow: prompt \(promptIDs.count) reaches maxContext "
                        + "\(args.maxContext)")
            }
            let requestedMaxNew = job.maxNew
                ?? (args.maxNewExplicit ? args.maxNew : 64)
            guard requestedMaxNew > 0 else {
                throw GFTokenizerError.invalidChatTemplate("max_new must be positive")
            }
            let temperature = job.temperature
                ?? (args.temperatureExplicit ? args.temperature : 0)
            guard temperature >= 0 else {
                throw GFTokenizerError.invalidChatTemplate(
                    "temperature must be non-negative")
            }
            let maxNew = min(requestedMaxNew, args.maxContext - promptIDs.count)
            let config = GenerationConfig(
                maxNewTokens: maxNew,
                temperature: temperature,
                topK: args.topK,
                topP: args.topP,
                repetitionPenalty: args.repetitionPenalty,
                seed: args.seed)

            engine.resetStatistics()
            var detokenizer = K3Detokenizer(tokenizer: tokenizer)
            var text = ""
            let stats = try engine.generate(
                promptTokens: promptIDs,
                config: config,
                maxNew: maxNew,
                prefillMode: prefillMode) { token in
                    text += detokenizer.push(token)
                }
            text += detokenizer.flush()
            let experts = stats.expertStreaming
            totalRead &+= experts.bytesRead
            totalCacheHits &+= experts.residentCacheHits
            succeeded += 1
            try writeK3BatchResult(
                K3BatchResultJSON(
                    index: index,
                    id: job.id,
                    text: text,
                    error: nil,
                    stop: String(describing: stats.reason),
                    prefillTokens: stats.prefillTokens,
                    newTokens: stats.newTokens,
                    timeToFirstTokenSeconds: stats.timeToFirstTokenSeconds,
                    decodeTokensPerSecond: stats.tokensPerSecond,
                    expertBytesRead: experts.bytesRead,
                    residentCacheHits: experts.residentCacheHits),
                encoder: encoder,
                stdout: stdout)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            failures += 1
            engine.reset()
            try writeK3BatchResult(
                K3BatchResultJSON(
                    index: index,
                    id: jobID,
                    text: nil,
                    error: "\(error)",
                    stop: nil,
                    prefillTokens: nil,
                    newTokens: nil,
                    timeToFirstTokenSeconds: nil,
                    decodeTokensPerSecond: nil,
                    expertBytesRead: nil,
                    residentCacheHits: nil),
                encoder: encoder,
                stdout: stdout)
        }
    }

    if !args.quiet {
        stderr.write(Data(("[batch jobs=\(rows.count) succeeded=\(succeeded) "
            + "failed=\(failures) read=\(totalRead / (1024 * 1024))MB "
            + "ramCacheHits=\(totalCacheHits)]\n").utf8))
    }
    return RunResult(exitCode: failures == 0 ? 0 : 1)
}

private func k3BatchPromptIDs(_ job: K3BatchJobJSON,
                              args: Args,
                              tokenizer: K3Tokenizer) throws -> [Int32] {
    if job.prompt != nil && job.messages != nil {
        throw GFTokenizerError.invalidChatTemplate(
            "batch row prompt and messages are mutually exclusive")
    }
    if let prompt = job.prompt {
        return try tokenizer.encode(prompt, allowSpecial: false).map(Int32.init)
    }
    if let rows = job.messages {
        let messages = try rows.map(k3Message)
        let options = K3ChatOptions(
            thinking: !args.noThinking,
            thinkingEffort: args.reasoningEffort.flatMap(K3ThinkingEffort.init(rawValue:)))
        return try K3ChatRenderer.renderToIDs(
            messages: messages, options: options, tokenizer: tokenizer).map(Int32.init)
    }
    throw GFTokenizerError.invalidChatTemplate(
        "batch row requires prompt or messages")
}

private func writeK3BatchResult(_ result: K3BatchResultJSON,
                                encoder: JSONEncoder,
                                stdout: FileHandle) throws {
    var data = try encoder.encode(result)
    data.append(0x0A)
    stdout.write(data)
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
