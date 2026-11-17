import Foundation
import Metal
import TinyTitan
import TinyTitanLib
import TinyTitanCore
import TinyTitanLegacyEngine

/// The CLI's K2-Horizon path.
///
/// The family has no `RealForwardRunner` graph, so this drives the probe
/// runtime directly: the same engine the server serves from and the bench
/// measures. A raw `--prompt` stays raw; `--messages-file` (and `--concise`)
/// render through the installed chat template, so the CLI's two prompt shapes
/// mean what they mean for every other family.
func runK2Probe(
    args: Args, modelURL: URL, prompt: Prompt, options: GenerationOptions,
    thinkingMode: ThinkingMode, reasoningEffort: ReasoningEffort?,
    stdout: FileHandle, stderr: FileHandle
) async -> RunResult {
    do {
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: modelURL)
        let identity = try ManifestReader.peekIdentity(directoryURL: modelURL)
        let slots = args.expertCacheSlots ?? (identity.weightBits == 4 ? 64 : 16)
        guard slots > 0 else {
            return errored(stderr, "--expert-cache-slots must be positive", 2)
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            return errored(stderr, "no Metal device", 1)
        }
        let metal = try MetalContext.runtimeDefault(device: device)
        let loaded: Model
        do {
            loaded = try Model.load(
                directoryURL: modelURL, device: metal.device,
                expecting: config, streamingMode: .pread(slotCount: slots),
                integrityPolicy: .sizeCheckTrustedReceipt)
        } catch let error as ModelError {
            guard case .trustedReceiptInvalid = error else { throw error }
            loaded = try Model.load(
                directoryURL: modelURL, device: metal.device,
                expecting: config, streamingMode: .pread(slotCount: slots))
        }
        let k2Thinking: ModelThinkingMode = thinkingMode == .on ? .on : .off
        let k2Effort = reasoningEffort.flatMap {
            ModelReasoningEffort(rawValue: $0.rawValue)
        }
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: modelURL,
            thinkingMode: k2Thinking, reasoningEffort: k2Effort)

        let promptIDs: [Int32]
        switch prompt {
        case .raw(let text):
            promptIDs = tokenizer.encode(text, addBOS: false)
        case .messages(let messages):
            let rendered = try tokenizer.applyChatTemplate(
                messages.map { message in
                    GFTokenizer.Message(
                        role: k2Role(message.role), content: message.content)
                })
            promptIDs = tokenizer.encode(rendered, addBOS: false)
        }
        guard !promptIDs.isEmpty else {
            return errored(stderr, "the prompt tokenized to nothing", 2)
        }
        let contextLimit = min(args.maxContext, K2ProbeRuntime.contextCeiling)
        guard promptIDs.count < contextLimit else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIDs.count) reaches maxContext \(contextLimit)",
                2)
        }
        let budget = min(options.maxTokens, contextLimit - promptIDs.count)

        var configuration = GenerationConfig()
        configuration.temperature = Float(options.temperature)
        configuration.topP = Float(options.topP)
        configuration.topK = options.topK
        configuration.presencePenalty = Float(options.presencePenalty)
        configuration.repetitionPenalty = Float(options.repetitionPenalty)
        configuration.seed = options.seed
        configuration.stopStrings = options.stop

        let runtime = try K2ProbeRuntime(
            model: loaded, context: metal, config: config,
            contextLimit: contextLimit, slotCount: slots,
            directory: modelURL)
        let result = try await runtime.generate(
            promptIDs: promptIDs, budget: budget,
            configuration: configuration, allowedTools: nil,
            tokenizer: tokenizer
        ) { event in
            // The answer goes to stdout; the model's thoughts do not. This is
            // the CLI's contract for every family.
            if case .content(let text) = event, !text.isEmpty {
                stdout.write(Data(text.utf8))
            }
        }
        if !args.quiet {
            let newTokens = result.completion.usage.completionTokens
            let rate =
                result.decodeSeconds > 0
                ? Double(newTokens) / result.decodeSeconds : 0
            let footer =
                "\n[stop=\(result.stopReason.rawValue) "
                + "prefill=\(promptIDs.count)tok/"
                + String(format: "%.2f", result.prefillSeconds) + "s "
                + "new=\(newTokens)tok "
                + "decode=" + String(format: "%.2f", result.decodeSeconds) + "s "
                + "tok/s=" + String(format: "%.3f", rate) + "]\n"
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

/// The facade's role, in the tokenizer's vocabulary.
private func k2Role(_ role: ChatMessage.Role) -> GFTokenizer.Role {
    switch role {
    case .system, .developer: return .system
    case .user: return .user
    case .assistant: return .assistant
    case .tool: return .tool
    }
}
