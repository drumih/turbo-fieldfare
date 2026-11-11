import Foundation
import Metal
import TinyTitan
import TinyTitanLib

/// The CLI's GGUF-as-is path: a directory of `.gguf` shards read in place,
/// with no conversion, no install and no receipt.
///
/// This exists for the Unsloth-style checkpoints the operator runs as they
/// sit on disk. The engine's other paths load a `.ssdai` install (a
/// `manifest.json` this directory does not carry), so a GGUF directory gets
/// its own branch: the GGUF reader opens the shards directly, the tokenizer
/// sidecar is derived from the GGUF's own metadata into
/// `<model dir>/tokenizer/` (a derived artifact, written once and reused),
/// and `GGUFDecoder` runs the forward pass the same way the gated model
/// tests drive it.
///
/// Prefill is the decoder's token-at-a-time walk (`GGUFDecoder.prefill`),
/// which fills the KV/GDN/PLE state; decode is one sampled token per step.
/// The ANE prefill sidecar is a separate build (TT-056) and is not engaged
/// here.
///
/// A raw `--prompt` stays raw; `--messages-file` (and `--concise`) render
/// through the tokenizer's ChatML template, the same rule the K2 branch
/// applies. Stop strings are not implemented on this path; generation stops
/// on the tokenizer's stop tokens (`<|im_end|>`, `<|endoftext|>`) or the
/// token budget.
func runGGUFAsIs(
    args: Args, modelURL: URL, prompt: Prompt, options: GenerationOptions,
    thinkingMode: ThinkingMode, reasoningEffort: ReasoningEffort?,
    stdout: FileHandle, stderr: FileHandle
) async -> RunResult {
    do {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return errored(stderr, "no Metal device", 1)
        }
        let metal = try MetalContext(device: device)
        let model = try GGUFModelReader.read(modelAt: modelURL)
        let tokenizer = try await ggufTokenizer(
            for: model, modelURL: modelURL, quiet: args.quiet,
            thinkingMode: thinkingMode, reasoningEffort: reasoningEffort,
            stderr: stderr)

        let promptIDs = try ggufPromptIDs(prompt, tokenizer: tokenizer)
        guard !promptIDs.isEmpty else {
            return errored(stderr, "the prompt tokenized to nothing", 2)
        }
        // The full-attention KV cache is allocated for the whole window up
        // front; capping it keeps an accidental --maxContext 262144 from
        // reserving hundreds of buffers the prompt will never use.
        let contextLimit = min(args.maxContext, 16_384)
        guard promptIDs.count < contextLimit else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIDs.count) reaches maxContext \(contextLimit)",
                2)
        }
        let budget = min(options.maxTokens, contextLimit - promptIDs.count)

        let decoder = try GGUFDecoder(
            context: metal, arch: .qwen38FlashNext, model: model, maxSeq: contextLimit,
            ramBudgetBytes: args.ramBudgetBytes)
        let sampler = CPUSampler(
            temperature: Float(options.temperature), topP: Float(options.topP),
            topK: options.topK,
            presencePenalty: Float(options.presencePenalty),
            repetitionPenalty: Float(options.repetitionPenalty), seed: options.seed)
        let generation = try ggufGenerate(
            decoder: decoder, tokenizer: tokenizer, promptIDs: promptIDs,
            budget: budget, sampler: sampler, stdout: stdout)
        if !args.quiet {
            let prefillRate =
                promptIDs.count > 0
                ? Double(promptIDs.count) / generation.prefillSeconds : 0
            let line =
                "prefill: \(promptIDs.count)tok/"
                + String(format: "%.2f", generation.prefillSeconds) + "s ("
                + String(format: "%.2f", prefillRate) + " tok/s)\n"
            stderr.write(Data(line.utf8))
            let rate =
                generation.decodeSeconds > 0
                ? Double(generation.tokens.count) / generation.decodeSeconds : 0
            let footer =
                "\n[stop=\(generation.stopReason) "
                + "prefill=\(promptIDs.count)tok/"
                + String(format: "%.2f", generation.prefillSeconds) + "s "
                + "new=\(generation.tokens.count)tok "
                + "decode=" + String(format: "%.2f", generation.decodeSeconds) + "s "
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

/// The sidecar, written once and reused. Re-deriving on every run would
/// rewrite three files for no effect, and a directory whose sidecar was
/// hand-tuned would lose the edit.
private func ggufTokenizer(
    for model: GGUFModel, modelURL: URL, quiet: Bool,
    thinkingMode: ThinkingMode, reasoningEffort: ReasoningEffort?,
    stderr: FileHandle
) async throws -> GFTokenizer {
    let sidecarFolder = modelURL.appendingPathComponent(
        GGUFTokenizerSidecar.folderName, isDirectory: true)
    if !FileManager.default.fileExists(
        atPath: sidecarFolder.appendingPathComponent(
            GGUFTokenizerSidecar.tokenizerFileName
        ).path)
    {
        try GGUFTokenizerSidecar.write(from: model, to: sidecarFolder)
        if !quiet {
            stderr.write(
                Data("tokenizer sidecar written to \(sidecarFolder.path)\n".utf8))
        }
    }
    return try await GFTokenizer.load(
        forModelDirectory: modelURL,
        thinkingMode: thinkingMode == .on ? .on : .off,
        reasoningEffort: reasoningEffort.flatMap {
            ModelReasoningEffort(rawValue: $0.rawValue)
        })
}

/// A raw prompt stays raw; a conversation renders through the tokenizer's
/// ChatML template and is encoded as text, the K2 branch's rule.
private func ggufPromptIDs(
    _ prompt: Prompt, tokenizer: GFTokenizer
) throws -> [Int32] {
    switch prompt {
    case .raw(let text):
        return tokenizer.encode(text, addBOS: false)
    case .messages(let messages):
        let rendered = try tokenizer.applyChatTemplate(
            messages.map { message in
                GFTokenizer.Message(
                    role: ggufRole(message.role), content: message.content)
            })
        return tokenizer.encode(rendered, addBOS: false)
    }
}

/// One generation: token-at-a-time prefill, then sampled decode with the
/// whole-sequence suffix-diff streaming, so multi-byte and multi-token
/// characters never tear mid-stream. At this decode rate the quadratic decode
/// cost is noise against the forward pass.
private func ggufGenerate(
    decoder: GGUFDecoder, tokenizer: GFTokenizer, promptIDs: [Int32],
    budget: Int, sampler: CPUSampler, stdout: FileHandle
) throws -> (tokens: [Int32], stopReason: String, prefillSeconds: Double, decodeSeconds: Double) {
    let generator = sampler.makeGenerator()
    let prefill = try decoder.prefill(tokens: promptIDs)
    var generated: [Int32] = []
    var decodedText = ""
    var stopReason = "length"
    var logits = prefill.logits
    var decodeSeconds = 0.0
    for index in 0..<budget {
        let picked = sampler.pick(
            logits, history: promptIDs + generated, using: generator)
        let token = Int32(picked)
        if tokenizer.stopTokenIDs.contains(token) {
            stopReason = "eos"
            break
        }
        generated.append(token)
        let text = tokenizer.decode(generated, skipSpecialTokens: true)
        if text.hasPrefix(decodedText), text.count > decodedText.count {
            let start = text.index(text.startIndex, offsetBy: decodedText.count)
            stdout.write(Data(String(text[start...]).utf8))
            decodedText = text
        }
        // The whole sequence so far: the PLE n-gram of the first generated
        // token reaches back into the prompt.
        let sequence = promptIDs + generated
        let position = promptIDs.count + index
        let stepStart = DispatchTime.now().uptimeNanoseconds
        logits = try decoder.decode(sequence: sequence, position: position)
        decodeSeconds += Double(DispatchTime.now().uptimeNanoseconds - stepStart) / 1e9
    }
    return (generated, stopReason, prefill.seconds, decodeSeconds)
}

/// A directory the as-is route should take: `.gguf` shards present, no
/// `manifest.json` (an install carries one and stays on the engine path,
/// whatever else sits beside it).
func isGGUFModelDirectory(_ url: URL, fileManager: FileManager = .default) -> Bool {
    var isDirectory: ObjCBool = false
    guard
        fileManager.fileExists(atPath: url.standardizedFileURL.path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else { return false }
    if fileManager.fileExists(
        atPath: url.appendingPathComponent("manifest.json").path)
    {
        return false
    }
    let names = (try? fileManager.contentsOfDirectory(atPath: url.path)) ?? []
    return names.contains { $0.hasSuffix(".gguf") }
}

/// The facade's role, in the tokenizer's vocabulary.
private func ggufRole(_ role: ChatMessage.Role) -> GFTokenizer.Role {
    switch role {
    case .system, .developer: return .system
    case .user: return .user
    case .assistant: return .assistant
    case .tool: return .tool
    }
}
