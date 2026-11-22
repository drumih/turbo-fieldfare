import Foundation
import TinyTitan
import TinyTitanLib

/// The CLI's Inkling2 GGUF route: an Unsloth Inkling-2-Mini `.gguf` (or a
/// directory holding exactly one) runs in place on the native decoder through
/// `Inkling2GGUFSession`, the generation path the loopback server shares.
///
/// A raw `--prompt` stays raw; `--messages-file` renders through Inkling's
/// message format. Generation stops on the GGUF's end-of-sequence/turn tokens
/// or the token budget.
func runInkling2GGUF(
    args: Args, model: URL, prompt: Prompt, options: GenerationOptions, stdout: FileHandle,
    stderr: FileHandle
) async -> RunResult {
    do {
        let loadStarted = Date()
        let session = try await Inkling2GGUFSession(
            gguf: model, maxContext: min(args.maxContext, 16_384))
        let loadSeconds = Date().timeIntervalSince(loadStarted)
        let text: String
        switch prompt {
        case .raw(let raw):
            text = raw
        case .messages(let messages):
            text = Inkling2GGUFSession.render(
                messages.map { message in
                    let role: Inkling2GGUFSession.Role
                    switch message.role {
                    case .system, .developer: role = .system
                    case .user: role = .user
                    case .assistant: role = .assistant
                    case .tool: role = .tool
                    }
                    return (role, message.content)
                })
        }
        let sampler = CPUSampler(
            temperature: Float(options.temperature), topP: Float(options.topP),
            topK: options.topK, presencePenalty: Float(options.presencePenalty),
            repetitionPenalty: Float(options.repetitionPenalty), seed: options.seed)
        let generation = try session.generate(
            promptIDs: session.tokenizer.encode(text), maxTokens: options.maxTokens,
            sampler: sampler
        ) { delta in stdout.write(Data(delta.utf8)) }
        stdout.write(Data("\n".utf8))
        if !args.quiet {
            let prefillRate = Double(generation.promptTokens) / max(generation.prefillSeconds, 1e-9)
            let decodeRate = Double(generation.tokens.count) / max(generation.decodeSeconds, 1e-9)
            let footer = String(
                format:
                    "[stop=%@ load=%.1fs prefill=%dtok/%.2fs (%.1f tok/s) new=%dtok decode=%.2fs tok/s=%.3f]\n",
                generation.stopReason, loadSeconds, generation.promptTokens,
                generation.prefillSeconds, prefillRate, generation.tokens.count,
                generation.decodeSeconds, decodeRate)
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch {
        stderr.write(Data("error: \(error)\n".utf8))
        return RunResult(exitCode: 1)
    }
}
