import Foundation
import Tokenizers
import TinyTitanCore

/// The checkpoint's own tokenizer, loaded without TinyTitan's ChatML/K2
/// dialect resolver.
///
/// `GFTokenizer` requires `<|endoftext|>` for ChatML and `<|ifm|im_start|>` for
/// K2; Nemotron writes `<|im_start|>`/`<|im_end|>` but has no
/// `<|endoftext|>`, so neither dialect fits and its check refuses the model's
/// own tokenizer. This wrapper uses `AutoTokenizer` and the checkpoint's
/// `chat_template.jinja` directly, which is the whole contract this model
/// needs: render the template, encode, decode.
public final class NemotronTokenizer {
    private let tokenizer: any Tokenizer
    private let template: String

    public init(directory: URL, thinking: Bool = false) async throws {
        guard
            let folder = GFTokenizer.resolvedTokenizerFolder(
                forModelDirectory: directory)
        else {
            throw NemotronInstallError.unsupported(
                "the install has no tokenizer/tokenizer.json")
        }
        tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let templateData = try BoundedMetadataRead.read(
            fileAt: folder.appendingPathComponent("chat_template.jinja"),
            maxBytes: 1024 * 1024)
        guard let raw = String(bytes: templateData, encoding: .utf8) else {
            throw NemotronInstallError.unsupported("the chat template is not UTF-8")
        }
        // The card's template defaults `enable_thinking` to true. A benchmark
        // and a CLI one-shot both want the answer, not the thoughts, so the
        // flag is set before the template's own `is defined` fallback.
        template = thinking ? raw : "{% set enable_thinking = false %}\n" + raw
    }

    public func encode(prompt: String, chat: Bool) throws -> [Int32] {
        let ids: [Int]
        if chat {
            var messages: [[String: any Sendable]] = []
            if let system = ProcessInfo.processInfo.environment[
                "TINYTITAN_NEMOTRON_SYSTEM"]
            {
                messages.append(["role": "system", "content": system])
            }
            messages.append(["role": "user", "content": prompt])
            ids = try tokenizer.applyChatTemplate(
                messages: messages,
                chatTemplate: .literal(template),
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: nil,
                additionalContext: nil)
        } else {
            ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
        }
        return ids.map(Int32.init)
    }

    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(
            tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }
}
