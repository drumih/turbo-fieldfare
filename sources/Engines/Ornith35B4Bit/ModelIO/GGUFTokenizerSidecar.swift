import Foundation

package enum GGUFTokenizerError: Error, Equatable, CustomStringConvertible {
    case missingMetadata(String)
    case tokenTypeCountMismatch(tokens: Int, types: Int)
    case invalidMerge(String)
    case emptyVocabulary

    package var description: String {
        switch self {
        case .missingMetadata(let key):
            return "GGUF metadata has no \(key)"
        case .tokenTypeCountMismatch(let tokens, let types):
            return
                "GGUF tokenizer lists \(tokens) tokens but \(types) token types"
        case .invalidMerge(let merge):
            return "GGUF merge \(merge.debugDescription) is not two space-separated pieces"
        case .emptyVocabulary:
            return "GGUF tokenizer has an empty vocabulary"
        }
    }
}

/// Turns the tokenizer GGUF already carries into the sidecar files the runtime's
/// loader reads: `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`
/// and `tokenizer/chat_template.jinja`.
///
/// This exists instead of a second tokenizer path in the engine. `GFTokenizer`
/// loads Hugging Face tokenizer data through swift-transformers, and everything
/// that data needs is in the GGUF: 248,320 byte-level tokens (already in the
/// GPT-2 unicode alphabet, which is what HF's `vocab` keys are), 247,587 merges,
/// the control tokens, and the chat template. Which pre-tokenizer to use is the
/// one fact not stored as data — GGUF names it (`tokenizer.ggml.pre = qwen35`)
/// and llama.cpp carries the regex in `llama-vocab.cpp` with the comment
/// "original regex from tokenizer.json", which is what `qwen35Pattern` here is.
package enum GGUFTokenizerSidecar {
    /// The subdirectory under a model directory the runtime loader reads
    /// (`GFTokenizer.tokenizerFolder`); exported files must land there.
    package static let folderName = "tokenizer"
    package static let tokenizerFileName = "tokenizer.json"
    package static let configFileName = "tokenizer_config.json"
    package static let templateFileName = "chat_template.jinja"

    package static func write(from model: GGUFModel, to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try tokenizerJSON(from: model).write(
            to: directory.appendingPathComponent(tokenizerFileName))
        try configJSON(from: model).write(
            to: directory.appendingPathComponent(configFileName))
        try chatTemplate(from: model).write(
            to: directory.appendingPathComponent(templateFileName),
            atomically: true, encoding: .utf8)
    }

    package static func tokenizerJSON(from model: GGUFModel) throws -> Data {
        let tokens = try strings(model, "tokenizer.ggml.tokens")
        let types = try ints(model, "tokenizer.ggml.token_type")
        let merges = try strings(model, "tokenizer.ggml.merges")
        guard !tokens.isEmpty else { throw GGUFTokenizerError.emptyVocabulary }
        guard tokens.count == types.count else {
            throw GGUFTokenizerError.tokenTypeCountMismatch(
                tokens: tokens.count, types: types.count)
        }

        var out = ""
        out.reserveCapacity(tokens.count * 16 + merges.count * 16 + 4096)
        out += #"{"version":"1.0","truncation":null,"padding":null,"added_tokens":["#
        var first = true
        for id in tokens.indices where types[id] == 3 || types[id] == 4 {
            if !first { out += "," }
            first = false
            out += #"{"id":"# + String(id)
            out += #","content":"# + escape(tokens[id])
            out += #","single_word":false,"lstrip":false,"rstrip":false,"#
            // GGML's token types carry the distinction the runtime needs:
            // CONTROL (3) tokens are special (EOS is filtered from output),
            // while USER_DEFINED (4) tokens are the ChatML barriers
            // (<tool_call>, <tool_response>, <think>) that must stay literal.
            let special = types[id] == 3 ? "true" : "false"
            out += #""normalized":false,"special":"# + special + "}"
        }
        out += "],\"normalizer\":null,\"pre_tokenizer\":"
        out +=
            #"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"#
            + escape(qwen35Pattern) + #"},"behavior":"Isolated","invert":false},"#
            + #"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},"#
        out += #""post_processor":null,"#
        out +=
            #""decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"#
        out += #""model":{"type":"BPE","dropout":null,"unk_token":null,"#
        out += #""continuing_subword_prefix":null,"end_of_word_suffix":null,"#
        out += #""fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"#
        for (id, token) in tokens.enumerated() {
            if id > 0 { out += "," }
            out += escape(token) + ":" + String(id)
        }
        out += "},\"merges\":["
        for (index, merge) in merges.enumerated() {
            let pieces = merge.split(separator: " ", omittingEmptySubsequences: false)
            guard pieces.count == 2 else { throw GGUFTokenizerError.invalidMerge(merge) }
            if index > 0 { out += "," }
            out += "[" + escape(String(pieces[0])) + "," + escape(String(pieces[1])) + "]"
        }
        out += "]}}"
        guard let data = out.data(using: .utf8) else {
            throw GGUFTokenizerError.missingMetadata("UTF-8 encoding of tokenizer.json")
        }
        return data
    }

    package static func configJSON(from model: GGUFModel) throws -> Data {
        let tokens = try strings(model, "tokenizer.ggml.tokens")
        guard !tokens.isEmpty else { throw GGUFTokenizerError.emptyVocabulary }
        func token(at key: String) -> String? {
            guard let id = model.metadata[key]?.integer, tokens.indices.contains(Int(id)) else {
                return nil
            }
            return tokens[Int(id)]
        }
        var object: [String: Any] = [
            "tokenizer_class": "Qwen2Tokenizer",
            "add_bos_token": false,
            "add_eos_token": false,
            "clean_up_tokenization_spaces": false,
            "model_max_length": model.architectureValue("context_length")?.integer ?? 262_144,
        ]
        if let bos = token(at: "tokenizer.ggml.bos_token_id") { object["bos_token"] = bos }
        if let eos = token(at: "tokenizer.ggml.eos_token_id") { object["eos_token"] = eos }
        if let pad = token(at: "tokenizer.ggml.padding_token_id") { object["pad_token"] = pad }
        return try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    }

    package static func chatTemplate(from model: GGUFModel) throws -> String {
        guard let template = model.metadata["tokenizer.chat_template"]?.string else {
            throw GGUFTokenizerError.missingMetadata("tokenizer.chat_template")
        }
        return template
    }

    private static func strings(_ model: GGUFModel, _ key: String) throws -> [String] {
        guard let values = model.metadata[key]?.strings else {
            throw GGUFTokenizerError.missingMetadata(key)
        }
        return values
    }

    private static func ints(_ model: GGUFModel, _ key: String) throws -> [Int] {
        guard let values = model.metadata[key]?.array else {
            throw GGUFTokenizerError.missingMetadata(key)
        }
        var result: [Int] = []
        result.reserveCapacity(values.count)
        for value in values {
            guard let integer = value.integer else {
                throw GGUFTokenizerError.missingMetadata(key)
            }
            result.append(Int(integer))
        }
        return result
    }

    /// JSON string escaping for a vocabulary key. Byte-level tokens can contain
    /// a quote or a backslash, so these are not cosmetic.
    private static func escape(_ value: String) -> String {
        var out = "\""
        out.reserveCapacity(value.utf8.count + 2)
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }

    /// `LLAMA_VOCAB_PRE_TYPE_QWEN35` from llama.cpp's `llama-vocab.cpp`, whose
    /// comment names it as the original regex from the checkpoint's
    /// tokenizer.json. Note the `\p{N}` (one digit) and `[\p{L}\p{M}]+` that
    /// differ from the Qwen2/GPT-2 patterns.
    static let qwen35Pattern =
        #"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
}
