import Foundation
import Tokenizers
import TinyTitanCore

enum GFDetokenizerError: Error, Equatable, CustomStringConvertible {
    case invalidTokenID(Int32)
    case invalidByteLevelToken(id: Int32, token: String)

    var description: String {
        switch self {
        case .invalidTokenID(let id):
            return "tokenizer cannot decode generated token id \(id)"
        case .invalidByteLevelToken(let id, let token):
            return "token \(id) is not valid ByteLevel text: \(token.debugDescription)"
        }
    }
}

/// The small portion of `tokenizer.json` needed by the streaming decoder.
/// Model vocabulary entries are deliberately not decoded here: there are
/// 248K of them, while only the added-token barriers and decoder kind matter.
@usableFromInline
struct GFByteLevelDecoderConfiguration: Sendable {
    struct AddedToken: Sendable {
        let content: String
        let special: Bool
    }

    private struct FileMetadata: Decodable {
        struct DecoderMetadata: Decodable { let type: String }
        struct AddedTokenMetadata: Decodable {
            let id: Int
            let content: String
            let special: Bool
        }

        let decoder: DecoderMetadata
        let addedTokens: [AddedTokenMetadata]

        enum CodingKeys: String, CodingKey {
            case decoder
            case addedTokens = "added_tokens"
        }
    }

    let addedTokens: [Int32: AddedToken]

    static func load(
        from tokenizerJSON: URL,
        tokenizer: any Tokenizer,
        maxBytes: UInt64 = ManifestReader.defaultMaxBytes
    ) throws -> Self {
        // `tokenizer.json` is the largest document the load path reads: 12,809,320
        // bytes for the installed 125B tokenizer measured on this host. It is
        // copied into the install like any other part of the model directory, so
        // it takes the manifest's ceiling rather than a number tuned to today's
        // vocab, which would refuse a legitimate tokenizer the day the vocab grows.
        let data = try BoundedMetadataRead.read(fileAt: tokenizerJSON, maxBytes: maxBytes)
        let metadata = try JSONDecoder().decode(FileMetadata.self, from: data)
        guard metadata.decoder.type == "ByteLevel" else {
            throw GFTokenizerError.unsupportedForDialect(
                "streaming decoder requires tokenizer.json decoder.type=ByteLevel")
        }

        var result: [Int32: AddedToken] = [:]
        result.reserveCapacity(metadata.addedTokens.count)
        for entry in metadata.addedTokens {
            guard let id = Int32(exactly: entry.id),
                tokenizer.convertIdToToken(entry.id) == entry.content,
                result[id] == nil
            else {
                throw GFTokenizerError.unsupportedForDialect(
                    "tokenizer.json contains inconsistent added-token metadata")
            }
            result[id] = AddedToken(content: entry.content, special: entry.special)
        }
        return Self(addedTokens: result)
    }

    /// Compatibility initializer for callers that construct `GFTokenizer`
    /// directly instead of loading its sidecar. Production loads always use
    /// `load(from:tokenizer:)`, which validates every added token.
    static func knownChatMLTokens(tokenizer: any Tokenizer) -> Self {
        let special = [
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>",
            "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
            "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
            "<tts_text_bos_single>",
        ]
        let literal = [
            "<tool_call>", "</tool_call>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
            "<tool_response>", "</tool_response>", "<think>", "</think>",
        ]
        var result: [Int32: AddedToken] = [:]
        for token in special {
            if let id = tokenizer.convertTokenToId(token),
                tokenizer.convertIdToToken(id) == token,
                let id32 = Int32(exactly: id)
            {
                result[id32] = AddedToken(content: token, special: true)
            }
        }
        for token in literal {
            if let id = tokenizer.convertTokenToId(token),
                tokenizer.convertIdToToken(id) == token,
                let id32 = Int32(exactly: id)
            {
                result[id32] = AddedToken(content: token, special: false)
            }
        }
        return Self(addedTokens: result)
    }
}

/// Incremental Hugging Face ByteLevel decoder used by generation loops.
///
/// Ordinary vocabulary pieces are inverse-mapped directly to bytes. Complete
/// UTF-8 is emitted immediately and at most three bytes of an unfinished scalar
/// remain buffered. Added tokens are decoder barriers; special added tokens are
/// filtered before decoding, matching `skipSpecialTokens: true`.
///
/// Work and retained state are bounded by the newly pushed token rather than
/// the complete generated history. Batch decoding remains on swift-transformers
/// so tests can compare the two independent implementations.
///
/// Public for the CPU engine, whose loop lives outside this module but needs
/// the same per-token deltas: the structured decoder recognises `<think>` and
/// `<tool_call>` by their literal text on the delta.
public struct GFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    @usableFromInline let configuration: GFByteLevelDecoderConfiguration
    @usableFromInline var pendingBytes: [UInt8] = []

    public init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer.tokenizer
        self.configuration = tokenizer.byteLevelDecoderConfiguration
        pendingBytes.reserveCapacity(8)
    }

    public mutating func push(_ id: Int32) throws -> String {
        guard id >= 0, let token = tokenizer.convertIdToToken(Int(id)) else {
            throw GFDetokenizerError.invalidTokenID(id)
        }

        if let added = configuration.addedTokens[id] {
            if added.special {
                // Hugging Face removes special IDs before ByteLevel decoding,
                // so they do not split a UTF-8 byte sequence.
                return ""
            }
            return drain(final: true) + added.content
        }

        for scalar in token.unicodeScalars {
            guard let byte = Self.byteLevelScalarToByte[scalar.value] else {
                throw GFDetokenizerError.invalidByteLevelToken(id: id, token: token)
            }
            pendingBytes.append(byte)
        }
        return drain(final: false)
    }

    public mutating func flush() -> String {
        drain(final: true)
    }

    /// Emits all bytes except a valid but incomplete UTF-8 scalar at the end.
    /// `String(decoding:as:)` supplies the reference decoder's replacement
    /// behavior for malformed byte sequences and a truncated final scalar.
    @usableFromInline
    mutating func drain(final: Bool) -> String {
        let boundary = final ? pendingBytes.count : Self.incompleteSuffixStart(pendingBytes)
        guard boundary > 0 else { return "" }
        let result = pendingBytes[..<boundary].lossyUTF8String
        pendingBytes.removeFirst(boundary)
        return result
    }

    /// Returns the start of the trailing, structurally valid prefix of a UTF-8
    /// scalar. When there is no incomplete suffix the returned index is `count`.
    @usableFromInline
    static func incompleteSuffixStart(_ bytes: [UInt8]) -> Int {
        guard let last = bytes.last, last >= 0x80 else { return bytes.count }
        var lead = bytes.count - 1
        while lead > 0, bytes[lead] & 0xC0 == 0x80,
            bytes.count - lead <= 3
        {
            lead -= 1
        }
        let first = bytes[lead]
        let expected: Int
        switch first {
        case 0xC2...0xDF: expected = 2
        case 0xE0...0xEF: expected = 3
        case 0xF0...0xF4: expected = 4
        default: return bytes.count
        }
        let available = bytes.count - lead
        guard available < expected,
            bytes[(lead + 1)...].allSatisfy({ $0 & 0xC0 == 0x80 })
        else {
            return bytes.count
        }
        if available >= 2 {
            let second = bytes[lead + 1]
            if first == 0xE0, second < 0xA0 { return bytes.count }
            if first == 0xED, second > 0x9F { return bytes.count }
            if first == 0xF0, second < 0x90 { return bytes.count }
            if first == 0xF4, second > 0x8F { return bytes.count }
        }
        return lead
    }

    /// Inverse GPT-2 / Hugging Face ByteLevel byte-to-Unicode mapping.
    @usableFromInline
    static let byteLevelScalarToByte: [UInt32: UInt8] = {
        var bytes: [Int] = Array(0x21...0x7E) + Array(0xA1...0xAC) + Array(0xAE...0xFF)
        var scalars: [Int] = bytes
        var next = 0
        for byte in 0...255 where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(0x100 + next)
            next += 1
        }
        return Dictionary(
            uniqueKeysWithValues: zip(scalars, bytes).map {
                (UInt32($0.0), UInt8($0.1))
            })
    }()
}
