import Foundation
import TinyTitanCore

/// Inkling 2 Mini's tokenizer: the checkpoint's `tokenizer.json` is a
/// byte-level BPE with a Unicode split pattern and 60 added/special tokens.
/// This is a faithful encode/decode of that file, not a Qwen tokenizer.
public struct Inkling2Tokenizer: Sendable {
    public struct AddedToken: Sendable, Equatable {
        public let id: Int
        public let content: String
        public let special: Bool
    }

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case missing(String)
        case invalid(String)
        case unknownToken(String)

        public var description: String {
            switch self {
            case .missing(let detail): "missing \(detail)"
            case .invalid(let detail): "invalid tokenizer: \(detail)"
            case .unknownToken(let token): "tokenizer has no id for \(token)"
            }
        }
    }

    private let vocabulary: [String: Int]
    private let idToToken: [Int: String]
    private let mergeRanks: [String: Int]
    private let added: [AddedToken]
    private let addedById: [Int: AddedToken]
    private let addedSorted: [AddedToken]
    private let pattern: String?
    private let byteToUnicode: [UInt8: Character]
    private let unicodeToByte: [Character: UInt8]

    public init(directory: URL) throws {
        let data = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("tokenizer.json"),
            maxBytes: 64 * 1024 * 1024)
        try self.init(json: data)
    }

    public init(json: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: json) as? [String: Any],
            let model = root["model"] as? [String: Any]
        else {
            throw Error.invalid("not a tokenizer.json object")
        }
        guard model["type"] as? String == "BPE" else {
            throw Error.invalid("expected a BPE model")
        }
        guard let vocabObject = model["vocab"] as? [String: Int], !vocabObject.isEmpty else {
            throw Error.missing("model.vocab")
        }
        let merges: [String]
        if let strings = model["merges"] as? [String] {
            merges = strings
        } else if let pairs = model["merges"] as? [[String]] {
            merges = pairs.map { $0.joined(separator: " ") }
        } else {
            throw Error.missing("model.merges")
        }
        guard !merges.isEmpty else { throw Error.missing("model.merges") }
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            guard ranks[merge] == nil else {
                throw Error.invalid("duplicate merge \(merge)")
            }
            ranks[merge] = rank
        }
        var addedTokens: [AddedToken] = []
        if let rawAdded = root["added_tokens"] as? [[String: Any]] {
            for entry in rawAdded {
                guard let id = entry["id"] as? Int,
                    let content = entry["content"] as? String
                else {
                    throw Error.invalid("added token without id/content")
                }
                addedTokens.append(
                    AddedToken(
                        id: id, content: content,
                        special: entry["special"] as? Bool ?? false))
            }
        }
        self.vocabulary = vocabObject
        self.idToToken = Dictionary(
            uniqueKeysWithValues: vocabObject.map { ($0.value, $0.key) })
        self.mergeRanks = ranks
        self.added = addedTokens
        self.addedById = Dictionary(
            uniqueKeysWithValues: addedTokens.map { ($0.id, $0) })
        self.addedSorted =
            addedTokens
            .filter(\.special)
            .sorted { $0.content.count > $1.content.count }
        self.pattern = Self.splitPattern(root["pre_tokenizer"])
        let mapping = Self.byteEncoder()
        self.byteToUnicode = mapping.forward
        self.unicodeToByte = mapping.reverse
    }

    public func encode(_ text: String) throws -> [Int] {
        var ids: [Int] = []
        var remaining = text[...]
        while !remaining.isEmpty {
            if let (token, range) = firstSpecial(in: remaining) {
                let prefix = remaining[..<range.lowerBound]
                if !prefix.isEmpty {
                    ids.append(contentsOf: try encodeOrdinary(String(prefix)))
                }
                ids.append(token.id)
                remaining = remaining[range.upperBound...]
            } else {
                ids.append(contentsOf: try encodeOrdinary(String(remaining)))
                remaining = ""
            }
        }
        return ids
    }

    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if let special = addedById[id] {
                bytes.append(contentsOf: Array(special.content.utf8))
                continue
            }
            guard let token = idToToken[id] else {
                continue
            }
            for scalar in token.unicodeScalars {
                if let byte = unicodeToByte[Character(scalar)] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: String(scalar).utf8)
                }
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    private func encodeOrdinary(_ text: String) throws -> [Int] {
        guard let pattern else {
            return try bpe(Self.byteLevel(text, map: byteToUnicode))
        }
        let regex = try NSRegularExpression(pattern: pattern)
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        if matches.isEmpty {
            return try bpe(Self.byteLevel(text, map: byteToUnicode))
        }
        var ids: [Int] = []
        for match in matches {
            let chunk = ns.substring(with: match.range)
            ids.append(contentsOf: try bpe(Self.byteLevel(chunk, map: byteToUnicode)))
        }
        return ids
    }

    private func bpe(_ symbols: [String]) throws -> [Int] {
        guard !symbols.isEmpty else { return [] }
        var parts = symbols
        while parts.count > 1 {
            var bestRank = Int.max
            for index in 0..<(parts.count - 1) {
                let key = parts[index] + " " + parts[index + 1]
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                }
            }
            if bestRank == Int.max { break }
            var merged: [String] = []
            merged.reserveCapacity(parts.count)
            var index = 0
            while index < parts.count {
                if index + 1 < parts.count,
                    mergeRanks[parts[index] + " " + parts[index + 1]] == bestRank
                {
                    merged.append(parts[index] + parts[index + 1])
                    index += 2
                } else {
                    merged.append(parts[index])
                    index += 1
                }
            }
            parts = merged
        }
        return try parts.map { token in
            guard let id = vocabulary[token] else {
                throw Error.unknownToken(token)
            }
            return id
        }
    }

    private func firstSpecial(
        in text: Substring
    ) -> (AddedToken, Range<Substring.Index>)? {
        var earliest: (AddedToken, Range<Substring.Index>)?
        for token in addedSorted {
            guard let range = text.range(of: token.content) else { continue }
            if let current = earliest, range.lowerBound >= current.1.lowerBound {
                continue
            }
            earliest = (token, range)
        }
        return earliest
    }

    private static func splitPattern(_ preTokenizer: Any?) -> String? {
        guard let dictionary = preTokenizer as? [String: Any] else { return nil }
        if let pattern = dictionary["pattern"] as? [String: Any],
            let regex = pattern["Regex"] as? String
        {
            return regex
        }
        if let sequence = dictionary["pretokenizers"] as? [[String: Any]] {
            for item in sequence {
                if let found = splitPattern(item) { return found }
            }
        }
        return nil
    }

    private static func byteLevel(
        _ text: String, map: [UInt8: Character]
    ) -> [String] {
        var symbols: [String] = []
        for byte in text.utf8 {
            if let character = map[byte] {
                symbols.append(String(character))
            }
        }
        return symbols
    }

    private static func byteEncoder() -> (
        forward: [UInt8: Character], reverse: [Character: UInt8]
    ) {
        var bytes: [Int] = []
        var chars: [Int] = []
        for value in 33...126 { bytes.append(value); chars.append(value) }
        for value in 161...172 { bytes.append(value); chars.append(value) }
        for value in 174...255 { bytes.append(value); chars.append(value) }
        var extra = 0
        for value in 0...255 where !bytes.contains(value) {
            bytes.append(value)
            chars.append(256 + extra)
            extra += 1
        }
        var forward: [UInt8: Character] = [:]
        var reverse: [Character: UInt8] = [:]
        for (byte, scalar) in zip(bytes, chars) {
            guard let unicode = UnicodeScalar(UInt32(scalar)) else { continue }
            let character = Character(unicode)
            forward[UInt8(byte)] = character
            reverse[character] = UInt8(byte)
        }
        return (forward, reverse)
    }
}
