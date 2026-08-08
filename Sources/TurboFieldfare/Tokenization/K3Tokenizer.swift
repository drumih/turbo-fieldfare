import Foundation

public enum K3TokenizerError: Error, CustomStringConvertible {
    case unreadableVocabulary(String)
    case malformedVocabularyLine(Int)
    case invalidBase64(Int)
    case duplicateTokenBytes(Int)
    case duplicateRank(Int)
    case rankOutOfRange(Int)
    case mergeableRankCount(expected: Int, actual: Int)
    case missingByteToken(UInt8)
    case specialTokenTableMismatch(String)

    public var description: String {
        switch self {
        case .unreadableVocabulary(let path):
            return "cannot read tiktoken vocabulary at \(path)"
        case .malformedVocabularyLine(let line):
            return "malformed tiktoken vocabulary line \(line) (expected \"<base64> <rank>\")"
        case .invalidBase64(let line):
            return "invalid base64 token on tiktoken vocabulary line \(line)"
        case .duplicateTokenBytes(let line):
            return "duplicate token bytes on tiktoken vocabulary line \(line)"
        case .duplicateRank(let rank):
            return "duplicate tiktoken rank \(rank)"
        case .rankOutOfRange(let rank):
            return "tiktoken rank \(rank) out of range"
        case .mergeableRankCount(let expected, let actual):
            return "tiktoken vocabulary holds \(actual) mergeable ranks, expected \(expected)"
        case .missingByteToken(let byte):
            return "tiktoken vocabulary is missing the single-byte token 0x\(String(byte, radix: 16))"
        case .specialTokenTableMismatch(let detail):
            return "K3 special-token table mismatch: \(detail)"
        }
    }
}

/// Kimi K3 tokenizer: byte-level tiktoken BPE over the pinned `tiktoken.model`
/// ranks file, plus the 256-slot special-token table from the official
/// `tokenization_kimi.py` / `tokenizer_config.json`.
///
/// Encoding mirrors `tiktoken.Encoding.encode` with the K3 `pat_str`:
/// the text is split into pattern pieces with ICU (`NSRegularExpression`;
/// the pattern uses ICU-legal `&&` class intersections and `(?i:…)` groups,
/// matching the Rust `fancy-regex` semantics tiktoken relies on), and each
/// piece's UTF-8 bytes are merged by rank. `allowSpecial: true` additionally
/// maps literal special-token strings (`<|open|>`, `[BOS]`, …) to their ids,
/// exactly like tiktoken's `allowed_special="all"`; `allowSpecial: false`
/// never matches specials, like `disallowed_special=()`.
///
/// Decoding mirrors `tiktoken.Encoding.decode`: mergeable ids emit their
/// bytes, special ids emit their literal name, and the concatenation is
/// decoded as UTF-8 with replacement semantics. Out-of-range ids decode to
/// U+FFFD (tiktoken would raise; the runtime prefers a total function).
public struct K3Tokenizer: @unchecked Sendable {
    /// Mergeable ranks in `tiktoken.model`.
    public static let baseVocabSize = 163_584
    /// Reserved special slots directly above the mergeable ranks.
    public static let specialSlotCount = 256
    /// `baseVocabSize + specialSlotCount`.
    public static let totalVocabSize = baseVocabSize + specialSlotCount

    public let vocabSize: Int

    // Named special ids (pinned by tokenizer_config.json's added_tokens_decoder).
    public let bosID = 163_584          // [BOS]
    public let eosID = 163_585          // [EOS]
    public let endOfMessageID = 163_586 // <|end_of_msg|>
    public let openID = 163_587         // <|open|>
    public let closeID = 163_588        // <|close|>
    public let sepID = 163_589          // <|sep|>
    public let startHeaderID = 163_590  // [start_header_id]
    public let endHeaderID = 163_591    // [end_header_id]
    public let eotID = 163_593          // [EOT]
    public let mediaBeginID = 163_602   // <|media_begin|>
    public let mediaContentID = 163_603 // <|media_content|>
    public let mediaEndID = 163_604     // <|media_end|>
    public let mediaPadID = 163_605     // <|media_pad|>
    public let osAgentModeID = 163_649  // <osagent_mode>
    public let unkID = 163_838          // [UNK]
    public let padID = 163_839          // [PAD]

    private let ranks: [[UInt8]: Int]
    private let tokenBytes: [[UInt8]]
    private let specialNameToID: [String: Int]
    private let specialIDToName: [Int: String]
    private let specialPattern: NSRegularExpression
    private let pieceCache = K3PieceCache()

    /// The 16 named slots from `tokenizer_config.json`; every other reserved
    /// slot is `<|reserved_token_<id>|>` per `tokenization_kimi.py`.
    private static let namedSpecials: [(id: Int, name: String)] = [
        (163_584, "[BOS]"),
        (163_585, "[EOS]"),
        (163_586, "<|end_of_msg|>"),
        (163_587, "<|open|>"),
        (163_588, "<|close|>"),
        (163_589, "<|sep|>"),
        (163_590, "[start_header_id]"),
        (163_591, "[end_header_id]"),
        (163_593, "[EOT]"),
        (163_602, "<|media_begin|>"),
        (163_603, "<|media_content|>"),
        (163_604, "<|media_end|>"),
        (163_605, "<|media_pad|>"),
        (163_649, "<osagent_mode>"),
        (163_838, "[UNK]"),
        (163_839, "[PAD]"),
    ]

    /// `pat_str` from `tokenization_kimi.py`, verbatim. ICU accepts the
    /// `[\p{Lu}…\p{M}&&[^\p{Han}]]` intersections and `(?i:…)` groups as-is.
    private static let patternString = [
        #"\p{Han}+"#,
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#,
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#,
        #"\p{N}{1,3}"#,
        #" ?[^\s\p{L}\p{N}]+[\r\n]*"#,
        #"\s*[\r\n]+"#,
        #"\s+(?!\S)"#,
        #"\s+"#,
    ].joined(separator: "|")

    private static let tokenPattern: NSRegularExpression = {
        // The pattern is a compile-time constant; a failure here is a bug.
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: patternString)
    }()

    public init(vocabURL: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: vocabURL)
        } catch {
            throw K3TokenizerError.unreadableVocabulary(vocabURL.path)
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw K3TokenizerError.unreadableVocabulary(vocabURL.path)
        }

        var ranks: [[UInt8]: Int] = [:]
        ranks.reserveCapacity(Self.baseVocabSize)
        var lineNumber = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            lineNumber += 1
            guard let space = line.lastIndex(of: " ") else {
                throw K3TokenizerError.malformedVocabularyLine(lineNumber)
            }
            guard let rank = Int(line[line.index(after: space)...]) else {
                throw K3TokenizerError.malformedVocabularyLine(lineNumber)
            }
            guard let decoded = Data(base64Encoded: Data(line[..<space].utf8)) else {
                throw K3TokenizerError.invalidBase64(lineNumber)
            }
            let bytes = [UInt8](decoded)
            guard ranks.updateValue(rank, forKey: bytes) == nil else {
                throw K3TokenizerError.duplicateTokenBytes(lineNumber)
            }
        }
        guard ranks.count == Self.baseVocabSize else {
            throw K3TokenizerError.mergeableRankCount(
                expected: Self.baseVocabSize, actual: ranks.count)
        }

        var tokenBytes = [[UInt8]](repeating: [], count: ranks.count)
        for (bytes, rank) in ranks {
            guard tokenBytes.indices.contains(rank) else {
                throw K3TokenizerError.rankOutOfRange(rank)
            }
            guard tokenBytes[rank].isEmpty else {
                throw K3TokenizerError.duplicateRank(rank)
            }
            tokenBytes[rank] = bytes
        }
        for byte: UInt8 in 0...255 {
            guard ranks[[byte]] != nil else {
                throw K3TokenizerError.missingByteToken(byte)
            }
        }

        // Special-token table: named slots from the pinned table, the rest
        // <|reserved_token_<id>|>. Asserted against the pinned layout so a
        // bad edit fails loudly at load, not as a wrong token stream.
        var nameToID: [String: Int] = [:]
        var idToName: [Int: String] = [:]
        var namedIDs = Set<Int>()
        for entry in Self.namedSpecials {
            guard entry.id >= Self.baseVocabSize,
                  entry.id < Self.baseVocabSize + Self.specialSlotCount else {
                throw K3TokenizerError.specialTokenTableMismatch(
                    "\(entry.name) id \(entry.id) outside the reserved range")
            }
            guard namedIDs.insert(entry.id).inserted,
                  nameToID.updateValue(entry.id, forKey: entry.name) == nil else {
                throw K3TokenizerError.specialTokenTableMismatch(
                    "duplicate entry for \(entry.name)")
            }
            idToName[entry.id] = entry.name
        }
        guard namedIDs.count == 16 else {
            throw K3TokenizerError.specialTokenTableMismatch(
                "expected 16 named slots, found \(namedIDs.count)")
        }
        let pinned: [(Int, Int)] = [
            (163_584, bosID), (163_585, eosID), (163_586, endOfMessageID),
            (163_587, openID), (163_588, closeID), (163_589, sepID),
            (163_590, startHeaderID), (163_591, endHeaderID), (163_593, eotID),
            (163_602, mediaBeginID), (163_603, mediaContentID),
            (163_604, mediaEndID), (163_605, mediaPadID), (163_649, osAgentModeID),
            (163_838, unkID), (163_839, padID),
        ]
        for (tableID, accessorID) in pinned where tableID != accessorID {
            throw K3TokenizerError.specialTokenTableMismatch(
                "named id \(tableID) disagrees with accessor \(accessorID)")
        }
        for id in Self.baseVocabSize..<(Self.baseVocabSize + Self.specialSlotCount)
        where idToName[id] == nil {
            let name = "<|reserved_token_\(id)|>"
            idToName[id] = name
            nameToID[name] = id
        }

        // tiktoken scans specials leftmost-first; the alternation order only
        // matters for same-position matches, which the fixed-width reserved
        // names cannot produce. Id order mirrors the reference dict order.
        let alternation = (Self.baseVocabSize..<(Self.baseVocabSize + Self.specialSlotCount))
            .map { NSRegularExpression.escapedPattern(for: idToName[$0]!) }
            .joined(separator: "|")
        do {
            specialPattern = try NSRegularExpression(pattern: alternation)
        } catch {
            throw K3TokenizerError.specialTokenTableMismatch(
                "special-token pattern failed to compile: \(error)")
        }

        self.ranks = ranks
        self.tokenBytes = tokenBytes
        self.specialNameToID = nameToID
        self.specialIDToName = idToName
        self.vocabSize = Self.totalVocabSize
    }

    /// Id of a special token by literal name (`"<|open|>"` → 163587); nil for
    /// ordinary text.
    public func specialID(forName name: String) -> Int? {
        specialNameToID[name]
    }

    /// Literal name of a special id (163587 → `"<|open|>"`); nil for
    /// mergeable-rank ids.
    public func specialName(forID id: Int) -> String? {
        specialIDToName[id]
    }

    /// Encode UTF-8 text to token ids. `allowSpecial: true` maps literal
    /// special-token strings to their ids (tiktoken `allowed_special="all"`);
    /// `false` encodes them as ordinary text (tiktoken `disallowed_special=()`).
    ///
    /// Marked `throws` for API stability; loading validates the vocabulary so
    /// encoding a `String` cannot actually fail.
    public func encode(_ text: String, allowSpecial: Bool = true) throws -> [Int] {
        if !allowSpecial {
            return encodeOrdinary(text)
        }
        var ids: [Int] = []
        let full = NSRange(text.startIndex..., in: text)
        var cursor = text.startIndex
        for match in specialPattern.matches(in: text, range: full) {
            guard let range = Range(match.range, in: text) else { continue }
            if range.lowerBound > cursor {
                ids.append(contentsOf: encodeOrdinary(String(text[cursor..<range.lowerBound])))
            }
            // Every alternation branch comes from the table, so the lookup
            // cannot miss.
            ids.append(specialNameToID[String(text[range])]!)
            cursor = range.upperBound
        }
        if cursor < text.endIndex {
            ids.append(contentsOf: encodeOrdinary(String(text[cursor...])))
        }
        return ids
    }

    /// Decode token ids to text. Mergeable ids emit their token bytes,
    /// special ids their literal names, unknown ids U+FFFD; the concatenated
    /// bytes decode as UTF-8 with replacement (tiktoken `errors="replace"`).
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if id >= 0, id < Self.baseVocabSize {
                bytes.append(contentsOf: tokenBytes[id])
            } else if let name = specialIDToName[id] {
                bytes.append(contentsOf: [UInt8](name.utf8))
            } else {
                bytes.append(contentsOf: [0xEF, 0xBF, 0xBD]) // U+FFFD
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - BPE

    /// Pattern-split then byte-pair encode; specials never match here.
    private func encodeOrdinary(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        var ids: [Int] = []
        let full = NSRange(text.startIndex..., in: text)
        for match in Self.tokenPattern.matches(in: text, range: full) {
            guard let range = Range(match.range, in: text) else { continue }
            ids.append(contentsOf: bytePairEncode([UInt8](text[range].utf8)))
        }
        return ids
    }

    /// Byte-pair encode one pattern piece. O(n log n): a min-heap over
    /// adjacent-pair ranks drives the merge order (ties break leftmost, like
    /// tiktoken's Rust core), and a linked list unlinks merged parts so each
    /// merge recomputes only its two neighboring pairs. Results are cached by
    /// piece, which is where repeated words/tags pay off across encodes.
    private func bytePairEncode(_ bytes: [UInt8]) -> [Int] {
        if let cached = pieceCache.lookup(bytes) {
            return cached
        }
        let result = bytePairMerge(bytes)
        pieceCache.store(bytes, result)
        return result
    }

    private func bytePairMerge(_ bytes: [UInt8]) -> [Int] {
        let count = bytes.count
        if count == 0 { return [] }
        if let rank = ranks[bytes] { return [rank] }

        // Part i covers bytes[partStart[i] ..< end(of: i)]; parts stay
        // contiguous, so the pair (i, next[i]) is bytes[partStart[i] ..< end(of: next[i])].
        let partStart = [Int](0..<count)
        var previous = [Int](repeating: 0, count: count)
        var next = [Int](repeating: 0, count: count)
        var alive = [Bool](repeating: true, count: count)
        for i in 0..<count {
            previous[i] = i - 1
            next[i] = i + 1 < count ? i + 1 : -1
        }

        func end(of i: Int) -> Int {
            next[i] >= 0 ? partStart[next[i]] : count
        }
        func pairRank(left i: Int) -> Int? {
            let j = next[i]
            guard j >= 0 else { return nil }
            return ranks[[UInt8](bytes[partStart[i]..<end(of: j)])]
        }

        var heap = K3MergeHeap()
        for i in 0..<(count - 1) {
            if let rank = pairRank(left: i) {
                heap.insert(rank: rank, index: i)
            }
        }

        while let (rank, i) = heap.removeMin() {
            let j = next[i]
            // Stale heap entries fail the rank re-check after an earlier
            // merge rewrote this pair; merged-away parts fail the liveness
            // check (without it their writes would corrupt live back-links).
            guard j >= 0, alive[i], alive[j], pairRank(left: i) == rank else { continue }
            next[i] = next[j]
            if next[i] >= 0 { previous[next[i]] = i }
            alive[j] = false
            if let leftRank = pairRank(left: i) {
                heap.insert(rank: leftRank, index: i)
            }
            let p = previous[i]
            if p >= 0, let leftRank = pairRank(left: p) {
                heap.insert(rank: leftRank, index: p)
            }
        }

        var result: [Int] = []
        var i = 0
        while i >= 0 {
            // Invariant: every surviving part is a mergeable token — singles
            // are validated at load and merges only produce ranked bytes.
            result.append(ranks[[UInt8](bytes[partStart[i]..<end(of: i)])]!)
            i = next[i]
        }
        return result
    }
}

/// Lock-protected piece → ids memo. Cleared wholesale past a bound; entries
/// are tiny and chat rendering reuses the same words and structural tags.
private final class K3PieceCache: @unchecked Sendable {
    private let lock = NSLock()
    private var map: [[UInt8]: [Int]] = [:]

    func lookup(_ piece: [UInt8]) -> [Int]? {
        lock.lock()
        defer { lock.unlock() }
        return map[piece]
    }

    func store(_ piece: [UInt8], _ ids: [Int]) {
        lock.lock()
        defer { lock.unlock() }
        if map.count > 65_536 { map.removeAll(keepingCapacity: true) }
        map[piece] = ids
    }
}

/// Min-heap of (rank, part index), ordered by rank then leftmost index —
/// the same tie-break as tiktoken's `_byte_pair_merge`.
private struct K3MergeHeap {
    private var items: [(rank: Int, index: Int)] = []

    var isEmpty: Bool { items.isEmpty }

    mutating func insert(rank: Int, index: Int) {
        items.append((rank, index))
        var child = items.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            guard Self.less(items[child], items[parent]) else { break }
            items.swapAt(child, parent)
            child = parent
        }
    }

    mutating func removeMin() -> (rank: Int, index: Int)? {
        guard let first = items.first else { return nil }
        if items.count == 1 {
            items.removeAll()
            return first
        }
        items[0] = items.removeLast()
        var parent = 0
        while true {
            var smallest = parent
            let left = 2 * parent + 1
            let right = left + 1
            if left < items.count, Self.less(items[left], items[smallest]) { smallest = left }
            if right < items.count, Self.less(items[right], items[smallest]) { smallest = right }
            guard smallest != parent else { break }
            items.swapAt(parent, smallest)
            parent = smallest
        }
        return first
    }

    private static func less(_ a: (rank: Int, index: Int), _ b: (rank: Int, index: Int)) -> Bool {
        a.rank != b.rank ? a.rank < b.rank : a.index < b.index
    }
}
