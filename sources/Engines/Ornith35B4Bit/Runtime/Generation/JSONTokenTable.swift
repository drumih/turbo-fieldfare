import Foundation
import Tokenizers

/// The vocabulary as byte strings, for the grammar to walk.
///
/// Built once per loaded model, not once per request: it is 248k `String`
/// lookups through the tokenizer plus the ByteLevel inverse mapping, which is
/// milliseconds of work that never changes while the model is resident.
///
/// Tokens are stored flat (one byte array plus offsets) because the mask builder
/// touches every one of them for every position it has not seen before; 248k
/// separate small arrays would spend the time in allocator traffic instead of
/// in the grammar.
///
/// A token that carries no grammar bytes -- a special token, an added marker
/// like `<|im_end|>`, or a token the ByteLevel mapping does not describe -- is
/// stored empty. Such a token is never allowed by a constraint: emitting it
/// would advance the document by nothing, which is exactly the loop a grammar
/// exists to prevent.
public struct JSONTokenTable: Sendable {
    /// Ids below this have a byte string in the table.
    public let vocab: Int
    /// Every token's bytes, concatenated.
    private let storage: [UInt8]
    /// `offsets[id]..<offsets[id + 1]` slices `storage`.
    private let offsets: [Int]
    /// Ids grouped by their first byte: a position only has to look at the
    /// tokens whose first byte the grammar can still accept.
    private let buckets: [[Int32]]
    /// The tokens that end a turn. A finished document may only be padded with
    /// whitespace or stopped, so these are the whole allowed set there.
    public let stopTokens: Set<Int32>

    /// Build from an installed tokenizer.
    public init(tokenizer: GFTokenizer) {
        let vocab = max(tokenizer.vocabSize, 1)
        let markers = tokenizer.nonByteTokenIDs
        var storage: [UInt8] = []
        storage.reserveCapacity(vocab * 4)
        var offsets: [Int] = []
        offsets.reserveCapacity(vocab + 1)
        var buckets = [[Int32]](repeating: [], count: 256)
        for id in 0..<vocab {
            offsets.append(storage.count)
            let value = Int32(id)
            guard !markers.contains(value),
                let text = tokenizer.tokenizer.convertIdToToken(id), !text.isEmpty
            else {
                continue
            }
            var bytes: [UInt8] = []
            bytes.reserveCapacity(text.utf8.count)
            var literal = true
            for scalar in text.unicodeScalars {
                guard let byte = GFDetokenizer.byteLevelScalarToByte[scalar.value] else {
                    literal = false
                    break
                }
                bytes.append(byte)
            }
            guard literal, !bytes.isEmpty else { continue }
            buckets[Int(bytes[0])].append(value)
            storage.append(contentsOf: bytes)
        }
        offsets.append(storage.count)
        self.vocab = vocab
        self.storage = storage
        self.offsets = offsets
        self.buckets = buckets
        self.stopTokens = tokenizer.stopTokenIDs
    }

    /// Build from explicit entries. This is the seam the tests use: the
    /// grammar's behaviour does not depend on a real tokenizer, only on the
    /// byte string each id stands for. Ids not named in `entries` have no
    /// bytes, exactly like a special token.
    public init(
        vocab: Int, entries: [(id: Int32, bytes: [UInt8])],
        stopTokens: Set<Int32> = []
    ) {
        let highest = entries.map { Int($0.id) }.max() ?? -1
        let count = max(vocab, highest + 1, 1)
        let byID = Dictionary(grouping: entries, by: \.id)
        var storage: [UInt8] = []
        var offsets = [Int](repeating: 0, count: count + 1)
        var buckets = [[Int32]](repeating: [], count: 256)
        var running = 0
        for id in 0..<count {
            offsets[id] = running
            guard let bytes = byID[Int32(id)]?.first?.bytes, !bytes.isEmpty else { continue }
            buckets[Int(bytes[0])].append(Int32(id))
            storage.append(contentsOf: bytes)
            running += bytes.count
        }
        offsets[count] = running
        self.vocab = count
        self.storage = storage
        self.offsets = offsets
        self.buckets = buckets
        self.stopTokens = stopTokens
    }

    /// The bytes a token stands for; empty when it has none.
    public func bytes(of id: Int32) -> ArraySlice<UInt8> {
        let index = Int(id)
        guard index >= 0, index + 1 < offsets.count, index < vocab else { return [] }
        return storage[offsets[index]..<offsets[index + 1]]
    }

    /// Ids whose first byte is `byte`. Empty when the table holds none.
    func tokens(startingWith byte: UInt8) -> [Int32] { buckets[Int(byte)] }

    /// Every byte a token in this position could start with, in order.
    var populatedFirstBytes: [UInt8] {
        (0..<256).compactMap { buckets[$0].isEmpty ? nil : UInt8($0) }
    }
}
