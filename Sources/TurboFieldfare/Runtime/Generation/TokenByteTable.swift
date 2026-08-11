import Foundation

/// Token id → raw UTF-8 bytes with the piece-table quirks both grammar filters
/// need: the SentencePiece space marker `▁` maps to a space, `<0xNN>`
/// byte-fallback pieces carry one raw byte, and whole-piece `<...>` markers are
/// blocked because the detokenizer strips them, which would desync the
/// automaton from the emitted text. `allowedMarkerIDs` opts specific markers
/// back in: the tool-call grammar needs `<|"|>` (the tokenizer's escape token)
/// as a literal five-byte string delimiter.
public final class TokenByteTable {
    public typealias PieceLookup = (Int32) -> String?

    private static let spaceMarker = "\u{2581}"

    private let piece: PieceLookup
    private let blockedIDs: Set<Int32>
    private let allowedMarkerIDs: Set<Int32>
    private var byteCache: [Int32: [UInt8]] = [:]
    private var blockedCache: Set<Int32> = []

    public init(pieceLookup: @escaping PieceLookup,
                blockedIDs: Set<Int32>,
                allowedMarkerIDs: Set<Int32> = []) {
        self.piece = pieceLookup
        self.blockedIDs = blockedIDs
        self.allowedMarkerIDs = allowedMarkerIDs
    }

    /// nil when the id is blocked, unknown, empty, or marker-shaped.
    public func bytes(for id: Int32) -> [UInt8]? {
        if blockedCache.contains(id) { return nil }
        if let cached = byteCache[id] { return cached }
        guard !blockedIDs.contains(id), let raw = piece(id), !raw.isEmpty else {
            blockedCache.insert(id)
            return nil
        }
        let resolved: [UInt8]?
        if let byte = Self.byteFallback(raw) {
            resolved = [byte]
        } else if Self.looksLikeMarker(raw), !allowedMarkerIDs.contains(id) {
            resolved = nil
        } else {
            resolved = Array(raw.replacingOccurrences(of: Self.spaceMarker,
                                                      with: " ").utf8)
        }
        guard let resolved else {
            blockedCache.insert(id)
            return nil
        }
        byteCache[id] = resolved
        return resolved
    }

    private static func byteFallback(_ token: String) -> UInt8? {
        guard token.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">") else {
            return nil
        }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }

    /// Whole-piece `<...>` shapes (e.g. `<unused12>`, `<start_of_image>`) are
    /// added tokens that signal a derailed generation; never let them through
    /// unless the caller opted the id in. This also blocks the vocab's ~97
    /// legitimate single-piece HTML tags (`<td>`, `<div>`) — accepted cost: the
    /// model can still spell them from smaller pieces.
    private static func looksLikeMarker(_ token: String) -> Bool {
        guard token.count > 2, token.hasPrefix("<"), token.hasSuffix(">") else {
            return false
        }
        return !token.dropFirst().dropLast().contains { $0 == "<" || $0 == ">" }
    }
}

/// A grammar that can veto sampled tokens inside the raw-completion loop.
public protocol TokenGrammarFilter: AnyObject {
    /// True when the token may be emitted next; state is committed only on
    /// success, so speculative calls during the fallback scan are free.
    func tryAccept(_ id: Int32) -> Bool
    /// Recorded by the loop when the GPU-sampled token was rejected.
    func noteVeto()
    var vetoCount: Int { get }
}
