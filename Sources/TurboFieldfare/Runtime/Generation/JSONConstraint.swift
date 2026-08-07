import Foundation

/// Byte-level JSON acceptor for constrained decoding.
///
/// Operates on raw UTF-8 bytes so tokenizer pieces that split multi-byte
/// codepoints (byte-fallback `<0xNN>` tokens, SentencePiece surface splits)
/// never confuse it: inside a string every byte ≥ 0x20 is legal, including
/// UTF-8 continuation bytes, so a codepoint split across two tokens is
/// accepted byte by byte.
///
/// The grammar is RFC 8259 JSON with one permissive edge: `\uXXXX` escapes are
/// validated as four hex digits without pairing surrogates. Root may be any
/// JSON value. A root-level bare number is the one case where "complete"
/// cannot be decided without lookahead (more digits may follow); `canStop`
/// reports it as stoppable while `isComplete` stays false until a terminator
/// byte lands.
public struct JSONByteAutomaton: Sendable, Equatable {
    /// Containers deeper than this are rejected; matches llama.cpp's default
    /// guard against runaway nesting.
    public static let maxDepth = 128

    private enum NumberPhase: Equatable {
        case afterMinus
        case leadingZero
        case intDigits
        case afterDot
        case fracDigits
        case afterExpMark
        case afterExpSign
        case expDigits

        /// Phases at which the number is a valid JSON number if it ends here.
        var isTerminal: Bool {
            switch self {
            case .leadingZero, .intDigits, .fracDigits, .expDigits: return true
            case .afterMinus, .afterDot, .afterExpMark, .afterExpSign: return false
            }
        }
    }

    private enum Container: Equatable { case object, array }

    private enum Mode: Equatable {
        case expectValue
        case expectValueOrClose   // right after '['
        case expectKeyOrClose     // right after '{'
        case expectKey            // after ',' inside an object
        case expectColon
        case inString(isKey: Bool)
        case stringEscape(isKey: Bool)
        case stringUnicode(isKey: Bool, remaining: Int)
        case inNumber(NumberPhase)
        case inLiteral([UInt8])   // remaining bytes of true/false/null
        case afterValue
        case complete
    }

    private var stack: [Container] = []
    private var mode: Mode = .expectValue

    public init() {}

    /// The root value is closed; only trailing whitespace may follow.
    public var isComplete: Bool { mode == .complete }

    /// EOS would leave valid JSON here. True at `complete`, and also for a
    /// root-level bare number sitting on a terminal phase.
    public var canStop: Bool {
        if mode == .complete { return true }
        if case .inNumber(let phase) = mode, stack.isEmpty { return phase.isTerminal }
        return false
    }

    /// Feed one byte. Returns false — leaving the automaton unchanged — when
    /// the byte cannot extend a valid JSON document.
    public mutating func consume(_ byte: UInt8) -> Bool {
        switch mode {
        case .expectValue, .expectValueOrClose:
            if isWhitespace(byte) { return true }
            if byte == UInt8(ascii: "]"), mode == .expectValueOrClose {
                return closeContainer()
            }
            return startValue(byte)

        case .expectKeyOrClose:
            if isWhitespace(byte) { return true }
            if byte == UInt8(ascii: "\"") { mode = .inString(isKey: true); return true }
            if byte == UInt8(ascii: "}") { return closeContainer() }
            return false

        case .expectKey:
            if isWhitespace(byte) { return true }
            if byte == UInt8(ascii: "\"") { mode = .inString(isKey: true); return true }
            return false

        case .expectColon:
            if isWhitespace(byte) { return true }
            if byte == UInt8(ascii: ":") { mode = .expectValue; return true }
            return false

        case .inString(let isKey):
            switch byte {
            case UInt8(ascii: "\""):
                mode = isKey ? .expectColon : endedValueMode()
                return true
            case UInt8(ascii: "\\"):
                mode = .stringEscape(isKey: isKey)
                return true
            case 0x00..<0x20:
                return false
            default:
                return true
            }

        case .stringEscape(let isKey):
            switch byte {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"),
                 UInt8(ascii: "b"), UInt8(ascii: "f"), UInt8(ascii: "n"),
                 UInt8(ascii: "r"), UInt8(ascii: "t"):
                mode = .inString(isKey: isKey)
                return true
            case UInt8(ascii: "u"):
                mode = .stringUnicode(isKey: isKey, remaining: 4)
                return true
            default:
                return false
            }

        case .stringUnicode(let isKey, let remaining):
            guard isHexDigit(byte) else { return false }
            mode = remaining == 1
                ? .inString(isKey: isKey)
                : .stringUnicode(isKey: isKey, remaining: remaining - 1)
            return true

        case .inNumber(let phase):
            if let next = numberPhase(phase, byte) {
                mode = .inNumber(next)
                return true
            }
            guard phase.isTerminal else { return false }
            // The number ends where a non-number byte lands; close the value
            // and let the terminator byte act on the new mode.
            let saved = mode
            mode = endedValueMode()
            if consume(byte) { return true }
            mode = saved
            return false

        case .inLiteral(let remaining):
            guard byte == remaining.first else { return false }
            let rest = Array(remaining.dropFirst())
            mode = rest.isEmpty ? endedValueMode() : .inLiteral(rest)
            return true

        case .afterValue:
            if isWhitespace(byte) { return true }
            switch (stack.last, byte) {
            case (.object, UInt8(ascii: ",")): mode = .expectKey; return true
            case (.object, UInt8(ascii: "}")): return closeContainer()
            case (.array, UInt8(ascii: ",")): mode = .expectValue; return true
            case (.array, UInt8(ascii: "]")): return closeContainer()
            default: return false
            }

        case .complete:
            return isWhitespace(byte)
        }
    }

    // MARK: - Transitions

    private mutating func startValue(_ byte: UInt8) -> Bool {
        switch byte {
        case UInt8(ascii: "\""):
            mode = .inString(isKey: false)
        case UInt8(ascii: "{"):
            guard stack.count < Self.maxDepth else { return false }
            stack.append(.object)
            mode = .expectKeyOrClose
        case UInt8(ascii: "["):
            guard stack.count < Self.maxDepth else { return false }
            stack.append(.array)
            mode = .expectValueOrClose
        case UInt8(ascii: "t"):
            mode = .inLiteral(Array("rue".utf8))
        case UInt8(ascii: "f"):
            mode = .inLiteral(Array("alse".utf8))
        case UInt8(ascii: "n"):
            mode = .inLiteral(Array("ull".utf8))
        case UInt8(ascii: "-"):
            mode = .inNumber(.afterMinus)
        case UInt8(ascii: "0"):
            mode = .inNumber(.leadingZero)
        case UInt8(ascii: "1")...UInt8(ascii: "9"):
            mode = .inNumber(.intDigits)
        default:
            return false
        }
        return true
    }

    private mutating func closeContainer() -> Bool {
        stack.removeLast()
        mode = stack.isEmpty ? .complete : .afterValue
        return true
    }

    private func endedValueMode() -> Mode {
        stack.isEmpty ? .complete : .afterValue
    }

    private func numberPhase(_ phase: NumberPhase, _ byte: UInt8) -> NumberPhase? {
        let isDigit = (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
        switch phase {
        case .afterMinus:
            guard isDigit else { return nil }
            return byte == UInt8(ascii: "0") ? .leadingZero : .intDigits
        case .leadingZero:
            if byte == UInt8(ascii: ".") { return .afterDot }
            if byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") { return .afterExpMark }
            return nil
        case .intDigits:
            if isDigit { return .intDigits }
            if byte == UInt8(ascii: ".") { return .afterDot }
            if byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") { return .afterExpMark }
            return nil
        case .afterDot:
            return isDigit ? .fracDigits : nil
        case .fracDigits:
            if isDigit { return .fracDigits }
            if byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") { return .afterExpMark }
            return nil
        case .afterExpMark:
            if isDigit { return .expDigits }
            if byte == UInt8(ascii: "+") || byte == UInt8(ascii: "-") { return .afterExpSign }
            return nil
        case .afterExpSign:
            return isDigit ? .expDigits : nil
        case .expDigits:
            return isDigit ? .expDigits : nil
        }
    }

    private func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }

    private func isHexDigit(_ byte: UInt8) -> Bool {
        (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
            || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains(byte)
            || (UInt8(ascii: "A")...UInt8(ascii: "F")).contains(byte)
    }
}

/// Filters sampled token ids through `JSONByteAutomaton` for one generation.
///
/// Token → bytes goes through the tokenizer's piece table, not `decode()`:
/// SentencePiece pieces carry the space marker `▁` (mapped to a space) and
/// byte-fallback pieces `<0xNN>` carry one raw byte. Special/control tokens
/// (turn, channel, tool markers) are never valid JSON output and are blocked
/// by id; unknown `<...>` marker-shaped pieces are blocked by pattern because
/// the detokenizer strips them from the output, which would desync the
/// automaton from the emitted text.
public final class JSONTokenFilter {
    public typealias PieceLookup = (Int32) -> String?

    private let piece: PieceLookup
    private let blockedIDs: Set<Int32>
    private let stopIDs: Set<Int32>
    private var automaton = JSONByteAutomaton()
    private var byteCache: [Int32: [UInt8]] = [:]
    private var blockedCache: Set<Int32> = []

    public convenience init(tokenizer: GFTokenizer) {
        let underlying = tokenizer.tokenizer
        self.init(
            pieceLookup: { underlying.convertIdToToken(Int($0)) },
            blockedIDs: [tokenizer.bosID, tokenizer.padID,
                         tokenizer.toolCallStartID, tokenizer.toolCallEndID,
                         tokenizer.toolResponseID, tokenizer.toolResponseEndID,
                         tokenizer.channelStartID, tokenizer.channelEndID],
            stopIDs: tokenizer.stopTokenIDs)
    }

    public init(pieceLookup: @escaping PieceLookup,
                blockedIDs: Set<Int32>,
                stopIDs: Set<Int32>) {
        self.piece = pieceLookup
        self.blockedIDs = blockedIDs.union(stopIDs)
        self.stopIDs = stopIDs
    }

    public var isComplete: Bool { automaton.isComplete }

    /// True when the token may be emitted next: stop tokens pass only where
    /// EOS would leave valid JSON; content tokens pass when every byte
    /// advances the automaton (committed on success).
    public func tryAccept(_ id: Int32) -> Bool {
        if stopIDs.contains(id) { return automaton.canStop }
        guard let bytes = bytes(for: id), !bytes.isEmpty else { return false }
        var candidate = automaton
        for byte in bytes {
            guard candidate.consume(byte) else { return false }
        }
        automaton = candidate
        return true
    }

    // MARK: - Token bytes

    private static let spaceMarker = "\u{2581}"

    private func bytes(for id: Int32) -> [UInt8]? {
        if blockedCache.contains(id) { return nil }
        if let cached = byteCache[id] { return cached }
        guard !blockedIDs.contains(id), let raw = piece(id), !raw.isEmpty else {
            blockedCache.insert(id)
            return nil
        }
        let resolved: [UInt8]?
        if let byte = Self.byteFallback(raw) {
            resolved = [byte]
        } else if Self.looksLikeMarker(raw) {
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
    /// added tokens the detokenizer strips; never let them through.
    private static func looksLikeMarker(_ token: String) -> Bool {
        guard token.count > 2, token.hasPrefix("<"), token.hasSuffix(">") else {
            return false
        }
        return !token.dropFirst().dropLast().contains { $0 == "<" || $0 == ">" }
    }
}
