import Foundation

/// Byte-level JSON acceptor for constrained decoding.
///
/// Operates on raw UTF-8 bytes so tokenizer pieces that split multi-byte
/// codepoints (byte-fallback `<0xNN>` tokens, SentencePiece surface splits)
/// never confuse it: inside a string every byte ≥ 0x20 is legal, including
/// UTF-8 continuation bytes, so a codepoint split across two tokens is
/// accepted byte by byte.
///
/// The grammar is RFC 8259 JSON with surrogate pairing enforced in `\uXXXX`
/// escapes: a high surrogate must be followed immediately by `\u` + a low
/// surrogate, and a lone low surrogate is rejected — Foundation's JSON
/// parsers refuse lone surrogates, so allowing them would break the
/// decodable-output guarantee. Root may be any JSON value. A root-level bare number is the one case where "complete"
/// cannot be decided without lookahead (more digits may follow); `canStop`
/// reports it as stoppable while `isComplete` stays false until a terminator
/// byte lands.
///
/// Inside strings the automaton validates UTF-8 sequence structure (lead and
/// continuation ranges per the WHATWG table, rejecting overlongs, surrogates
/// and > U+10FFFF), so a document it completes is always decodable text — a
/// grammar-driven fallback can never splice invalid bytes into a string.
public struct JSONByteAutomaton: Sendable, Equatable {
    /// Containers deeper than this are rejected; matches llama.cpp's default
    /// guard against runaway nesting. It also bounds the unbounded recursion in
    /// `GemmaToolCallParser`'s `object()`/`value()`.
    public static let maxDepth = 128

    /// Which language the automaton accepts. `.strict` is RFC 8259 — the
    /// `--force-json` path, unchanged. `.gemmaToolArguments` is the tool-call
    /// argument dialect the Gemma 4 chat template renders and
    /// `GemmaToolCallParser` reads: an object root with BARE keys at every
    /// depth, strings delimited by the `<|"|>` escape token instead of quotes,
    /// and bounded numbers with no exponent.
    public enum Dialect: Sendable, Equatable {
        case strict
        case gemmaToolArguments
    }

    /// The tokenizer's escape token, used verbatim as the dialect's string
    /// delimiter. It has no proper border, so its KMP failure function is
    /// identically zero: on a mismatch the matcher resets to zero and
    /// re-dispatches the current byte, which is byte for byte the leftmost scan
    /// `GemmaToolCallParser.gemmaString()` performs.
    private static let gemmaDelimiter = Array(#"<|"|>"#.utf8)

    /// Digits accepted on each side of the decimal point in the Gemma dialect.
    /// `Decimal(string:)` returns nil past an implementation-defined magnitude
    /// and the parser turns that nil into `.malformed`, so bounding the literal
    /// removes Foundation's boundary from the subset argument.
    private static let maximumNumberDigits = 15

    private enum NumberPhase: Equatable {
        case afterMinus
        case leadingZero
        case intDigits(Int)
        case afterDot
        case fracDigits(Int)
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

    /// Mid-UTF-8-sequence state inside a string: how many continuation bytes
    /// are still owed, the constrained range for the first of them (E0, ED, F0
    /// and F4 leads restrict it; nil means the generic 0x80...0xBF), and the
    /// scalar accumulated so far — the Gemma dialect needs the finished scalar
    /// to decide whether it grapheme-merges with a following delimiter byte.
    private struct UTF8Pending: Equatable {
        var remaining: Int
        var first: ClosedRange<UInt8>?
        var value: UInt32
    }

    /// Which production a string belongs to. One implementation of the escape
    /// and surrogate rules serves all three, so the hardened `\uXXXX` handling
    /// cannot drift between dialects.
    private enum StringKind: Equatable { case objectKey, value, gemmaValue }

    /// `GemmaToolCallParser` lexes an `[Character]`, so a raw scalar that
    /// grapheme-joins with an adjacent ASCII character hides that character from
    /// it. Both directions have to be blocked inside a `<|"|>` string.
    private enum GraphemeGuard: Equatable {
        case none
        /// The previous scalar is Grapheme_Cluster_Break=Prepend: it would
        /// swallow a following `<` (the closing delimiter) or `\` (an escape).
        case forward
        /// The previous character is structurally significant to the parser —
        /// the opening delimiter's `>`, or the last character of an escape — so
        /// a combining scalar here would fuse with it and hide it.
        case backward
    }

    private enum Mode: Equatable {
        case expectValue
        case expectValueOrClose   // right after '['
        case expectKeyOrClose     // right after '{'
        case expectKey            // after ',' inside an object
        case expectColon
        case inString(kind: StringKind, utf8: UTF8Pending?)
        case stringEscape(kind: StringKind)
        case stringUnicode(kind: StringKind, remaining: Int, value: UInt16, expectLow: Bool)
        // A completed high surrogate owes exactly `\` + `u` + a low escape.
        case stringHighSurrogate(kind: StringKind, sawBackslash: Bool)
        case inBareKey                                  // Gemma dialect only
        case gemmaOpen(matched: Int)                    // 1...4 delimiter bytes
        // `delimiter > 0` implies `utf8 == nil`: a partial delimiter is ASCII.
        case inGemmaString(delimiter: Int, utf8: UTF8Pending?, guard: GraphemeGuard)
        case inNumber(NumberPhase)
        case inLiteral([UInt8])   // remaining bytes of true/false/null
        case afterValue
        case complete
    }

    private var stack: [Container] = []
    private var mode: Mode = .expectValue

    public let dialect: Dialect

    public init(dialect: Dialect = .strict) {
        self.dialect = dialect
    }

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
            if byte == UInt8(ascii: "}") { return closeContainer() }
            return startKey(byte)

        case .expectKey:
            if isWhitespace(byte) { return true }
            return startKey(byte)

        case .expectColon:
            if isWhitespace(byte) { return true }
            if byte == UInt8(ascii: ":") { mode = .expectValue; return true }
            return false

        case .inBareKey:
            if isWhitespace(byte) { mode = .expectColon; return true }
            if byte == UInt8(ascii: ":") { mode = .expectValue; return true }
            return Self.isBareKeyByte(byte)

        case .inString(let kind, let utf8):
            if let pending = utf8 {
                guard (pending.first ?? 0x80...0xBF).contains(byte) else { return false }
                let rest = pending.remaining - 1
                mode = .inString(kind: kind,
                                 utf8: rest == 0
                                     ? nil
                                     : UTF8Pending(remaining: rest, first: nil,
                                                   value: pending.value << 6 | UInt32(byte & 0x3F)))
                return true
            }
            switch byte {
            case UInt8(ascii: "\""):
                mode = closedStringMode(kind)
                return true
            case UInt8(ascii: "\\"):
                mode = .stringEscape(kind: kind)
                return true
            case 0x00..<0x20:
                return false
            case 0x20...0x7F:
                return true
            default:
                guard let pending = Self.utf8Lead(byte) else { return false }
                mode = .inString(kind: kind, utf8: pending)
                return true
            }

        case .gemmaOpen(let matched):
            guard byte == Self.gemmaDelimiter[matched] else { return false }
            mode = matched == Self.gemmaDelimiter.count - 1
                ? .inGemmaString(delimiter: 0, utf8: nil, guard: .backward)
                : .gemmaOpen(matched: matched + 1)
            return true

        case .inGemmaString(let delimiter, let utf8, let graphemeGuard):
            if let pending = utf8 {
                guard (pending.first ?? 0x80...0xBF).contains(byte) else { return false }
                let accumulated = pending.value << 6 | UInt32(byte & 0x3F)
                let rest = pending.remaining - 1
                if rest == 0 {
                    guard let scalar = Unicode.Scalar(accumulated) else { return false }
                    if graphemeGuard == .backward, Self.mergesBackward(scalar) { return false }
                    mode = .inGemmaString(delimiter: 0, utf8: nil,
                                          guard: Self.mergesForward(scalar) ? .forward : .none)
                    return true
                }
                let next = UTF8Pending(remaining: rest, first: nil, value: accumulated)
                // Prune sequence prefixes every continuation byte would have to
                // reject, which would strand the fallback scan mid-codepoint:
                // after `CC` under a backward guard all 64 completions are
                // combining marks.
                guard graphemeGuard != .backward
                        || Self.rangeHasGraphemeBase(Self.pendingScalarRange(next)) else {
                    return false
                }
                mode = .inGemmaString(delimiter: 0, utf8: next, guard: graphemeGuard)
                return true
            }
            if delimiter > 0 {
                if byte == Self.gemmaDelimiter[delimiter] {
                    mode = delimiter == Self.gemmaDelimiter.count - 1
                        ? endedValueMode()
                        : .inGemmaString(delimiter: delimiter + 1, utf8: nil, guard: .none)
                    return true
                }
                mode = .inGemmaString(delimiter: 0, utf8: nil, guard: .none)
                return consume(byte)
            }
            switch byte {
            case UInt8(ascii: "<"):
                // A Prepend scalar would swallow this byte in the parser's
                // grapheme-cluster lexer, hiding the closing delimiter.
                guard graphemeGuard != .forward else { return false }
                mode = .inGemmaString(delimiter: 1, utf8: nil, guard: .none)
                return true
            case UInt8(ascii: "\\"):
                guard graphemeGuard != .forward else { return false }
                mode = .stringEscape(kind: .gemmaValue)
                return true
            case 0x09, 0x0A, 0x0D:
                mode = .inGemmaString(delimiter: 0, utf8: nil, guard: .none)
                return true
            case 0x00..<0x20:
                return false
            case 0x20...0x7F:
                // ASCII never joins backwards onto an ASCII character, so it
                // always clears the guard and is always available — which is
                // what keeps the fallback scan from running out of options.
                mode = .inGemmaString(delimiter: 0, utf8: nil, guard: .none)
                return true
            default:
                guard let pending = Self.utf8Lead(byte) else { return false }
                guard graphemeGuard != .backward
                        || Self.rangeHasGraphemeBase(Self.pendingScalarRange(pending)) else {
                    return false
                }
                mode = .inGemmaString(delimiter: 0, utf8: pending, guard: graphemeGuard)
                return true
            }

        case .stringEscape(let kind):
            switch byte {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"),
                 UInt8(ascii: "b"), UInt8(ascii: "f"), UInt8(ascii: "n"),
                 UInt8(ascii: "r"), UInt8(ascii: "t"):
                mode = openStringMode(kind)
                return true
            case UInt8(ascii: "u"):
                mode = .stringUnicode(kind: kind, remaining: 4, value: 0, expectLow: false)
                return true
            default:
                return false
            }

        case .stringUnicode(let kind, let remaining, let value, let expectLow):
            guard let digit = hexValue(byte) else { return false }
            let accumulated = value << 4 | UInt16(digit)
            if remaining > 1 {
                // Prune escape prefixes no digit can complete, or the fallback
                // scan strands the automaton mid-escape with the whole
                // vocabulary vetoed: after `\udc2` every hex digit spells a
                // lone low surrogate, and under `\u` + expectLow every prefix
                // outside D8...DF can never reach one.
                let shift = UInt16(4 * (remaining - 1))
                let lowest = accumulated << shift
                let highest = lowest | ((1 << shift) - 1)
                let lowSurrogates: ClosedRange<UInt16> = 0xDC00...0xDFFF
                let reachable = expectLow
                    ? (lowest...highest).overlaps(lowSurrogates)
                    : !(lowest >= lowSurrogates.lowerBound
                        && highest <= lowSurrogates.upperBound)
                guard reachable else { return false }
                mode = .stringUnicode(kind: kind, remaining: remaining - 1,
                                      value: accumulated, expectLow: expectLow)
                return true
            }
            let isHigh = (0xD800...0xDBFF).contains(accumulated)
            let isLow = (0xDC00...0xDFFF).contains(accumulated)
            if expectLow {
                guard isLow else { return false }
                mode = openStringMode(kind)
                return true
            }
            if isLow { return false }               // lone low surrogate
            mode = isHigh
                ? .stringHighSurrogate(kind: kind, sawBackslash: false)
                : openStringMode(kind)
            return true

        case .stringHighSurrogate(let kind, let sawBackslash):
            if !sawBackslash {
                guard byte == UInt8(ascii: "\\") else { return false }
                mode = .stringHighSurrogate(kind: kind, sawBackslash: true)
                return true
            }
            guard byte == UInt8(ascii: "u") else { return false }
            mode = .stringUnicode(kind: kind, remaining: 4, value: 0, expectLow: true)
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

    /// An object key. `.strict` wants `"`; the Gemma dialect wants a bare
    /// identifier — the chat template renders keys unquoted at every depth
    /// (`format_argument` propagates `escape_keys=False` into nested mappings)
    /// and the parser rejects quoted ones, which is failure cause #1 in the
    /// upstream report.
    private mutating func startKey(_ byte: UInt8) -> Bool {
        switch dialect {
        case .strict:
            guard byte == UInt8(ascii: "\"") else { return false }
            mode = .inString(kind: .objectKey, utf8: nil)
            return true
        case .gemmaToolArguments:
            guard Self.isBareKeyByte(byte) else { return false }
            mode = .inBareKey
            return true
        }
    }

    private mutating func startValue(_ byte: UInt8) -> Bool {
        if dialect == .gemmaToolArguments {
            // The tool-call body is always an argument object.
            guard !stack.isEmpty || byte == UInt8(ascii: "{") else { return false }
            if byte == UInt8(ascii: "<") {
                mode = .gemmaOpen(matched: 1)
                return true
            }
            // The parser's `"`-delimited path silently eats interior
            // whitespace, so a quoted string would decode to something other
            // than what was generated. Only the escape-token delimiter passes.
            guard byte != UInt8(ascii: "\"") else { return false }
        }
        switch byte {
        case UInt8(ascii: "\""):
            mode = .inString(kind: .value, utf8: nil)
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
            mode = .inNumber(.intDigits(1))
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

    /// Back into the string body after an escape. The escape's raw text is
    /// ASCII, so nothing can be swallowed forwards — but its last character is
    /// one the parser's `escapedFragment` compares literally, so a combining
    /// scalar must not fuse onto it.
    private func openStringMode(_ kind: StringKind) -> Mode {
        kind == .gemmaValue
            ? .inGemmaString(delimiter: 0, utf8: nil, guard: .backward)
            : .inString(kind: kind, utf8: nil)
    }

    private func closedStringMode(_ kind: StringKind) -> Mode {
        kind == .objectKey ? .expectColon : endedValueMode()
    }

    /// The bare-key alphabet `GemmaToolCallParser.objectKey()` accepts,
    /// narrowed to ASCII so the trie and the fallback scan stay byte-local.
    private static func isBareKeyByte(_ byte: UInt8) -> Bool {
        switch byte {
        case UInt8(ascii: "A")...UInt8(ascii: "Z"),
             UInt8(ascii: "a")...UInt8(ascii: "z"),
             UInt8(ascii: "0")...UInt8(ascii: "9"),
             UInt8(ascii: "_"), UInt8(ascii: "-"),
             UInt8(ascii: "."), UInt8(ascii: "$"):
            return true
        default:
            return false
        }
    }

    /// True when this scalar grapheme-merges with a following `<`, which would
    /// hide the closing `<|"|>` from `GemmaToolCallParser`: its lexer scans an
    /// `[Character]`, so a Grapheme_Cluster_Break=Prepend scalar (U+0600,
    /// U+0D4E, …) swallows the delimiter's first byte and the string never
    /// terminates. Asking the stdlib instead of hardcoding the class keeps the
    /// check exact for whatever Unicode version is linked. The same probe
    /// covers `\`, because Prepend joins with anything that follows it.
    private static func mergesForward(_ scalar: Unicode.Scalar) -> Bool {
        guard !scalar.isASCII else { return false }
        var probe = String(scalar)
        probe.append("<")
        return probe.count == 1
    }

    /// True when this scalar joins onto the preceding character — a combining
    /// mark, ZWJ, and everything else Unicode lets attach backwards. Any ASCII
    /// base answers the same question here (the base-sensitive rules are Hangul,
    /// emoji-ZWJ and regional indicators, none of which an ASCII character
    /// triggers), so one representative probe covers every position the dialect
    /// guards.
    private static func mergesBackward(_ scalar: Unicode.Scalar) -> Bool {
        guard !scalar.isASCII else { return false }
        var probe = ">"
        probe.unicodeScalars.append(scalar)
        return probe.count == 1
    }

    /// The exact set of scalars an unfinished UTF-8 sequence can still reach,
    /// honouring the lead byte's restricted first-continuation range.
    private static func pendingScalarRange(_ pending: UTF8Pending) -> ClosedRange<UInt32> {
        let range = pending.first ?? 0x80...0xBF
        let tailBits = UInt32(6 * (pending.remaining - 1))
        let head = pending.value << 6
        let lowest = (head | UInt32(range.lowerBound & 0x3F)) << tailBits
        let highest = ((head | UInt32(range.upperBound & 0x3F)) << tailBits)
            | ((1 << tailBits) - 1)
        return lowest...highest
    }

    /// True when some scalar in the range does not join backwards. Scanning
    /// stops at the first one, which is the answer for all but the runs of pure
    /// combining marks the check exists to catch.
    private static func rangeHasGraphemeBase(_ range: ClosedRange<UInt32>) -> Bool {
        for value in range {
            guard let scalar = Unicode.Scalar(value) else { continue }
            if !mergesBackward(scalar) { return true }
        }
        return false
    }

    private func numberPhase(_ phase: NumberPhase, _ byte: UInt8) -> NumberPhase? {
        let isDigit = (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
        let isExponentMark = byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E")
        let allowsExponent = dialect == .strict
        let digitCap = dialect == .strict ? Int.max : Self.maximumNumberDigits
        switch phase {
        case .afterMinus:
            guard isDigit else { return nil }
            return byte == UInt8(ascii: "0") ? .leadingZero : .intDigits(1)
        case .leadingZero:
            if byte == UInt8(ascii: ".") { return .afterDot }
            if isExponentMark, allowsExponent { return .afterExpMark }
            return nil
        case .intDigits(let digits):
            if isDigit { return digits < digitCap ? .intDigits(digits + 1) : nil }
            if byte == UInt8(ascii: ".") { return .afterDot }
            if isExponentMark, allowsExponent { return .afterExpMark }
            return nil
        case .afterDot:
            return isDigit ? .fracDigits(1) : nil
        case .fracDigits(let digits):
            if isDigit { return digits < digitCap ? .fracDigits(digits + 1) : nil }
            if isExponentMark, allowsExponent { return .afterExpMark }
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

    /// WHATWG UTF-8 lead-byte table: continuation count owed plus the
    /// restricted range for the first continuation where the lead demands one
    /// (rejects overlongs, surrogates and codepoints past U+10FFFF). `value`
    /// seeds the scalar accumulator with the lead's payload bits. Returns nil
    /// for bytes that cannot start a sequence (bare continuations, C0/C1,
    /// F5...FF).
    private static func utf8Lead(_ byte: UInt8) -> UTF8Pending? {
        switch byte {
        case 0xC2...0xDF: return UTF8Pending(remaining: 1, first: nil, value: UInt32(byte & 0x1F))
        case 0xE0:        return UTF8Pending(remaining: 2, first: 0xA0...0xBF, value: UInt32(byte & 0x0F))
        case 0xE1...0xEC: return UTF8Pending(remaining: 2, first: nil, value: UInt32(byte & 0x0F))
        case 0xED:        return UTF8Pending(remaining: 2, first: 0x80...0x9F, value: UInt32(byte & 0x0F))
        case 0xEE...0xEF: return UTF8Pending(remaining: 2, first: nil, value: UInt32(byte & 0x0F))
        case 0xF0:        return UTF8Pending(remaining: 3, first: 0x90...0xBF, value: UInt32(byte & 0x07))
        case 0xF1...0xF3: return UTF8Pending(remaining: 3, first: nil, value: UInt32(byte & 0x07))
        case 0xF4:        return UTF8Pending(remaining: 3, first: 0x80...0x8F, value: UInt32(byte & 0x07))
        default:          return nil
        }
    }

    private func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }

    private func hexValue(_ byte: UInt8) -> UInt8? {
        switch byte {
        case UInt8(ascii: "0")...UInt8(ascii: "9"): return byte - UInt8(ascii: "0")
        case UInt8(ascii: "a")...UInt8(ascii: "f"): return byte - UInt8(ascii: "a") + 10
        case UInt8(ascii: "A")...UInt8(ascii: "F"): return byte - UInt8(ascii: "A") + 10
        default: return nil
        }
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
public final class JSONTokenFilter: TokenGrammarFilter {
    public typealias PieceLookup = TokenByteTable.PieceLookup

    private let table: TokenByteTable
    private let stopIDs: Set<Int32>
    private var automaton = JSONByteAutomaton()

    public convenience init(tokenizer: GFTokenizer, extraStopIDs: Set<Int32> = []) {
        let underlying = tokenizer.tokenizer
        self.init(
            pieceLookup: { underlying.convertIdToToken(Int($0)) },
            blockedIDs: [tokenizer.bosID, tokenizer.padID,
                         tokenizer.toolCallStartID, tokenizer.toolCallEndID,
                         tokenizer.toolResponseID, tokenizer.toolResponseEndID,
                         tokenizer.channelStartID, tokenizer.channelEndID],
            stopIDs: tokenizer.stopTokenIDs.union(extraStopIDs))
    }

    public init(pieceLookup: @escaping PieceLookup,
                blockedIDs: Set<Int32>,
                stopIDs: Set<Int32>) {
        self.table = TokenByteTable(pieceLookup: pieceLookup,
                                    blockedIDs: blockedIDs.union(stopIDs))
        self.stopIDs = stopIDs
    }

    public var isComplete: Bool { automaton.isComplete }

    /// Times the grammar rejected the sampled token, forcing the
    /// probability-ordered fallback. Recorded by the generation loop.
    public private(set) var vetoCount = 0

    public func noteVeto() { vetoCount += 1 }

    /// Raw bytes of the last content token `tryAccept` committed (empty for
    /// an accepted stop token). The generation loop emits THESE bytes instead
    /// of detokenizer text under forceJSON: the detokenizer's `cleanUp` pass
    /// (` ,` → `,` — swift-transformers defaults it on) and its strict flush
    /// of byte-fallback tails can both drop grammar-validated bytes, breaking
    /// the valid-JSON guarantee at the output while the automaton believes it
    /// held.
    public private(set) var lastAcceptedBytes: [UInt8] = []

    /// True when the token may be emitted next: stop tokens pass only where
    /// EOS would leave valid JSON; content tokens pass when every byte
    /// advances the automaton (committed on success).
    public func tryAccept(_ id: Int32) -> Bool {
        if stopIDs.contains(id) {
            guard automaton.canStop else { return false }
            lastAcceptedBytes = []
            return true
        }
        guard let bytes = table.bytes(for: id), !bytes.isEmpty else { return false }
        var candidate = automaton
        for byte in bytes {
            guard candidate.consume(byte) else { return false }
        }
        automaton = candidate
        lastAcceptedBytes = bytes
        return true
    }
}

/// Reassembles the grammar-validated byte stream into printable `String`
/// deltas, holding back a trailing UTF-8 sequence a token boundary split.
/// The automaton guarantees the bytes are structurally valid UTF-8, so the
/// only incomplete point is the tail.
public struct UTF8StreamAssembler: Sendable {
    private var pending: [UInt8] = []

    public init() {}

    public mutating func push(_ bytes: [UInt8]) -> String {
        pending.append(contentsOf: bytes)
        let cut = Self.completePrefixLength(pending)
        guard cut > 0 else { return "" }
        let out = String(decoding: pending[..<cut], as: UTF8.self)
        pending.removeFirst(cut)
        return out
    }

    /// Truncation by max-new can end mid-codepoint; the incomplete tail is
    /// dropped (mirrors the detokenizer). A COMPLETE document never loses
    /// bytes here — the automaton refuses to close a string mid-sequence.
    public mutating func flush() -> String {
        defer { pending = [] }
        let cut = Self.completePrefixLength(pending)
        return cut > 0 ? String(decoding: pending[..<cut], as: UTF8.self) : ""
    }

    private static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        var i = bytes.count
        var trailing = 0
        while i > 0, trailing < 3, bytes[i - 1] & 0xC0 == 0x80 {
            i -= 1
            trailing += 1
        }
        guard i > 0 else { return bytes.count }
        let needed: Int
        switch bytes[i - 1] {
        case 0xC2...0xDF: needed = 1
        case 0xE0...0xEF: needed = 2
        case 0xF0...0xF4: needed = 3
        default: return bytes.count   // ASCII tail: nothing pending
        }
        return trailing >= needed ? bytes.count : i - 1
    }
}
