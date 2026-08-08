import Foundation

/// The Gemma marker ids the tool-call grammar treats as terminal symbols.
public struct ToolCallMarkerIDs: Sendable, Equatable {
    public var toolCallStart: Int32      // <|tool_call>
    public var toolCallEnd: Int32        // <tool_call|>
    public var toolResponse: Int32       // <|tool_response>
    public var toolResponseEnd: Int32    // <tool_response|>
    public var channelStart: Int32       // <|channel>
    public var channelEnd: Int32         // <channel|>
    public var escape: Int32             // <|"|>
    public var endOfTurn: Int32          // <turn|>
    public var eos: Int32
    public var bos: Int32
    public var pad: Int32

    public init(toolCallStart: Int32, toolCallEnd: Int32, toolResponse: Int32,
                toolResponseEnd: Int32, channelStart: Int32, channelEnd: Int32,
                escape: Int32, endOfTurn: Int32, eos: Int32, bos: Int32, pad: Int32) {
        self.toolCallStart = toolCallStart
        self.toolCallEnd = toolCallEnd
        self.toolResponse = toolResponse
        self.toolResponseEnd = toolResponseEnd
        self.channelStart = channelStart
        self.channelEnd = channelEnd
        self.escape = escape
        self.endOfTurn = endOfTurn
        self.eos = eos
        self.bos = bos
        self.pad = pad
    }

    public init(tokenizer: GFTokenizer) {
        self.init(toolCallStart: tokenizer.toolCallStartID,
                  toolCallEnd: tokenizer.toolCallEndID,
                  toolResponse: tokenizer.toolResponseID,
                  toolResponseEnd: tokenizer.toolResponseEndID,
                  channelStart: tokenizer.channelStartID,
                  channelEnd: tokenizer.channelEndID,
                  escape: tokenizer.escapeTokenID,
                  endOfTurn: tokenizer.endOfTurnID,
                  eos: tokenizer.eosID,
                  bos: tokenizer.bosID,
                  pad: tokenizer.padID)
    }
}

/// Byte trie over the declared tool names; `{` only leaves a terminal node, so
/// a generated name is always one of the declared ones. Shared prefixes cost
/// nothing: a node can be terminal AND have children, which is what makes
/// `get` and `get_all` both reachable.
///
/// Names outside the alphabet `OpenAIToolName.isValid` guarantees (1...64 bytes
/// of ASCII `[-0-9A-Za-z_]`) are dropped rather than forced: that alphabet is
/// exactly what `GemmaToolCallParser.identifier()` reads back, so forcing
/// anything else would spell a name the parser cannot return. Dropping every
/// name leaves the trie empty, which blocks `<|tool_call>` outright.
struct ToolCallNameTrie {
    static let maximumNameBytes = 64

    private struct Node {
        var children: [UInt8: Int] = [:]
        var isTerminal = false
    }

    private var nodes: [Node] = [Node()]

    let root = 0

    init(_ names: Set<String>) {
        for name in names.sorted() {
            let bytes = Array(name.utf8)
            guard (1...Self.maximumNameBytes).contains(bytes.count),
                  bytes.allSatisfy(Self.isNameByte) else { continue }
            var node = root
            for byte in bytes {
                if let next = nodes[node].children[byte] {
                    node = next
                } else {
                    nodes.append(Node())
                    let next = nodes.count - 1
                    nodes[node].children[byte] = next
                    node = next
                }
            }
            nodes[node].isTerminal = true
        }
    }

    var isEmpty: Bool { nodes.count == 1 }

    func child(_ node: Int, _ byte: UInt8) -> Int? { nodes[node].children[byte] }

    func isTerminal(_ node: Int) -> Bool { nodes[node].isTerminal }

    private static func isNameByte(_ byte: UInt8) -> Bool {
        switch byte {
        case 45, 48...57, 65...90, 95, 97...122: true
        default: false
        }
    }
}

/// Constrains the assistant turn so that every `<|tool_call>` block it opens is
/// a `GemmaToolCallParser`-decodable call for a declared tool.
///
/// The accepted language is a strict SUBSET of what `GemmaToolCallParser`
/// accepts, so a region this filter closes always parses: bare object keys at
/// every depth (the parser rejects quoted ones), `<|"|>`-delimited strings (the
/// `"`-delimited path silently eats whitespace), bounded numbers (the parser
/// turns out-of-range literals into `malformed`), a name forced from the
/// declared set, and no raw scalar that would grapheme-merge with the closing
/// delimiter.
///
/// The filter owns only the tool markers and the bytes between them: free prose
/// before the first call is unconstrained and reaches the caller through the
/// detokenizer exactly as it does without the filter.
public final class ToolCallTokenFilter: TokenGrammarFilter {
    private enum Phase {
        case free                            // prose; the filter is transparent
        case prefix(matched: Int)            // matching "call:" byte by byte
        case name(node: Int)                 // walking the trie
        case arguments(JSONByteAutomaton)    // dialect .gemmaToolArguments
        case afterCall                       // a call closed; the turn continues
    }

    private static let callPrefix = Array("call:".utf8)

    private let table: TokenByteTable
    private let markers: ToolCallMarkerIDs
    private let trie: ToolCallNameTrie
    /// Markers that can never appear between `<|tool_call>` and `<tool_call|>`.
    private let blockedInsideCall: Set<Int32>
    private var phase: Phase = .free
    /// How far the last rejected token got before the grammar refused it. A
    /// token that reaches the argument object and dies there is not a signal
    /// about the tool name, even though the committed phase is still `.name`.
    private var lastRejectionPhase: Phase?
    /// Region bytes accepted so far. Deliberately outside the copy-on-try
    /// snapshot: the fallback scan can try hundreds of candidates per token and
    /// a region runs to 256 KiB.
    private var body: [UInt8] = []

    public convenience init(tokenizer: GFTokenizer, allowedNames: Set<String>) {
        let underlying = tokenizer.tokenizer
        self.init(pieceLookup: { underlying.convertIdToToken(Int($0)) },
                  markers: ToolCallMarkerIDs(tokenizer: tokenizer),
                  allowedNames: allowedNames)
    }

    public init(pieceLookup: @escaping TokenByteTable.PieceLookup,
                markers: ToolCallMarkerIDs,
                allowedNames: Set<String>) {
        self.markers = markers
        self.trie = ToolCallNameTrie(allowedNames)
        let blocked: Set<Int32> = [markers.toolCallStart, markers.toolCallEnd,
                                   markers.toolResponse, markers.toolResponseEnd,
                                   markers.channelStart, markers.channelEnd,
                                   markers.endOfTurn, markers.eos,
                                   markers.bos, markers.pad]
        self.blockedInsideCall = blocked
        self.table = TokenByteTable(pieceLookup: pieceLookup,
                                    blockedIDs: blocked.subtracting([markers.escape]),
                                    allowedMarkerIDs: [markers.escape])
    }

    /// Times the grammar vetoed the sampled token, forcing the
    /// probability-ordered fallback. Recorded by the generation loop.
    public private(set) var vetoCount = 0

    /// Of those, the ones where the rejected token died while the function name
    /// was still being spelled — the signal that the model wanted a tool that
    /// was not declared, as opposed to one that mis-wrote its arguments.
    public private(set) var nameVetoCount = 0

    /// Grammar-validated bytes (`call:` … `}`) of the region the last accepted
    /// token closed; nil for every other token. The decoder parses THESE bytes
    /// instead of `tokenizer.decode(_:skipSpecialTokens:)`, whose `cleanUp`
    /// pass rewrites ` ,` → `,` and ` 's` → `'s` inside string arguments.
    public private(set) var closedCallBody: [UInt8]?

    /// Calls closed so far this generation.
    public private(set) var emittedCalls = 0

    /// A region is open: the generation was cut before the call closed.
    public var isInsideCall: Bool {
        switch phase {
        case .prefix, .name, .arguments: true
        case .free, .afterCall: false
        }
    }

    public func noteVeto() {
        vetoCount += 1
        switch lastRejectionPhase ?? phase {
        case .prefix, .name: nameVetoCount += 1
        case .free, .arguments, .afterCall: break
        }
    }

    /// True when the token may be emitted next. State is committed only on
    /// success, so the loop's speculative fallback scan is free.
    public func tryAccept(_ id: Int32) -> Bool {
        closedCallBody = nil
        lastRejectionPhase = phase
        switch phase {
        case .free:
            if id == markers.toolCallStart {
                // With nothing declared there is no name the parser could
                // resolve, so the block never opens and the turn stays prose.
                guard !trie.isEmpty else { return false }
                phase = .prefix(matched: 0)
                body.removeAll(keepingCapacity: true)
                return true
            }
            // Closing markers with nothing open are `malformed` in the decoder.
            if id == markers.toolCallEnd || id == markers.toolResponseEnd { return false }
            // `<|tool_response>` before any call is the orphan stop the server
            // currently turns into a 500.
            if id == markers.toolResponse { return emittedCalls > 0 }
            return true

        case .afterCall:
            // The template emits `<|tool_response>` after an unanswered call and
            // never `<turn|>`; keeping the set to these two also pins
            // `rawStopReason` to `.toolCalls` for the prompt cache.
            if id == markers.toolCallStart {
                phase = .prefix(matched: 0)
                body.removeAll(keepingCapacity: true)
                return true
            }
            return id == markers.toolResponse

        case .prefix, .name, .arguments:
            if id == markers.toolCallEnd {
                guard case .arguments(let automaton) = phase, automaton.isComplete else {
                    return false
                }
                emittedCalls += 1
                closedCallBody = body
                body.removeAll(keepingCapacity: true)
                phase = .afterCall
                return true
            }
            guard !blockedInsideCall.contains(id),
                  let bytes = table.bytes(for: id), !bytes.isEmpty,
                  body.count + bytes.count <= GemmaToolCallParser.maximumBytes else {
                return false
            }
            var candidate = phase
            for byte in bytes {
                guard Self.consume(byte, into: &candidate, trie: trie) else {
                    lastRejectionPhase = candidate
                    return false
                }
            }
            phase = candidate
            body.append(contentsOf: bytes)
            return true
        }
    }

    private static func consume(_ byte: UInt8,
                                into phase: inout Phase,
                                trie: ToolCallNameTrie) -> Bool {
        switch phase {
        case .prefix(let matched):
            guard byte == callPrefix[matched] else { return false }
            phase = matched == callPrefix.count - 1
                ? .name(node: trie.root)
                : .prefix(matched: matched + 1)
            return true

        case .name(let node):
            if let next = trie.child(node, byte) {
                phase = .name(node: next)
                return true
            }
            // The template renders no whitespace around the name, so `{` on a
            // terminal node is the only way out of the trie.
            guard byte == UInt8(ascii: "{"), trie.isTerminal(node) else { return false }
            var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
            guard automaton.consume(byte) else { return false }
            phase = .arguments(automaton)
            return true

        case .arguments(var automaton):
            guard automaton.consume(byte) else { return false }
            phase = .arguments(automaton)
            return true

        case .free, .afterCall:
            return false
        }
    }
}
