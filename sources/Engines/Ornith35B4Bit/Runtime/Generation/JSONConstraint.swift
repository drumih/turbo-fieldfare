import Foundation

/// One generation's structured-output state: the document parsed so far, and
/// the token set that may extend it.
///
/// The constraint never looks at probabilities. It answers one question per
/// token position -- *which tokens can still produce a document the schema
/// allows* -- and the sampler answers the other one. That split is what keeps
/// constrained output as close to unconstrained output as the schema permits.
///
/// ## Cost
///
/// The answer is computed per *position*, not per token: `allowedMask()` walks
/// the token table's byte strings through the grammar and caches the result
/// under the whole grammar state, so the second token inside a string costs
/// nothing and the thousandth character of a string costs nothing either. The
/// walk is pruned twice over -- only the first bytes the grammar still accepts
/// are tried, and only the tokens that start with them -- so the expensive full
/// scan happens for a handful of positions in a document, not for every token.
///
/// ## Threading
///
/// unchecked-invariant: one constraint belongs to one generation. The decode
/// loop is the only caller of `observe`, and it is the same task that calls
/// `allowedMask`, so no lock is needed; the class is a reference only so that
/// `GenerationConfig` (a value type copied per call) can carry it without
/// copying a 32 KiB mask and a grammar stack per token.
public final class JSONConstraint: @unchecked Sendable {
    /// The vocabulary the sampler draws from. Ids at or above the token
    /// table's own vocabulary are never allowed -- they have no bytes, and
    /// leaving them unmasked would let the model step outside the document.
    public let vocab: Int
    private let table: JSONTokenTable
    private var grammar: JSONGrammar
    /// Whether the token chosen at the previous position was nothing but
    /// whitespace.
    ///
    /// A JSON grammar allows whitespace between any two tokens, which means it
    /// allows whitespace forever -- and a model asked for JSON by a prompt that
    /// does not prime it will spend the whole cap on newlines. That is not
    /// hypothetical: with a `{"type": "boolean"}` schema and the prompt "say
    /// hello in a friendly way", the dense 2B emitted 32 whitespace tokens and
    /// stopped on `length`, because whitespace was the only allowed token its
    /// distribution liked. One whitespace-only token between two structural
    /// tokens is enough for any document -- pretty-printed JSON included, since
    /// an indent is one token -- so a second one in a row is refused. That
    /// makes the stall impossible without making the document impossible.
    private var lastTokenWasWhitespace = false
    /// The mask cache. The key is the whole position: the grammar *and* the
    /// whitespace rule, because the same grammar state offers a different set
    /// after a whitespace token than after a structural one.
    private struct Position: Hashable {
        let grammar: JSONGrammar
        let afterWhitespace: Bool
    }
    private var cache: [Position: LogitMask] = [:]
    private var violations = 0

    public init(table: JSONTokenTable, node: JSONSchemaNode, vocab: Int? = nil) {
        self.table = table
        self.vocab = max(vocab ?? table.vocab, table.vocab)
        self.grammar = JSONGrammar(node: node)
    }

    /// Whether a complete document has been produced. Trailing whitespace and
    /// a stop token may still follow; nothing else may.
    public var isComplete: Bool { grammar.isComplete }

    /// How many generated tokens the grammar rejected. Non-zero means the mask
    /// failed, which is a bug in this file rather than something a client did,
    /// so the generation loop turns it into a named error instead of shipping
    /// invalid JSON.
    public var violationCount: Int { violations }

    /// The tokens allowed at the current position. The returned mask is shared
    /// with the cache -- callers must not mutate it.
    public func allowedMask() -> LogitMask {
        let position = Position(grammar: grammar, afterWhitespace: lastTokenWasWhitespace)
        if let cached = cache[position] { return cached }
        var mask = LogitMask(vocab: vocab)
        if grammar.isComplete {
            // The document is done. Whitespace tokens would keep it valid but
            // would let the model pad indefinitely; the stop tokens end the
            // turn, and that is the whole answer here.
            for id in table.stopTokens where Int(id) < vocab { mask.insert(id) }
        } else {
            for byte in table.populatedFirstBytes {
                var first = grammar
                guard first.consume(byte) else { continue }
                for id in table.tokens(startingWith: byte) {
                    var probe = first
                    var accepted = true
                    for next in table.bytes(of: id).dropFirst()
                    where !probe.consume(next) {
                        accepted = false
                        break
                    }
                    // A legal token that leads only to a dead end is not a
                    // choice: the mask exists to keep the model inside the
                    // document, not merely inside the grammar's byte rules.
                    // Nor is a second whitespace-only token in a row, which
                    // would let the model pad the whole response.
                    if accepted, probe.canComplete,
                        !(lastTokenWasWhitespace && isWhitespaceOnly(id))
                    {
                        mask.insert(id)
                    }
                }
            }
        }
        cache[position] = mask
        return mask
    }

    /// Whether a token's bytes are all JSON whitespace. A token with no bytes
    /// is never in the mask, so it is not whitespace for this purpose.
    private func isWhitespaceOnly(_ id: Int32) -> Bool {
        let bytes = table.bytes(of: id)
        guard !bytes.isEmpty else { return false }
        return bytes.allSatisfy(JSONGrammar.isWhitespace)
    }

    /// Advance the document by a generated token. Returns false when the token
    /// was not allowed, which the loop reports rather than ignores.
    @discardableResult
    public func observe(_ id: Int32) -> Bool {
        let bytes = table.bytes(of: id)
        lastTokenWasWhitespace = isWhitespaceOnly(id)
        guard !bytes.isEmpty else {
            // A special token adds nothing to the document; it is only legal
            // once the document is complete, and the loop stops there anyway.
            if !grammar.isComplete {
                violations += 1
                return false
            }
            return true
        }
        for byte in bytes {
            guard grammar.consume(byte) else {
                violations += 1
                return false
            }
        }
        return true
    }

    /// The cache's size, so a test can assert that repeated positions reuse
    /// their mask instead of rescanning the vocabulary.
    var cachedPositionCount: Int { cache.count }

    /// Whether the previous token was whitespace only, exposed for tests.
    var previousTokenWasWhitespace: Bool { lastTokenWasWhitespace }
}
