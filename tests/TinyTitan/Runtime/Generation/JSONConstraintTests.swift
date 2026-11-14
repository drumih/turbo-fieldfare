import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The token-set side of structured output: which ids a grammar position
/// allows, and how the document advances as they are chosen.
///
/// These run on a hand-built table, so they say nothing about any particular
/// tokenizer -- the point is the contract between the grammar and the mask.
@Suite struct JSONConstraintTests {
    /// `{ } " a : 1 , [ ] x` plus a stop token, and five ids with no bytes at
    /// all so the "never allowed" rule is exercised.
    private static let table = JSONTokenTable(
        vocab: 16,
        entries: [
            (0, Array("{".utf8)), (1, Array("}".utf8)), (2, Array("\"".utf8)),
            (3, Array("a".utf8)), (4, Array(":".utf8)), (5, Array("1".utf8)),
            (6, Array(",".utf8)), (7, Array("[".utf8)), (8, Array("]".utf8)),
            (9, Array("x".utf8)), (10, []),
        ],
        stopTokens: [10])

    private static let objectWithA = JSONSchemaNode.object(
        properties: ["a": .scalar([.integer])], required: ["a"], additional: false)

    private func constraint(_ node: JSONSchemaNode = objectWithA) -> JSONConstraint {
        JSONConstraint(table: Self.table, node: node)
    }

    private func allowed(_ mask: LogitMask) -> [Int32] {
        (0..<Int32(mask.vocab)).filter { mask.contains($0) }
    }

    @Test func theMaskOffersOnlyTokensTheGrammarAccepts() {
        let constraint = constraint()
        #expect(allowed(constraint.allowedMask()) == [0])
        #expect(constraint.observe(0))
        // After `{`, the only key the schema allows is "a": quote, a, quote.
        #expect(allowed(constraint.allowedMask()) == [2])
        #expect(constraint.observe(2))
        #expect(allowed(constraint.allowedMask()) == [3])
        #expect(constraint.observe(3))
        #expect(allowed(constraint.allowedMask()) == [2])
        #expect(constraint.observe(2))
        #expect(allowed(constraint.allowedMask()) == [4])
        #expect(constraint.observe(4))
        // The value node is an integer, so only a number may start here.
        #expect(allowed(constraint.allowedMask()) == [5])
        #expect(constraint.observe(5))
        // `{"a":1` is not a document: the object still has to close. The
        // comma is grammatically legal but leads to a key position with no key
        // left, so the mask has already dropped it.
        #expect(!constraint.isComplete)
        #expect(allowed(constraint.allowedMask()) == [1, 5])
        #expect(constraint.observe(1))
        #expect(constraint.isComplete)
        // A finished document may only be stopped; padding forever is not an
        // answer.
        #expect(allowed(constraint.allowedMask()) == [10])
    }

    /// Ids with no bytes are never offered, whatever the position: emitting one
    /// would advance the document by nothing.
    @Test func byteLessIdsAreNeverAllowed() {
        let constraint = constraint()
        for _ in 0..<6 {
            let mask = constraint.allowedMask()
            for id in Int32(11)..<Int32(16) { #expect(!mask.contains(id)) }
            #expect(!mask.contains(9), "`x` is not part of a JSON object")
            _ = constraint.observe(allowed(mask).first ?? 10)
        }
    }

    @Test func aWrongTokenIsReportedRatherThanAbsorbed() {
        let constraint = constraint()
        // `x` is a fine string character and a fine JSON *value* nowhere.
        #expect(!constraint.observe(9))
        #expect(constraint.violationCount == 1)
        #expect(constraint.observe(0))
        #expect(constraint.violationCount == 1)
    }

    /// Two positions of the same shape share one mask: the expensive walk over
    /// the vocabulary happens once per distinct position, not once per token.
    @Test func aRepeatedPositionReusesItsMask() {
        let constraint = constraint(.any)
        let stringStart = constraint.allowedMask()
        #expect(!stringStart.isEmpty)
        #expect(constraint.cachedPositionCount == 1)
        // A string value: opening quote, then characters (all allowed), so the
        // "inside a string" position recurs and is computed once.
        #expect(constraint.observe(2))
        let first = constraint.allowedMask()
        #expect(constraint.observe(3))
        let second = constraint.allowedMask()
        #expect(first == second)
        #expect(constraint.cachedPositionCount == 2)
    }

    @Test func aCompleteDocumentStopsOnlyWhereTheTableSaysSo() {
        let noStops = JSONConstraint(
            table: JSONTokenTable(
                vocab: 11, entries: [(0, Array("{".utf8)), (1, Array("}".utf8))]),
            node: JSONSchemaNode.object(properties: [:], required: [], additional: true))
        #expect(noStops.observe(0))
        #expect(noStops.observe(1))
        #expect(noStops.isComplete)
        #expect(
            noStops.allowedMask().isEmpty, "with no stop token a complete document cannot continue")
    }

    /// A greedy walk over nothing but the masks: the document the model is
    /// *allowed* to write is exactly the document the schema describes.
    @Test func aGreedyWalkProducesTheSchemaDocument() throws {
        let constraint = constraint()
        var text = ""
        for _ in 0..<16 {
            guard let next = allowed(constraint.allowedMask()).min(), next != 10 else { break }
            text += Self.table.bytes(of: next).lossyUTF8String
            #expect(constraint.observe(next))
        }
        #expect(text == #"{"a":1}"#)
        #expect(constraint.isComplete)
        let parsed = try JSONSerialization.jsonObject(with: Data(text.utf8))
        #expect(parsed is [String: Any])
    }

    /// The same walk with a schema that constrains a value and a nested array,
    /// on a table that can spell the property and enum names.
    @Test func aSchemaWithAnEnumAndAnArrayIsFollowedExactly() throws {
        var entries: [(id: Int32, bytes: [UInt8])] = [
            (0, Array("{".utf8)), (1, Array("}".utf8)), (2, Array("\"".utf8)),
            (3, Array(":".utf8)), (4, Array(",".utf8)), (5, Array("[".utf8)),
            (6, Array("]".utf8)), (7, Array("-".utf8)),
        ]
        for (offset, digit) in "0123456789".utf8.enumerated() {
            entries.append((Int32(8 + offset), [digit]))
        }
        for (offset, letter) in "abcdefghijklmnopqrstuvwxyz".utf8.enumerated() {
            entries.append((Int32(18 + offset), [letter]))
        }
        entries.append((44, []))
        let table = JSONTokenTable(vocab: 48, entries: entries, stopTokens: [44])
        let node = JSONSchemaNode.object(
            properties: [
                "mode": .enumeration([#""fast""#, #""slow""#]),
                "tags": .array(items: .enumeration([#""x""#, #""y""#])),
            ],
            required: ["mode"], additional: false)
        let constraint = JSONConstraint(table: table, node: node)
        var text = ""
        for _ in 0..<64 {
            let mask = constraint.allowedMask()
            guard let next = allowed(mask).min(), next != 44 else { break }
            text += table.bytes(of: next).lossyUTF8String
            #expect(constraint.observe(next))
        }
        #expect(constraint.isComplete)
        // `mode` is required, so the walk must have written it before closing,
        // and `tags` may only hold the two enumerated values.
        let parsed = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        #expect(parsed?["mode"] as? String == "fast")
        if let tags = parsed?["tags"] as? [String] {
            #expect(tags.allSatisfy { $0 == "x" || $0 == "y" })
        } else {
            #expect(parsed?["tags"] == nil, "tags is optional and may be absent")
        }
    }
}

/// The whitespace rule, on a table that can spell a whitespace-only token the
/// way a real vocabulary does.
@Suite struct JSONConstraintWhitespaceTests {
    private static let table = JSONTokenTable(
        vocab: 16,
        entries: [
            (0, Array("{".utf8)), (1, Array("}".utf8)), (2, Array("\"".utf8)),
            (3, Array("a".utf8)), (4, Array(":".utf8)), (5, Array("1".utf8)),
            (6, Array("\n".utf8)), (7, Array("  ".utf8)), (8, Array("\n  ".utf8)),
            (9, Array("\n{".utf8)), (10, []),
        ],
        stopTokens: [10])
    private static let objectWithA = JSONSchemaNode.object(
        properties: ["a": .scalar([.integer])], required: ["a"], additional: false)

    private func allowed(_ constraint: JSONConstraint) -> [Int32] {
        let mask = constraint.allowedMask()
        return (0..<Int32(mask.vocab)).filter { mask.contains($0) }
    }

    /// One whitespace-only token is allowed -- both `\n` and a token carrying
    /// an indent -- and a second one in a row is not.
    @Test func aWhitespaceRunIsAllowedButNotRepeated() {
        let constraint = JSONConstraint(table: Self.table, node: Self.objectWithA)
        #expect(allowed(constraint).contains(6))
        #expect(constraint.observe(6))
        #expect(constraint.previousTokenWasWhitespace)
        #expect(
            !allowed(constraint).contains(7), "a second whitespace token would pad the response")
        #expect(!allowed(constraint).contains(8))
        // A token that carries whitespace *and* the structural byte is still
        // fine, because it makes progress.
        #expect(allowed(constraint).contains(9))
        #expect(constraint.observe(9))
        #expect(!constraint.previousTokenWasWhitespace)
    }

    /// The pretty-printed shape a model actually writes: a newline with an
    /// indent in one token, then the value.
    @Test func aPrettyPrintedDocumentIsStillReachable() {
        let constraint = JSONConstraint(table: Self.table, node: Self.objectWithA)
        #expect(constraint.observe(9))  // "\n{"
        #expect(constraint.observe(7))  // "  " inside the object
        #expect(allowed(constraint).contains(2))
    }
}
