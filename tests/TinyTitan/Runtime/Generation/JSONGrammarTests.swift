import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The JSON grammar, on bytes alone. Every case here is a document a model
/// could emit, so this is the layer that decides whether structured output is
/// correct; the tokenizer only turns byte strings into token ids.
@Suite struct JSONGrammarTests {
    private func grammar(_ node: JSONSchemaNode = .any) -> JSONGrammar { JSONGrammar(node: node) }

    private func accepts(_ text: String, _ node: JSONSchemaNode = .any) -> Bool {
        grammar(node).acceptsDocument(bytes: Array(text.utf8))
    }

    private func acceptsPrefix(_ text: String, _ node: JSONSchemaNode = .any) -> Bool {
        grammar(node).accepts(bytes: Array(text.utf8))
    }

    @Test func everyJsonValueShapeIsAccepted() {
        let documents = [
            "{}", "[]", "0", "-0", "1", "-1", "1234567890", "1.5", "-0.25",
            "1e10", "1E10", "1e+10", "1e-10", "0.5e2",
            "true", "false", "null", "\"\"", "\"a\"", "\"a b\"",
            "\"\\\"\"", "\"\\\\\"", "\"\\n\"", "\"\\u00e9\"", "\"héllo\"",
            "{\"a\":1}", "{\"a\":{\"b\":[1,2,3]}}", "[[[[1]]]]",
            "[1,true,null,\"x\",{},[]]", " {\"a\" : 1 } ", "{\n\t\"a\": null\n}",
            "\"東京\"",
        ]
        for document in documents {
            #expect(accepts(document), "should accept \(document)")
        }
    }

    @Test func malformedJsonIsRejected() {
        let documents = [
            "", "{", "[", "{\"a\"", "{\"a\":", "[1,", "[1,]", "{\"a\":1,}",
            "{,}", "{\"a\" 1}", "{\"a\"::1}", "{a:1}", "{'a':1}", "[,1]",
            "01", "-", "+1", ".5", "1.", "1e", "1e+", "--1", "0x1",
            "tru", "truex", "nul", "True", "NULL",
            "\"unterminated", "\"a\" \"b\"", "{\"a\":1}}", "[]]", "{}x",
            "{\"a\":1}{\"b\":2}", "[1 2]", "\"a\nb\"", "NaN", "Infinity",
        ]
        for document in documents {
            #expect(!accepts(document), "should reject \(document)")
        }
    }

    /// A prefix is what the mask is built from: mid-document states are legal,
    /// they are just not complete.
    @Test func prefixesAreAcceptedButNotComplete() {
        for prefix in ["{", "{\"a\"", "{\"a\":", "{\"a\":1", "[1,", "\"abc", "tru"] {
            #expect(acceptsPrefix(prefix), "prefix \(prefix) should be accepted")
            #expect(!accepts(prefix), "prefix \(prefix) should not be a document")
        }
    }

    @Test func whitespaceSeparatesTokensButNotInsideThem() {
        #expect(accepts("  \n\t{ \"a\" : [ 1 , 2 ] }  \r\n"))
        #expect(!accepts("{\"a\" : 1 2}"))
        #expect(!acceptsPrefix("tr ue"))
        #expect(!acceptsPrefix("1 .5"))
        #expect(!acceptsPrefix("[1 .5]"))
    }

    // MARK: - Schema

    private func integerProperty(_ name: String) -> JSONSchemaNode {
        .object(properties: [name: .scalar([.integer])], required: [name], additional: false)
    }

    @Test func requiredPropertiesMustAppearBeforeTheObjectCloses() {
        let node = integerProperty("a")
        #expect(!accepts("{}", node))
        #expect(!accepts("{\"a\":1", node))
        #expect(accepts("{\"a\":1}", node))
        #expect(accepts("{ \"a\" : 1 }", node))
        // A second key is not in `properties` and additional is false.
        #expect(!accepts("{\"a\":1,\"b\":2}", node))
        #expect(!acceptsPrefix("{\"a\":1,\"b\"", node))
    }

    @Test func anUnconstrainedObjectStillRefusesBadJson() {
        let node = JSONSchemaNode.object(properties: [:], required: [], additional: true)
        #expect(accepts("{}", node))
        #expect(accepts("{\"anything\":[1]}", node))
        #expect(!accepts("[1]", node), "json_object means an object at the top level")
        #expect(!accepts("1", node))
    }

    @Test func propertyOrderIsFreeAndDuplicatesAreRefused() {
        let node = JSONSchemaNode.object(
            properties: ["a": .scalar([.integer]), "b": .scalar([.string])],
            required: ["a", "b"], additional: false)
        #expect(accepts("{\"a\":1,\"b\":\"x\"}", node))
        #expect(accepts("{\"b\":\"x\",\"a\":1}", node))
        #expect(!accepts("{\"a\":1,\"a\":2,\"b\":\"x\"}", node))
    }

    @Test func arraysUseTheirItemSchema() {
        let node = JSONSchemaNode.array(items: .scalar([.integer]))
        #expect(accepts("[]", node))
        #expect(accepts("[1,2,3]", node))
        #expect(!accepts("[\"x\"]", node))
        #expect(!accepts("[1,]", node))
        let untyped = JSONSchemaNode.array(items: nil)
        #expect(accepts("[1,\"x\",{}]", untyped))
    }

    @Test func integerAndNumberDifferOnFractionsAndExponents() {
        #expect(accepts("-12", .scalar([.integer])))
        #expect(!accepts("1.5", .scalar([.integer])))
        #expect(!accepts("1e3", .scalar([.integer])))
        #expect(accepts("1.5", .scalar([.number])))
        #expect(accepts("1e3", .scalar([.number])))
    }

    @Test func scalarUnionsAcceptEitherKind() {
        let nullable = JSONSchemaNode.scalar([.string, .null])
        #expect(accepts("\"x\"", nullable))
        #expect(accepts("null", nullable))
        #expect(!accepts("1", nullable))
        #expect(!accepts("true", nullable))
    }

    @Test func enumsMatchTheirCanonicalSpelling() {
        let node = JSONSchemaNode.enumeration(["\"red\"", "\"green\""])
        #expect(accepts("\"red\"", node))
        #expect(accepts("\"green\"", node))
        #expect(!accepts("\"blue\"", node))
        #expect(!accepts("red", node))
        #expect(!accepts("\"re\"", node))
        // Numbers: a completed literal may continue into a longer candidate.
        let numbers = JSONSchemaNode.enumeration(["1", "12"])
        #expect(accepts("1", numbers))
        #expect(accepts("12", numbers))
        #expect(!accepts("13", numbers))
        #expect(!accepts("123", numbers))
        // A completed enum value still has to be followed by valid JSON.
        #expect(accepts("[1,12]", JSONSchemaNode.array(items: numbers)))
        #expect(!accepts("[1,13]", JSONSchemaNode.array(items: numbers)))
    }

    @Test func enumValuesFollowTheValueIntoTheDocumentGrammar() {
        let node = JSONSchemaNode.object(
            properties: ["kind": .enumeration(["\"a\"", "\"b\""])],
            required: ["kind"], additional: false)
        #expect(accepts("{\"kind\":\"a\"}", node))
        #expect(!accepts("{\"kind\":\"c\"}", node))
        #expect(!acceptsPrefix("{\"kind\":\"c\"", node))
    }
}

/// The schema compiler: what it accepts, and what it refuses by name.
@Suite struct JSONSchemaCompileTests {
    private func compile(_ json: String) throws -> JSONSchemaNode {
        let value = try JSONDecoder().decode(JSONValue.self, from: Data(json.utf8))
        return try JSONSchemaNode.compile(value)
    }

    @Test func commonSchemasCompile() throws {
        let node = try compile(
            """
            {"type":"object","properties":{
               "name":{"type":"string"},
               "count":{"type":"integer"},
               "tags":{"type":"array","items":{"type":"string"}},
               "mode":{"enum":["fast","slow"]}
             },"required":["name"],"additionalProperties":false}
            """)
        guard case .object(let properties, let required, let additional) = node else {
            Issue.record("expected an object node, got \(node)")
            return
        }
        #expect(required == ["name"])
        #expect(!additional)
        #expect(properties["count"] == .scalar([.integer]))
        #expect(properties["tags"] == .array(items: .scalar([.string])))
        #expect(properties["mode"] == .enumeration(["\"fast\"", "\"slow\""]))
    }

    @Test func annotationsAndBooleanSchemasAreAccepted() throws {
        #expect(try compile("true") == .any)
        #expect(
            try compile(
                """
                {"type":"string","title":"Name","description":"a name","default":"x","x-vendor":1}
                """) == .scalar([.string]))
    }

    @Test func unsupportedKeywordsAreRefusedByName() throws {
        let refusals = [
            ##"{"$ref":"#/definitions/x"}"##,
            #"{"type":"string","pattern":"^a+$"}"#,
            #"{"type":"number","minimum":0}"#,
            #"{"anyOf":[{"type":"string"}]}"#,
            #"{"allOf":[{"type":"string"}]}"#,
            #"{"not":{"type":"string"}}"#,
            #"{"type":"array","items":{"type":"string"},"maxItems":3}"#,
            #"{"type":"string","format":"date-time"}"#,
            #"{"type":"object","additionalProperties":{"type":"string"}}"#,
            #"{"type":"object","unknownKeyword":1}"#,
        ]
        for refusal in refusals {
            #expect(throws: JSONSchemaCompileError.self, "should refuse \(refusal)") {
                _ = try compile(refusal)
            }
        }
        do {
            _ = try compile(#"{"type":"string","pattern":"^a+$"}"#)
            Issue.record("expected a refusal")
        } catch let error as JSONSchemaCompileError {
            guard case .unsupported(let keyword, let at) = error else {
                Issue.record("unexpected \(error)")
                return
            }
            #expect(keyword == "pattern")
            #expect(at == "$")
        }
    }

    @Test func malformedAndUnsatisfiableSchemasAreRefused() {
        #expect(throws: JSONSchemaCompileError.self) { _ = try self.compile(#"{"type":"strin"}"#) }
        #expect(throws: JSONSchemaCompileError.self) { _ = try self.compile(#"{"enum":[]}"#) }
        #expect(throws: JSONSchemaCompileError.self) {
            _ = try self.compile(
                #"{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false,"required":["b"]}"#
            )
        }
        #expect(throws: JSONSchemaCompileError.self) {
            // A completion (properties) beside an enum would be ignored.
            _ = try self.compile(#"{"enum":["a"],"properties":{"x":{"type":"string"}}}"#)
        }
        #expect(throws: JSONSchemaCompileError.self) {
            // `enum` values that share a prefix cannot be matched unambiguously.
            _ = try self.compile(#"{"enum":[1,12]}"#)
        }
        #expect(throws: JSONSchemaCompileError.self) {
            // A string needing escapes has more than one legal spelling.
            _ = try self.compile(#"{"enum":["a\"b"]}"#)
        }
    }

    @Test func containerUnionsAreRefused() throws {
        #expect(throws: JSONSchemaCompileError.self) {
            _ = try self.compile(#"{"type":["object","null"]}"#)
        }
        // A union of scalars is fine: the first byte decides.
        #expect(try self.compile(#"{"type":["string","null"]}"#) == .scalar([.string, .null]))
    }

    @Test func propertiesAndItemsImplyTheirContainer() throws {
        #expect(
            try compile(#"{"properties":{"a":{"type":"string"}}}"#)
                == .object(properties: ["a": .scalar([.string])], required: [], additional: true))
        #expect(try compile(#"{"items":{"type":"string"}}"#) == .array(items: .scalar([.string])))
        #expect(throws: JSONSchemaCompileError.self) {
            _ = try self.compile(#"{"type":"array","properties":{"a":{"type":"string"}}}"#)
        }
    }

    /// `schemas` chained object schemas, each reaching the next through one
    /// property (or through `items` when `viaItems`), plus the string leaf at the
    /// bottom: the compiler walks `schemas + 1` levels, the deepest at depth
    /// `schemas`.
    private func nestedSchemas(_ schemas: Int, viaItems: Bool = false) -> String {
        let opener =
            viaItems
            ? #"{"type":"array","items":"#
            : #"{"type":"object","properties":{"a":"#
        let closer = viaItems ? "}" : "}}"
        return String(repeating: opener, count: schemas)
            + #"{"type":"string"}"#
            + String(repeating: closer, count: schemas)
    }

    @Test func theNestingCapIsExactlyWhereItSaysItIs() throws {
        // The deepest schema the cap still compiles, and one level past it -- so
        // the guard is pinned, not merely "small". Both recursion edges count:
        // `properties` reaches a child through a name, `items` reaches one.
        let limit = JSONSchemaNode.maximumNestingDepth
        for viaItems in [false, true] {
            _ = try compile(nestedSchemas(limit, viaItems: viaItems))
            #expect(throws: JSONSchemaCompileError.self) {
                _ = try self.compile(nestedSchemas(limit + 1, viaItems: viaItems))
            }
        }
    }

    @Test func aRefusedDepthIsAMalformedSchemaErrorThatNamesTheCap() {
        var message = ""
        do {
            _ = try compile(nestedSchemas(JSONSchemaNode.maximumNestingDepth + 1))
        } catch let error as JSONSchemaCompileError {
            message = error.description
        } catch {
            Issue.record("expected a schema error, got \(error)")
        }
        #expect(message.contains("deeper than \(JSONSchemaNode.maximumNestingDepth) levels"))
    }

    @Test func theDecoderBoundsWhatTheCompilerCanBeHanded() throws {
        // Measured on this toolchain: 512 nesting levels decode, 513 throw. The
        // cap above sits inside that bound, so a document too deep for either
        // one is refused rather than walked -- and this expectation is the
        // tripwire if a future Foundation ever lifts its own limit, which is the
        // only thing that would make the recursion here unbounded again.
        let depth = 513
        let document =
            String(repeating: "{\"a\":", count: depth)
            + "null" + String(repeating: "}", count: depth)
        #expect(throws: (any Error).self) {
            _ = try JSONDecoder().decode(JSONValue.self, from: Data(document.utf8))
        }
    }
}
