import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("OpenAI request decoding")
struct OpenAIRequestDecodingTests {
    private typealias Rejection = (message: String, param: String?, code: String)

    private func decode(_ body: String) throws -> OpenAIChatRequest {
        try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.utf8))
    }

    private func request(_ fields: String) -> String {
        """
        {"model":"m","messages":[{"role":"user","content":"x"}],\(fields)}
        """
    }

    private func decodeRejection(_ body: String) -> Rejection? {
        do {
            _ = try decode(body)
            Issue.record("request decoded instead of failing")
            return nil
        } catch ServerRequestError.invalid(let message, let param, let code) {
            return (message, param, code)
        } catch {
            Issue.record("decoding threw \(error) rather than a ServerRequestError")
            return nil
        }
    }

    private func validationRejection(_ body: String) -> Rejection? {
        do {
            _ = try OpenAIRequestValidator.validate(try decode(body), modelID: "m")
            Issue.record("request validated instead of failing")
            return nil
        } catch ServerRequestError.invalid(let message, let param, let code) {
            return (message, param, code)
        } catch {
            Issue.record("validation threw \(error) rather than a ServerRequestError")
            return nil
        }
    }

    // A `response_format` that is not an object used to fail the typed decode,
    // so the caller was answered "malformed JSON request" instead of being told
    // which field the server could not read.
    @Test(arguments: [#""json_object""#, "[]", "7", "true"])
    func nonObjectResponseFormatNamesTheFieldInsteadOfReadingAsMalformedJSON(
        _ value: String
    ) throws {
        let refusal = try #require(
            validationRejection(request(#""response_format":\#(value)"#)))
        #expect(refusal.message.contains("must be an object"))
        #expect(refusal.param == "response_format")
        #expect(refusal.code == "invalid_value")
    }

    @Test(arguments: ["7", "[]", #"{"name":"s"}"#])
    func responseFormatTypeThatIsNotAStringIsInvalid(_ value: String) throws {
        let refusal = try #require(
            validationRejection(request(#""response_format":{"type":\#(value)}"#)))
        #expect(refusal.message.contains("response_format.type must be a string"))
        #expect(refusal.param == "response_format")
        #expect(refusal.code == "invalid_value")
    }

    @Test(arguments: ["{}", #"{"type":null}"#])
    func responseFormatWithoutATypeStillReportsItAsRequired(_ value: String) throws {
        let refusal = try #require(
            validationRejection(request(#""response_format":\#(value)"#)))
        #expect(refusal.message.contains("response_format.type is required"))
        #expect(refusal.param == "response_format")
        #expect(refusal.code == "invalid_value")
    }

    @Test func responseFormatTextStillValidatesAfterTheShapeChange() throws {
        let validated = try OpenAIRequestValidator.validate(
            try decode(request(#""response_format":{"type":"text"}"#)), modelID: "m")
        #expect(validated.messages.count == 1)
    }

    @Test func responseFormatJSONObjectIsStillUnsupportedAfterTheShapeChange() throws {
        let refusal = try #require(
            validationRejection(request(#""response_format":{"type":"json_object"}"#)))
        #expect(refusal.message.contains("structured output"))
        #expect(refusal.param == "response_format")
        #expect(refusal.code == "unsupported_value")
    }

    // The key sweep runs before the typed decode, so the answer names the key
    // the caller misspelled rather than whatever DecodingError another field
    // raises first.
    @Test func unknownKeyIsNamedEvenBesideAMistypedDeclaredField() throws {
        let refusal = try #require(
            decodeRejection(request(#""max_token":4,"temperature":"hot""#)))
        #expect(refusal.message.contains("max_token"))
        #expect(refusal.param == "max_token")
        #expect(refusal.code == "unknown_parameter")
    }

    @Test func unsupportedKeyIsNamedEvenBesideAMistypedDeclaredField() throws {
        let refusal = try #require(
            decodeRejection(request(#""logit_bias":{"1":2},"temperature":"hot""#)))
        #expect(refusal.message == "logit_bias is not supported")
        #expect(refusal.param == "logit_bias")
        #expect(refusal.code == "unsupported_value")
    }

    // openai-python serializes an option the caller never set as an explicit
    // `null`, which asks for nothing and must not be refused on presence.
    @Test(arguments: [
        #""logit_bias":null"#,
        #""web_search_options":null"#,
        #""foo":null"#,
        #""n":null"#,
        #""response_format":null"#,
    ])
    func aKeySetToNullIsTreatedAsAbsent(_ field: String) throws {
        let validated = try OpenAIRequestValidator.validate(
            try decode(request(field)), modelID: "m")
        #expect(validated.messages.count == 1)
    }

    @Test func anUnsupportedKeyCarryingAValueIsStillRefused() throws {
        let refusal = try #require(decodeRejection(request(#""logit_bias":{}"#)))
        #expect(refusal.param == "logit_bias")
        #expect(refusal.code == "unsupported_value")
    }

    // A Character-counted bound is no bound: one base scalar plus combining
    // marks is a single Character however many bytes it carries, so the whole
    // key came back in `param` and, quoted, in `message`. Bounded in bytes,
    // the echo stays within the 64-byte limit plus the ellipsis.
    @Test func aKeyMadeOfCombiningMarksIsBoundedInBytes() throws {
        let key = "a" + String(repeating: "\u{0301}", count: 100_000)
        let refusal = try #require(decodeRejection(request(#""\#(key)":1"#)))
        #expect(refusal.code == "unknown_parameter")
        let param = try #require(refusal.param)
        #expect(param.hasSuffix("..."))
        #expect(param.utf8.count <= 64 + 3, "param is \(param.utf8.count) bytes")
        #expect(refusal.message.utf8.count < 1_024,
                "message is \(refusal.message.utf8.count) bytes")
    }

    @Test func legacyFunctionsAreNamedEvenBesideAMistypedDeclaredField() throws {
        let refusal = try #require(
            decodeRejection(request(#""functions":[{"name":"f"}],"temperature":"hot""#)))
        #expect(refusal.message == "legacy functions are not supported; use tools")
        #expect(refusal.param == "functions")
        #expect(refusal.code == "unsupported_value")
    }

    // The body cap is 5 MiB, so no rejection may echo an arbitrary slice of it.
    @Test func responseFormatTypeIsBoundedInTheRejection() throws {
        let long = String(repeating: "t", count: 200)
        let refusal = try #require(
            validationRejection(request(#""response_format":{"type":"\#(long)"}"#)))
        #expect(refusal.message.contains(String(repeating: "t", count: 64) + "..."))
        #expect(!refusal.message.contains(String(repeating: "t", count: 65)))
        #expect(refusal.code == "invalid_value")
    }

    @Test func toolNameIsBoundedInTheRejection() throws {
        let long = String(repeating: "z", count: 200)
        let refusal = try #require(validationRejection(request(#"""
        "tools":[{"type":"function",
          "function":{"name":"\#(long)","parameters":{"type":"object"}}}]
        """#)))
        #expect(refusal.message.contains(String(repeating: "z", count: 64) + "..."))
        #expect(!refusal.message.contains(String(repeating: "z", count: 65)))
        #expect(refusal.code == "invalid_tool_name")
    }

    // The key sweep must not swallow the wrong-typed-value class: HTTPServer
    // answers a DecodingError with `invalid_json`, and a declared key carrying
    // the wrong type is not an unknown field.
    @Test func aWrongTypedDeclaredValueAloneStillThrowsADecodingError() {
        do {
            _ = try decode(request(#""temperature":"not-a-number""#))
            Issue.record("a wrong-typed temperature decoded")
        } catch is DecodingError {
        } catch {
            Issue.record("decoding threw \(error) rather than a DecodingError")
        }
    }
}
