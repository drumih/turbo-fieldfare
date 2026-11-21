import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2TokenizerTests {
    private func json() throws -> Data {
        let object: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": ["a": 0, "b": 1, "ab": 2, "Ġ": 3, "Ġab": 4, "c": 5],
                "merges": ["a b", "Ġ ab"],
            ],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    [
                        "type": "Split",
                        "pattern": ["Regex": "\\S+|\\s+"],
                    ]
                ],
            ],
            "added_tokens": [
                ["id": 6, "content": "<|x|>", "special": true]
            ],
        ]
        return try JSONSerialization.data(withJSONObject: object)
    }

    @Test func mergesAndSpecials() throws {
        let tokenizer = try Inkling2Tokenizer(json: try json())
        #expect(try tokenizer.encode("ab") == [2])
        #expect(try tokenizer.encode("<|x|>") == [6])
        #expect(try tokenizer.encode("ab <|x|> ab") == [2, 3, 6, 3, 2])
    }

    @Test func decodesBackToText() throws {
        let tokenizer = try Inkling2Tokenizer(json: try json())
        let text = "ab <|x|> ab"
        let ids = try tokenizer.encode(text)
        #expect(tokenizer.decode(ids) == text)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func matchesTheReferenceTokenizer() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tokenizer = try Inkling2Tokenizer(
            directory: URL(fileURLWithPath: path))
        let cases: [(String, [Int])] = [
            ("Hello, world!", [32_269, 11, 150_770, 0]),
            ("The capital of France is", [371, 151_451, 149_879, 151_194, 149_884]),
            (
                "<|message_user|>hi<|end_message|>",
                [200_000, 1_910, 200_010]
            ),
            ("a\n\nb", [64, 68_014, 65]),
        ]
        for (text, expected) in cases {
            #expect(try tokenizer.encode(text) == expected, "\(text)")
        }
        #expect(tokenizer.decode([200_000, 1_910, 200_010]) == "<|message_user|>hi<|end_message|>")
    }

    @Test func unknownTokenIsRefused() throws {
        let broken: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": ["a": 0],
                "merges": ["a a"],
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: broken)
        let tokenizer = try Inkling2Tokenizer(json: data)
        #expect(throws: Inkling2Tokenizer.Error.self) {
            try tokenizer.encode("aa")
        }
    }
}
