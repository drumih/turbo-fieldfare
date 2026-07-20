import CryptoKit
import Foundation
import TurboFieldfareAppCore
import Testing

@Suite("App prompt presets")
struct AppPromptPresetTests {
    @Test("Catalog has exactly seven unique prompts in the approved order")
    func catalogIdentity() {
        let presets = AppPromptPreset.all

        #expect(presets.map(\.id) == [
            "paris",
            "fibonacci",
            "fieldfare",
            "cosine-similarity",
            "recommendation-systems",
            "matrix-multiplication",
            "metal-matmul",
        ])
        #expect(Set(presets.map(\.id)).count == 7)
        #expect(Set(presets.map(\.title)).count == 7)
        #expect(Set(presets.map(\.prompt)).count == 7)
        #expect(presets.allSatisfy { !$0.prompt.isEmpty })
        #expect(AppPromptPreset.primary.count == 3)
        #expect(AppPromptPreset.secondary.count == 4)
    }

    @Test("Prompt bytes match the accepted catalog")
    func exactPromptHashes() {
        let expected = [
            "paris": "7d977c14f3236c90d02e3ac77ce3f5864ee0c4a95e68ed2afca05e98db0f5e11",
            "cosine-similarity": "b0b3a392029aa32aea6d392ef738501b69eba55f5fbc46702e01265897f477d0",
            "fieldfare": "d4f955dbf02cca9e05302d50fbe9dcf03aa27076534f1ed29f0fdea681fa58cc",
            "fibonacci": "8ba84822675c09fb7c68a66701ff2c6fe58db84daf05a9458430b9604ceb3631",
            "recommendation-systems": "d46fa2cee4f50216afd3601e8957c782d7b3955f23e5a3588d1011bf7789ee27",
            "matrix-multiplication": "308d3d062ba6a93b38fc950b6282a0f313f0042caa109224e09f925829738609",
            "metal-matmul": "582a1286e387789ddc4351552bed2aef939649aff74d56c711ccf0d2c677d932",
        ]

        for preset in AppPromptPreset.all {
            let digest = SHA256.hash(data: Data(preset.prompt.utf8))
                .map { String(format: "%02x", $0) }
                .joined()
            #expect(digest == expected[preset.id])
        }
    }

    @Test("Presets contain no embedded chat wrappers")
    func chatTemplateContract() {
        let forbiddenMarkers = [
            "<|turn>", "<turn|>", "<|channel>", "<channel|>",
            "<start_of_turn>", "<end_of_turn>",
        ]

        for preset in AppPromptPreset.all {
            #expect(!forbiddenMarkers.contains { preset.prompt.contains($0) })
        }
    }

    @Test("Historical first three prompts occupy primary showcase slots")
    func primaryShowcase() {
        #expect(AppPromptPreset.primary.map(\.id) == [
            "paris", "fibonacci", "fieldfare",
        ])
        #expect(AppPromptPreset.secondary[0].id == "cosine-similarity")
        #expect(AppPromptPreset.secondary.last?.id == "metal-matmul")
        #expect(AppPromptPreset.primary[2].prompt.contains("winter migration to Britain"))
    }
}
