import TurboFieldfareAppCore
import Testing

@Suite("App prompt presets")
struct AppPromptPresetTests {
    @Test("Catalog has unique nonempty identifiers and titles")
    func catalogIdentity() {
        let presets = AppPromptPreset.all

        #expect(Set(presets.map(\.id)).count == presets.count)
        #expect(Set(presets.map(\.title)).count == presets.count)
        #expect(presets.allSatisfy { !$0.prefix.isEmpty })
        #expect(presets.count == 6)
        #expect(AppPromptPreset.primary.count == 3)
        #expect(AppPromptPreset.secondary.count == 3)
    }

    @Test("Primary presets are exact raw-completion prefixes")
    func exactPrimaryPrefixes() {
        #expect(AppPromptPreset.primary[0].prefix ==
            "Paris is the capital and largest city of France. Situated on the River Seine, the city is known for")
        #expect(AppPromptPreset.primary[1].prefix == """
            An iterative Python function can return the first n Fibonacci numbers without recursion. It returns an empty list when n is zero, uses O(n) time, and includes a small correctness check:

            ```python
            def fibonacci(n: int) -> list[int]:
            """)
        #expect(AppPromptPreset.primary[2].prefix ==
            "The fieldfare (Turdus pilaris) is a grey-headed, chestnut-backed thrush that reaches Britain and Ireland from northern Europe each autumn. Its noisy winter flocks roam the countryside in search of berries, and")
        #expect(AppPromptPreset.primary[2].id == "fieldfare")
        #expect(AppPromptPreset.primary[2].title == "Meet the fieldfare")
        #expect(!AppPromptPreset.all.contains { $0.id == "metal-gemv" })
    }

    @Test("Presets contain no chat wrappers or imperative requests")
    func rawCompletionContract() {
        let forbiddenOpenings = ["Write ", "Explain ", "Give me ", "Rewrite "]
        let forbiddenMarkers = ["<start_of_turn>", "<end_of_turn>", "user\n", "assistant\n"]

        for preset in AppPromptPreset.all {
            #expect(!forbiddenOpenings.contains { preset.prefix.hasPrefix($0) })
            #expect(!forbiddenMarkers.contains { preset.prefix.contains($0) })
        }
    }

    @Test("Primary presets include the Fieldfare easter egg")
    func fieldfarePreset() {
        let preset = AppPromptPreset.primary.first { $0.id == "fieldfare" }

        #expect(preset?.title == "Meet the fieldfare")
        #expect(preset?.prefix ==
            "The fieldfare (Turdus pilaris) is a grey-headed, chestnut-backed thrush that reaches Britain and Ireland from northern Europe each autumn. Its noisy winter flocks roam the countryside in search of berries, and")
        #expect(!AppPromptPreset.all.contains { $0.id == "software" })
        #expect(!AppPromptPreset.all.contains { $0.id == "lighthouse" })
    }

    @Test("Secondary presets include cosine similarity")
    func cosineSimilarityPreset() {
        let preset = AppPromptPreset.secondary.first { $0.id == "cosine-similarity" }

        #expect(preset?.title == "Dot product and cosine similarity")
        #expect(preset?.prefix == """
            The unit vectors u = [1, 0] and v = [0, 1] are orthogonal. Their dot product is 0, both lengths are 1, and their cosine similarity is exactly 0. A compact check is:

            ```python
            assert dot([1, 0], [0, 1]) == 0
            assert cosine_similarity([1, 0], [0, 1]) ==
            """)
        #expect(!AppPromptPreset.all.contains { $0.id == "solar-system" })
    }

    @Test("Secondary presets include recommendation systems and matrix multiplication")
    func substantialExplanationPresets() {
        let recommendations = AppPromptPreset.secondary.first { $0.id == "recommendation-systems" }
        let matrix = AppPromptPreset.secondary.first { $0.id == "matrix-multiplication" }

        #expect(recommendations?.title == "How recommendations learn")
        #expect(recommendations?.prefix.contains("Matrix factorization learns a low-dimensional vector") == true)
        #expect(recommendations?.prefix.hasSuffix("## The sparse interaction matrix") == true)
        #expect(matrix?.title == "Matrix multiplication")
        #expect(matrix?.prefix.contains("C[2,2] = 3 x 6 + 4 x 8 = 50") == true)
        #expect(matrix?.prefix.hasSuffix("## Why the inner dimensions match") == true)
    }
}
