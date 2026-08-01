import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelSourceProbeTests {
    @Test func buildsAResolveURLForAFile() throws {
        let url = try #require(ModelSourceProbe.fileURL(
            repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
            revision: "main",
            fileName: "config.json"))
        #expect(url.absoluteString
            == "https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/resolve/main/config.json")
    }

    @Test func pinsToAnExactRevisionWhenGiven() throws {
        let url = try #require(ModelSourceProbe.fileURL(
            repoID: "owner/model",
            revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
            fileName: ModelSourceProbe.indexFileName))
        #expect(url.absoluteString == "https://huggingface.co/owner/model/resolve/"
            + "0d77464eeb233a2da68ebf9d7dc4edaac7db956d/model.safetensors.index.json")
    }

    @Test func classifiesAGGUFOnlyRepository() {
        let files = [
            ".gitattributes",
            "README.md",
            "gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
            "gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q8_0.gguf",
        ]
        #expect(ModelSourceProbe.classify(fileNames: files) == .ggufOnly)
    }

    @Test func classifiesSafetensorsMissingItsConfig() {
        let files = ["model-00001-of-00003.safetensors", "model.safetensors.index.json"]
        #expect(ModelSourceProbe.classify(fileNames: files) == .safetensorsWithoutConfig)
    }

    @Test func classifiesAnEmptyOrUnrecognisedRepository() {
        #expect(ModelSourceProbe.classify(fileNames: ["README.md"]) == .unrecognised)
        #expect(ModelSourceProbe.classify(fileNames: []) == .unrecognised)
    }

    /// A GGUF repo is the most likely thing a user pastes, so its explanation
    /// has to name the format and say what to look for instead. "no config.json"
    /// is true but tells the user nothing actionable.
    @Test func ggufExplanationNamesTheFormatAndTheAlternative() {
        let message = ModelSourceProbe.explanation(for: .ggufOnly, repoID: "owner/model-GGUF")
        #expect(message.contains("GGUF"))
        #expect(message.lowercased().contains("mlx"))
        #expect(message.contains("owner/model-GGUF"))
    }

    @Test func everyRepositoryKindHasANonEmptyExplanation() {
        for kind in [ModelSourceProbe.RepositoryKind.ggufOnly,
                     .safetensorsWithoutConfig,
                     .unrecognised] {
            #expect(ModelSourceProbe.explanation(for: kind, repoID: "owner/model").isEmpty == false)
        }
    }

    @Test func buildsTheRepositoryMetadataURL() throws {
        let url = try #require(ModelSourceProbe.repositoryInfoURL(repoID: "owner/model"))
        #expect(url.absoluteString == "https://huggingface.co/api/models/owner/model")
    }

    @Test func alwaysUsesHTTPS() throws {
        let url = try #require(ModelSourceProbe.fileURL(
            repoID: "owner/model", revision: "main", fileName: "config.json"))
        #expect(url.scheme == "https")
        #expect(url.host == "huggingface.co")
    }
}
