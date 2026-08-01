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

    @Test func alwaysUsesHTTPS() throws {
        let url = try #require(ModelSourceProbe.fileURL(
            repoID: "owner/model", revision: "main", fileName: "config.json"))
        #expect(url.scheme == "https")
        #expect(url.host == "huggingface.co")
    }
}
