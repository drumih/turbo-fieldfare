import Foundation
import Testing

@testable import TinyTitanCLICore

/// The GGUF-as-is branch is taken on a directory shape: `.gguf` files present,
/// no `manifest.json`. A directory with a manifest is an install and must stay
/// on the engine path whatever else sits beside it, and anything that is not a
/// directory is not a model at all.
@Suite struct GGUFModelDirectoryTests {
    private func makeDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-detect-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func aDirectoryWithGGUFShardsAndNoManifestIsTheAsIsRoute() throws {
        let url = makeDirectory()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("x".utf8).write(
            to: url.appendingPathComponent("model-00001-of-00002.gguf"))
        #expect(isGGUFModelDirectory(url))
    }

    @Test func aDirectoryWithAManifestIsAnInstall() throws {
        let url = makeDirectory()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("x".utf8).write(
            to: url.appendingPathComponent("model-00001-of-00002.gguf"))
        try Data("{}".utf8).write(to: url.appendingPathComponent("manifest.json"))
        #expect(!isGGUFModelDirectory(url))
    }

    @Test func aDirectoryWithoutGGUFFilesIsNot() throws {
        let url = makeDirectory()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{}".utf8).write(to: url.appendingPathComponent("config.json"))
        #expect(!isGGUFModelDirectory(url))
    }

    @Test func aMissingPathIsNot() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-detect-missing-\(UUID().uuidString)")
        #expect(!isGGUFModelDirectory(missing))
    }
}
