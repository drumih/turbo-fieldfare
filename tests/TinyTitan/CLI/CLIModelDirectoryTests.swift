import Foundation
import Testing
import TinyTitan

@testable import TinyTitanCLICore
import TinyTitanCore

/// TT-036: pointing `--model` at a path that is not there used to fail with
/// "installed tokenizer is missing chat_template.jinja; reinstall the model",
/// which sends the reader to reinstall a model that is simply not at that path.
/// The CLI now names the path, and that is what this pins.
@Suite struct CLIModelDirectoryTests {
    @Test func missingModelDirectoryIsNamedInTheError() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-no-such-model-\(UUID().uuidString)", isDirectory: true)
        let args = try Args.parse(["--model", missing.path, "--prompt", "hi", "--max-new", "4"])
        let stdout = Pipe()
        let stderr = Pipe()

        let result = await run(
            args: args,
            stdout: stdout.fileHandleForWriting,
            stderr: stderr.fileHandleForWriting)
        stderr.fileHandleForWriting.closeFile()
        let text = try #require(
            String(
                data: stderr.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8))

        #expect(result.exitCode != 0)
        #expect(text.contains("model directory not found: \(missing.standardizedFileURL.path)"))
        #expect(!text.contains("chat_template.jinja"))
    }
}
