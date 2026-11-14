import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The GPU engine against the CPU engine on a dense Qwen 3.5 install.
///
/// The CPU engine is this family's oracle: it was written against the numpy
/// reference and `TinyTitanBench cpu35` checks it with the three continuations the
/// reference defines itself by. This runs the *GPU* engine on the same three
/// prompts and asserts it produces the same next token, which is the bar the
/// port had to clear before the family could be offered on that engine.
///
/// Skipped unless `TINYTITAN_DENSE_GPU_EQUIV=1`. It needs a release CLI and a real
/// install, and it is a model run, so the preconditions in `AGENTS.md` apply
/// (macOS 26+, Swift 6.4+, disk, acceptable `memory_pressure`, no other model
/// process).
///
/// Set `TINYTITAN_DENSE_EQUIV_MODEL` to choose the install; it defaults to the 2B
/// at 4 bits, the model the family was built against.
@Suite("Dense GPU equivalence")
struct DenseGPUEquivalenceTests {

    /// The three checks `TinyTitanBench cpu35` (and the numpy reference) define
    /// correctness with: the prompt, and the next token it must produce.
    static let oracleChecks: [(prompt: String, nextToken: String)] = [
        ("Once upon a", "time"),
        ("The capital of France is", "Paris"),
        ("The quick brown fox jumps over the lazy", "dog"),
    ]

    private func repositoryRoot() -> URL {
        // <root>/tests/TinyTitan/CPUEngine/<this file>
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // CPUEngine
            .deletingLastPathComponent()  // TinyTitan
            .deletingLastPathComponent()  // tests
            .deletingLastPathComponent()  // <root>
    }

    private func runCLI(_ executable: URL, prompt: String) throws -> String {
        let process = Process()
        process.executableURL = executable
        process.arguments = [
            "--model",
            ProcessInfo.processInfo
                .environment["TINYTITAN_DENSE_EQUIV_MODEL"] ?? "models/qwen3.5_2B_4Bit",
            "--prompt", prompt,
            "--max-new", "1",
            "--temperature", "0",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let output = data.lossyUTF8String
        // A non-zero exit is a failure of the engine, not of the expectation,
        // so it is reported with the output that explains it.
        try #require(
            process.terminationStatus == 0,
            "CLI exited \(process.terminationStatus): \(output)")
        return output
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_GPU_EQUIV"] != nil))
    func theGPUEngineContinuesTheOraclesOwnChecks() throws {
        let root = repositoryRoot()
        let executable = root.appendingPathComponent(".build/release/TinyTitanCLI")
        try #require(
            FileManager.default.isExecutableFile(atPath: executable.path),
            "build it first: swift build -c release --product TinyTitanCLI")

        let model =
            ProcessInfo.processInfo.environment["TINYTITAN_DENSE_EQUIV_MODEL"]
            ?? "models/qwen3.5_2B_4Bit"
        try #require(
            FileManager.default.fileExists(
                atPath: root.appendingPathComponent(model).path),
            "no install at \(model)")

        for check in Self.oracleChecks {
            let output = try runCLI(executable, prompt: check.prompt)
            #expect(
                output.contains(check.nextToken),
                "\(check.prompt) -> expected \(check.nextToken), got: \(output)")
        }
    }
}
