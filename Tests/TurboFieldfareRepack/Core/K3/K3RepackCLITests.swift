import Foundation
import Testing

/// CLI surface for the K3 repack profile (`--model-family kimi-k3`,
/// `--trunk-quant int4|int8`), exercising the built debug executable like
/// `RepackCLITests` does.
@Suite(.serialized)
struct K3RepackCLITests {
    @Test func unknownModelFamilyIsRejected() throws {
        let output = temporaryOutput("family")
        defer { clean(output) }
        let result = try run(["--output", output, "--model-family", "bogus"])
        #expect(result.status == 2)
        #expect(result.stderr.contains("unknown model family"))
    }

    @Test func trunkQuantRequiresKimiK3Family() throws {
        let output = temporaryOutput("quant-scope")
        defer { clean(output) }
        let result = try run(["--output", output, "--trunk-quant", "int8"])
        #expect(result.status == 2)
        #expect(result.stderr.contains("--trunk-quant requires --model-family kimi-k3"))
    }

    @Test func unknownTrunkQuantIsRejected() throws {
        let output = temporaryOutput("quant-value")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--model-family", "kimi-k3",
            "--trunk-quant", "int9",
        ])
        #expect(result.status == 2)
        #expect(result.stderr.contains("unknown trunk quantization"))
    }

    @Test func kimiK3ResumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("k3-resume")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--model-family", "kimi-k3",
            "--resume",
        ])
        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func expertReuseRequiresExplicitTargetQuant() throws {
        let output = temporaryOutput("reuse-quant")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--model-family", "kimi-k3",
            "--reuse-existing-experts",
        ])
        #expect(result.status == 2)
        #expect(result.stderr.contains("requires an explicit --trunk-quant"))
    }

    @Test func helpMentionsKimiK3() throws {
        let result = try run(["--help"])
        #expect(result.status == 0)
        #expect(result.stdout.contains("--model-family kimi-k3"))
        #expect(result.stdout.contains("--trunk-quant"))
        #expect(result.stdout.contains("--reuse-existing-experts"))
    }

    private func run(_ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String) {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/TurboFieldfareRepack")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            String(decoding: out, as: UTF8.self),
            String(decoding: err, as: UTF8.self))
    }

    private func temporaryOutput(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-k3cli-\(tag)-\(UUID().uuidString).gturbo")
    }

    private func clean(_ output: String) {
        for path in [
            output,
            output + ".partial",
            output + ".resume.json",
            output + ".install-state",
            output + ".install.lock",
        ] {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
}
