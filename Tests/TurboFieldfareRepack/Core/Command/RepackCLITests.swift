import Foundation
import Testing

@Suite(.serialized)
struct RepackCLITests {
    @Test func resumeAndDiscardAreMutuallyExclusive() throws {
        let output = temporaryOutput("exclusive")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
            "--discard-partial",
        ])

        #expect(result.status == 2)
        #expect(result.stderr.contains("mutually exclusive"))
    }

    @Test func resumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("missing-resume")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func discardWithoutStateReportsAnError() throws {
        let output = temporaryOutput("missing-discard")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
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
            .appendingPathComponent("turbofieldfare-cli-\(tag)-\(UUID().uuidString).gturbo")
    }

    private func clean(_ output: String) {
        for path in [
            output,
            output + ".partial",
            output + ".install-state",
            output + ".install-state.cleanup",
            output + ".install.lock",
        ] {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
}
