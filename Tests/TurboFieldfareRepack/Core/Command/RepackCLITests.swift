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
    @Test func visionActivationWithoutPreparedStateReportsAnError() throws {
        let output = temporaryOutput("vision-missing-activation")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--activate-vision-install",
            "--vision-output", output,
            "--text-model", "/tmp/missing-text.gturbo",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func visionDiscardWithoutStateReportsAnError() throws {
        let output = temporaryOutput("vision-missing-discard")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--vision-output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func visionRemovalWithoutInstalledPackReportsAnError() throws {
        let output = temporaryOutput("vision-missing-remove")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--remove-vision-install",
            "--vision-output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    /// Discard is handled before the other vision modes, so accepting the
    /// combination silently performed the discard and exited 0 without ever
    /// removing, verifying or activating anything.
    @Test func visionModesAreMutuallyExclusive() throws {
        let output = temporaryOutput("vision-mode-clash")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        for mode in ["--remove-vision-install", "--verify-vision-install",
                     "--activate-vision-install"] {
            let result = try run(["--discard-partial", mode, "--vision-output", output])
            #expect(result.status == 2, "\(mode) with --discard-partial was accepted")
            #expect(result.stderr.contains("mutually exclusive"))
        }
    }

    /// Verify and activate read an installed or prepared pack; a transfer flag
    /// is a request they cannot honour, not a no-op.
    @Test func verifyAndActivateRejectTransferFlags() throws {
        let output = temporaryOutput("vision-flag-clash")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        for mode in ["--verify-vision-install", "--activate-vision-install"] {
            for flag in ["--overwrite", "--resume"] {
                let result = try run([mode, "--vision-output", output,
                                      "--text-model", "scratch/gemma4.gturbo", flag])
                #expect(result.status == 2, "\(mode) accepted \(flag)")
            }
        }
    }

    @Test func remoteVisionResumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("vision-missing-resume")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--vision-output", output,
            "--text-model", "/tmp/missing-text.gturbo",
            "--resume",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

}
