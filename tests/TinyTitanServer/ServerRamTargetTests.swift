import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

/// `--ram-budget` names a target for the whole server, and the floor under which
/// that target cannot be honoured is a hard argument error rather than a warning.
///
/// It used to accept 1G and 2G and quietly land on the ~4.7 GiB floor (weights
/// plus the 8-slot minimum cache), which is how `--ram 8` came to be reported as
/// 11.5 GiB of real use: the flag was a cache budget, not a process target.
@Suite struct ServerRamTargetTests {
    @Test func refusesATargetBelowTheFloor() {
        for tiny in ["1G", "2G", "3G", "3221225472"] {
            #expect(throws: ServerArgumentError.self) {
                try ServerArguments.parse(
                    ["--model", "/m", "--ram-budget", tiny],
                    environment: [:])
            }
        }
    }

    @Test func acceptsTheFloorAndAbove() throws {
        for accepted in ["4G", "4GiB", "8G", "16G", "32G"] {
            let arguments = try ServerArguments.parse(
                ["--model", "/m", "--ram-budget", accepted], environment: [:])
            #expect(
                arguments.expertCacheBudgetBytes
                    == RuntimeConfiguration.parseBudgetBytes(accepted))
        }
    }

    /// The message has to name the real reason, not just that the value is small.
    @Test func theRefusalNamesTheFloor() {
        do {
            _ = try ServerArguments.parse(
                ["--model", "/m", "--ram-budget", "2G"],
                environment: [:])
            Issue.record("2G should have been refused")
        } catch let error as ServerArgumentError {
            #expect(error.description.contains("at least 4G"), "\(error)")
            #expect(error.description.contains("4.7G"), "\(error)")
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }
}
