import Foundation

/// Turns install progress into one terse stderr line, rate-limited so a
/// 14.6 GB transfer reports movement without scrolling.
///
/// Without this the installer printed nothing at all between the command and
/// "Installed", for the whole transfer: no bytes, no percentage, no sign it was
/// alive. The data was already there - both installers take a progress closure,
/// and `main.swift` passed none.
public struct InstallProgressReporter: Sendable {
    private let write: @Sendable (String) -> Void
    private let minimumInterval: Double
    private let state = State()

    private final class State: @unchecked Sendable {
        var lastReport: Double = 0
        var lastPhase: String = ""
    }

    public init(minimumInterval: Double = 2,
                write: @escaping @Sendable (String) -> Void = { message in
                    FileHandle.standardError.write(Data(message.utf8))
                }) {
        self.minimumInterval = minimumInterval
        self.write = write
    }

    public func report(_ progress: ModelInstallProgress, now: Double) {
        guard let line = Self.line(for: progress) else { return }
        let phase = Self.phase(of: progress)
        // Rate-limited per phase, so the byte counter throttles while a phase
        // change is always announced.
        if phase == state.lastPhase, now - state.lastReport < minimumInterval {
            return
        }
        state.lastPhase = phase
        state.lastReport = now
        write(line + "\n")
    }

    public func callAsFunction(_ progress: ModelInstallProgress) {
        report(progress, now: Date().timeIntervalSince1970)
    }

    static func phase(of progress: ModelInstallProgress) -> String {
        switch progress {
        case .downloadingMetadata: "metadata"
        case .planning: "planning"
        case .checkingDisk: "disk"
        case .reservingOutput: "reserving"
        case .copyingPayload: "copying"
        case .hashingOutput: "hashing"
        case .finalizing: "finalizing"
        }
    }

    static func line(for progress: ModelInstallProgress) -> String? {
        switch progress {
        case .downloadingMetadata:
            return "[install] reading source metadata"
        case let .planning(downloadBytes, outputBytes):
            return "[install] planned: \(bytes(downloadBytes)) to download, "
                + "\(bytes(outputBytes)) to write"
        case .checkingDisk:
            return "[install] checking free space"
        case let .reservingOutput(bytes: reserved):
            return "[install] reserving \(bytes(reserved))"
        case let .copyingPayload(reused, downloaded, total):
            let done = reused + downloaded
            let percent = total == 0 ? 0 : Int((Double(done) / Double(total)) * 100)
            return "[install] \(bytes(done)) of \(bytes(total)) (\(percent)%)"
                + (reused > 0 ? ", \(bytes(reused)) already on disk" : "")
        case let .hashingOutput(file):
            return "[install] verifying \(file)"
        case .finalizing:
            return "[install] finalizing"
        }
    }

    static func bytes(_ count: UInt64) -> String {
        let units = ["B", "KiB", "MiB", "GiB"]
        var value = Double(count)
        var unit = 0
        while value >= 1024, unit < units.count - 1 {
            value /= 1024
            unit += 1
        }
        return unit == 0
            ? "\(count) B"
            : String(format: "%.1f %@", value, units[unit])
    }
}
