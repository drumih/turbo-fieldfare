import Foundation
import Synchronization

/// Spawns `TurboFieldfareServer` as a plain child process — no launchd, since
/// unlike the decode service this does not need to survive the app quitting
/// (`AppModel.stopServerForTermination()` stops it explicitly). Stopping
/// sends `SIGTERM`, which `ServerTerminationSignals` already turns into a
/// graceful shutdown that exits with status 0; any other exit status is
/// reported as `.failed`, whether that is a startup argument error or a
/// mid-run crash — one path for both, the same way `terminationHandler`
/// fires for either.
public final class ProcessServerController: AppServerController, @unchecked Sendable {
    private let executableURL: URL
    private let lock = NSLock()
    private var process: Process?

    public init(executableURL: URL = ProcessServerController.defaultExecutableURL()) {
        self.executableURL = executableURL
    }

    public func start(arguments: [String],
                      onStateChange: @escaping @Sendable (AppServerState) -> Void) {
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
            onStateChange(.failed(message:
                "server executable is missing at \(executableURL.path); run swift build -c release before starting the server"))
            return
        }

        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.executableURL = executableURL
        process.arguments = arguments
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let stdoutBuffer = Mutex(Data())
        let stderrBuffer = Mutex(Data())
        let reportedPort = Mutex<Int?>(nil)

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text: String? = stdoutBuffer.withLock { buffer in
                buffer.append(data)
                return String(data: buffer, encoding: .utf8)
            }
            guard let text, let port = Self.port(fromReadyOutput: text) else { return }
            let isNewReport = reportedPort.withLock { existing -> Bool in
                guard existing == nil else { return false }
                existing = port
                return true
            }
            if isNewReport { onStateChange(.running(port: port)) }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            stderrBuffer.withLock { $0.append(data) }
        }

        process.terminationHandler = { finished in
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            if finished.terminationStatus == 0 {
                onStateChange(.stopped)
                return
            }
            let stderrText = stderrBuffer.withLock {
                String(decoding: $0, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            onStateChange(.failed(message: Self.errorMessage(
                stderr: stderrText, status: finished.terminationStatus)))
        }

        lock.lock()
        self.process = process
        lock.unlock()

        do {
            try process.run()
        } catch {
            lock.lock()
            self.process = nil
            lock.unlock()
            onStateChange(.failed(message: "failed to launch server: \(error)"))
        }
    }

    public func stop() {
        lock.lock()
        let process = self.process
        lock.unlock()
        process?.terminate()
    }

    private static func port(fromReadyOutput text: String) -> Int? {
        guard let readyRange = text.range(of: "TurboFieldfareServer ready") else { return nil }
        let tail = text[readyRange.lowerBound...]
        guard let markerRange = tail.range(of: "http://127.0.0.1:") else { return nil }
        let digits = tail[markerRange.upperBound...].prefix { $0.isNumber }
        return digits.isEmpty ? nil : Int(digits)
    }

    private static func errorMessage(stderr: String, status: Int32) -> String {
        guard !stderr.isEmpty else { return "server exited with status \(status)" }
        return stderr.hasPrefix("error: ") ? String(stderr.dropFirst(7)) : stderr
    }

    public static func defaultExecutableURL() -> URL {
        Bundle.main.executableURL!
            .deletingLastPathComponent()
            .appendingPathComponent("TurboFieldfareServer")
    }
}
