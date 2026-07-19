import Foundation
@testable import TurboFieldfareRepackCore
import Synchronization
import Testing
@testable import TurboFieldfareAppCore

@Suite struct RepackModelInstallerClientTests {
    @Test func mapsCoreProgressAndCompletion() async throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("scripted.gturbo")
        let client = RepackModelInstallerClient { outputDirectory, progress in
            progress(.downloadingMetadata)
            progress(.planning(downloadBytes: 10, outputBytes: 20))
            progress(.checkingDisk(DiskSpaceRequirement(path: "/",
                                                        requiredBytes: 30,
                                                        availableBytes: 40)))
            progress(.reservingOutput(bytes: 20))
            progress(.copyingPayload(downloadedBytes: 4, totalBytes: 10))
            progress(.hashingOutput("model_weights.bin"))
            progress(.finalizing)
            return outputDirectory
        }

        var events: [AppModelInstallEvent] = []
        for try await event in client.installDefaultModel(outputDirectory: output) {
            events.append(event)
        }

        #expect(events == [
            .checking,
            .downloadingMetadata,
            .planning,
            .checking,
            .reservingOutput,
            .copyingPayload(doneBytes: 4, totalBytes: 10),
            .hashingOutput("model_weights.bin"),
            .finalizing,
            .installed(output.standardizedFileURL),
        ])
    }

    @Test func cancelPropagatesCancellationToConsumer() async throws {
        let started = Flag()
        let client = RepackModelInstallerClient { outputDirectory, progress in
            progress(.downloadingMetadata)
            started.set()
            try await Task.sleep(for: .seconds(60))
            return outputDirectory
        }
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("cancelled.gturbo")
        let consume = Task {
            for try await _ in client.installDefaultModel(outputDirectory: output) {}
        }

        for _ in 0..<200 where !started.value {
            await Task.yield()
        }
        #expect(started.value)
        client.cancel()

        await #expect(throws: CancellationError.self) {
            try await consume.value
        }
    }
}

private final class Flag: Sendable {
    private let storage = Mutex(false)
    var value: Bool { storage.withLock { $0 } }
    func set() { storage.withLock { $0 = true } }
}
