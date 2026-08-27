import Foundation
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareDecodeService
import TurboFieldfareDecodeProtocol

@Suite struct DecodeServiceOutboxTests {
    @Test func cancellationFollowedByThrownCancellationWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .cancelled(diagnostics(stopReason: .cancelled)),
            finishError: AppInferenceError.cancelled)

        #expect(event.kind == .cancelled)
        #expect(event.generationID == generationID)
        #expect(event.computedPrefillTokens == 7)
    }

    @Test func failureFollowedByThrownErrorWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .failed(.unknown("first"), partial: nil),
            finishError: AppInferenceError.unknown("second"))

        #expect(event.kind == .failed)
        #expect(event.generationID == generationID)
        #expect(event.error == "first")
    }

    @Test func abrokenLineageIsItsOwnTerminalKind() throws {
        let outbox = DecodeServiceOutbox(generationID: UUID())
        let event = try firstTerminal(
            from: outbox,
            published: .failed(.conversationLineageLost("prefill cursor mismatch"),
                               partial: nil),
            finishError: AppInferenceError.cancelled)

        #expect(event.kind == .lineageLost)
        #expect(event.error?.contains("prefill cursor mismatch") == true)
        #expect(event.error?.contains("Start a new chat") == true)
    }

    @Test func anordinaryFailureStaysFailed() throws {
        let outbox = DecodeServiceOutbox(generationID: UUID())
        let event = try firstTerminal(
            from: outbox,
            published: .failed(.unknown("something else"), partial: nil),
            finishError: AppInferenceError.cancelled)
        #expect(event.kind == .failed)
    }

    /// Image encoding produces no progress or tokens, so the outbox must emit
    /// memory-only events during that otherwise silent interval.
    @Test func aSilentGenerationStillReportsMemory() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let first = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(first.kind == .memory)
        #expect(first.generationID == generationID)
        #expect(try #require(first.currentMemoryBytes) > 0)

        let second = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(second.kind == .memory)

        outbox.publish(.prefillProgress(done: 4, total: 8))
        var event = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while event.kind == .memory {
            event = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(event.kind == .prefill)
        #expect(event.prefillDone == 4)
        #expect(event.currentMemoryBytes != nil)

        outbox.finish(error: AppInferenceError.cancelled)
        #expect(writerFinished.wait(timeout: .now() + 5) == .success)
    }

    @Test func liveEventsCarryTheImageTowerFigure() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(
            generationID: generationID,
            towerBytes: { 1_144_373_248 })
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let idle = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(idle.kind == .memory)
        #expect(idle.visionTowerMappedBytes == 1_144_373_248)

        outbox.publish(.prefillProgress(done: 1, total: 2))
        var event = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while event.kind == .memory {
            event = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(event.kind == .prefill)
        #expect(event.visionTowerMappedBytes == 1_144_373_248)

        outbox.finish(error: AppInferenceError.cancelled)
        #expect(writerFinished.wait(timeout: .now() + 5) == .success)
    }

    private func firstTerminal(
        from outbox: DecodeServiceOutbox,
        published event: AppInferenceEvent,
        finishError: Error
    ) throws -> DecodeServiceEvent {
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        outbox.publish(event)
        let terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        outbox.finish(error: finishError)

        #expect(writerFinished.wait(timeout: .now() + 2) == .success)
        #expect(pipe.fileHandleForReading.readDataToEndOfFile().isEmpty)
        return terminal
    }

    private func diagnostics(stopReason: AppStopReason) -> AppDiagnostics {
        AppDiagnostics(
            generatedTokens: 0,
            stopReason: stopReason,
            computedPrefillTokens: 7,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
    }
}
