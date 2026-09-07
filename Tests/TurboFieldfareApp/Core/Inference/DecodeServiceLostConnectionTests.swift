import Foundation
import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

/// A service the client can no longer reach is one loss, whatever was asked
/// of it.
///
/// The router tears the connection down when the service dies while idle and
/// nothing tells the window. The next request then found no handles: a reset
/// reported that as an unknown failure and a restore as a refused replay, so
/// the model stayed advertised as ready with no Retry Load, and the held
/// lineage was ended for a service that was simply gone. `connectionLost` is
/// the class the app turns into a failed load.
@Suite struct DecodeServiceLostConnectionTests {
    @Test func connectionLostDuringResetIsReportedAsConnectionLost() async throws {
        let requests = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: requests.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading)
        defer {
            client.shutdownForTermination()
            try? requests.fileHandleForReading.close()
        }
        let peer = Task.detached {
            _ = try DecodeFrameCodec.read(
                DecodeServiceCommand.self, from: requests.fileHandleForReading)
            try responses.fileHandleForWriting.close()
        }
        do {
            try await client.resetConversation(epoch: UUID())
            Issue.record("reset succeeded after the peer disconnected")
        } catch AppInferenceError.connectionLost {
        } catch {
            Issue.record("transport failure did not reach Retry Load: \(error)")
        }
        try await peer.value
    }

    @Test func aRestoreWithNoConnectionIsALostConnection() async {
        let client = DecodeServiceInferenceClient()
        do {
            _ = try await client.restoreConversation(
                AppConversationLineage(tokenIDs: [1, 2], committedTurns: 1),
                epoch: UUID(), options: AppRuntimeOptions(),
                maxContextTokens: 8_192) { _, _ in }
            Issue.record("a restore with no connection returned")
        } catch AppInferenceError.connectionLost {
        } catch {
            Issue.record("a restore with no connection threw \(error)")
        }
    }

    @Test func aResetWithNoConnectionIsALostConnection() async {
        let client = DecodeServiceInferenceClient()
        do {
            try await client.resetConversation(epoch: UUID())
            Issue.record("a reset with no connection returned")
        } catch AppInferenceError.connectionLost {
        } catch {
            Issue.record("a reset with no connection threw \(error)")
        }
    }
}
