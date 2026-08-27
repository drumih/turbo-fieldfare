import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct DecodeServiceConnectionInvalidationTests {
    @Test func lostResetAcknowledgementDiscardsTheDeadConnection() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading)
        #expect(client.connectionIsInstalled)
        weak var releasedRouter: DecodeServiceResponseRouter?
        releasedRouter = client.installedRouter

        let reset = Task {
            try await client.resetConversation(epoch: UUID())
        }
        try await Task.sleep(for: .milliseconds(20))
        try responses.fileHandleForWriting.close()

        await #expect(throws: (any Error).self) { try await reset.value }
        let deadline = ContinuousClock.now + .seconds(5)
        while client.connectionIsInstalled, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!client.connectionIsInstalled,
                "the router ended but its handles stayed installed")

        let releaseDeadline = ContinuousClock.now + .seconds(5)
        while releasedRouter != nil, ContinuousClock.now < releaseDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(releasedRouter == nil,
                "the invalidated connection retained its response router")

        try? commands.fileHandleForReading.close()
    }
}
