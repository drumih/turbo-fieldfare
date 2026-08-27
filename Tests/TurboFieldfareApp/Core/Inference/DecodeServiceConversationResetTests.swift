import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct DecodeServiceConversationResetTests {
    @Test func aResetWithNoConnectionFailsRatherThanReportingSuccess() async throws {
        let client = DecodeServiceInferenceClient()
        await #expect(throws: AppInferenceError.self) {
            try await client.resetConversation(epoch: UUID())
        }
    }
}
