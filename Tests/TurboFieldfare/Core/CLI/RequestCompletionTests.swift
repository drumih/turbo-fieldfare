import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

@Suite struct RequestCompletionTests {
    enum Failure: Error, Equatable { case operation, drain }

    @Test func footerDoesNotReportTheSecondFinishAsDrainTime() async throws {
        // The runtime drains within operation; timing the outer finish cannot
        // report that wait and previously printed a misleading drain field.
        let result = try await withFinishedRequest(operation: { 42 }, finish: {})
        let footer = loadedRequestFooter(result.timing)
        #expect(footer.hasPrefix("[request="))
        #expect(footer.hasSuffix("s]\n"))
        #expect(!footer.contains("drain="))
    }

    @Test func successFinishesExactlyOnceBeforeReturning() async throws {
        var events: [String] = []
        let result = try await withFinishedRequest(operation: {
            events.append("operation")
            return 42
        }, finish: { events.append("finish") })
        #expect(result.value == 42)
        #expect(events == ["operation", "finish"])
    }

    @Test func successfulOperationPropagatesDrainFailureWithoutRetry() async {
        var finishes = 0
        do {
            _ = try await withFinishedRequest(operation: { 42 }, finish: {
                finishes += 1
                throw Failure.drain
            })
            Issue.record("finish failure must prevent success")
        } catch {
            #expect(error as? Failure == .drain)
        }
        #expect(finishes == 1)
    }

    @Test func primaryFailureDrainsBeforeThrowing() async {
        var drained = false
        do {
            _ = try await withFinishedRequest(operation: { () -> Int in
                throw Failure.operation
            }, finish: { drained = true })
            Issue.record("operation must fail")
        } catch {
            #expect(drained)
            #expect(error as? Failure == .operation)
        }
    }

    @Test func primaryAndDrainFailuresRetainBoth() async {
        do {
            _ = try await withFinishedRequest(operation: { () -> Int in
                throw Failure.operation
            }, finish: { throw Failure.drain })
            Issue.record("operation must fail")
        } catch {
            let combined = error as? RequestCompletionFailure
            #expect(combined?.primary as? Failure == .operation)
            #expect(combined?.drain as? Failure == .drain)
            #expect(requestFailureExitCode(error) == 1)
            #expect(String(describing: error).contains("request drain also failed"))
        }
    }

    @Test func cancellationAndDrainFailureKeepExit130() async {
        var finishes = 0
        do {
            _ = try await withFinishedRequest(operation: { () -> Int in
                throw CancellationError()
            }, finish: {
                finishes += 1
                throw Failure.drain
            })
            Issue.record("operation must cancel")
        } catch {
            #expect(requestPrimaryError(error) is CancellationError)
            #expect(requestFailureExitCode(error) == 130)
            #expect((error as? RequestCompletionFailure)?.drain as? Failure == .drain)
        }
        #expect(finishes == 1)
    }

    @Test func nestedRuntimeCleanupRetainsPrimaryClassificationAndAllContext() {
        let error = RequestCompletionFailure(
            primary: RequestCompletionFailure(primary: CancellationError(), drain: Failure.operation),
            drain: Failure.drain)
        #expect(requestPrimaryError(error) is CancellationError)
        #expect(requestFailureExitCode(error) == 130)
        #expect(String(describing: error).contains("operation"))
        #expect(String(describing: error).contains("drain"))
    }
}
