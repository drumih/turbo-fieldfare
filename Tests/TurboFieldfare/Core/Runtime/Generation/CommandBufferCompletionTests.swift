import Foundation
import Metal
import Testing

@testable import TurboFieldfare

@Suite struct CommandBufferCompletionTests {
    private static var visionRuntimeSource: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboFieldfare/Runtime/Vision/VisionRuntime.swift")
    }

    private static func nsError(domain: String = "MTLCommandBufferErrorDomain",
                                code: Int = 1,
                                description: String,
                                userInfo extra: [String: Any] = [:]) -> NSError {
        var info: [String: Any] = [NSLocalizedDescriptionKey: description]
        info.merge(extra) { current, _ in current }
        return NSError(domain: domain, code: code, userInfo: info)
    }

    @Test func completedBufferWithNoErrorIsNotAFailure() {
        #expect(metalCommandBufferFailureDetail(label: "prefill layer=3",
                                                status: .completed,
                                                error: nil) == nil)
    }

    @Test func errorStatusWithNoErrorObjectIsStillAFailure() throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(label: "prefill layer=3",
                                            status: .error,
                                            error: nil))
        #expect(detail.contains("status=error"))
        #expect(detail.contains("label=prefill layer=3"))
        #expect(detail.contains("error=<none>"))
    }

    @Test(arguments: [
        MTLCommandBufferStatus.notEnqueued,
        .enqueued,
        .committed,
        .scheduled,
    ])
    func nonCompletedStatusIsAFailure(_ status: MTLCommandBufferStatus) throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(label: nil, status: status, error: nil))
        #expect(detail.contains("status=\(metalCommandBufferStatusName(status))"))
    }

    @Test func interactivityKillDetailPreservesTheIOGPUToken() throws {
        let description =
            "Impacting Interactivity "
            + "(0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"
        let detail = try #require(
            metalCommandBufferFailureDetail(
                label: "prefill start=8064 count=128 layer=12 phase=qkv_attention",
                status: .error,
                error: Self.nsError(description: description)))

        #expect(detail.contains("kIOGPUCommandBufferCallbackErrorImpactingInteractivity"))
        #expect(detail.contains("0000000e"))
        #expect(detail.contains("domain=MTLCommandBufferErrorDomain"))
        #expect(detail.contains("code=1"))
        #expect(detail.contains("layer=12"))
    }

    @Test func missingLabelIsReportedRatherThanOmitted() throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(label: nil,
                                            status: .error,
                                            error: Self.nsError(description: "boom")))
        #expect(detail.contains("label=<none>"))
    }

    @Test func emptyLabelIsDistinguishedFromAbsentLabel() throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(label: "",
                                            status: .error,
                                            error: Self.nsError(description: "boom")))
        #expect(detail.contains("label=<empty>"))
    }

    @Test func userInfoKeysAreNamedButValuesAreNotLeaked() throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(
                label: "x",
                status: .error,
                error: Self.nsError(description: "boom",
                                    userInfo: ["TFSecretPayload": "do-not-leak-me"])))

        #expect(detail.contains("userInfoKeys="))
        #expect(detail.contains("TFSecretPayload"))
        #expect(!detail.contains("do-not-leak-me"))
    }

    @Test func completedStatusCarryingAnErrorIsAFailure() throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(label: "x",
                                            status: .completed,
                                            error: Self.nsError(description: "boom")))
        #expect(detail.contains("status=completed"))
        #expect(detail.contains("description=boom"))
    }

    @Test func statusNamesCoverEveryCase() {
        #expect(metalCommandBufferStatusName(.notEnqueued) == "notEnqueued")
        #expect(metalCommandBufferStatusName(.enqueued) == "enqueued")
        #expect(metalCommandBufferStatusName(.committed) == "committed")
        #expect(metalCommandBufferStatusName(.scheduled) == "scheduled")
        #expect(metalCommandBufferStatusName(.completed) == "completed")
        #expect(metalCommandBufferStatusName(.error) == "error")
    }

    @Test func realCompletedCommandBufferDoesNotThrow() throws {
        let context = try MetalContext()
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        commandBuffer.label = "test empty"
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer)
    }

    @Test func realNonCompletedCommandBufferThrows() throws {
        let context = try MetalContext()
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        commandBuffer.label = "test uncommitted"

        do {
            try checkCommandBufferError(commandBuffer)
            Issue.record("expected a failure for a buffer that never completed")
        } catch let error as MetalError {
            #expect("\(error)".contains("test uncommitted"))
        }
    }

    /// The shared formatter covers status-only failures only if every vision
    /// wait routes through it. A direct `buffer.error` check accepts `.error`
    /// with no error object and can consume incomplete GPU output.
    @Test func visionRuntimeUsesStatusAwareCommandBufferChecks() throws {
        let source = try String(contentsOf: Self.visionRuntimeSource, encoding: .utf8)
        #expect(!source.contains("if let error = entry.buffer.error"))
        #expect(!source.contains("if let error = commandBuffer.error"))
        let checkedWaits = source.components(
            separatedBy: "metalCommandBufferFailureDetail(").count - 1
        #expect(checkedWaits == 3,
                "expected all three VisionRuntime waits to preserve status diagnostics")
    }
}
