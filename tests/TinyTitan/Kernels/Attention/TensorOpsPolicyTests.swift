import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// K0's policy is a pure function of a capability value and a mode, so these
/// tests never ask the host what it is: every capability below is synthetic.
@Suite struct TensorOpsPolicyTests {
    private func capabilities(
        apple10: Bool,
        metal4: Bool = true,
        ane: Bool = true,
        ram: UInt64 = 24 << 30
    ) -> HardwareCapabilities {
        HardwareCapabilities(
            chipModel: "MacTest,1",
            gpuName: "Synthetic GPU",
            gpuFamily: apple10 ? .apple10 : .apple9,
            supportsApple10TensorOps: apple10,
            metal4Available: metal4,
            anePresent: ane,
            physicalMemoryBytes: ram,
            osVersion: "26.0.0")
    }

    @Test func syntheticValueCarriesEveryProbedFact() {
        let caps = self.capabilities(apple10: false, metal4: false, ane: false, ram: 8 << 30)
        #expect(caps.chipModel == "MacTest,1")
        #expect(caps.gpuFamily == .apple9)
        #expect(!caps.supportsApple10TensorOps)
        #expect(!caps.metal4Available)
        #expect(!caps.anePresent)
        #expect(caps.physicalMemoryBytes == 8 << 30)
        #expect(HardwareCapabilities.GPUFamily.apple9 < .apple10)
        #expect(HardwareCapabilities.GPUFamily.apple11 > .apple10)
    }

    @Test func autoFallsBackWithoutTheCapability() {
        let plan = TensorOpsPolicy.plan(
            capabilities: self.capabilities(apple10: false), mode: .auto)
        #expect(!plan.attemptsPipeline)
        #expect(plan.reason.contains("does not support Apple10"))
    }

    @Test func autoAttemptsWithTheCapability() {
        let plan = TensorOpsPolicy.plan(
            capabilities: self.capabilities(apple10: true), mode: .auto)
        #expect(plan.attemptsPipeline)
        #expect(plan.reason.contains("available"))
    }

    @Test func offNeverAttemptsEvenWithTheCapability() {
        let plan = TensorOpsPolicy.plan(
            capabilities: self.capabilities(apple10: true), mode: .off)
        #expect(!plan.attemptsPipeline)
        #expect(plan.reason.contains("TINYTITAN_TENSOROPS=off"))
    }

    @Test func onAttemptsWithoutTheCapability() {
        let plan = TensorOpsPolicy.plan(
            capabilities: self.capabilities(apple10: false), mode: .on)
        #expect(plan.attemptsPipeline)
        #expect(plan.reason.contains("forced"))
    }

    @Test func modeParsingIsExplicitAndATypoIsReported() {
        #expect(TensorOpsMode.resolve(environment: [:]).mode == .auto)
        #expect(TensorOpsMode.resolve(environment: ["TINYTITAN_TENSOROPS": "auto"]).mode == .auto)
        #expect(TensorOpsMode.resolve(environment: ["TINYTITAN_TENSOROPS": "off"]).mode == .off)
        #expect(TensorOpsMode.resolve(environment: ["TINYTITAN_TENSOROPS": "on"]).mode == .on)
        let typo = TensorOpsMode.resolve(
            environment: ["TINYTITAN_TENSOROPS": "onnn"])
        #expect(typo.mode == .auto)
        #expect(typo.note?.contains("not auto, off, or on") == true)
    }

    @Test func decisionLogLineNamesPathModeAndReason() {
        let decision = TensorOpsDecision(
            mode: .auto, path: .fallback,
            reason: "device does not support Apple10 MPP tensor operations")
        #expect(decision.logLine.contains("tensor-ops prefill: fallback"))
        #expect(decision.logLine.contains("mode=auto"))
        #expect(decision.logLine.contains("does not support Apple10"))
    }
}
