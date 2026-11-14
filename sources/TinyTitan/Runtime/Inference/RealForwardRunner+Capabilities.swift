import Foundation
import TinyTitanCore

extension RealForwardRunner {
    /// The capability-injecting form (K0). `package` keeps the seam off the
    /// public surface while letting the CLI and the test target hand in a
    /// synthetic machine.
    package convenience init(
        model: Model, context: MetalContext, maxContext: Int,
        slots: Int = 1,
        runtimeConfiguration: RuntimeConfiguration = .production,
        enableSpeculativeGDN: Bool,
        capabilities: HardwareCapabilities?
    ) throws {
        let bp = Builder()
        let cfg = model.config
        let profile = try Self.buildCore(
            bp, model: model, context: context, maxContext: maxContext, slots: slots,
            runtimeConfiguration: runtimeConfiguration,
            enableSpeculativeGDN: enableSpeculativeGDN,
            capabilities: capabilities)
        try Self.buildScratch(
            bp, model: model, context: context, cfg: cfg, profile: profile)
        self.init(bp)
    }

    /// K0: the one startup line the capability seam produces. The decision is
    /// recorded when the prefill kernel is built, so a forced `on` that fell
    /// back on a pipeline failure reports the fallback, not the attempt.
    ///
    /// K5: the projections and the FFN build their own tensor-ops pipelines
    /// under the same policy, so each records a line too. They are emitted as
    /// one diagnostic (the caller logs the string once) and the subject keeps
    /// them distinguishable.
    package var prefillTensorOpsLogLine: String {
        var lines = [
            prefillAttention.tensorOpsDecision.logLine(subject: "prefill attention")
        ]
        if let projections = prefillMPPAffineInt4 {
            lines.append(
                projections.tensorOpsDecision.logLine(subject: "prefill projections"))
        }
        if let ffn = prefillSharedExpert.tensorOpsDecision {
            lines.append(ffn.logLine(subject: "prefill ffn"))
        }
        return lines.joined(separator: "\n")
    }
}
