import Foundation
import TinyTitanCore

extension StreamingMTPDecoder {
    public func diagnosticLines(result: TinyTitanCore.RawDecodeResult, includePhases: Bool)
        -> [String]
    {
        let stats = statistics
        let decodeRate =
            result.decodeSeconds > 0
            ? Double(result.newTokens) / result.decodeSeconds : 0
        var lines = [
            String(
                format:
                    "TinyTitan mtp drafted=%d accepted=%d acceptance=%.1f%% "
                    + "target_passes=%d emitted_per_pass=%.3f "
                    + "prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f "
                    + "memory_required_mib=%.1f memory_budget_mib=%.1f",
                stats.draftedTokens, stats.acceptedTokens, stats.acceptanceRate * 100,
                stats.targetBackbonePasses, stats.emittedTokensPerTargetPass,
                result.prefillSeconds, result.decodeSeconds, decodeRate,
                Double(memoryPlan.requiredBytes) / 1_048_576,
                Double(memoryPlan.budgetBytes) / 1_048_576)
        ]
        if includePhases, stats.targetBackbonePasses > 0 {
            let passes = Double(stats.targetBackbonePasses)
            let ms: (UInt64) -> Double = { Double($0) / passes / 1_000_000 }
            lines.append(
                String(
                    format:
                        "TinyTitan mtp-phases per_pass_ms proposal=%.3f checkpoint=%.3f "
                        + "verify=%.3f verify_backbone=%.3f verify_head=%.3f "
                        + "verify_argmax=%.3f commit=%.3f rollback=%.3f passes=%d",
                    ms(stats.proposalNanos), ms(stats.checkpointNanos), ms(stats.verifyNanos),
                    ms(stats.verifyBackboneNanos), ms(stats.verifyHeadNanos),
                    ms(stats.verifyArgmaxNanos), ms(stats.commitNanos), ms(stats.rollbackNanos),
                    stats.targetBackbonePasses))
        }
        return lines
    }
}
