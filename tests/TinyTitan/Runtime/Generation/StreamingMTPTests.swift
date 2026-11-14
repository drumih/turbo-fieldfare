import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

@Suite struct StreamingMTPTests {
    @Test func defaultPlanFitsStrictBudget() throws {
        let slots = StreamingMTPMemoryPlan.expertSlots
        let plan = try StreamingMTPMemoryPlan(
            residentTensorBytes: 43 * 1_048_576,
            expertStrideBytes: 1_769_472,
            targetRollbackBytes: 62 * 1_048_576,
            scratchBytes: 8 * 1_048_576)
        #expect(plan.streamedExpertCacheBytes == slots * 1_769_472)
        #expect(plan.draftKVBytes == 128 * 1_048_576)
        // The default budget (384 MiB) must cover the sum of every component.
        #expect(
            plan.requiredBytes == 43 * 1_048_576 + slots * 1_769_472
                + 128 * 1_048_576 + 62 * 1_048_576 + 8 * 1_048_576)
        #expect(plan.requiredBytes < plan.budgetBytes)
    }

    @Test func expertSlotDefaultIsWithinTheAllowedSet() {
        #expect(
            StreamingMTPMemoryPlan.allowedExpertSlots
                .contains(StreamingMTPMemoryPlan.defaultExpertSlots))
        // The env override is read once into a `let`, so a test cannot flip it
        // in-process. What it can pin is that an unset/invalid override leaves
        // the default in force, which is the property callers depend on.
        #expect(
            StreamingMTPMemoryPlan.expertSlots
                == StreamingMTPMemoryPlan.defaultExpertSlots
                || ProcessInfo.processInfo.environment["TINYTITAN_MTP_EXPERT_SLOTS"] != nil)
    }

    @Test func draftKVAccountingUsesSelectedPrecision() throws {
        func bytes(_ precision: KVCachePrecision) throws -> Int {
            try StreamingMTPMemoryPlan(
                residentTensorBytes: 0,
                expertStrideBytes: 0,
                draftKVTokens: 65_536,
                kvCachePrecision: precision,
                targetRollbackBytes: 0,
                scratchBytes: 0
            ).draftKVBytes
        }
        #expect(try bytes(.fp16) == 128 * 1_048_576)
        #expect(try bytes(.int8) == 68 * 1_048_576)
        #expect(try bytes(.int4) == 36 * 1_048_576)
    }

    @Test func oversizedResidentDesignIsRejected() {
        #expect(throws: StreamingMTPError.self) {
            _ = try StreamingMTPMemoryPlan(
                budgetMiB: 256,
                residentTensorBytes: 300 * 1_048_576,
                expertStrideBytes: 1_769_472,
                targetRollbackBytes: 62 * 1_048_576,
                scratchBytes: 8 * 1_048_576)
        }
    }

    @Test func architectureIsOneLayerAndBounded() {
        let arch = ArchConfig.qwen36MTP
        #expect(arch.family == .qwen36MTP)
        #expect(arch.numLayers == 1)
        #expect(arch.fullAttentionLayerMask == [1])
        #expect(arch.slidingWindow == 65_536)
        #expect(arch.topKExperts == 8)
        #expect(!arch.tieWordEmbeddings)
    }

    @Test func statisticsMeasureAcceptedSpeedup() {
        var statistics = MTPStatistics()
        statistics.record(accepted: true, emitted: 2, targetPasses: 1)
        statistics.record(accepted: false, emitted: 1, targetPasses: 1)
        #expect(statistics.acceptanceRate == 0.5)
        #expect(statistics.emittedTokensPerTargetPass == 1.5)
    }

    @Test func modelLineageRejectsCrossModelWeightSharing() {
        #expect(
            Model.mtpLineagesAreCompatible(
                sidecarID: "ornith-1.5-35b-a3b-mtp-4bit",
                targetID: "ornith-1.5-35b-a3b-8bit"))
        #expect(
            Model.mtpLineagesAreCompatible(
                sidecarID: "qwen3.6-35b-a3b-mtp-4bit",
                targetID: "qwen3.6-35b-a3b-4bit"))
        #expect(
            !Model.mtpLineagesAreCompatible(
                sidecarID: "ornith-1.5-35b-a3b-mtp-4bit",
                targetID: "qwen3.6-35b-a3b-4bit"))
    }

    // MARK: - Toy sidecar integration (T27)

    /// The MTP sidecar toy mirrors the real installed sidecar schema: it
    /// loads under the qwen36_mtp expectation, resolves its adapter tensors,
    /// and shares the target's embedding/lm_head buffers (weight sharing, no
    /// copy) via `sharingTargetWeights`.
    @Test func mtpSidecarLoadsAndSharesTargetWeights() throws {
        let targetDir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: targetDir) }
        let sidecarDir = try QwenToySynthetic.writeMTP()
        defer { try? FileManager.default.removeItem(at: sidecarDir) }
        let ctx = try MetalContext()
        let target = try Model.load(
            directoryURL: targetDir,
            device: ctx.device,
            expecting: .qwenToy())
        let sidecar = try Model.load(
            directoryURL: sidecarDir,
            device: ctx.device,
            expecting: .qwenToyMTP())

        #expect(target.config.family == .qwen36)
        #expect(sidecar.config.family == .qwen36MTP)
        // MTP adapter tensors resolve against the sidecar's resident index.
        _ = try sidecar.mtpProjection()
        _ = try sidecar.mtpEmbeddingNorm()
        _ = try sidecar.mtpHiddenNorm()

        let bound = try sidecar.sharingTargetWeights(from: target)
        #expect(try bound.embedding().buffer === target.embedding().buffer)
        #expect(try bound.lmHead().buffer === target.lmHead().buffer)
    }

    /// Full MTP decode-loop smoke on the toy pair: load the target and the
    /// sidecar, construct `StreamingMTPDecoder`, and run a short greedy
    /// completion via `runRawCompletion` (which routes StreamingMTPDecoder
    /// producers through `runStreamingMTPCompletion`). Asserts tokens are
    /// emitted and the acceptance metrics are reported on the decoder.
    @Test func verifyScheduleControlDefaultsToPairAndFailsClosed() throws {
        #expect(try RuntimeMTPVerifySchedule.environmentValue([:]) == .pair)
        #expect(
            try RuntimeMTPVerifySchedule.environmentValue(
                ["TINYTITAN_MTP_VERIFY": "pair"]) == .pair)
        #expect(
            try RuntimeMTPVerifySchedule.environmentValue(
                ["TINYTITAN_MTP_VERIFY": "tile"]) == .tile)
        #expect(throws: StreamingMTPError.self) {
            try RuntimeMTPVerifySchedule.environmentValue(
                ["TINYTITAN_MTP_VERIFY": "fast"])
        }
        #expect(throws: StreamingMTPError.self) {
            try RuntimeMTPVerifySchedule.environmentValue(
                ["TINYTITAN_MTP_VERIFY": ""])
        }
    }

    @Test func mtpDecodeLoopEmitsTokensAndReportsAcceptance() async throws {
        let targetDir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: targetDir) }
        let sidecarDir = try QwenToySynthetic.writeMTP()
        defer { try? FileManager.default.removeItem(at: sidecarDir) }
        let ctx = try MetalContext()
        let tokenizer = try await GFTokenizer.load(
            from: ChatMLTemplateTests.fixtureFolder())
        let target = try Model.load(
            directoryURL: targetDir,
            device: ctx.device,
            expecting: .qwenToy())
        let sidecar = try Model.load(
            directoryURL: sidecarDir,
            device: ctx.device,
            expecting: .qwenToyMTP())
        let decoder = try StreamingMTPDecoder(
            targetModel: target,
            mtpSidecar: sidecar,
            context: ctx,
            maxContext: 64)
        let scratch = try RawCompletionScratch(
            context: ctx,
            vocab: tokenizer.vocabSize)

        var emitted = 0
        let result = try await runRawCompletion(
            producer: decoder,
            tokenizer: tokenizer,
            promptIds: [1, 2],
            config: GenerationConfig(maxNewTokens: 4, temperature: 0),
            context: ctx,
            scratch: scratch,
            onProgress: { progress in
                if case .token = progress { emitted += 1 }
            })

        #expect(result.newTokens > 0, "MTP loop emitted no tokens")
        #expect(emitted > 0)
        // Greedy verification always runs at least one advance for a 4-token
        // budget, so the acceptance statistics are reported (drafted >= 1).
        #expect(
            decoder.statistics.draftedTokens > 0,
            "no draft/verify cycle ran; acceptance metrics were not reported")
        #expect(decoder.statistics.acceptedTokens >= 0)
        #expect(decoder.statistics.emittedTokens >= result.newTokens)
        #expect(decoder.targetPosition == decoder.target.continuationPosition)
    }
}
