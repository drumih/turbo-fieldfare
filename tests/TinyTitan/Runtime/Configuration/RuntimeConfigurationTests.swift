import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct RuntimeConfigurationTests {
    @Test func publicContextChoicesReachQwenMaximum() {
        #expect(
            RuntimeConfiguration.supportedContextTokens
                == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
        #expect(
            RuntimeConfiguration.supportedContextTokens.last
                == RuntimeConfiguration.nativeMaximumContextTokens)
        #expect(
            RuntimeConfiguration.supportedYaRNContextTokens == [
                524_288, 1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216,
            ])
    }

    /// The expert-cache floor is the list's own smallest rung, and the list is
    /// ascending.
    ///
    /// The first half is what stops the named constant and the literal drifting
    /// apart -- the floor decides the cache for a model whose manifest cannot be
    /// read, so a silent 8 that is no longer the smallest rung is a wrong cache
    /// size with nothing to say so. The second half is what the derivations
    /// assume: the nearest-rung walk takes the earliest element on a tie, which
    /// is the *smaller* rung only while the list climbs.
    @Test func expertCacheSlotFloorIsTheSmallestRung() {
        #expect(
            RuntimeConfiguration.allowedExpertCacheSlots.first
                == RuntimeConfiguration.minimumExpertCacheSlots)
        #expect(RuntimeConfiguration.minimumExpertCacheSlots == 8)
        #expect(
            RuntimeConfiguration.allowedExpertCacheSlots
                == RuntimeConfiguration.allowedExpertCacheSlots.sorted())
        #expect(
            Set(RuntimeConfiguration.allowedExpertCacheSlots).count
                == RuntimeConfiguration.allowedExpertCacheSlots.count)
        // Every rung is a legal `expertCacheSlots:` value, so the floor is not a
        // number the validator would refuse.
        let runtime = try? RuntimeConfiguration(expertCacheSlots: 8)
        #expect(runtime?.expertCacheSlots == 8)
    }

    @Test func productionDefaultsAreStable() throws {
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: 16,
            expertCachePolicy: .lfu,
            rdadvisePolicy: .off,
            prefillEnabled: true,
            prefillChunkTokens: 128,
            prefillAttentionPath: .fullTensorOps2DPreferred,
            forceLogitsHead: false)
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
        #expect(runtime.decodeExpertExecution == .hitFixup)
        #expect(runtime.kvCachePrecision == .int8)
        #expect(runtime.ropeScalingMode == .none)
    }

    @Test func contextScalingValidationIsFailClosed() throws {
        let native = try RuntimeConfiguration()
        try native.validate(maxContext: 262_144)
        #expect(throws: RuntimeConfigurationError.self) {
            try native.validate(maxContext: 524_288)
        }
        let yarn = try RuntimeConfiguration(
            ropeScalingMode: .yarn,
            yarnContextTokens: 524_288)
        try yarn.validate(maxContext: 524_288)
        #expect(throws: RuntimeConfigurationError.self) {
            try yarn.validate(maxContext: 1_048_576)
        }
    }

    @Test func retainedControlsReachTypedRuntime() throws {
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true,
            decodeExpertExecution: .barrier)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
        #expect(runtime.decodeExpertExecution == .barrier)
    }

    @Test func decodeExpertExecutionEnvironmentIsFailClosed() throws {
        #expect(try RuntimeDecodeExpertExecution.environmentValue([:]) == .hitFixup)
        #expect(
            try RuntimeDecodeExpertExecution.environmentValue([
                "TINYTITAN_DECODE_EXPERT_EXECUTION": "barrier"
            ]) == .barrier)
        #expect(
            try RuntimeDecodeExpertExecution.environmentValue([
                "TINYTITAN_DECODE_EXPERT_EXECUTION": "gpu-residency"
            ]) == .gpuResidency)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeDecodeExpertExecution.environmentValue([
                "TINYTITAN_DECODE_EXPERT_EXECUTION": "typo"
            ])
        }
    }

    @Test func expertIOSynchronizationEnvironmentIsFailClosed() throws {
        #expect(try RuntimeExpertIOSynchronization.environmentValue([:]) == .host)
        #expect(
            try RuntimeExpertIOSynchronization.environmentValue([
                "TINYTITAN_EXPERT_IO_SYNC": "event"
            ]) == .event)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeExpertIOSynchronization.environmentValue([
                "TINYTITAN_EXPERT_IO_SYNC": "typo"
            ])
        }
    }

    /// The keep-warm knob is off by default and fail-closed on anything that is
    /// not a bounded integer: an operator who sets `200.0` or `abc` and gets a
    /// silently disabled arm would read a control as a measurement.
    @Test func gpuKeepWarmEnvironmentIsOffByDefaultAndFailClosed() throws {
        #expect(try RuntimeGPUKeepWarm.environmentValue([:]) == nil)
        #expect(
            try RuntimeGPUKeepWarm.environmentValue([
                "TINYTITAN_GPU_KEEPWARM_US": "200"
            ]) == 200)
        #expect(
            try RuntimeGPUKeepWarm.environmentValue([
                "TINYTITAN_GPU_KEEPWARM_US": "\(RuntimeGPUKeepWarm.minimumIntervalUs)"
            ]) == RuntimeGPUKeepWarm.minimumIntervalUs)
        #expect(
            try RuntimeGPUKeepWarm.environmentValue([
                "TINYTITAN_GPU_KEEPWARM_US": "\(RuntimeGPUKeepWarm.maximumIntervalUs)"
            ]) == RuntimeGPUKeepWarm.maximumIntervalUs)
        for rejected in ["abc", "200.0", "0", "99", "20001", "999999", "-200", ""] {
            // The knob moved to TinyTitanCore with the TT-097 engine contract,
            // and Core cannot name the runtime's RuntimeConfigurationError, so
            // it throws its own. The fail-closed intent and the message are
            // the assertions; the type follows the module.
            #expect(throws: GPUKeepWarmError.self) {
                try RuntimeGPUKeepWarm.environmentValue([
                    "TINYTITAN_GPU_KEEPWARM_US": rejected
                ])
            }
        }
    }

    @Test func expertIOSubmissionEnvironmentIsFailClosed() throws {
        #expect(try RuntimeExpertIOSubmission.environmentValue([:]) == .deferred)
        #expect(
            try RuntimeExpertIOSubmission.environmentValue([
                "TINYTITAN_EXPERT_IO_SUBMISSION": "immediate"
            ]) == .immediate)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeExpertIOSubmission.environmentValue([
                "TINYTITAN_EXPERT_IO_SUBMISSION": "typo"
            ])
        }
    }

    @Test func expertIOTimelineEnvironmentIsFailClosed() throws {
        // The shipped behaviour is one model-wide timeline; splitting it is opt-in.
        #expect(try RuntimeExpertIOTimeline.environmentValue([:]) == .shared)
        #expect(
            try RuntimeExpertIOTimeline.environmentValue([
                "TINYTITAN_EXPERT_IO_TIMELINE": "per-layer"
            ]) == .perLayer)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeExpertIOTimeline.environmentValue([
                "TINYTITAN_EXPERT_IO_TIMELINE": "typo"
            ])
        }
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1_024, 2_048, 4_096])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) throws {
        let runtime = try RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
