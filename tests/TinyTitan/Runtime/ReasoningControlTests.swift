import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The per-family reasoning contract: templates that define only the binary
/// thinking switch reject effort levels instead of faking them, and the
/// Qwen3.8-Flash-Next template's levels resolve with its xhigh default.
@Suite("Reasoning control")
struct ReasoningControlTests {
    @Test("Family capability mapping")
    func familyMapping() {
        #expect(ModelFamily.qwen36.reasoningControl == .binaryThinking)
        #expect(ModelFamily.qwen36MTP.reasoningControl == .binaryThinking)
        #expect(
            ModelFamily.qwen38flash.reasoningControl
                == .thinkingWithEffortLevels(
                    defaultEffort: .xhigh, supported: [.low, .medium, .xhigh]))
        #expect(
            ModelFamily.k2Horizon.reasoningControl
                == .effortOnly(
                    defaultEffort: .high, supported: [.low, .medium, .high]))
    }

    @Test("Nil effort always validates")
    func nilEffortPasses() throws {
        for family in [ModelFamily.qwen36, .qwen36MTP, .qwen38flash] {
            try family.validateReasoning(thinkingMode: .off, effort: nil)
            try family.validateReasoning(thinkingMode: .on, effort: nil)
        }
    }

    @Test("Binary families reject every effort level")
    func binaryFamiliesRejectEffort() {
        for effort in ModelReasoningEffort.allCases {
            #expect(
                throws: ModelReasoningControlError.effortUnsupported(
                    family: .qwen36, effort: effort)
            ) {
                try ModelFamily.qwen36.validateReasoning(
                    thinkingMode: .on, effort: effort)
            }
        }
    }

    @Test("Effort families require thinking on")
    func effortRequiresThinking() throws {
        #expect(
            throws: ModelReasoningControlError.effortRequiresThinkingOn(
                effort: .low)
        ) {
            try ModelFamily.qwen38flash.validateReasoning(
                thinkingMode: .off, effort: .low)
        }
        try ModelFamily.qwen38flash.validateReasoning(
            thinkingMode: .on, effort: .low)
    }

    @Test("Effective effort resolves the template default")
    func effectiveEffort() {
        #expect(
            ModelFamily.qwen38flash.effectiveReasoningEffort(
                thinkingMode: .on, effort: nil) == .xhigh)
        #expect(
            ModelFamily.qwen38flash.effectiveReasoningEffort(
                thinkingMode: .on, effort: .medium) == .medium)
        #expect(
            ModelFamily.qwen38flash.effectiveReasoningEffort(
                thinkingMode: .off, effort: .low) == nil)
        #expect(
            ModelFamily.qwen36.effectiveReasoningEffort(
                thinkingMode: .on, effort: nil) == nil)
    }

    @Test("Environment resolution accepts only the template's levels")
    func environmentResolution() {
        #expect(ModelReasoningEffort.resolved(environment: [:]) == nil)
        #expect(
            ModelReasoningEffort.resolved(
                environment: ["TINYTITAN_REASONING_EFFORT": "XHigh"]) == .xhigh)
        #expect(
            ModelReasoningEffort.resolved(
                environment: ["TINYTITAN_REASONING_EFFORT": "high"]) == .high)
    }

    // MARK: Thinking levels

    @Test("Each family offers exactly the levels its template renders")
    func supportedLevels() {
        #expect(ModelFamily.qwen36.supportedReasoningLevels == [.off, .on])
        #expect(ModelFamily.qwen36MTP.supportedReasoningLevels == [.off, .on])
        #expect(
            ModelFamily.qwen38flash.supportedReasoningLevels
                == [.off, .low, .medium, .xhigh])
        #expect(
            ModelFamily.qwen38flashMTP.supportedReasoningLevels
                == [.off, .low, .medium, .xhigh])
        #expect(
            ModelFamily.k2Horizon.supportedReasoningLevels
                == [.low, .medium, .high])
        #expect(CPUModelFamily.qwen35Dense.supportedReasoningLevels == [.off, .on])
    }

    /// The two effort families do not accept each other's levels: Qwen3.8
    /// raises on `high`, K2 raises on `xhigh`. A shared enum plus a per-family
    /// list is what keeps that from becoming a template exception at runtime.
    @Test("Each effort family rejects the other's level")
    func effortSetsAreNotInterchangeable() {
        #expect(
            throws: ModelReasoningControlError.effortUnsupported(
                family: .qwen38flash, effort: .high)
        ) {
            try ModelFamily.qwen38flash.validateReasoning(
                thinkingMode: .on, effort: .high)
        }
        #expect(
            throws: ModelReasoningControlError.effortUnsupported(
                family: .k2Horizon, effort: .xhigh)
        ) {
            try ModelFamily.k2Horizon.validateReasoning(
                thinkingMode: .on, effort: .xhigh)
        }
        for effort in [ModelReasoningEffort.low, .medium, .high] {
            #expect(throws: Never.self) {
                try ModelFamily.k2Horizon.validateReasoning(
                    thinkingMode: .on, effort: effort)
            }
        }
        // K2 has no thinking-off prompt at all, even without an effort.
        #expect(
            throws: ModelReasoningControlError.thinkingRequired(family: .k2Horizon)
        ) {
            try ModelFamily.k2Horizon.validateReasoning(
                thinkingMode: .off, effort: nil)
        }
        #expect(!ModelFamily.k2Horizon.supportsThinkingOff)
        #expect(ModelFamily.qwen38flash.supportsThinkingOff)
    }

    @Test("Levels map to the thinking switch and effort")
    func runtimeMapping() throws {
        for family in [ModelFamily.qwen36, .qwen36MTP] {
            let off = try family.runtimeReasoning(for: .off)
            #expect(off.thinking == .off && off.effort == nil)
            let on = try family.runtimeReasoning(for: .on)
            #expect(on.thinking == .on && on.effort == nil)
        }
        let off38 = try ModelFamily.qwen38flash.runtimeReasoning(for: .off)
        #expect(off38.thinking == .off && off38.effort == nil)
        let expected: [(ReasoningLevel, ModelReasoningEffort)] =
            [(.low, .low), (.medium, .medium), (.xhigh, .xhigh)]
        for (level, effort) in expected {
            let settings = try ModelFamily.qwen38flash.runtimeReasoning(for: level)
            #expect(settings.thinking == .on && settings.effort == effort)
        }
        let cpuOff = try CPUModelFamily.qwen35Dense.runtimeReasoning(for: .off)
        #expect(cpuOff.thinking == .off && cpuOff.effort == nil)
        let cpuOn = try CPUModelFamily.qwen35Dense.runtimeReasoning(for: .on)
        #expect(cpuOn.thinking == .on && cpuOn.effort == nil)
    }

    @Test("Every supported level yields settings the family validates")
    func mappedSettingsValidate() throws {
        for family in [
            ModelFamily.qwen36, .qwen36MTP, .qwen38flash,
            .qwen38flashMTP, .k2Horizon,
        ] {
            for level in family.supportedReasoningLevels {
                let settings = try family.runtimeReasoning(for: level)
                try family.validateReasoning(
                    thinkingMode: settings.thinking,
                    effort: settings.effort)
            }
        }
    }

    @Test("A level the template does not render throws")
    func unsupportedLevelsThrow() {
        let binary: [ReasoningLevel] = [.off, .on]
        for level in ReasoningLevel.allCases where !binary.contains(level) {
            #expect(
                throws: ReasoningLevelError.unsupported(
                    family: "qwen36", level: level, supported: binary)
            ) {
                try ModelFamily.qwen36.runtimeReasoning(for: level)
            }
            #expect(
                throws: ReasoningLevelError.unsupported(
                    family: "qwen3_5_dense", level: level, supported: binary)
            ) {
                try CPUModelFamily.qwen35Dense.runtimeReasoning(for: level)
            }
        }
        let effort: [ReasoningLevel] = [.off, .low, .medium, .xhigh]
        for level in [ReasoningLevel.on, .minimal, .high, .max] {
            #expect(
                throws: ReasoningLevelError.unsupported(
                    family: "qwen38flash", level: level, supported: effort)
            ) {
                try ModelFamily.qwen38flash.runtimeReasoning(for: level)
            }
        }
    }

    @Test("Display names and wire form")
    func displayNames() throws {
        #expect(
            ReasoningLevel.allCases
                == [.off, .on, .minimal, .low, .medium, .high, .xhigh, .max])
        #expect(ReasoningLevel.xhigh.displayName == "extra high")
        for level in ReasoningLevel.allCases where level != .xhigh {
            #expect(level.displayName == level.rawValue)
        }
        let data = try JSONEncoder().encode([ReasoningLevel.xhigh])
        #expect(data.lossyUTF8String == "[\"xhigh\"]")
        let error = ReasoningLevelError.unsupported(
            family: "qwen38flash", level: .high, supported: [.off, .xhigh])
        #expect(error.description.contains("off, extra high"))
    }

    @Test("CPU family sampling defaults")
    func cpuSamplingDefaults() {
        let sampling = CPUModelFamily.qwen35Dense.samplingDefaults
        #expect(sampling.temperature == 0.6)
        #expect(sampling.topP == 0.95)
        #expect(sampling.topK == GenerationDefaults.topK)
    }
}
