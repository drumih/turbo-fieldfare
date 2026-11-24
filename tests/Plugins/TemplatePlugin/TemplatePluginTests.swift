import Foundation
import Testing
import TinyTitanCore
import TinyTitanPluginAPI

@testable import TemplatePlugin

/// The golden base (TPP/1): every requirement the template implements, checked
/// without a model, so a plugin copied from it starts from code that passes.
@Suite struct TemplatePluginTests {
    let install = InstallProbe(
        directory: URL(fileURLWithPath: "/nonexistent/template"), manifest: nil,
        declaredFamily: TemplatePlugin.family)

    @Test func probesOnlyItsOwnFamily() {
        #expect(TemplatePlugin.probe(install) == .exact)
        let other = InstallProbe(
            directory: install.directory, manifest: nil, declaredFamily: "qwen36")
        #expect(TemplatePlugin.probe(other) == nil)
        #expect(TemplatePlugin.protocolVersion == .current)
    }

    @Test func declaresEveryParameterItCannotHonour() {
        let table = TemplatePlugin.parameters(for: install)
        #expect(!table.kvCachePrecision.accepts(.int8))
        #expect(table.kvCachePrecision.accepts(.fp16))
        #expect(!table.concurrentSequences.accepts(2))
        #expect(!table.expertStreaming.accepts(true))
        #expect(table.ropeScaling.accepts(.none))
    }

    @Test func theLoadPlanHonoursTheWindowAndTheBudget() {
        #expect(TemplateLoadPlan.make(requested: 262_144, ramBudget: nil).context == 4_096)
        #expect(TemplateLoadPlan.make(requested: 1_024, ramBudget: nil).context == 1_024)
        let budgeted = TemplateLoadPlan.make(requested: 4_096, ramBudget: 400)
        #expect(budgeted.context == 100)
        #expect(budgeted.plannedBytes <= 400)
    }

    @Test func theRunnerCountsAndRefusesAWrongPosition() throws {
        let runner = TemplateRunner(
            model: TemplateModel(directory: install.directory), maxContext: 16)
        try runner.feed([3, 4], at: 0)
        #expect(runner.nextGreedy() == 5)
        #expect(runner.logits()[5] == 1)
        #expect(throws: TemplateError.position(expected: 2, got: 0)) {
            try runner.feed([9], at: 0)
        }
    }

    @Test func aSnapshotRestoresTheSameNextToken() throws {
        let runner = TemplateRunner(
            model: TemplateModel(directory: install.directory), maxContext: 16)
        try runner.feed([7, 8, 9], at: 0)
        let snapshot = try runner.captureInferenceState(maximumBytes: nil)
        runner.reset()
        #expect(runner.continuationPosition == 0)
        try runner.restoreInferenceState(snapshot, auxiliary: nil)
        #expect(runner.continuationPosition == 3)
        #expect(runner.nextGreedy() == 10)
        #expect(throws: InferenceStateSnapshotError.exceedsLimit(bytes: 12, limit: 4)) {
            try runner.captureInferenceState(maximumBytes: 4)
        }
    }

    @Test func theFactoryPlansThroughThePlugin() throws {
        let factory = TemplatePlugin.engineFactory(
            host: HostServices(log: { _ in }, cacheDirectory: URL(fileURLWithPath: "/tmp")))
        let options = SessionBuildOptions(
            modelDirectory: install.directory, context: nil, maxContext: 8_192, slots: 1,
            promptCacheMaximumEntries: 1, promptCacheMemoryLimitBytes: 0,
            promptCacheDiskDirectory: nil, promptCacheDiskLimitBytes: 0, prefillChunkTokens: nil,
            kvCachePrecision: .fp16, hostKVWindowTokens: nil, ropeScalingMode: .none,
            thinkingMode: .off, reasoningEffort: nil, expertCacheSlots: nil,
            expertCacheBudgetBytes: nil, mtpModelDirectory: nil, mtpMemoryMiB: 0,
            integrityPolicy: nil, rdAdvicePolicy: nil, forceLogitsHead: true)
        let plan = try factory.prepareLoad(options)
        #expect(plan.architecture.vocabSize == TemplateModel.vocabulary)
        #expect((plan.engineConfiguration as? TemplateLoadPlan)?.context == 4_096)
        #expect(try factory.metadata(install.directory).family == TemplatePlugin.family)
    }
}
