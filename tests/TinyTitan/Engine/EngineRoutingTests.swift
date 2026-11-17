import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore
@testable import TinyTitanLib

@Suite struct EngineRoutingTests {
    @Test func serverSessionLoadsThroughAnInjectedEngineFactory() async throws {
        let directory = try QwenToySynthetic.write(weightBits: 4, qsa: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenizer = directory.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.copyItem(
            at: ChatMLTemplateTests.fixtureFolder(), to: tokenizer)
        let factory = EngineFactory(
            buildModel: { directory, arch, context in
                StubEngineModel(
                    try Model.load(
                        directoryURL: directory,
                        device: try #require(context).device,
                        expecting: arch))
            },
            buildRunner: { model, options in
                let concrete = try #require((model as? StubEngineModel)?.model)
                return StubEngineRunner(
                    try RealForwardRunner(
                        model: concrete,
                        context: try #require(options.context),
                        maxContext: options.maxContext,
                        slots: options.slots,
                        runtimeConfiguration: RuntimeConfiguration()))
            })

        let session = try await ServerModelSession.load(
            modelDirectory: directory,
            factory: factory,
            maxContext: 64,
            warmExperts: false)

        #expect(session.maximumContext == 64)
        #expect(await session.model is StubEngineModel)
        #expect(await session.runner is StubEngineRunner)
    }
}

private struct StubEngineModel: EngineModel {
    let model: Model

    init(_ model: Model) { self.model = model }
    var config: ArchConfig { model.config }
    var modelID: String { model.modelID }
    var sourceSnapshotHash: String? { model.sourceSnapshotHash }
    var routedExpertWeightBits: Int { model.routedExpertWeightBits }
    func routedExpertCacheSlotCount() -> Int? { model.routedExpertCacheSlotCount() }
    func saveExpertUsageProfile() { model.saveExpertUsageProfile() }
}

private final class StubEngineRunner: EngineRunner, @unchecked Sendable {
    let runner: RealForwardRunner

    init(_ runner: RealForwardRunner) { self.runner = runner }
    var statistics: EngineStatistics { runner.statistics }
    func resetStatistics() { runner.resetStatistics() }
    var continuationPosition: Int { runner.continuationPosition }
    func prepareForContinuation(expectedPosition: Int) throws {
        try runner.prepareForContinuation(expectedPosition: expectedPosition)
    }
    var maxContext: Int { runner.maxContext }
    func reset() { runner.reset() }
    func reset(slot: Int) { runner.reset(slot: slot) }
    func resetSequence(slot: Int) async { await runner.resetSequence(slot: slot) }
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await runner.produce(token: token, position: position, into: logits)
    }
    func produce(
        token: Int32, position: Int, slot: Int, into logits: MTLBuffer
    ) async throws {
        try await runner.produce(token: token, position: position, slot: slot, into: logits)
    }
    func produceBatch(
        _ rows: [(token: Int32, position: Int, slot: Int)], logits: [MTLBuffer]
    ) async throws {
        try await runner.produceBatch(rows, logits: logits)
    }
    func prefillChunked(
        tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig, into logits: MTLBuffer, onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await runner.prefillChunked(
            tokens: tokens, startPosition: startPosition, outputMode: outputMode,
            config: config, into: logits, onProgress: onProgress)
    }
    func prefillChunked(
        tokens: ArraySlice<Int32>, startPosition: Int, slot: Int,
        outputMode: PrefillOutputMode, config: PrefillRuntimeConfig,
        into logits: MTLBuffer, onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await runner.prefillChunked(
            tokens: tokens, startPosition: startPosition, slot: slot,
            outputMode: outputMode, config: config, into: logits, onProgress: onProgress)
    }
    var supportsInferenceStateRestore: Bool { runner.supportsInferenceStateRestore }
    func captureInferenceState(maximumBytes: Int?) throws -> InferenceStateSnapshot {
        try runner.captureInferenceState(maximumBytes: maximumBytes)
    }
    func restoreInferenceState(
        _ snapshot: InferenceStateSnapshot, auxiliary: InferenceAuxiliaryState?
    ) throws {
        try runner.restoreInferenceState(snapshot, auxiliary: auxiliary)
    }
    func captureAuxiliaryState() throws -> InferenceAuxiliaryState? {
        try runner.captureAuxiliaryState()
    }
    func decodeExpertIO() -> (hits: UInt64, misses: UInt64, bytes: UInt64)? {
        runner.decodeExpertIO()
    }
    func expertStreamingStatistics() -> ExpertStreamingStatistics {
        runner.expertStreamingStatistics()
    }
    var prefetchRingSummary: String? { runner.prefetchRingSummary }
    var prefillTensorOpsLogLine: String { runner.prefillTensorOpsLogLine }
    func kernelGPUTimingSummary() -> [(role: String, millis: Double, count: Int)] {
        runner.kernelGPUTimingSummary()
    }
    func kernelGPUOccupancy() -> (busyMillis: Double, spanMillis: Double) {
        runner.kernelGPUOccupancy()
    }
    func kernelGPUGaps() -> [(transition: String, millis: Double, count: Int)] {
        runner.kernelGPUGaps()
    }
    func resetKernelGPUTimings() { runner.resetKernelGPUTimings() }
}
