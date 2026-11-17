import Foundation
import Metal
import TinyTitan
import TinyTitanLib
import TinyTitanLegacyEngine

/// Measures one calibration arm by loading its engine and timing a fresh
/// generation. Loading is outside the measurement: the sample is the engine's
/// own `completionTokens / decodeSeconds`, so a slow install does not look like
/// a slow arm.
///
/// Every arm is a load-time decision in this tree (cache slots and prefill
/// chunk), so each sample owns an engine and unloads it afterwards. That is
/// also why the bounded-I/O arm is not measured here: it is a process
/// environment switch, and flipping it inside a process would leak into every
/// other engine.
public struct CalibrationMeasurer: Sendable {
    public let directory: URL
    public let contextTokens: Int
    public let tokens: Int
    public let device: MTLDevice

    public init(
        directory: URL, contextTokens: Int, tokens: Int,
        device: MTLDevice
    ) {
        self.directory = directory
        self.contextTokens = contextTokens
        self.tokens = tokens
        self.device = device
    }

    public enum MeasureError: Error, CustomStringConvertible {
        case noDecodeTime(arm: String)

        public var description: String {
            switch self {
            case .noDecodeTime(let arm):
                return "arm \(arm) produced no decode time"
            }
        }
    }

    /// One sample for one arm, in decode tokens per second.
    public func measure(_ arm: CalibrationArm) async throws -> Double {
        let configuration = EngineConfiguration(
            contextWindow: contextTokens,
            cachePrecision: .eightBit,
            maxConcurrentGenerations: 1,
            prefillChunkTokens: arm.knobs.prefillChunkTokens,
            expertCacheSlots: arm.knobs.expertCacheSlots,
            forceLogitsHead: true,
            logSink: { _ in })
        let engine = try await Engine(
            directory: directory, device: device,
            factory: EngineFactories.sixXGPU(), configuration: configuration)
        let session = await engine.session(
            system: "You are a calibration probe. Answer in one paragraph.")
        let warm = [
            ChatMessage(
                role: .user,
                content: "Warm the engine by naming three colours, then stop.")
        ]
        _ = try await session.respond(
            to: warm,
            options: GenerationOptions(maxTokens: 8, temperature: 0),
            onEvent: { _ in })
        let prompt = [
            ChatMessage(
                role: .user,
                content:
                    "Explain how a CPU cache hierarchy works, covering L1, L2, "
                    + "L3, inclusion and coherence, in one paragraph.")
        ]
        let summary = try await session.respond(
            to: prompt,
            options: GenerationOptions(maxTokens: tokens, temperature: 0),
            onEvent: { _ in })
        await engine.unload()
        guard summary.decodeSeconds > 0, summary.completionTokens > 0 else {
            throw MeasureError.noDecodeTime(arm: arm.name)
        }
        return Double(summary.completionTokens) / summary.decodeSeconds
    }
}
