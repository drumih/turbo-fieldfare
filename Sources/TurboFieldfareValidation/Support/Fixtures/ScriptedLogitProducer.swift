import Foundation
import Metal
import TurboFieldfare

/// Test `LogitProducer` that writes scripted logits, independent of the kernel
/// stack. The `step` closure maps `(inputToken, callIndex)` to a logit spec, so
/// token-keyed automata can script deterministic greedy sequences regardless of
/// how many tokens the prompt prefilled.
public final class ScriptedLogitProducer: LogitProducer, @unchecked Sendable {
    public enum Step: Sendable {
        case argmax(Int32)
        case vector([Float])
    }

    public let vocabSize: Int
    private let step: @Sendable (Int32, Int) -> Step
    private var calls = 0

    public init(vocabSize: Int, step: @escaping @Sendable (Int32, Int) -> Step) {
        self.vocabSize = vocabSize
        self.step = step
    }

    public func reset() { calls = 0 }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        let spec = step(token, calls)
        calls += 1
        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
        switch spec {
        case .argmax(let token):
            for i in 0..<vocabSize { ptr[i] = Float16(-30.0) }
            if Int(token) >= 0 && Int(token) < vocabSize { ptr[Int(token)] = Float16(30.0) }
        case .vector(let values):
            for i in 0..<vocabSize { ptr[i] = Float16(i < values.count ? values[i] : -30.0) }
        }
    }
}
