import Foundation
import Metal
import Synchronization
import TinyTitanCore

/// Test `LogitProducer` that writes scripted logits, independent of the kernel
/// stack. The `step` closure maps `(inputToken, callIndex)` to a logit spec, so
/// token-keyed automata can script deterministic greedy sequences regardless of
/// how many tokens the prompt prefilled.
/// unchecked-invariant: a test fixture replaying a fixed script, driven by one
/// test at a time. Not used in production.
public final class ScriptedLogitProducer: LogitProducer, @unchecked Sendable {
    public enum Step: Sendable {
        case argmax(Int32)
        case vector([Float])
    }

    public let vocabSize: Int
    private let step: @Sendable (Int32, Int) -> Step
    // `produce` may be invoked concurrently by the generation loop, so the
    // call counter is lock-protected rather than relying on @unchecked
    // Sendable for unsynchronized mutation. The stdlib `Mutex` is used
    // instead of `NSLock` because the read happens in an `async` context.
    private let callsLock = Mutex<Int>(0)

    public init(vocabSize: Int, step: @escaping @Sendable (Int32, Int) -> Step) {
        self.vocabSize = vocabSize
        self.step = step
    }

    public func reset() {
        callsLock.withLock { $0 = 0 }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        let callIndex = callsLock.withLock { value -> Int in
            defer { value += 1 }
            return value
        }
        let spec = step(token, callIndex)
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
