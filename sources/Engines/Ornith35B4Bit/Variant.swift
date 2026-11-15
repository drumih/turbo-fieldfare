// Provenance: 7.0 Ornith 1.5 35B-A3B 4-bit variant (the 7.0 release focus).
// The GPU engine in this target is a spec-sanctioned copy (§1.2) of the
// qwen36-family runtime; the generic pieces (MetalContext, manifest, geometry)
// come from TinyTitanCore. Statics grounded in the install manifest
// (hidden 2048, 40 layers, 256 experts top-8, 16 Q / 2 KV heads).
import Foundation
import Metal
import TinyTitanCore

/// The Ornith 1.5 35B-A3B at 4-bit, served by the qwen36-family GPU engine with
/// SSD expert streaming.
/// unchecked-invariant: the runner, model and Metal context are owned by one
/// variant instance and driven serially by a single generation session; the
/// async engine calls are bridged to the sync EngineVariant surface one at a
/// time, never concurrently.
public final class Ornith35B4BitVariant: EngineVariant, @unchecked Sendable {
    public static let id: VariantID = {
        do { return try VariantID("ornith-1.5-35b-4bit") } catch {
            fatalError("invalid variant id literal: \(error)")
        }
    }()

    public static let source = SourcePin(
        repo: "ornith-ai/Ornith-1.5-35B-A3B",
        sha: "10fbf86fed7ecee4a061f8b499a618f46001cac1")

    public static let geometry = GeometryContract(required: [
        "family": .string("qwen36"),
        "hiddenSize": .int(2048),
    ])

    public static let quant: VariantQuant = {
        do { return VariantQuant(defaultSpec: try QuantSpec(bits: 4, group: 64)) } catch {
            fatalError("invalid quant spec literal: \(error)")
        }
    }()

    public static let performance = PerformanceProfile(
        expertCacheBytes: 8 << 30, prefetchDepth: 32, prefillChunk: 128,
        routerTopKSimd: true, attentionSimdPartial: true, keepCacheWired: true)

    public static let sampling = SamplingProfile(
        temperature: 0.6, topK: 20, topP: 0.95, presencePenalty: 0, thinking: nil)

    public static let recipe = ModelRecipe(
        variantID: id, renames: [:], keepBF16Stems: [],
        headBits: nil, embeddingBits: nil)

    private let model: Model
    private let runner: RealForwardRunner
    private let device: MTLDevice
    private let logits: MTLBuffer
    private var position: Int = 0
    private var lastToken: Int32 = 0

    public init(context: EngineContext, model handle: LoadedModelHandle) throws {
        guard let directory = handle.directoryURL else {
            throw VariantError.notPorted(Self.id.raw)
        }
        let dev = context.metal ?? MTLCreateSystemDefaultDevice()
        guard let dev else { throw MetalError.noDevice }
        device = dev
        let metalContext = try MetalContext(device: dev)
        model = try Model.load(directoryURL: directory, device: dev)
        runner = try RealForwardRunner(
            model: model, context: metalContext, maxContext: 4096)
        let vocab = runner.cfg.vocabSize
        guard
            let buf = dev.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("ornith logits")
        }
        logits = buf
    }

    /// Greedy generation: prefill the prompt, then argmax-decode `maxNew` tokens.
    public func generate(prompt: [Int], maxNew: Int) async throws -> [Int] {
        runner.reset()
        let ids = prompt.map { Int32($0) }
        let result = try await runner.prefillChunked(
            tokens: ids[...], startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: PrefillRuntimeConfig.defaultChunked,
            into: logits, onProgress: { _ in })
        position = result.newPosition
        var produced: [Int] = []
        var next: Int32
        switch result.seed {
        case .greedyToken(let t): next = Int32(t)
        case .logitsWritten: next = argmax()
        }
        for _ in 0..<maxNew {
            produced.append(Int(next))
            lastToken = next
            try await runner.produce(token: next, position: position, into: logits)
            position += 1
            next = argmax()
        }
        return produced
    }

    private func argmax() -> Int32 {
        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: runner.cfg.vocabSize)
        var best = 0
        var bestVal = ptr[0]
        for i in 1..<runner.cfg.vocabSize where ptr[i] > bestVal {
            bestVal = ptr[i]; best = i
        }
        return Int32(best)
    }

    // MARK: EngineVariant sync surface (bridged to the async engine)

    /// unchecked-invariant: the box is published once before the semaphore is
    /// signalled and read only after, so the semaphore orders every access.
    private final class Bridge: @unchecked Sendable {
        var error: Error?
        var token: Int32 = 0
        let sem = DispatchSemaphore(value: 0)
    }

    private func runSync(_ body: @escaping @Sendable (Bridge) async -> Void) throws -> Int32 {
        let box = Bridge()
        Task {
            await body(box); box.sem.signal()
        }
        box.sem.wait()
        if let error = box.error { throw error }
        return box.token
    }

    public func prefill(_ tokens: [Int], state: inout SessionState) throws {
        let first = try runSync { box in
            do {
                self.runner.reset()
                let ids = tokens.map { Int32($0) }
                let r = try await self.runner.prefillChunked(
                    tokens: ids[...], startPosition: 0,
                    outputMode: .greedyIfAvailable,
                    config: PrefillRuntimeConfig.defaultChunked,
                    into: self.logits, onProgress: { _ in })
                self.position = r.newPosition
                switch r.seed {
                case .greedyToken(let t): box.token = Int32(t)
                case .logitsWritten: box.token = self.argmax()
                }
            } catch { box.error = error }
        }
        lastToken = first
        state.position = position
        state.tokens = tokens
    }

    public func decode(_ state: inout SessionState) throws -> Int {
        let token = lastToken
        let next = try runSync { box in
            do {
                try await self.runner.produce(
                    token: token, position: self.position, into: self.logits)
                self.position += 1
                box.token = self.argmax()
            } catch { box.error = error }
        }
        lastToken = next
        state.position = position
        state.tokens.append(Int(token))
        return Int(token)
    }

    public static func qualify() throws -> Qualification {
        throw VariantError.notPorted(id.raw)
    }
}
