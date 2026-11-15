import Testing
import Foundation
import Metal
import TinyTitanCore
@testable import Ornith35B4Bit
@testable import TinyTitan

/// Parity of the 7.0 Ornith variant against the 6.x runtime GPU engine on the
/// same install (the 7.0 release gate). Model-gated: set
/// TINYTITAN_ORNITH_PARITY_MODEL to the Ornith 1.5 35B-A3B 4-bit install.
struct OrnithParityTests {
    private static func install() -> URL? {
        guard let p = ProcessInfo.processInfo.environment["TINYTITAN_ORNITH_PARITY_MODEL"]
        else { return nil }
        return URL(fileURLWithPath: p)
    }

    /// The 6.x reference: original runtime Model + RealForwardRunner, greedy.
    private static func referenceGreedy(directory: URL, prompt: [Int], maxNew: Int) async throws
        -> [Int]
    {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        let ctx = try MetalContext(device: dev)
        let model = try TinyTitan.Model.load(directoryURL: directory, device: dev)
        let runner = try TinyTitan.RealForwardRunner(model: model, context: ctx, maxContext: 4096)
        let vocab = runner.cfg.vocabSize
        guard
            let logits = dev.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("parity logits")
        }
        func argmax() -> Int32 {
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            var best = 0; var bv = ptr[0]
            for i in 1..<vocab where ptr[i] > bv { bv = ptr[i]; best = i }
            return Int32(best)
        }
        runner.reset()
        let ids = prompt.map { Int32($0) }
        let r = try await runner.prefillChunked(
            tokens: ids[...], startPosition: 0, outputMode: .greedyIfAvailable,
            config: TinyTitan.PrefillRuntimeConfig.defaultChunked,
            into: logits, onProgress: { _ in })
        var pos = r.newPosition
        var next: Int32
        switch r.seed {
        case .greedyToken(let t): next = Int32(t)
        case .logitsWritten: next = argmax()
        }
        var out: [Int] = []
        for _ in 0..<maxNew {
            out.append(Int(next))
            try await runner.produce(token: next, position: pos, into: logits)
            pos += 1
            next = argmax()
        }
        return out
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_ORNITH_PARITY_MODEL"] != nil,
            "requires the Ornith install; set TINYTITAN_ORNITH_PARITY_MODEL"))
    func greedyParityAgainst6x() async throws {
        let directory = try #require(Self.install())
        let prompt = [1, 100, 200, 300, 400]
        let maxNew = 24

        let expected = try await Self.referenceGreedy(
            directory: directory, prompt: prompt, maxNew: maxNew)

        let context = EngineContext(
            metal: MTLCreateSystemDefaultDevice(),
            physicalRAMBytes: 24 * 1024 * 1024 * 1024)
        let handle = LoadedModelHandle(tensors: [:], directoryURL: directory)
        let variant = try Ornith35B4BitVariant(context: context, model: handle)
        let produced = try await variant.generate(prompt: prompt, maxNew: maxNew)

        #expect(
            produced == expected,
            "7.0 Ornith variant diverged from 6.x: \(produced) vs \(expected)")
    }
}
