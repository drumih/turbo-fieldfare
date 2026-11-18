import Testing
import Foundation
import Metal
@testable import TinyTitanCore
@testable import DenseGPU
@testable import TinyTitan

/// The dense-GPU engine's byte-parity oracle (TT-097 Phase 3 Task 13, the
/// restored `TINYTITAN_DENSE_GPU_EQUIV` bar): the shared DenseGPU engine and
/// the 6.x runtime GPU engine — the same config-driven runner DenseGPU was
/// copied from — driven on the SAME dense install with the SAME greedy
/// request must produce IDENTICAL tokens. This is the engine pair that can
/// legitimately be byte-identical (same kernels, same graph, same summation
/// order); the dense CPU engine is a different hardware path and is
/// deliberately NOT the reference (the server-vs-CLI disagreement across that
/// boundary is expected and recorded).
///
/// Both legs force the logits head (`forceLogitsHead: true`) as the server
/// does, so the greedy choice is read from written logits on both sides and
/// no fused-head shortcut can mask an engine difference. The two legs are
/// written out concretely because both modules export a `RealForwardRunner`
/// class; one shared generic would make the type lookup ambiguous.
///
/// Model-gated: set TINYTITAN_DENSE_GPU_PARITY_MODEL to a dense `.ssdai`
/// install (e.g. models/qwen3.5_4B_4Bit).
struct DenseGPUParityTests {
    private static func install() -> URL? {
        guard let p = ProcessInfo.processInfo.environment["TINYTITAN_DENSE_GPU_PARITY_MODEL"]
        else { return nil }
        return URL(fileURLWithPath: p)
    }

    private static func makeLogits(_ dev: MTLDevice, vocab: Int) throws -> MTLBuffer {
        guard
            let logits = dev.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("dense-GPU parity logits")
        }
        return logits
    }

    private static func argmax(_ logits: MTLBuffer, vocab: Int) -> Int32 {
        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        var best = 0
        var bv = ptr[0]
        for i in 1..<vocab where ptr[i] > bv { bv = ptr[i]; best = i }
        return Int32(best)
    }

    /// The 6.x oracle leg: the runtime's own GPU engine at the manifest's
    /// dense geometry.
    private static func referenceGreedy(
        directory: URL, prompt: [Int], maxNew: Int
    ) async throws -> [Int] {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        // ArchConfig is a TinyTitanCore type; `resolved` is TinyTitan's
        // extension, so both imports are what makes this member visible
        // (MemberImportVisibility).
        let denseArch = try ArchConfig.resolved(
            forFamily: TinyTitan.ModelFamily.qwen35Dense, directoryURL: directory)
        let model = try TinyTitan.Model.load(
            directoryURL: directory, device: dev, expecting: denseArch)
        let ctx = try MetalContext(device: dev)
        let runtime = try TinyTitan.RuntimeConfiguration(forceLogitsHead: true)
        let runner = try TinyTitan.RealForwardRunner(
            model: model, context: ctx, maxContext: 4096,
            runtimeConfiguration: runtime)
        let vocab = runner.cfg.vocabSize
        let logits = try makeLogits(dev, vocab: vocab)
        func argmaxNow() -> Int32 { argmax(logits, vocab: vocab) }

        runner.reset()
        let ids = prompt.map { Int32($0) }
        let r = try await runner.prefillChunked(
            tokens: ids[...], startPosition: 0, outputMode: .logits,
            config: PrefillRuntimeConfig.defaultChunked,
            into: logits, onProgress: { _ in })
        var pos = r.newPosition
        var next = argmaxNow()
        var out: [Int] = []
        for _ in 0..<maxNew {
            out.append(Int(next))
            try await runner.produce(token: next, position: pos, into: logits)
            pos += 1
            next = argmaxNow()
        }
        return out
    }

    /// The engine under test: the shared dense-GPU target, geometry resolved
    /// by its own dense-only load path.
    private static func denseGreedy(
        directory: URL, prompt: [Int], maxNew: Int
    ) async throws -> [Int] {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        let model = try DenseGPU.Model.load(directoryURL: directory, device: dev)
        let ctx = try MetalContext(device: dev)
        let runtime = try DenseGPU.RuntimeConfiguration(forceLogitsHead: true)
        let runner = try DenseGPU.RealForwardRunner(
            model: model, context: ctx, maxContext: 4096,
            runtimeConfiguration: runtime)
        let vocab = runner.cfg.vocabSize
        let logits = try makeLogits(dev, vocab: vocab)
        func argmaxNow() -> Int32 { argmax(logits, vocab: vocab) }

        runner.reset()
        let ids = prompt.map { Int32($0) }
        let r = try await runner.prefillChunked(
            tokens: ids[...], startPosition: 0, outputMode: .logits,
            config: PrefillRuntimeConfig.defaultChunked,
            into: logits, onProgress: { _ in })
        var pos = r.newPosition
        var next = argmaxNow()
        var out: [Int] = []
        for _ in 0..<maxNew {
            out.append(Int(next))
            try await runner.produce(token: next, position: pos, into: logits)
            pos += 1
            next = argmaxNow()
        }
        return out
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_GPU_PARITY_MODEL"] != nil,
            "requires a dense install; set TINYTITAN_DENSE_GPU_PARITY_MODEL"))
    func denseGPUParityAgainst6x() async throws {
        let directory = try #require(Self.install())
        let prompt = [1, 100, 200, 300, 400]
        let maxNew = 24

        let expected = try await Self.referenceGreedy(
            directory: directory, prompt: prompt, maxNew: maxNew)
        let produced = try await Self.denseGreedy(
            directory: directory, prompt: prompt, maxNew: maxNew)

        #expect(
            produced == expected,
            "DenseGPU diverged from the 6.x GPU oracle: \(produced) vs \(expected)")
        #expect(!produced.isEmpty, "an empty greedy run asserts nothing")
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_GPU_PARITY_MODEL"] != nil,
            "requires a dense install; set TINYTITAN_DENSE_GPU_PARITY_MODEL"))
    func denseGPUToolPromptParityAgainst6x() async throws {
        let directory = try #require(Self.install())
        let folder = try #require(GFTokenizer.tokenizerFolder(forModelDirectory: directory))
        let tokenizer = try await GFTokenizer.load(from: folder, thinkingMode: .off)
        let prompt = try tokenizer.encodeToolChat(
            messages: [
                GFTokenizer.Message(role: .user, content: "Use get_weather to check Berlin.")
            ],
            tools: [
                .init(
                    name: "get_weather", description: "Get weather",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "city": .object(["type": .string("string")])
                        ]),
                        "required": .array([.string("city")]),
                    ]))
            ])

        let promptIDs = prompt.map(Int.init)
        let expected = try await Self.referenceGreedy(
            directory: directory, prompt: promptIDs, maxNew: 32)
        let produced = try await Self.denseGreedy(
            directory: directory, prompt: promptIDs, maxNew: 32)

        #expect(produced == expected, "tool-prompt tokens diverged: \(produced) vs \(expected)")
    }
}
