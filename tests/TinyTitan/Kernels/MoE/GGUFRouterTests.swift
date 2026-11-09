import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

@Suite struct GGUFRouterTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func routerLogitsMatchTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let router = try GGUFRouter(context: context, arch: arch, model: model)
        var rng = SeedTree(0x9017).key("router-x")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer, let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.bufferAllocationFailed("router buffers")
        }
        try router.encode(commandBuffer: cb, layer: 0, x: xBuffer)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        // CPU reference from the F32 weights.
        let location = try #require(model.tensor(named: "blk.0.ffn_gate_inp.weight"))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
        let weights: [Float] = data.withUnsafeBytes { raw in
            let count = raw.count / 4
            let pointer = raw.bindMemory(to: Float.self)
            return (0..<count).map { pointer[$0] }
        }
        let selected = router.select(topK: arch.topKExperts)
        #expect(selected.experts.count == arch.topKExperts)
        #expect(abs(selected.weights.reduce(0, +) - 1) < 1e-3)

        // The CPU top-10 from the same logits must agree in ranking.
        var cpuLogits = [Float](repeating: 0, count: arch.numExperts)
        let xf = x.map(Float.init)
        for e in 0..<arch.numExperts {
            var acc: Float = 0
            let base = e * arch.hiddenSize
            for n in 0..<arch.hiddenSize { acc += weights[base + n] * xf[n] }
            cpuLogits[e] = acc
        }
        let cpuTop = (0..<arch.numExperts).sorted { cpuLogits[$0] > cpuLogits[$1] }
            .prefix(arch.topKExperts)
        let agreement = Set(cpuTop).intersection(selected.experts).count
        print("router top-10 overlap with CPU: \(agreement)/\(arch.topKExperts)")
        #expect(agreement >= arch.topKExperts - 1)
    }
}
