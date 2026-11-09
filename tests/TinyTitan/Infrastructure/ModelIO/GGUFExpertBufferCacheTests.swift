import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The per-expert fetch path: real byte ranges read into Metal buffers, the
/// right block type per matrix, an LRU that evicts, and a fused GEMV on the
/// bytes the cache produced compared with the CPU.
@Suite struct GGUFExpertBufferCacheTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func fetchesRealExpertsWithTheRightFormatsAndSizes() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let cache = try GGUFExpertBufferCache(
            model: model, device: context.device, capacity: 4)

        let layer0 = try cache.expert(layer: 0, expert: 0)
        #expect(layer0.gate.type == .iq2S)
        #expect(layer0.up.type == .iq2S)
        #expect(layer0.down.type == .iq4NL)
        #expect(layer0.gate.rows == 640 && layer0.gate.columns == 2_560)
        #expect(layer0.down.rows == 2_560 && layer0.down.columns == 640)
        #expect(layer0.fetchedBytes == 1_971_200)

        #expect(try cache.expert(layer: 8, expert: 3).down.type == .q2_0)
        #expect(try cache.expert(layer: 14, expert: 511).gate.type == .iq3XXS)
        #expect(try cache.expert(layer: 47, expert: 0).gate.type == .iq4XS)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func cachedBytesFeedTheFusedGEMVLikeTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let cache = try GGUFExpertBufferCache(
            model: model, device: context.device, capacity: 2)
        let expert = try cache.expert(layer: 0, expert: 5)

        var rng = SeedTree(0xB0).key("expert-cache-x")
        let x = (0..<2_560).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let xBuffer,
            let yBuffer = context.device.makeBuffer(
                length: 640 * MemoryLayout<UInt16>.size, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            throw MetalError.bufferAllocationFailed("expert cache test buffers")
        }
        let gemv = try GGUFExpertGEMV(context: context)
        try gemv.encode(
            commandBuffer: commandBuffer, weights: expert.gate.buffer,
            x: xBuffer, y: yBuffer, rows: 640, columns: 2_560, type: expert.gate.type)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }

        // CPU reference from the same byte range.
        let weights = try GGUFExpertLayout.weights(layer: 0, expert: 5, in: model)
        let handle = try FileHandle(forReadingFrom: model.shards[weights.gate.shardIndex].url)
        defer { try? handle.close() }
        try handle.seek(toOffset: weights.gate.fileOffset)
        let raw = try #require(try handle.read(upToCount: Int(weights.gate.byteCount)))
        let values = try GGUFDequant.decode(type: weights.gate.type, bytes: [UInt8](raw))
        let pointer = yBuffer.contents().assumingMemoryBound(to: UInt16.self)
        var worst: Float = 0
        for row in 0..<640 {
            var reference: Float = 0
            let base = row * 2_560
            for n in 0..<2_560 { reference += values[base + n] * Float(x[n]) }
            let got = Float(Float16(bitPattern: pointer[row]))
            worst = max(worst, abs(got - reference) / max(abs(reference), 1e-3))
        }
        print("cached expert GEMV worst relative error: \(worst)")
        #expect(worst < 5e-3)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresColdExpertFetch() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let cache = try GGUFExpertBufferCache(
            model: model, device: context.device, capacity: 1)
        // Layer order walks the shard files, so this is a real cold read rather
        // than a warm page-cache hit on one region.
        let start = DispatchTime.now().uptimeNanoseconds
        let fetches = 12
        for index in 0..<fetches {
            let layer = (index * 4) % 48
            _ = try cache.expert(layer: layer, expert: index)
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let perExpertMs = seconds * 1000 / Double(fetches)
        let bytes = 1_971_200
        print(
            "expert fetch: \(String(format: "%.2f", perExpertMs)) ms/expert, "
                + "\(String(format: "%.2f", Double(bytes) / (perExpertMs / 1000) / 1e9)) GB/s; "
                + "480 experts/token without cache hits = "
                + "\(String(format: "%.0f", perExpertMs * 480)) ms")
        #expect(perExpertMs > 0)
    }
}
