import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The shared-expert stage on real GGUF weights: Q6_K gate/up, Q8_0 down and
/// the F32 scalar gate, against a CPU reference.
@Suite struct GGUFSharedExpertStageTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func sharedExpertMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let stage = try GGUFSharedExpertStage(context: context, arch: arch, model: model)

        var rng = SeedTree(0x5A1E).key("shared-x")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer,
            let out = context.device.makeBuffer(
                length: arch.hiddenSize * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("shared expert buffers") }
        memset(out.contents(), 0, arch.hiddenSize * MemoryLayout<UInt16>.size)
        try stage.encode(commandBuffer: cb, layer: 0, x: xBuffer, out: out)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }
        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        let gpu = (0..<arch.hiddenSize).map { Float(Float16(bitPattern: pointer[$0])) }

        // CPU reference.
        func tensor(_ name: String) throws -> [Float] {
            let location = try #require(model.tensor(named: name))
            let shard = model.shards[location.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            defer { try? handle.close() }
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            let data = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
            if location.tensor.type == .f32 {
                return data.withUnsafeBytes { raw in
                    let count = raw.count / 4
                    let pointer = raw.bindMemory(to: Float.self)
                    return (0..<count).map { pointer[$0] }
                }
            }
            return try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](data))
        }
        let gate = try tensor("blk.0.ffn_gate_shexp.weight")
        let up = try tensor("blk.0.ffn_up_shexp.weight")
        let down = try tensor("blk.0.ffn_down_shexp.weight")
        let scalar = try tensor("blk.0.ffn_gate_inp_shexp.weight")
        let xf = x.map(Float.init)
        var act = [Float](repeating: 0, count: arch.moeIntermediateSize)
        for row in 0..<arch.moeIntermediateSize {
            var g: Float = 0
            var u: Float = 0
            let base = row * arch.hiddenSize
            for n in 0..<arch.hiddenSize {
                g += gate[base + n] * xf[n]
                u += up[base + n] * xf[n]
            }
            let gh = Float(Float16(g))
            let uh = Float(Float16(u))
            act[row] = Float(Float16(gh / (1 + exp(-gh)) * uh))
        }
        var gateScalar: Float = 0
        for n in 0..<arch.hiddenSize { gateScalar += scalar[n] * xf[n] }
        let scale = Float(Float16(1 / (1 + exp(-gateScalar))))
        var reference = [Float](repeating: 0, count: arch.hiddenSize)
        for row in 0..<arch.hiddenSize {
            var acc: Float = 0
            let base = row * arch.moeIntermediateSize
            for n in 0..<arch.moeIntermediateSize { acc += down[base + n] * act[n] }
            reference[row] = scale * Float(Float16(acc))
        }
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for i in 0..<arch.hiddenSize {
            let diff = Double(gpu[i] - reference[i])
            sqDiff += diff * diff
            sqRef += Double(reference[i] * reference[i])
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("shared expert RMS relative error: \(relative)")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }
}
