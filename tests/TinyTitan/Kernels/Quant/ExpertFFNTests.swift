import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// One routed expert's whole FFN on the GPU: gate and up from IQ2_S blocks,
/// SiLU(gate)·up, then down from IQ4_NL blocks. Checked against a CPU
/// decode-then-matmul, then timed per token (48 layers x 10 experts).
@Suite struct ExpertFFNTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Expert {
        let gateBlocks: [UInt8]
        let upBlocks: [UInt8]
        let downBlocks: [UInt8]
        let gate: [Float]
        let up: [Float]
        let down: [Float]
    }

    private static func load(_ model: GGUFModel) throws -> Expert {
        let weights = try GGUFExpertLayout.weights(layer: 0, expert: 0, in: model)
        func bytes(_ range: GGUFTensorRange) throws -> [UInt8] {
            let handle = try FileHandle(forReadingFrom: model.shards[range.shardIndex].url)
            try handle.seek(toOffset: range.fileOffset)
            let data = try #require(try handle.read(upToCount: Int(range.byteCount)))
            try handle.close()
            return [UInt8](data)
        }
        let gateBytes = try bytes(weights.gate)
        let upBytes = try bytes(weights.up)
        let downBytes = try bytes(weights.down)
        return Expert(
            gateBlocks: gateBytes, upBlocks: upBytes, downBlocks: downBytes,
            gate: try GGUFDequant.decode(type: weights.gate.type, bytes: gateBytes),
            up: try GGUFDequant.decode(type: weights.up.type, bytes: upBytes),
            down: try GGUFDequant.decode(type: weights.down.type, bytes: downBytes))
    }

    private static func buffer(_ context: MetalContext, _ bytes: [UInt8]) -> MTLBuffer? {
        context.device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)
    }

    private static func halfBuffer(_ context: MetalContext, _ values: [Float16]) -> MTLBuffer? {
        let bits = values.map(\.bitPattern)
        return bits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
    }

    /// Runs gate, up, SiLU multiply and down; returns the final 2560-vector.
    private static func runFFN(
        _ context: MetalContext,
        expert: Expert,
        x: [Float16],
        gateKernel: IQ2SGEMV,
        downKernel: IQ4NLGEMV
    ) throws -> (output: [Float16], seconds: Double) {
        guard
            let gateW = buffer(context, expert.gateBlocks),
            let upW = buffer(context, expert.upBlocks),
            let downW = buffer(context, expert.downBlocks),
            let xBuffer = halfBuffer(context, x),
            let gateY = context.device.makeBuffer(
                length: 640 * MemoryLayout<UInt16>.size, options: .storageModeShared),
            let upY = context.device.makeBuffer(
                length: 640 * MemoryLayout<UInt16>.size, options: .storageModeShared),
            let actY = context.device.makeBuffer(
                length: 640 * MemoryLayout<UInt16>.size, options: .storageModeShared),
            let outY = context.device.makeBuffer(
                length: 2_560 * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("expert FFN buffers")
        }
        let start = DispatchTime.now().uptimeNanoseconds
        // gate and up share one command buffer; they are independent.
        guard let first = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try gateKernel.encode(
            commandBuffer: first, weights: gateW, x: xBuffer, y: gateY,
            rows: 640, columns: 2_560)
        try gateKernel.encode(
            commandBuffer: first, weights: upW, x: xBuffer, y: upY,
            rows: 640, columns: 2_560)
        first.commit()
        first.waitUntilCompleted()
        if let error = first.error { throw MetalError.invalidState("\(error)") }

        // SiLU(gate) * up on the host (640 values; negligible next to the GEMVs).
        let gateP = gateY.contents().assumingMemoryBound(to: UInt16.self)
        let upP = upY.contents().assumingMemoryBound(to: UInt16.self)
        let actP = actY.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<640 {
            let g = Float(Float16(bitPattern: gateP[i]))
            let u = Float(Float16(bitPattern: upP[i]))
            let silu = g / (1 + exp(-g))
            actP[i] = Float16(silu * u).bitPattern
        }
        guard let second = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try downKernel.encode(
            commandBuffer: second, weights: downW, x: actY, y: outY,
            rows: 2_560, columns: 640)
        second.commit()
        second.waitUntilCompleted()
        if let error = second.error { throw MetalError.invalidState("\(error)") }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let outP = outY.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<2_560).map { Float16(bitPattern: outP[$0]) }, seconds)
    }

    private static func cpuReference(_ expert: Expert, x: [Float16]) -> [Float] {
        func dot(_ weights: [Float], rows: Int, columns: Int, _ vector: [Float]) -> [Float] {
            (0..<rows).map { row in
                var acc: Float = 0
                let base = row * columns
                for n in 0..<columns { acc += weights[base + n] * vector[n] }
                return acc
            }
        }
        let xf = x.map(Float.init)
        // Mirror the GPU: gate and up are written as half, so the SiLU reads
        // the rounded values, and the activation is rounded to half before the
        // down projection reads it.
        let gate = dot(expert.gate, rows: 640, columns: 2_560, xf)
            .map { Float(Float16($0)) }
        let up = dot(expert.up, rows: 640, columns: 2_560, xf)
            .map { Float(Float16($0)) }
        var act = [Float](repeating: 0, count: 640)
        for i in 0..<640 {
            let g = gate[i]
            act[i] = Float(Float16(g / (1 + exp(-g)) * up[i]))
        }
        return dot(expert.down, rows: 2_560, columns: 640, act)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func expertFFNMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let expert = try Self.load(model)
        var rng = SeedTree(0xF7).key("expert-ffn-x")
        let x = (0..<2_560).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let (output, _) = try Self.runFFN(
            context, expert: expert, x: x,
            gateKernel: IQ2SGEMV(context: context),
            downKernel: IQ4NLGEMV(context: context))
        let reference = Self.cpuReference(expert, x: x)
        // RMS-relative, not per-element: single elements near zero amplify a
        // small absolute difference, which says nothing about the vector.
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for i in 0..<2_560 {
            let diff = Double(Float(output[i]) - reference[i])
            sqDiff += diff * diff
            sqRef += Double(reference[i] * reference[i])
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("expert FFN RMS relative error: \(relative)")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresExpertFFNPerToken() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let expert = try Self.load(model)
        var rng = SeedTree(0xA1).key("expert-ffn-bench")
        let x = (0..<2_560).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let gateKernel = try IQ2SGEMV(context: context)
        let downKernel = try IQ4NLGEMV(context: context)
        _ = try Self.runFFN(
            context, expert: expert, x: x, gateKernel: gateKernel, downKernel: downKernel)
        let iterations = 20
        var total: Double = 0
        for _ in 0..<iterations {
            let (_, seconds) = try Self.runFFN(
                context, expert: expert, x: x, gateKernel: gateKernel, downKernel: downKernel)
            total += seconds
        }
        let perExpertMs = total * 1000 / Double(iterations)
        let expertsPerToken = 48.0 * 10.0
        let computeMs = perExpertMs * expertsPerToken
        print(
            "expert FFN: \(String(format: "%.3f", perExpertMs)) ms per expert; "
                + "\(String(format: "%.0f", computeMs)) ms/token -> "
                + "\(String(format: "%.2f", 1000.0 / computeMs)) tok/s compute ceiling")
        #expect(perExpertMs > 0)
    }
}
