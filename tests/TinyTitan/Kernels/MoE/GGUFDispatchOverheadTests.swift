import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Where a token's host time goes: one command buffer carrying many dispatches
/// against many buffers carrying one each. Same kernels, same buffers.
@Suite struct GGUFDispatchOverheadTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func batchesAgainstSingletons() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let mapped = try GGUFMappedShards(model: model, device: context.device)
        let weight = try #require(try mapped.weight(named: "blk.0.attn_qkv.weight"))
        let rows = 640
        let columns = 2_560
        var rng = SeedTree(0x0D15).key("dispatch-overhead")
        let x = (0..<columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer,
            let y = context.device.makeBuffer(length: rows * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("dispatch overhead") }
        let pso = try context.pipeline("q6k_gemv_acc4")

        func encodeOne(_ cb: MTLCommandBuffer) throws {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            enc.setBuffer(xBuffer, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            var m = UInt32(rows)
            var n = UInt32(columns)
            enc.setBytes(&m, length: 4, index: 3)
            enc.setBytes(&n, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        let dispatches = 60
        // Warm up both shapes.
        for _ in 0..<2 {
            guard let cb = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            for _ in 0..<dispatches { try encodeOne(cb) }
            cb.commit()
            cb.waitUntilCompleted()
        }

        func batched(_ rounds: Int) throws -> (wall: Double, gpu: Double) {
            var wall: UInt64 = 0
            var gpu: Double = 0
            for _ in 0..<rounds {
                let start = DispatchTime.now().uptimeNanoseconds
                guard let cb = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                for _ in 0..<dispatches { try encodeOne(cb) }
                cb.commit()
                cb.waitUntilCompleted()
                wall &+= DispatchTime.now().uptimeNanoseconds - start
                gpu += cb.gpuEndTime - cb.gpuStartTime
            }
            return (Double(wall) / 1e9, gpu)
        }

        func singletons(_ rounds: Int) throws -> (wall: Double, gpu: Double) {
            var wall: UInt64 = 0
            var gpu: Double = 0
            for _ in 0..<rounds {
                let start = DispatchTime.now().uptimeNanoseconds
                for _ in 0..<dispatches {
                    guard let cb = context.queue.makeCommandBuffer() else {
                        throw MetalError.commandEncoderFailed
                    }
                    try encodeOne(cb)
                    cb.commit()
                    cb.waitUntilCompleted()
                    gpu += cb.gpuEndTime - cb.gpuStartTime
                }
                wall &+= DispatchTime.now().uptimeNanoseconds - start
            }
            return (Double(wall) / 1e9, gpu)
        }

        let rounds = 10
        // Distinct buffers: the decoder's routed command buffer references ~30
        // different expert buffers, and Metal validates the resources a command
        // buffer touches at commit.
        var distinct: [GGUFMappedShards.Weight] = []
        for layer in 0..<10 {
            for role in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"] {
                let name = "blk.\(layer).\(role)"
                if let weight = try mapped.weight(named: name) { distinct.append(weight) }
            }
        }
        let distinctCount = min(distinct.count, dispatches)
        func encodeDistinct(_ cb: MTLCommandBuffer, _ index: Int) throws {
            let weight = distinct[index % distinctCount]
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            enc.setBuffer(xBuffer, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            var m = UInt32(rows)
            var n = UInt32(columns)
            enc.setBytes(&m, length: 4, index: 3)
            enc.setBytes(&n, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        func batchedDistinct(_ rounds: Int) throws -> Double {
            var wall: UInt64 = 0
            for _ in 0..<rounds {
                let start = DispatchTime.now().uptimeNanoseconds
                guard let cb = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                for index in 0..<dispatches { try encodeDistinct(cb, index) }
                cb.commit()
                cb.waitUntilCompleted()
                wall &+= DispatchTime.now().uptimeNanoseconds - start
            }
            return Double(wall) / 1e9
        }
        let distinctWall = try batchedDistinct(rounds)
        print(
            "per dispatch, distinct buffers batched: "
                + "\(String(format: "%.3f", distinctWall * 1000 / Double(rounds * dispatches))) ms")

        let single = try singletons(rounds)
        let batch = try batched(rounds)
        let perDispatchSingle = single.wall * 1000 / Double(rounds * dispatches)
        let perDispatchBatch = batch.wall * 1000 / Double(rounds * dispatches)
        print(
            "per dispatch: single-buffer \(String(format: "%.3f", perDispatchSingle)) ms, "
                + "batched \(String(format: "%.3f", perDispatchBatch)) ms")
        print(
            "gpu share: single \(String(format: "%.1f", single.gpu * 1000 / Double(rounds))) ms/"
                + "round, batched \(String(format: "%.1f", batch.gpu * 1000 / Double(rounds))) ms/round"
        )
        #expect(perDispatchBatch > 0)
    }
}
