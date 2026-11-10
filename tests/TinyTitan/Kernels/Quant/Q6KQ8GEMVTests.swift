import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Q6_K and Q8_0 fused GEMV on the real shared-expert matrices.
@Suite struct Q6KQ8GEMVTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Matrix {
        let bytes: [UInt8]
        let rows: Int
        let columns: Int
        let values: [Float]
        let type: GGUFType
    }

    private static func load(_ model: GGUFModel, name: String, rows: Int, columns: Int) throws
        -> Matrix
    {
        let location = try #require(model.tensor(named: name))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
        return Matrix(
            bytes: [UInt8](data), rows: rows, columns: columns,
            values: try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](data)),
            type: location.tensor.type)
    }

    private static func run(
        _ context: MetalContext, matrix: Matrix, x: [Float16], iterations: Int
    ) throws -> (values: [Float16], seconds: Double) {
        let encode: (MTLCommandBuffer, MTLBuffer, MTLBuffer, MTLBuffer) throws -> Void
        if matrix.type == .q6K {
            let kernel = try Q6KGEMV(context: context)
            encode = { cb, w, xx, y in
                try kernel.encode(
                    commandBuffer: cb, weights: w, x: xx, y: y,
                    rows: matrix.rows, columns: matrix.columns)
            }
        } else {
            let kernel = try Q80GEMV(context: context)
            encode = { cb, w, xx, y in
                try kernel.encode(
                    commandBuffer: cb, weights: w, x: xx, y: y,
                    rows: matrix.rows, columns: matrix.columns)
            }
        }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let w = context.device.makeBuffer(
                bytes: matrix.bytes, length: matrix.bytes.count, options: .storageModeShared),
            let xBuffer,
            let y = context.device.makeBuffer(
                length: matrix.rows * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("GEMV buffers") }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard let cb = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try encode(cb, w, xBuffer, y)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw MetalError.invalidState("\(error)") }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let pointer = y.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<matrix.rows).map { Float16(bitPattern: pointer[$0]) }, seconds)
    }

    private static func compare(
        _ matrix: Matrix, x: [Float16], gpu: [Float16]
    ) -> Double {
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for row in 0..<matrix.rows {
            var ref: Float = 0
            let base = row * matrix.columns
            for n in 0..<matrix.columns { ref += matrix.values[base + n] * Float(x[n]) }
            let refHalf = Float(Float16(ref))
            let diff = Double(Float(gpu[row]) - refHalf)
            sqDiff += diff * diff
            sqRef += Double(refHalf * refHalf)
        }
        return (sqDiff / max(sqRef, 1e-12)).squareRoot()
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func q6kGEMVMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.load(
            model, name: "blk.0.ffn_gate_shexp.weight", rows: 640, columns: 2_560)
        try #require(matrix.type == .q6K)
        var rng = SeedTree(0x66).key("q6k-gemv-x")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let (gpu, _) = try Self.run(context, matrix: matrix, x: x, iterations: 1)
        let relative = Self.compare(matrix, x: x, gpu: gpu)
        print("Q6_K GEMV RMS relative error: \(relative)")
        #expect(relative < 5e-3)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func q8GEMVMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.load(
            model, name: "blk.0.ffn_down_shexp.weight", rows: 2_560, columns: 640)
        try #require(matrix.type == .q8_0)
        var rng = SeedTree(0x88).key("q8-gemv-x")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let (gpu, _) = try Self.run(context, matrix: matrix, x: x, iterations: 1)
        let relative = Self.compare(matrix, x: x, gpu: gpu)
        print("Q8_0 GEMV RMS relative error: \(relative)")
        #expect(relative < 5e-3)
    }
}

/// The Q6_K GEMV's own number and rate on this machine.
@Suite struct Q6KGEMVThroughputTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresTheGEMVRateAndMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let location = try #require(model.tensor(named: "blk.0.attn_qkv.weight"))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        let rows = 640
        let columns = 2_560
        let rowBytes = (columns / 256) * 210
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(try handle.read(upToCount: rows * rowBytes))
        let values = try GGUFDequant.decode(type: .q6K, bytes: [UInt8](data))
        var rng = SeedTree(0x6B6B).key("q6k-throughput")
        let x = (0..<columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let w = context.device.makeBuffer(bytes: [UInt8](data), length: data.count),
            let xBuffer,
            let y = context.device.makeBuffer(length: rows * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("q6k throughput buffers") }
        let kernel = try Q6KGEMV(context: context)
        func run(_ iterations: Int) throws -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<iterations {
                guard let cb = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                try kernel.encode(
                    commandBuffer: cb, weights: w, x: xBuffer, y: y,
                    rows: rows, columns: columns)
                cb.commit()
                cb.waitUntilCompleted()
                if let error = cb.error { throw MetalError.invalidState("\(error)") }
            }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        }
        _ = try run(2)
        let iterations = 200
        let seconds = try run(iterations)
        let elements = Double(rows * columns * iterations)
        let rate = elements / seconds / 1e9
        print("Q6_K GEMV: \(String(format: "%.2f", rate)) G elements/s")

        // Numbers must equal the CPU dot.
        let pointer = y.contents().assumingMemoryBound(to: UInt16.self)
        let xf = x.map(Float.init)
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for row in 0..<rows {
            var acc: Float = 0
            let base = row * columns
            for n in 0..<columns { acc += values[base + n] * xf[n] }
            let reference = Float(Float16(acc))
            let got = Float(Float16(bitPattern: pointer[row]))
            sqDiff += Double(got - reference) * Double(got - reference)
            sqRef += Double(reference * reference)
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("Q6_K GEMV RMS: \(relative)")
        #expect(relative < 5e-3)
        #expect(rate > 0)
    }
}

/// The four-accumulator Q6_K GEMV against the scalar kernel it would replace,
/// on the same weights and input, before it is allowed near the decoder.
@Suite struct Q6KGEMVAccumulatorTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func accumulatorsMatchTheScalarKernel() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let rows = 640
        let columns = 2_560
        let location = try #require(model.tensor(named: "blk.0.attn_qkv.weight"))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(
            try handle.read(upToCount: rows * (columns / 256) * 210))
        var rng = SeedTree(0xACC4).key("q6k-acc")
        let x = (0..<columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let w = context.device.makeBuffer(bytes: [UInt8](data), length: data.count),
            let xBuffer
        else { throw MetalError.bufferAllocationFailed("acc test buffers") }

        func run(_ pipeline: String) throws -> [Float16] {
            let pso = try context.pipeline(pipeline)
            guard
                let y = context.device.makeBuffer(
                    length: rows * 2, options: .storageModeShared),
                let cb = context.queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw MetalError.commandEncoderFailed }
            enc.setComputePipelineState(pso)
            enc.setBuffer(w, offset: 0, index: 0)
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
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw MetalError.invalidState("\(error)") }
            let pointer = y.contents().assumingMemoryBound(to: UInt16.self)
            return (0..<rows).map { Float16(bitPattern: pointer[$0]) }
        }
        let scalar = try run("q6k_gemv_simd")
        let acc4 = try run("q6k_gemv_acc4")
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for index in 0..<rows {
            let a = Double(Float(scalar[index]))
            let b = Double(Float(acc4[index]))
            sqDiff += (a - b) * (a - b)
            sqRef += a * a
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("Q6_K acc4 vs scalar RMS: \(relative)")
        #expect(relative < 1e-4)

        // Paired throughput: alternate the two kernels within each round so
        // machine drift cancels in the per-round difference.
        func time(_ pipeline: String, _ iterations: Int) throws -> Double {
            let pso = try context.pipeline(pipeline)
            guard
                let y = context.device.makeBuffer(
                    length: rows * 2, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("acc timing") }
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<iterations {
                guard let cb = context.queue.makeCommandBuffer(),
                    let enc = cb.makeComputeCommandEncoder()
                else { throw MetalError.commandEncoderFailed }
                enc.setComputePipelineState(pso)
                enc.setBuffer(w, offset: 0, index: 0)
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
                cb.commit()
                cb.waitUntilCompleted()
            }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        }
        _ = try time("q6k_gemv_simd", 5)
        _ = try time("q6k_gemv_acc4", 5)
        let elements = Double(rows * columns)
        for round in 0..<3 {
            let a = try time("q6k_gemv_simd", 50)
            let b = try time("q6k_gemv_acc4", 50)
            let rateA = elements * 50 / a / 1e9
            let rateB = elements * 50 / b / 1e9
            print(
                "round \(round): scalar \(String(format: "%.2f", rateA)) G/s, "
                    + "acc4 \(String(format: "%.2f", rateB)) G/s, "
                    + "paired \(String(format: "%+.2f", rateB - rateA)) G/s")
        }
    }
}
