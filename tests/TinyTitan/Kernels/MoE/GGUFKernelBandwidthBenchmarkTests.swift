import Foundation
import Metal
import Testing

@testable import TinyTitan

/// Weight-read bandwidth of every non-expert GEMV of one GDN and one
/// full-attention layer plus the output head, each timed alone on the decoder's
/// own mapped weights. This is the per-kernel profile the 10 tok/s plan's GPU
/// step is measured against (`docs/plan-gguf-decode-10toks.md`, step 2.1).
/// Each figure is the best of five 20-dispatch command buffers, which keeps a
/// shared host's background load out of the number. Opt-in
/// (`TINYTITAN_KERNEL_BENCH=1` plus a model).
@Suite struct GGUFKernelBandwidthBenchmarkTests {
    private static var enabled: Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment["TINYTITAN_GGUF_MODEL"] != nil
            && environment["TINYTITAN_KERNEL_BENCH"] == "1"
    }

    @Test(.enabled(if: enabled))
    func gemvBandwidthPerTensor() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"])
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let decoder = try GGUFDecoder(
            context: context, arch: .qwen38FlashNext, model: model, maxSeq: 64,
            ramBudgetBytes: 10 << 30)
        let gemv = try GGUFExpertGEMV(context: context)
        let supported: Set<GGUFType> = [.q6K, .q8_0, .iq2S, .iq3XXS, .iq4NL, .iq4XS, .q2_0]
        let names = model.tensors.keys.filter { name in
            (name.hasPrefix("blk.0.") || name.hasPrefix("blk.3.") || name == "output.weight")
                && !name.contains("_exps")
        }.sorted()
        let repeats = 20
        var totalBytes = 0.0
        var totalMs = 0.0
        for name in names {
            guard let weight = try decoder.mapped.weight(named: name),
                supported.contains(weight.type)
            else { continue }
            let blocks = UInt64(weight.columns) / weight.type.blockSize
            let bytes = Double(UInt64(weight.rows) * blocks * weight.type.blockBytes)
            let x = try #require(
                context.device.makeBuffer(length: weight.columns * 2, options: .storageModeShared))
            memset(x.contents(), 0x3C, x.length)  // 0x3C3C ≈ 1.06 in half
            let y = try #require(
                context.device.makeBuffer(length: weight.rows * 2, options: .storageModeShared))
            var best = Double.infinity
            for trial in 0..<6 {
                let commandBuffer = try #require(context.queue.makeCommandBuffer())
                for _ in 0..<(trial == 0 ? 1 : repeats) {
                    try gemv.encode(
                        commandBuffer: commandBuffer, weights: weight.buffer,
                        weightsOffset: weight.offset, x: x, y: y,
                        rows: weight.rows, columns: weight.columns, type: weight.type)
                }
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                #expect(commandBuffer.error == nil, "\(name)")
                if trial > 0 {
                    let ms =
                        (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
                        / Double(repeats) * 1000
                    best = min(best, ms)
                }
            }
            totalBytes += bytes
            totalMs += best
            print(
                String(
                    format: "bw %-28@ %-7@ %6dx%-6d %7.2f MB %7.3f ms %6.1f GB/s",
                    name as NSString, weight.type.name as NSString, weight.rows,
                    weight.columns, bytes / 1e6, best, bytes / best / 1e6))
        }
        print(
            String(
                format: "bw total %.1f MB %.3f ms %.1f GB/s", totalBytes / 1e6, totalMs,
                totalBytes / totalMs / 1e6))
        #expect(totalMs > 0)
    }

    /// The routed stage's three batched GEMVs (gate, up, down of the ten
    /// routed experts) timed alone, so a kernel change is measured on its
    /// own dispatch rather than inside the five-dispatch stage.
    @Test(.enabled(if: enabled))
    func routedBatchBandwidth() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"])
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let cache = try GGUFExpertBufferCache(
            model: model, device: context.device, capacity: arch.topKExperts)
        let batch = try GGUFExpertBatchGEMV(context: context)
        let width = arch.moeIntermediateSize
        let hidden = arch.hiddenSize
        let k = arch.topKExperts
        func buffer(_ halves: Int) throws -> MTLBuffer {
            let b = try #require(
                context.device.makeBuffer(length: halves * 2, options: .storageModeShared))
            memset(b.contents(), 0x2C, b.length)  // small positive halves
            return b
        }
        let x = try buffer(hidden)
        let act = try buffer(width * k)
        let gateOut = try buffer(width * k)
        let downOut = try buffer(hidden * k)
        let experts = Array(stride(from: 5, to: 5 + 41 * k, by: 41))
        let repeats = 20
        for layer in [1, 24, 47] {
            let fetched = try cache.experts(layer: layer, experts: experts)
            let parts: [(String, [GGUFExpertMatrix], MTLBuffer, Int, MTLBuffer, Int)] = [
                ("gate", fetched.map(\.gate), x, 0, gateOut, width),
                ("up", fetched.map(\.up), x, 0, gateOut, width),
                ("down", fetched.map(\.down), act, width, downOut, hidden),
            ]
            for (label, matrices, input, xStride, output, yStride) in parts {
                let rows = label == "down" ? hidden : width
                let columns = label == "down" ? width : hidden
                let type = matrices[0].type
                let bytes = Double(
                    k * rows * (columns / Int(type.blockSize)) * Int(type.blockBytes))
                var best = Double.infinity
                for trial in 0..<6 {
                    let commandBuffer = try #require(context.queue.makeCommandBuffer())
                    for _ in 0..<(trial == 0 ? 1 : repeats) {
                        try batch.encode(
                            commandBuffer: commandBuffer, matrices: matrices,
                            x: input, xStride: xStride, y: output, yStride: yStride)
                    }
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                    #expect(commandBuffer.error == nil)
                    if trial > 0 {
                        best = min(
                            best,
                            (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
                                / Double(repeats) * 1000)
                    }
                }
                print(
                    String(
                        format: "rb L%-2d %-4@ %-7@ %5.2f MB %6.3f ms %6.1f GB/s", layer,
                        label as NSString, type.name as NSString, bytes / 1e6, best,
                        bytes / best / 1e6))
            }
        }
    }

    /// What an encoder boundary costs on the GPU: 200 tiny dependent
    /// dispatches, each in its own compute encoder versus all in one serial
    /// encoder. The decode layer opens ~25 encoders, so this prices the
    /// "one encoder per command buffer" step before it is built.
    @Test(.enabled(if: enabled))
    func encoderBoundaryCost() throws {
        let context = try MetalContext()
        let pso = try context.pipeline("q8_0_gemv_vec")
        let rows = 8
        let columns = 256
        let w = try #require(
            context.device.makeBuffer(length: rows * columns / 32 * 34, options: .storageModeShared)
        )
        memset(w.contents(), 0, w.length)
        let x = try #require(
            context.device.makeBuffer(length: columns * 2, options: .storageModeShared))
        let y = try #require(
            context.device.makeBuffer(length: columns * 2, options: .storageModeShared))
        var m = UInt32(rows)
        var n = UInt32(columns)
        func dispatch(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(pso)
            encoder.setBuffer(w, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.setBuffer(y, offset: 0, index: 2)
            encoder.setBytes(&m, length: 4, index: 3)
            encoder.setBytes(&n, length: 4, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let count = 200
        for mode in ["separate", "single", "single+barrier"] {
            var best = Double.infinity
            for trial in 0..<6 {
                let commandBuffer = try #require(context.queue.makeCommandBuffer())
                if mode == "separate" {
                    for _ in 0..<count {
                        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
                        dispatch(encoder)
                        encoder.endEncoding()
                    }
                } else {
                    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
                    for _ in 0..<count {
                        dispatch(encoder)
                        if mode == "single+barrier" { encoder.memoryBarrier(scope: .buffers) }
                    }
                    encoder.endEncoding()
                }
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if trial > 0 {
                    best = min(best, (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1e6)
                }
            }
            print(
                String(
                    format: "enc %-15@ %7.2f us per dispatch", mode as NSString,
                    best / Double(count)))
        }
    }

    /// The machine's read ceiling: a plain float4 streaming sum over 512 MB,
    /// the most a GEMV can approach. Every GB/s above is judged against it.
    @Test(.enabled(if: enabled))
    func streamingReadCeiling() throws {
        let context = try MetalContext()
        let source = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void stream_sum(device const float4* a [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   constant uint& n [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]],
                                   uint threads [[threads_per_grid]]) {
                float4 acc = 0.0f;
                for (uint i = gid; i < n; i += threads) { acc += a[i]; }
                if (dot(acc, 1.0f) == 12345.0f) { out[0] = 1.0f; }
            }
            """
        let library = try context.device.makeLibrary(source: source, options: nil)
        let function = try #require(library.makeFunction(name: "stream_sum"))
        let pso = try context.device.makeComputePipelineState(function: function)
        let bytes = 512 << 20
        let a = try #require(context.device.makeBuffer(length: bytes, options: .storageModePrivate))
        let out = try #require(context.device.makeBuffer(length: 16, options: .storageModeShared))
        var n = UInt32(bytes / 16)
        for threads in [16_384, 65_536, 262_144] {
            var best = Double.infinity
            for trial in 0..<6 {
                let commandBuffer = try #require(context.queue.makeCommandBuffer())
                let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
                encoder.setComputePipelineState(pso)
                encoder.setBuffer(a, offset: 0, index: 0)
                encoder.setBuffer(out, offset: 0, index: 1)
                encoder.setBytes(&n, length: 4, index: 2)
                encoder.dispatchThreads(
                    MTLSize(width: threads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if trial > 0 {
                    best = min(best, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
                }
            }
            print(
                String(
                    format: "ceiling %7d threads %6.1f GB/s", threads, Double(bytes) / best / 1e9))
        }
    }
}
