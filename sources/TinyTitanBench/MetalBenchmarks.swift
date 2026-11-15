//
//  MetalBenchmarks.swift
//  TinyTitanBench
//
//  The Metal kernel benchmarks: routed-MoE decode, GDN, and attention-head
//  shapes. Split out of `main.swift` so each file holds one family of
//  measurements; the dispatch that selects between them is in `main.swift`.
//

import Foundation
import Metal
import TinyTitan
import TinyTitanCore

/// A Metal object the harness needs but could not create. The benchmark
/// harnesses used to force-unwrap these, so a machine under memory pressure
/// crashed instead of reporting which allocation failed.
enum BenchHarnessError: Error, CustomStringConvertible {
    case metalObjectUnavailable(String)
    case badConfiguration(String)

    var description: String {
        switch self {
        case .metalObjectUnavailable(let what):
            return "could not create the Metal \(what) for this benchmark"
        case .badConfiguration(let why):
            return "refusing to run: \(why)"
        }
    }
}

/// Read an optional integer knob, refusing rather than reading as "off" when the
/// operator set it to something that is not a number: a benchmark arm that
/// silently disappears on a typo reports the control as a measurement.
func envIntFlag(_ name: String) throws -> Int? {
    guard let raw = ProcessInfo.processInfo.environment[name] else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty { return nil }
    guard let value = Int(trimmed) else {
        throw BenchHarnessError.badConfiguration("\(name) must be an integer, got \"\(raw)\"")
    }
    return value
}

/// A command buffer from the queue, or a named error.
func requireCommandBuffer(_ queue: MTLCommandQueue) throws -> MTLCommandBuffer {
    guard let cb = queue.makeCommandBuffer() else {
        throw BenchHarnessError.metalObjectUnavailable("command buffer")
    }
    return cb
}

extension TinyTitanBench {

    /// GDN fused input-projection GEMV at the real qwen36 shapes
    /// (qkvDim 8192, valueDim 4096, ab 32 each, N=2048). The weight read is
    /// the metric: qkv + z + a + b ~= 12.7 MB/layer. Variants:
    /// gdn_inproj (production 8-row), gdn_inproj_xsh8 (8-row + tgmem x),
    /// gdn_inproj_r16 (16-row device x), gdn_inproj_xsh16 (16-row + tgmem).
    /// lint:allow-long TinyTitanBench is a development harness, not a shipped
    /// product: each run* is one linear measurement script whose setup,
    /// dispatch and reporting only make sense read top to bottom.
    static func runGDN(
        kernelName: String,
        iterations: Int,
        context: MetalContext
    ) throws {
        let device = context.device
        // Default shape is the Qwen 3.6 GDN layer. TINYTITAN_BENCH_GDN_SHAPE=qwen38
        // selects Qwen3.8-Flash-Next's (16 k-heads x 128 + 48 v-heads x 128
        // for qkv, 48 x 128 for z, 48 for a/b, hidden 2560), the shape the
        // decode profile's gdn.inproj number comes from.
        let qwen38 = ProcessInfo.processInfo.environment["TINYTITAN_BENCH_GDN_SHAPE"] == "qwen38"
        let qkvRows: UInt32 = qwen38 ? 16384 : 8192
        let zRows: UInt32 = qwen38 ? 6144 : 4096
        let abRows: UInt32 = qwen38 ? 48 : 32
        let N: UInt32 = qwen38 ? 2560 : 2048
        let groupCount = Int(N) / 64

        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw BenchHarnessError.metalObjectUnavailable("buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let qkvW = try makeBuffer(Int(qkvRows) * Int(N) / 2, 0x12)
        let qkvS = try makeBuffer(Int(qkvRows) * groupCount * 2, 0x01)
        let qkvB = try makeBuffer(Int(qkvRows) * groupCount * 2, 0x00)
        let zW = try makeBuffer(Int(zRows) * Int(N) / 2, 0x34)
        let zS = try makeBuffer(Int(zRows) * groupCount * 2, 0x01)
        let zB = try makeBuffer(Int(zRows) * groupCount * 2, 0x00)
        let aW = try makeBuffer(Int(abRows) * Int(N) / 2, 0x56)
        let aS = try makeBuffer(Int(abRows) * groupCount * 2, 0x01)
        let aB = try makeBuffer(Int(abRows) * groupCount * 2, 0x00)
        let bW = try makeBuffer(Int(abRows) * Int(N) / 2, 0x78)
        let bS = try makeBuffer(Int(abRows) * groupCount * 2, 0x01)
        let bB = try makeBuffer(Int(abRows) * groupCount * 2, 0x00)
        let x = try makeBuffer(Int(N) * 2, 0)
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(N) {
            xPtr[i] = Float16(Float(i + 1) * 0.001).bitPattern
        }
        let qkvY = try makeBuffer(Int(qkvRows) * 2, 0)
        let zY = try makeBuffer(Int(zRows) * 2, 0)
        let aY = try makeBuffer(Int(abRows) * 2, 0)
        let bY = try makeBuffer(Int(abRows) * 2, 0)

        let kernelName2: String
        let rowsPerTG: Int
        switch kernelName {
        case "gdn_inproj_xsh8":
            kernelName2 = "gdn_in_proj_gemv_simd_xsh8"
            rowsPerTG = 8
        case "gdn_inproj_r16":
            kernelName2 = "gdn_in_proj_gemv_simd_r16"
            rowsPerTG = 16
        case "gdn_inproj_xsh16":
            kernelName2 = "gdn_in_proj_gemv_simd_xsh16"
            rowsPerTG = 16
        case "gdn_inproj_u4":
            kernelName2 = "gdn_in_proj_gemv_simd_u4"
            rowsPerTG = 8
        case "gdn_inproj_sk4":
            kernelName2 = "gdn_in_proj_gemv_simd_sk4"
            rowsPerTG = 2
        case "gdn_inproj_bw":
            kernelName2 = "gdn_in_proj_gemv_simd_bw"
            rowsPerTG = 8
        default:
            kernelName2 = "gdn_in_proj_gemv_simd"
            rowsPerTG = 8
        }
        // sk4 runs four simdgroups per row, so its threadgroup is wider than
        // rows * 32.
        let threadsPerTG = kernelName == "gdn_inproj_sk4" ? 256 : rowsPerTG * 32
        let pso = try context.pipeline(
            kernelName2, constants: [],
            maxTotalThreadsPerThreadgroup: threadsPerTG)

        // Full GDN decode chain (gdn_chain): in_proj + conv + qk_norm +
        // delta-step + gated_norm in one command buffer, mirroring the
        // decode's linear-attention block. Measures the extras' cost.
        let chain = kernelName == "gdn_chain"
        let convPSO = try context.pipeline("gdn_conv_mix_decode")
        let qkNormPSO = try context.pipeline(
            "gdn_qk_norm",
            constants: [MetalFunctionConstant(index: 95, value: .uint32(128))])
        let deltaPSO = try context.pipeline("gdn_delta_step_decode")
        let gatedNormPSO = try context.pipeline(
            "gdn_gated_norm",
            constants: [MetalFunctionConstant(index: 95, value: .uint32(128))])
        let convTail = try makeBuffer(3 * Int(qkvRows), 0x44)  // [K-1, qkvDim] halfs
        let convW = try makeBuffer(Int(qkvRows) * 4 * 2, 0x55)  // [qkvDim, K] bfloat
        let convOut = try makeBuffer(Int(qkvRows) * 2, 0)
        let aLog = try makeBuffer(Int(abRows) * 2, 0x60)
        let dtBias = try makeBuffer(Int(abRows) * 2, 0x60)
        let state = try makeBuffer(Int(abRows) * 128 * 128 * 4, 0)  // FP32 [Hv, Dv, Dk]
        let deltaY = try makeBuffer(Int(zRows) * 2, 0)
        let gatedW = try makeBuffer(Int(zRows) * 2, 0x66)
        let gatedOut = try makeBuffer(Int(zRows) * 2, 0)

        let totalRows = Int(qkvRows + zRows + 2 * abRows)
        let bytes = UInt64(
            Int(qkvRows) * Int(N) / 2 + Int(zRows) * Int(N) / 2
                + 2 * Int(abRows) * Int(N) / 2)
        var qkvVar = qkvRows
        var zVar = zRows
        var abVar = abRows
        var nVar = N

        let cb = try requireCommandBuffer(context.queue)
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qkvW, offset: 0, index: 0)
        enc.setBuffer(qkvS, offset: 0, index: 1)
        enc.setBuffer(qkvB, offset: 0, index: 2)
        enc.setBuffer(zW, offset: 0, index: 3)
        enc.setBuffer(zS, offset: 0, index: 4)
        enc.setBuffer(zB, offset: 0, index: 5)
        enc.setBuffer(aW, offset: 0, index: 6)
        enc.setBuffer(aS, offset: 0, index: 7)
        enc.setBuffer(aB, offset: 0, index: 8)
        enc.setBuffer(bW, offset: 0, index: 9)
        enc.setBuffer(bS, offset: 0, index: 10)
        enc.setBuffer(bB, offset: 0, index: 11)
        enc.setBuffer(x, offset: 0, index: 12)
        enc.setBuffer(qkvY, offset: 0, index: 13)
        enc.setBuffer(zY, offset: 0, index: 14)
        enc.setBuffer(aY, offset: 0, index: 15)
        enc.setBuffer(bY, offset: 0, index: 16)
        enc.setBytes(&qkvVar, length: MemoryLayout<UInt32>.size, index: 17)
        enc.setBytes(&zVar, length: MemoryLayout<UInt32>.size, index: 18)
        enc.setBytes(&abVar, length: MemoryLayout<UInt32>.size, index: 19)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 20)
        for _ in 0..<iterations {
            enc.setComputePipelineState(pso)
            enc.dispatchThreadgroups(
                MTLSize(width: (totalRows + rowsPerTG - 1) / rowsPerTG, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
            if chain {
                // conv: tail(0) qkv(1) convW(2) out(3) channels(4) taps(5)
                enc.setComputePipelineState(convPSO)
                enc.setBuffer(convTail, offset: 0, index: 0)
                enc.setBuffer(qkvY, offset: 0, index: 1)
                enc.setBuffer(convW, offset: 0, index: 2)
                enc.setBuffer(convOut, offset: 0, index: 3)
                var ch = qkvRows
                var taps: UInt32 = 4
                enc.setBytes(&ch, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: Int(ch), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                // qk_norm: convOut(0) kHeads(1) keyDim(2) rowStride(3)
                enc.setComputePipelineState(qkNormPSO)
                enc.setBuffer(convOut, offset: 0, index: 0)
                var kHeads: UInt32 = 16
                var keyDim: UInt32 = 128
                var rowStride = qkvRows
                enc.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 1)
                enc.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 2)
                enc.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: 2 * 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                // delta: convOut(0) aProj(1) bProj(2) aLog(3) dtBias(4) state(5) y(6) dims 7-10
                enc.setComputePipelineState(deltaPSO)
                enc.setBuffer(convOut, offset: 0, index: 0)
                enc.setBuffer(aY, offset: 0, index: 1)
                enc.setBuffer(bY, offset: 0, index: 2)
                enc.setBuffer(aLog, offset: 0, index: 3)
                enc.setBuffer(dtBias, offset: 0, index: 4)
                enc.setBuffer(state, offset: 0, index: 5)
                enc.setBuffer(deltaY, offset: 0, index: 6)
                var vHeads: UInt32 = 32
                var vDim: UInt32 = 128
                enc.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 7)
                enc.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 8)
                enc.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 9)
                enc.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 10)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(vHeads), height: Int(vDim) / 4, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
                // gated_norm: y(0) z(1) weight(2) out(3) vHeads(4) valueDim(5)
                enc.setComputePipelineState(gatedNormPSO)
                enc.setBuffer(deltaY, offset: 0, index: 0)
                enc.setBuffer(zY, offset: 0, index: 1)
                enc.setBuffer(gatedW, offset: 0, index: 2)
                enc.setBuffer(gatedOut, offset: 0, index: 3)
                enc.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(vHeads), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("COMMAND BUFFER ERROR: \(err)")
        }

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let gbPerSec = Double(bytes) / perIteration / 1_000_000_000
        let theoretical = 100.0
        print(
            "kernel=\(kernelName) iterations=\(iterations) "
                + "total=\(String(format: "%.4f", totalSeconds))s "
                + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print(
            "bytes/launch=\(bytes) "
                + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
                + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak"
        )
    }

}
