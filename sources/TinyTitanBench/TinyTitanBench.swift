import Foundation
import Metal
import TinyTitan
import TinyTitanCore

/// Kernel micro-benchmarks. Two families:
///
/// 1. The fused QKV GEMV (baseline/bandwidth/unroll2/ulong2): the decode
///    attention block's dominant kernel in isolation (synthetic buffers, no
///    model pipeline). Reports GB/s against the M3's theoretical peak
///    (~100 GB/s).
///
/// 2. The routed-MoE decode kernels (moe_phase1 / moe_phase2 / moe): the
///    phase-1 gate/up GEMV and the phase-2 down+reduce GEMV with synthetic
///    8-expert blobs at the real 4-bit shapes (F=512, D=2816, topK=8),
///    matching the decode routedCB dispatch. Reports GB/s of weight reads.
///    The decode's routedCB measured 14.4 ms/token (~48 GB/s at the real
///    shapes); this bench isolates which phase and access pattern is slow.
///
/// Usage: TinyTitanBench [kernelName] [iterations]
///   qkv family: baseline (default), bandwidth, unroll2, ulong2
///   moe family: moe_phase1, moe_phase2, moe
///   gdn family: gdn_inproj (baseline), gdn_inproj_xsh, gdn_inproj_r16
@main
struct TinyTitanBench {

    static func main() throws {
        let kernelName =
            CommandLine.arguments.count > 1
            ? CommandLine.arguments[1] : "baseline"
        let iterations =
            CommandLine.arguments.count > 2
            ? Int(CommandLine.arguments[2]) ?? 300 : 300

        // Before the Metal context: these measure the CPU and must not
        // touch the GPU, whose bandwidth is the thing being competed for.
        if try runCPUCommand(kernelName, iterations: iterations) { return }

        let context = try MetalContext.runtimeDefault()
        let device = context.device
        print("device: \(device.name)")

        if kernelName.hasPrefix("gdn") {
            try runGDN(kernelName: kernelName, iterations: iterations, context: context)
            return
        }

        if kernelName.hasPrefix("moe") {
            try runMoE(kernelName: kernelName, iterations: iterations, context: context)
            return
        }

        if kernelName.hasPrefix("head") {
            try runHead(kernelName: kernelName, iterations: iterations, context: context)
            return
        }

        // Gated QKV shape (the dominant decode GEMV): qRows = 2*qDim = 8192,
        // kvRows = 2048, n = 2816.
        let qRows: UInt32 = 8192
        let kvRows: UInt32 = 2048
        let n: UInt32 = 2816
        let rowBytes = Int(n) / 2
        let groupCount = Int(n) / 64

        let pso = try context.pipeline(
            kernelName == "baseline"
                ? "dequant_int4_qkv_gemv_simd"
                : "dequant_int4_qkv_gemv_simd_\(kernelName)",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard
                let buf = device.makeBuffer(
                    length: bytes,
                    options: .storageModeShared)
            else {
                throw BenchHarnessError.metalObjectUnavailable("buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let qW = try makeBuffer(Int(qRows) * rowBytes, 0x12)
        let qS = try makeBuffer(Int(qRows) * groupCount * 2, 0x01)
        let qB = try makeBuffer(Int(qRows) * groupCount * 2, 0x00)
        let kW = try makeBuffer(Int(kvRows) * rowBytes, 0x34)
        let kS = try makeBuffer(Int(kvRows) * groupCount * 2, 0x01)
        let kB = try makeBuffer(Int(kvRows) * groupCount * 2, 0x00)
        let vW = try makeBuffer(Int(kvRows) * rowBytes, 0x56)
        let vS = try makeBuffer(Int(kvRows) * groupCount * 2, 0x01)
        let vB = try makeBuffer(Int(kvRows) * groupCount * 2, 0x00)
        let x = try makeBuffer(Int(n) * 2, 0x77)
        let qOut = try makeBuffer(Int(qRows) * 2, 0)
        let kOut = try makeBuffer(Int(kvRows) * 2, 0)
        let vOut = try makeBuffer(Int(kvRows) * 2, 0)

        let totalRows = Int(qRows + 2 * kvRows)
        let rowsPerThreadgroup = 8
        let threadgroups = (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        let bytesPerLaunch = UInt64(totalRows) * UInt64(rowBytes)

        var qVar = qRows
        var kvVar = kvRows
        var nVar = n

        let cb = try requireCommandBuffer(context.queue)
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qW, offset: 0, index: 0)
        enc.setBuffer(qS, offset: 0, index: 1)
        enc.setBuffer(qB, offset: 0, index: 2)
        enc.setBuffer(kW, offset: 0, index: 3)
        enc.setBuffer(kS, offset: 0, index: 4)
        enc.setBuffer(kB, offset: 0, index: 5)
        enc.setBuffer(vW, offset: 0, index: 6)
        enc.setBuffer(vS, offset: 0, index: 7)
        enc.setBuffer(vB, offset: 0, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        enc.setBuffer(qOut, offset: 0, index: 10)
        enc.setBuffer(kOut, offset: 0, index: 11)
        enc.setBuffer(vOut, offset: 0, index: 12)
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        for _ in 0..<iterations {
            enc.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: rowsPerThreadgroup * 32,
                    height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let gbPerSec = Double(bytesPerLaunch) / perIteration / 1_000_000_000
        let theoretical = 100.0
        print(
            "kernel=\(kernelName) iterations=\(iterations) "
                + "total=\(String(format: "%.4f", totalSeconds))s "
                + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print(
            "bytes/launch=\(bytesPerLaunch) "
                + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
                + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak"
        )
    }

}
