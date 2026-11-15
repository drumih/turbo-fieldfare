//
//  MetalHeadBenchmarks.swift
//  TinyTitanBench
//
//  The vocabulary-head GEMV benchmark: the one 8-bit op every install runs per
//  token, in its single-row and two-row forms. Split out of
//  `MetalBenchmarks.swift` under the 500-line rule; the dispatch stays in
//  `main.swift`.
//

import Foundation
import Metal
import TinyTitan
import TinyTitanCore

extension TinyTitanBench {
    /// The vocabulary head GEMV at the 35B family's shape (248,320 x 2048),
    /// the one GEMV every install runs at 8-bit. head_affine8 is the shipped
    /// generic affine kernel at bits=8 (measured 89.9 GB/s, at the ceiling;
    /// an 8-byte-per-lane specialization measured 91.7, noise, and was not
    /// kept), head_affine4 the same kernel at bits=4 (51.8 GB/s), head_int4
    /// the int4 head kernel (91.5). TINYTITAN_BENCH_HEAD_SHAPE=qwen38 takes
    /// 248,320 x 2560.
    /// One synthetic activation row per batch row, distinct per row so a
    /// two-row head cannot pass by reading row 0 twice.
    private static func fillHeadInput(_ x: MTLBuffer, n: UInt32, rows: Int) {
        let pointer = x.contents().assumingMemoryBound(to: UInt16.self)
        for row in 0..<rows {
            for i in 0..<Int(n) {
                pointer[row * Int(n) + i] =
                    Float16(Float((i + row * 31) % 97 + 1) * 0.001).bitPattern
            }
        }
    }

    /// Which kernel a `head_*` bench name selects, and how many activation
    /// rows one launch carries. The int4 four-row probe is refused: it was
    /// measured at the two-row launch's cost for twice the rows and is not a
    /// shipped path, so a bench run must not imply one.
    private struct HeadKernel {
        let bits: Int
        let kernel: String
        let rowsPerLaunch: Int
    }

    private static func headKernel(named kernelName: String) throws -> HeadKernel {
        let fourRows = kernelName.hasSuffix("_four_rows")
        let twoRows = kernelName.hasSuffix("_two_rows")
        let suffix = fourRows ? "4_simd" : (twoRows ? "2_simd" : "_simd")
        let base =
            kernelName
            .replacingOccurrences(of: "_four_rows", with: "")
            .replacingOccurrences(of: "_two_rows", with: "")
        let bits: Int
        let kernel: String
        switch base {
        case "head_affine8":
            bits = 8
            kernel = "affine_quant_gemv\(suffix)"
        case "head_affine4":
            bits = 4
            kernel = "affine_quant_gemv\(suffix)"
        case "head_int4":
            if fourRows {
                throw BenchHarnessError.badConfiguration(
                    "the int4 four-row head was measured at the two-row launch's cost "
                        + "for twice the rows and is not shipped; bench head_int4_two_rows")
            }
            bits = 4
            kernel = "dequant_int4_gemv\(suffix)"
        default:
            bits = 8
            kernel = "affine_quant_gemv\(suffix)"
        }
        return HeadKernel(
            bits: bits, kernel: kernel, rowsPerLaunch: fourRows ? 4 : (twoRows ? 2 : 1))
    }

    static func runHead(
        kernelName: String,
        iterations: Int,
        context: MetalContext
    ) throws {
        let device = context.device
        let qwen38 = ProcessInfo.processInfo.environment["TINYTITAN_BENCH_HEAD_SHAPE"] == "qwen38"
        let rows: UInt32 = 248_320
        let n: UInt32 = qwen38 ? 2560 : 2048
        let groupCount = Int(n) / 64
        let selection = try Self.headKernel(named: kernelName)
        let bits = selection.bits
        let kernel = selection.kernel
        let xRows = selection.rowsPerLaunch
        let rowBytes = Int(n) * bits / 8
        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw BenchHarnessError.metalObjectUnavailable("buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let w = try makeBuffer(Int(rows) * rowBytes, 0x5A)
        let s = try makeBuffer(Int(rows) * groupCount * 2, 0x3C)
        let bb = try makeBuffer(Int(rows) * groupCount * 2, 0x00)
        let x = try makeBuffer(Int(n) * 2 * xRows, 0)
        Self.fillHeadInput(x, n: n, rows: xRows)
        let y = try makeBuffer(Int(rows) * 2 * xRows, 0)
        let constants =
            kernel.hasPrefix("affine")
            ? [MetalFunctionConstant(index: 100, value: .uint32(UInt32(bits)))] : []
        let pso = try context.pipeline(
            kernel, constants: constants,
            maxTotalThreadsPerThreadgroup: 256)
        var rowsVar = rows
        var nVar = n
        let threadgroups = (Int(rows) + 7) / 8
        let bytes = UInt64(rows) * UInt64(rowBytes + groupCount * 4)
        // TINYTITAN_BENCH_HEAD_CADENCE_MS: one launch per command buffer with
        // a wall-clock gap between them, mimicking decode's once-per-token
        // cadence instead of back-to-back steady state.
        // TINYTITAN_BENCH_HEAD_KEEPWARM_US: run a one-threadgroup dispatch on a
        // second queue throughout those gaps, to test whether the cadence
        // penalty is GPU power state at all, and whether a stream can hold it.
        let cadenceMs = try envIntFlag("TINYTITAN_BENCH_HEAD_CADENCE_MS")
        let keepWarmUs = try envIntFlag("TINYTITAN_BENCH_HEAD_KEEPWARM_US")
        if keepWarmUs != nil, cadenceMs == nil {
            throw BenchHarnessError.badConfiguration(
                "TINYTITAN_BENCH_HEAD_KEEPWARM_US needs TINYTITAN_BENCH_HEAD_CADENCE_MS: "
                    + "back-to-back launches leave no idle window to hold warm")
        }
        if let cadenceMs {
            let keepWarm: GPUKeepWarmStream?
            if let keepWarmUs {
                keepWarm = try GPUKeepWarmStream(
                    device: device, pso: pso, columns: n,
                    rowBytes: rowBytes, groupBytes: groupCount * 2,
                    input: x, intervalUs: keepWarmUs)
                try keepWarm?.calibrate()
            } else {
                keepWarm = nil
            }
            try runHeadCadence(
                queue: context.queue, pso: pso, w: w, s: s, biases: bb, x: x, y: y,
                rows: rowsVar, n: nVar, threadgroups: threadgroups,
                iterations: iterations, cadenceMs: cadenceMs,
                bytes: bytes, kernel: kernel, bits: bits,
                keepWarm: keepWarm, keepWarmUs: keepWarmUs)
            return
        }
        let cb = try requireCommandBuffer(context.queue)
        let enc = try {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw BenchHarnessError.metalObjectUnavailable("compute encoder")
            }
            return enc
        }()
        enc.setComputePipelineState(pso)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(s, offset: 0, index: 1)
        enc.setBuffer(bb, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        enc.setBytes(&rowsVar, length: 4, index: 5)
        enc.setBytes(&nVar, length: 4, index: 6)
        for _ in 0..<iterations {
            enc.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let total = cb.gpuEndTime - cb.gpuStartTime
        let per = total / Double(iterations)
        let yPtr = y.contents().assumingMemoryBound(to: UInt16.self)
        let y0 = Float(Float16(bitPattern: yPtr[0]))
        let yLast = Float(Float16(bitPattern: yPtr[Int(rows) - 1]))
        print(
            "kernel=\(kernel) bits=\(bits) n=\(n) iterations=\(iterations) "
                + "per_launch=\(String(format: "%.1f", per * 1_000_000))us "
                + "achieved=\(String(format: "%.1f", Double(bytes) / per / 1e9)) GB/s "
                + "rows_per_launch=\(xRows) "
                + "per_row_us=\(String(format: "%.1f", per * 1_000_000 / Double(xRows))) "
                + "per_row_gbs=\(String(format: "%.1f", Double(bytes) / per / 1e9 / Double(xRows))) "
                + "y0=\(y0) yLast=\(yLast)")
    }

    /// Cadence arm for `runHead`: one launch per command buffer with a
    /// wall-clock gap between them, measuring the GPU-state penalty decode's
    /// once-per-token cadence pays over back-to-back steady state.
    private static func runHeadCadence(
        queue: MTLCommandQueue,
        pso: MTLComputePipelineState,
        w: MTLBuffer,
        s: MTLBuffer,
        biases: MTLBuffer,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: UInt32,
        n: UInt32,
        threadgroups: Int,
        iterations: Int,
        cadenceMs: Int,
        bytes: UInt64,
        kernel: String,
        bits: Int,
        keepWarm: GPUKeepWarmStream?,
        keepWarmUs: Int?
    ) throws {
        var rowsVar = rows
        var nVar = n
        var gpuSeconds: Double = 0
        var samples: [Double] = []
        guard iterations >= 2 else {
            throw BenchHarnessError.badConfiguration(
                "the cadence arm needs at least 2 iterations: it discards the first and divides by iterations-1"
            )
        }
        keepWarm?.start()
        // A thrown allocation below would otherwise leave the worker spinning
        // until the process exits; the success path stops it on the next lines
        // so the dispatch count it returns is still the one reported.
        var keepWarmStopped = false
        defer {
            if !keepWarmStopped { _ = keepWarm?.stop() }
        }
        for i in 0..<iterations {
            let cb = try requireCommandBuffer(queue)
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw BenchHarnessError.metalObjectUnavailable("compute encoder")
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(w, offset: 0, index: 0)
            enc.setBuffer(s, offset: 0, index: 1)
            enc.setBuffer(biases, offset: 0, index: 2)
            enc.setBuffer(x, offset: 0, index: 3)
            enc.setBuffer(y, offset: 0, index: 4)
            enc.setBytes(&rowsVar, length: 4, index: 5)
            enc.setBytes(&nVar, length: 4, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if i > 0 {
                let seconds = cb.gpuEndTime - cb.gpuStartTime
                gpuSeconds += seconds
                samples.append(seconds)
            }
            if i + 1 < iterations {
                Thread.sleep(forTimeInterval: Double(cadenceMs) / 1000)
            }
        }
        let spinDispatches = keepWarm.map { $0.stop() } ?? 0
        keepWarmStopped = true
        let spinProbe = keepWarm?.probeMicroseconds ?? 0
        let per = gpuSeconds / Double(iterations - 1)
        let sorted = samples.sorted()
        let median =
            sorted.count.isMultiple(of: 2)
            ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
            : sorted[sorted.count / 2]
        print(
            "kernel=\(kernel) bits=\(bits) n=\(n) iterations=\(iterations) "
                + "cadence_ms=\(cadenceMs) "
                + "keepwarm_us=\(keepWarmUs.map(String.init) ?? "off") "
                + "keepwarm_dispatches=\(spinDispatches) "
                + "keepwarm_probe_us=\(String(format: "%.1f", spinProbe)) "
                + "per_launch=\(String(format: "%.1f", per * 1_000_000))us "
                + "median=\(String(format: "%.1f", median * 1_000_000))us "
                + "min=\(String(format: "%.1f", sorted[0] * 1_000_000))us "
                + "max=\(String(format: "%.1f", sorted[sorted.count - 1] * 1_000_000))us "
                + "achieved=\(String(format: "%.1f", Double(bytes) / per / 1e9)) GB/s")
    }
}
