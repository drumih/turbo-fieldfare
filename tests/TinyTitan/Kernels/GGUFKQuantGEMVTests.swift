import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct GGUFKQuantGEMVTests {
    private static var benchmarkEnabled: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_BENCH"] == "1"
    }

    private static func fp16(_ value: Float) -> [UInt8] {
        let bits = Float16(value).bitPattern
        return [UInt8(bits & 0xFF), UInt8(bits >> 8)]
    }

    /// Deterministic blocks with every quant/scale bit pattern exercised and a
    /// distinct `d`/`dmin` per block, so a kernel that reads the wrong block,
    /// row or scale cannot match the host decode by accident.
    private static func weights(type: GGUFType, rows: Int, columns: Int) -> [UInt8] {
        let blocks = rows * (columns / 256)
        let blockBytes = type == .q2K ? 84 : 144
        var bytes = (0..<(blocks * blockBytes)).map { UInt8(($0 &* 37 &+ 11) & 0xFF) }
        let dOffset = type == .q2K ? 80 : 0
        for block in 0..<blocks {
            let base = block * blockBytes + dOffset
            let d = 0.01 + 0.002 * Float(block % 7)
            let dmin = 0.005 + 0.001 * Float(block % 5)
            bytes.replaceSubrange(base..<(base + 2), with: fp16(d))
            bytes.replaceSubrange((base + 2)..<(base + 4), with: fp16(dmin))
        }
        return bytes
    }

    private static func input(_ count: Int) -> [Float] {
        (0..<count).map { Float(($0 * 17) % 23) / 23.0 - 0.5 }
    }

    /// Host reference: decode each row with the bit-exact `GGUFDequant` and
    /// take the dot product in Double. Returns the value and the sum of
    /// |w·x| terms, which scales the float-reassociation tolerance.
    private static func reference(
        bytes: [UInt8], x: [Float], rows: Int, columns: Int, type: GGUFType
    ) throws -> [(value: Double, magnitude: Double)] {
        let rowBytes = (columns / 256) * (type == .q2K ? 84 : 144)
        return try (0..<rows).map { row in
            let decoded = try GGUFDequant.decode(
                type: type, bytes: bytes[(row * rowBytes)..<((row + 1) * rowBytes)])
            var value = 0.0
            var magnitude = 0.0
            for (w, v) in zip(decoded, x) {
                value += Double(w) * Double(v)
                magnitude += abs(Double(w) * Double(v))
            }
            return (value, magnitude)
        }
    }

    @Test(arguments: GGUFKQuantGEMV.Kernel.allCases)
    func matchesDequantizedDotProduct(kernel: GGUFKQuantGEMV.Kernel) throws {
        let context = try MetalContext()
        let gemv = try GGUFKQuantGEMV(context: context)
        // 1 and 37 rows leave partial tiles for every geometry; 2,304 columns
        // (nine blocks) leave the four block lanes uneven.
        for type in [GGUFType.q2K, .q4K] {
            for (rows, columns) in [(1, 256), (37, 256), (37, 2_304), (64, 1_280)] {
                let bytes = Self.weights(type: type, rows: rows, columns: columns)
                let x = Self.input(columns)
                let expected = try Self.reference(
                    bytes: bytes, x: x, rows: rows, columns: columns, type: type)
                let got = try gemv.gemv(
                    weights: bytes, x: x, rows: rows, columns: columns, type: type,
                    kernel: kernel)
                #expect(got.count == rows)
                for row in 0..<rows {
                    let error = abs(Double(got[row]) - expected[row].value)
                    let bound = 1e-5 * expected[row].magnitude + 1e-5
                    #expect(
                        error <= bound,
                        "\(kernel) \(type.name) \(rows)x\(columns) row \(row): \(got[row]) vs \(expected[row].value)"
                    )
                }
            }
        }
    }

    /// Paired, interleaved A/B of every geometry on a 2,304 x 2,304 projection
    /// (Inkling2's hidden size). Each round runs every kernel once, in a
    /// rotated order, as one command buffer of `dispatches` GEMVs timed on the
    /// GPU, so the number is the per-projection cost of a batched executor.
    /// Opt-in: `TINYTITAN_KERNEL_BENCH=1 swift test -c release --filter GGUFKQuantGEMVTests`.
    @Test(.enabled(if: benchmarkEnabled))
    func pairedGeometryBenchmark() throws {
        let context = try MetalContext()
        let gemv = try GGUFKQuantGEMV(context: context)
        let rows = 2_304
        let columns = 2_304
        let rounds = 7
        let dispatches = 200
        let kernels = GGUFKQuantGEMV.Kernel.allCases
        let x = Self.input(columns)
        for type in [GGUFType.q2K, .q4K] {
            let bytes = Self.weights(type: type, rows: rows, columns: columns)
            let prepared = try gemv.prepare(
                weights: bytes, rows: rows, columns: columns, type: type)
            // Warm every pipeline and check the geometries agree on this shape.
            let baseline = try gemv.gemv(prepared: prepared, x: x, kernel: kernels[0])
            for kernel in kernels.dropFirst() {
                let other = try gemv.gemv(prepared: prepared, x: x, kernel: kernel)
                let worst = zip(baseline, other).map { abs($0 - $1) }.max() ?? 0
                #expect(worst < 1e-2, "\(kernel) disagrees with \(kernels[0]) by \(worst)")
            }
            var samples: [GGUFKQuantGEMV.Kernel: [Double]] = [:]
            for round in 0..<rounds {
                for offset in 0..<kernels.count {
                    let kernel = kernels[(round + offset) % kernels.count]
                    let seconds = try gemv.gpuSecondsPerDispatch(
                        prepared: prepared, kernel: kernel, dispatches: dispatches)
                    samples[kernel, default: []].append(seconds)
                }
            }
            let bytesRead = Double(bytes.count)
            for kernel in kernels {
                let sorted = (samples[kernel] ?? []).sorted()
                #expect(sorted.count == rounds && sorted[0] > 0)
                let median = sorted[sorted.count / 2]
                let line = String(
                    format: "%@ %@ %dx%d: median %.4f ms (min %.4f, max %.4f) %.1f GB/s",
                    type.name, "\(kernel)", rows, columns, median * 1e3,
                    (sorted.first ?? 0) * 1e3, (sorted.last ?? 0) * 1e3,
                    bytesRead / median / 1e9)
                FileHandle.standardError.write(Data((line + "\n").utf8))
            }
        }
    }
}
