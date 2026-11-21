import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct GGUFKQuantGEMVTests {
    private static func pattern(_ count: Int) -> [UInt8] {
        (0..<count).map { UInt8(($0 * 37 + 11) & 0xFF) }
    }

    private static func fp16(_ value: Float) -> [UInt8] {
        let bits = Float16(value).bitPattern
        return [UInt8(bits & 0xFF), UInt8(bits >> 8)]
    }

    private static func weights(type: GGUFType, rows: Int, columns: Int) -> [UInt8] {
        let blocks = columns / 256
        let blockBytes = type == .q2K ? 84 : 144
        var bytes = pattern(rows * blocks * blockBytes)
        for block in 0..<(rows * blocks) {
            let base = block * blockBytes
            let dOffset = type == .q2K ? 80 : 0
            bytes.replaceSubrange(
                (base + dOffset)..<(base + dOffset + 2), with: fp16(0.02))
            bytes.replaceSubrange(
                (base + dOffset + 2)..<(base + dOffset + 4), with: fp16(0.01))
        }
        return bytes
    }

    private static func input(_ count: Int) -> [Float] {
        (0..<count).map { Float(($0 * 17) % 23) / 23.0 - 0.5 }
    }

    @Test func matchesDequantizedDotProduct() throws {
        let context = try MetalContext()
        let gemv = try GGUFKQuantGEMV(context: context)
        for type in [GGUFType.q2K, .q4K] {
            let rows = 8
            let columns = 256
            let bytes = Self.weights(type: type, rows: rows, columns: columns)
            let x = Self.input(columns)
            let got = try gemv.gemv(
                weights: bytes, x: x, rows: rows, columns: columns, type: type)
            for row in 0..<rows {
                let blockBytes = type == .q2K ? 84 : 144
                let start = row * blockBytes
                let decoded = try GGUFDequant.decode(
                    type: type, bytes: Array(bytes[start..<(start + blockBytes)]))
                let expected = zip(decoded, x).reduce(Float(0)) {
                    $0 + $1.0 * $1.1
                }
                #expect(
                    abs(got[row] - expected) < 1e-3,
                    "\(type.name) row \(row): \(got[row]) vs \(expected)")
            }
        }
    }

    @Test func gemvMicrobenchmark() throws {
        let context = try MetalContext()
        let gemv = try GGUFKQuantGEMV(context: context)
        let rows = 2_304
        let columns = 2_304
        let x = Self.input(columns)
        for type in [GGUFType.q2K, .q4K] {
            let bytes = Self.weights(type: type, rows: rows, columns: columns)
            let prepared = try gemv.prepare(
                weights: bytes, rows: rows, columns: columns, type: type)
            _ = try gemv.gemv(prepared: prepared, x: x)
            let iterations = 20
            let start = Date()
            for _ in 0..<iterations {
                _ = try gemv.gemv(prepared: prepared, x: x)
            }
            let seconds = Date().timeIntervalSince(start)
            let bytesRead = rows * (columns / 256) * (type == .q2K ? 84 : 144)
            print(
                "\(type.name) GEMV \(rows)x\(columns): "
                    + "\(seconds / Double(iterations) * 1000) ms, "
                    + "\(Double(bytesRead) / seconds / 1e9) GB/s")
        }
    }
}
