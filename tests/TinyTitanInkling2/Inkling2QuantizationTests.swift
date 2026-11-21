import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2QuantizationTests {
    @Test func e2m1AndE4m3Tables() {
        #expect(Inkling2Quantization.e2m1Value(0x0) == 0)
        #expect(Inkling2Quantization.e2m1Value(0x1) == 0.5)
        #expect(Inkling2Quantization.e2m1Value(0x7) == 6)
        #expect(Inkling2Quantization.e2m1Value(0x9) == -0.5)
        #expect(Inkling2Quantization.e4m3Value(0x38) == 1)
        #expect(Inkling2Quantization.e4m3Value(0xB8) == -1)
        #expect(Inkling2Quantization.e4m3Value(0x00) == 0)
        #expect(Inkling2Quantization.e4m3Value(0x01) == 0.001953125)
    }

    @Test func nvfp4LowNibbleFirst() throws {
        // One row, 16 values: byte 0x21 -> low 1 (0.5), high 2 (1.0), then zeros.
        var packed = Data(repeating: 0, count: 8)
        packed[0] = 0x21
        let scales = Data([0x38])  // E4M3 1.0
        let output = try Inkling2Quantization.dequantizeNVFP4(
            packed: packed, scales: scales, outerScale: 1,
            rows: 1, columns: 16)
        #expect(output[0] == 0.5)
        #expect(output[1] == 1.0)
        #expect(output[2] == 0)
    }

    @Test func nvfp4OuterScaleMultiplies() throws {
        var packed = Data(repeating: 0, count: 8)
        packed[0] = 0x02  // first value 1.0
        let scales = Data([0x38])
        let output = try Inkling2Quantization.dequantizeNVFP4(
            packed: packed, scales: scales, outerScale: 0.25,
            rows: 1, columns: 16)
        #expect(output[0] == 0.25)
    }

    @Test func mxfp8UsesPowerOfTwoScale() throws {
        let weights = Data(repeating: 0x38, count: 32)  // 1.0
        let output = try Inkling2Quantization.dequantizeMXFP8(
            weights: weights, scales: Data([127]),
            rows: 1, columns: 32)
        #expect(output.allSatisfy { $0 == 1 })
        let doubled = try Inkling2Quantization.dequantizeMXFP8(
            weights: weights, scales: Data([128]),
            rows: 1, columns: 32)
        #expect(doubled.allSatisfy { $0 == 2 })
    }

    @Test func malformedSizesAreRefused() {
        #expect(throws: Inkling2Quantization.Error.self) {
            try Inkling2Quantization.dequantizeNVFP4(
                packed: Data(count: 1), scales: Data(count: 1),
                outerScale: 1, rows: 1, columns: 16)
        }
        #expect(throws: Inkling2Quantization.Error.self) {
            try Inkling2Quantization.dequantizeMXFP8(
                weights: Data(count: 31), scales: Data(count: 1),
                rows: 1, columns: 32)
        }
    }
}
