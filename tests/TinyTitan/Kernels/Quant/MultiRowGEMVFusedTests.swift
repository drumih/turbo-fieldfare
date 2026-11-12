import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The fused decode projections against their multi-row form: GDN's
/// qkv/z/a/b input projection (`gdn_in_proj_gemv_simd`) and full attention's
/// q/k/v (`dequant_int4_qkv_gemv_simd`) each become one multi-row dispatch of
/// four or three segments. Every activation row must come out byte-identical
/// to the fused decode kernel run on that row alone.
@Suite struct MultiRowGEMVFusedTests {
    typealias Fixtures = MultiRowGEMVFixtures

    static let rowCounts = [1, 2, 5, 8, 9]

    struct GDNCase: Sendable, CustomTestStringConvertible {
        let config: LinearAttentionConfig
        let hiddenSize: Int
        let specialize: Bool
        let abBF16: Bool
        var testDescription: String {
            "qkv\(config.qkvDim) n\(hiddenSize) fc=\(specialize) abBF16=\(abBF16)"
        }
    }

    /// Qwen3.8's GDN geometry (16 key / 48 value heads of 128, hidden 2560)
    /// with the decode pipeline constant-folded as the runner builds it, and
    /// a small unspecialized one.
    static let qwen38 = LinearAttentionConfig(
        numKHeads: 16, numVHeads: 48, keyHeadDim: 128, valueHeadDim: 128,
        convKernelSize: 4, outputGate: .sigmoid)
    static let small = LinearAttentionConfig(
        numKHeads: 2, numVHeads: 4, keyHeadDim: 32, valueHeadDim: 32, convKernelSize: 4)

    @Test(arguments: [
        GDNCase(config: qwen38, hiddenSize: 2560, specialize: true, abBF16: false),
        GDNCase(config: qwen38, hiddenSize: 2560, specialize: true, abBF16: true),
        GDNCase(config: small, hiddenSize: 320, specialize: false, abBF16: false),
    ])
    func gdnInputProjectionMatchesDecodePerRow(testCase: GDNCase) throws {
        let context = try MetalContext()
        let config = testCase.config
        let n = testCase.hiddenSize
        let gdn = try GDN(
            context: context, config: config,
            specializedHiddenSize: testCase.specialize ? n : nil,
            abBF16: testCase.abBF16)
        var generator = Fixtures.Generator(seed: 0x4D52_0D17 &+ UInt64(config.qkvDim))
        let abFormat: MultiRowGEMV.Format = testCase.abBF16 ? .bf16 : .int4
        let outputs: [(view: TensorView, format: MultiRowGEMV.Format, rows: Int)] = [
            (
                try Fixtures.matrix(
                    device: context.device, format: .int4, rows: config.qkvDim,
                    columns: n, pad: 2, generator: &generator), .int4, config.qkvDim
            ),
            (
                try Fixtures.matrix(
                    device: context.device, format: .int4, rows: config.valueDim,
                    columns: n, pad: 2, generator: &generator), .int4, config.valueDim
            ),
            (
                try Fixtures.matrix(
                    device: context.device, format: abFormat, rows: config.numVHeads,
                    columns: n, pad: 2, generator: &generator), abFormat, config.numVHeads
            ),
            (
                try Fixtures.matrix(
                    device: context.device, format: abFormat, rows: config.numVHeads,
                    columns: n, pad: 2, generator: &generator), abFormat, config.numVHeads
            ),
        ]
        let maxRows = Self.rowCounts.max() ?? 1
        let x = try Fixtures.activations(
            device: context.device, count: maxRows * n, generator: &generator)
        let multiRow = MultiRowGEMV(context: context)
        for rows in Self.rowCounts {
            let actual = try outputs.map {
                try Fixtures.zeroed(device: context.device, halves: rows * $0.rows)
            }
            let expected = try (0..<rows).map { _ in
                try outputs.map { try Fixtures.zeroed(device: context.device, halves: $0.rows) }
            }
            try Fixtures.run(context) { cb in
                for row in 0..<rows {
                    try gdn.encodeInputProjections(
                        commandBuffer: cb, x: x, xOffset: row * n * 2,
                        qkv: outputs[0].view, qkvOut: expected[row][0],
                        z: outputs[1].view, zOut: expected[row][1],
                        a: outputs[2].view, aOut: expected[row][2],
                        b: outputs[3].view, bOut: expected[row][3],
                        hiddenSize: n)
                }
                try multiRow.encode(
                    commandBuffer: cb,
                    segments: zip(outputs, actual).map { output, y in
                        .init(
                            view: output.view, format: output.format, rows: output.rows,
                            y: y, yRowStride: output.rows)
                    },
                    x: x, xRowStride: n, activationRows: rows, columns: n)
            }
            for row in 0..<rows {
                for (index, output) in outputs.enumerated() {
                    Fixtures.expectIdentical(
                        expected[row][index], actual[index], halves: output.rows,
                        actualOffsetHalves: row * output.rows,
                        "GDN \(testCase.testDescription) segment \(index) W=\(rows) row \(row)")
                }
            }
        }
    }

    struct QKVCase: Sendable, CustomTestStringConvertible {
        let qRows: Int
        let kvRows: Int
        let n: Int
        var testDescription: String { "q\(qRows) kv\(kvRows) n\(n)" }
    }

    /// 4096/2048/2816 is one of `FusedQKVGEMV`'s constant-folded shapes;
    /// 12288/512/2560 is Qwen3.8's packed [query ; gate] with its k and v.
    @Test(arguments: [
        QKVCase(qRows: 4096, kvRows: 2048, n: 2816),
        QKVCase(qRows: 12288, kvRows: 512, n: 2560),
        QKVCase(qRows: 96, kvRows: 32, n: 320),
    ])
    func fusedQKVMatchesDecodePerRow(testCase: QKVCase) throws {
        let context = try MetalContext()
        let fused = try FusedQKVGEMV(context: context)
        let n = testCase.n
        var generator = Fixtures.Generator(seed: 0x4D52_0C4B &+ UInt64(testCase.qRows))
        let rowsPerSegment = [testCase.qRows, testCase.kvRows, testCase.kvRows]
        let views = try rowsPerSegment.map {
            try Fixtures.matrix(
                device: context.device, format: .int4, rows: $0, columns: n, pad: 2,
                generator: &generator)
        }
        let maxRows = Self.rowCounts.max() ?? 1
        // The decode wrapper binds x at offset zero, so each row gets its own.
        let rowInputs = try (0..<maxRows).map { _ in
            try Fixtures.activations(device: context.device, count: n, generator: &generator)
        }
        let x = try Fixtures.zeroed(device: context.device, halves: maxRows * n)
        for (row, input) in rowInputs.enumerated() {
            memcpy(x.contents() + row * n * 2, input.contents(), n * 2)
        }
        let multiRow = MultiRowGEMV(context: context)
        for rows in Self.rowCounts {
            let expected = try rowsPerSegment.map {
                try Fixtures.zeroed(device: context.device, halves: rows * $0)
            }
            let actual = try rowsPerSegment.map {
                try Fixtures.zeroed(device: context.device, halves: rows * $0)
            }
            try Fixtures.run(context) { cb in
                for row in 0..<rows {
                    try fused.encode(
                        commandBuffer: cb,
                        qWeights: views[0].buffer, qWeightsOffset: Int(views[0].offset),
                        qScales: views[0].buffer, qScalesOffset: Int(views[0].scaleOffset),
                        qBiases: views[0].buffer, qBiasesOffset: Int(views[0].biasOffset),
                        kWeights: views[1].buffer, kWeightsOffset: Int(views[1].offset),
                        kScales: views[1].buffer, kScalesOffset: Int(views[1].scaleOffset),
                        kBiases: views[1].buffer, kBiasesOffset: Int(views[1].biasOffset),
                        vWeights: views[2].buffer, vWeightsOffset: Int(views[2].offset),
                        vScales: views[2].buffer, vScalesOffset: Int(views[2].scaleOffset),
                        vBiases: views[2].buffer, vBiasesOffset: Int(views[2].biasOffset),
                        x: rowInputs[row],
                        qOut: expected[0], qOutOffset: row * testCase.qRows * 2,
                        kOut: expected[1], kOutOffset: row * testCase.kvRows * 2,
                        vOut: expected[2], vOutOffset: row * testCase.kvRows * 2,
                        qRows: UInt32(testCase.qRows), kvRows: UInt32(testCase.kvRows),
                        n: UInt32(n))
                }
                try multiRow.encode(
                    commandBuffer: cb,
                    segments: (0..<3).map {
                        .init(
                            view: views[$0], format: .int4, rows: rowsPerSegment[$0],
                            y: actual[$0], yRowStride: rowsPerSegment[$0])
                    },
                    x: x, xRowStride: n, activationRows: rows, columns: n)
            }
            for index in 0..<3 {
                Fixtures.expectIdentical(
                    expected[index], actual[index], halves: rows * rowsPerSegment[index],
                    "QKV \(testCase.testDescription) segment \(index) W=\(rows)")
            }
        }
    }
}
