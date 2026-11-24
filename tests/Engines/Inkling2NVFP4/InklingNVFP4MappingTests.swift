import Foundation
import Testing

@testable import Inkling2NVFP4

/// TT-125, model-free: the checkpoint-to-graph tensor map and the expert repack
/// arithmetic, on synthetic inputs whose expected output is written out by hand.
@Suite struct InklingNVFP4MappingTests {
    /// Runs a repack over copies of `values` and `scales` and returns its output.
    func repacked(
        _ values: [UInt8], _ scales: [UInt8], outputBytes: Int,
        _ body: (UnsafeRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer) -> Void
    ) -> [UInt8] {
        let v = UnsafeMutableRawPointer.allocate(byteCount: values.count, alignment: 16)
        let s = UnsafeMutableRawPointer.allocate(byteCount: scales.count, alignment: 16)
        let o = UnsafeMutableRawPointer.allocate(byteCount: outputBytes, alignment: 16)
        defer {
            v.deallocate()
            s.deallocate()
            o.deallocate()
        }
        v.copyMemory(from: values, byteCount: values.count)
        s.copyMemory(from: scales, byteCount: scales.count)
        o.initializeMemory(as: UInt8.self, repeating: 0xEE, count: outputBytes)
        body(v, s, o)
        return Array(
            UnsafeBufferPointer(start: o.assumingMemoryBound(to: UInt8.self), count: outputBytes))
    }

    @Test func gateUpRowsGoFromInterleavedToConcatenated() {
        // 6 rows: gate 0, up 0, gate 1, up 1, gate 2, up 2 -> gate 0,1,2, up 0,1,2.
        let order = (0..<6).map { InklingNVFP4Repack.sourceRow($0, rows: 6, interleaved: true) }
        #expect(order == [0, 2, 4, 1, 3, 5])
        #expect(InklingNVFP4Repack.sourceRow(4, rows: 6, interleaved: false) == 4)
    }

    @Test func nvfp4BlockPutsValueJAndJPlus8InOneByte() {
        // One row of 64 values: value i has code i % 16; codes pair-packed.
        var values: [UInt8] = []
        for pair in 0..<32 {
            let low = UInt8((2 * pair) % 16)
            let high = UInt8((2 * pair + 1) % 16)
            values.append(low | (high << 4))
        }
        let scales: [UInt8] = [0x38, 0x40, 0x48, 0x50]
        let out = repacked(values, scales, outputBytes: 36) {
            InklingNVFP4Repack.nvfp4(
                values: $0, scales: $1, rows: 1, columns: 64, interleaved: false, into: $2)
        }
        #expect(Array(out[0..<4]) == scales)
        // Each 16-group holds codes 0...15: byte j = code j | code (j+8) << 4.
        var group: [UInt8] = []
        for j in 0..<8 { group.append(UInt8(j) | (UInt8(j + 8) << 4)) }
        for g in 0..<4 { #expect(Array(out[(4 + 8 * g)..<(12 + 8 * g)]) == group) }
    }

    /// The word-wide repack the cache runs equals the nibble-at-a-time
    /// definition on arbitrary bytes (16 rows x 128 columns, interleaved rows).
    @Test func wordWideNVFP4RepackEqualsTheReference() {
        var generator = SystemRandomNumberGenerator()
        let rows = 16
        let columns = 128
        let values = (0..<(rows * columns / 2)).map { _ in
            UInt8.random(in: 0...255, using: &generator)
        }
        let scales = (0..<(rows * columns / 16)).map { _ in
            UInt8.random(in: 0...255, using: &generator)
        }
        let bytes = InklingNVFP4Repack.nvfp4Bytes(rows: rows, columns: columns)
        for interleaved in [false, true] {
            let fast = repacked(values, scales, outputBytes: bytes) {
                InklingNVFP4Repack.nvfp4(
                    values: $0, scales: $1, rows: rows, columns: columns,
                    interleaved: interleaved, into: $2)
            }
            let reference = repacked(values, scales, outputBytes: bytes) {
                InklingNVFP4Repack.nvfp4Reference(
                    values: $0, scales: $1, rows: rows, columns: columns,
                    interleaved: interleaved, into: $2)
            }
            #expect(fast == reference, "interleaved \(interleaved)")
        }
    }

    @Test func mxfp8BlockIsScalePadThenValues() {
        let values = [UInt8](0..<64)
        let scales: [UInt8] = [127, 130]
        let out = repacked(values, scales, outputBytes: 68) {
            InklingNVFP4Repack.mxfp8(
                values: $0, scales: $1, rows: 1, columns: 64, interleaved: false, into: $2)
        }
        #expect(out[0] == 127 && out[1] == 0 && Array(out[2..<34]) == Array(0..<32))
        #expect(out[34] == 130 && out[35] == 0 && Array(out[36..<68]) == Array(32..<64))
    }

    @Test func convolutionKernelsAreTransposedWithTapsReversed() {
        // Stored [taps = 4][channels = 2]; read [channels][taps], taps reversed.
        let stored: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let read = InklingNVFP4TensorMap.apply(
            .transposeReversed, to: stored, rows: 4, columns: 2)
        #expect(read == [7, 5, 3, 1, 8, 6, 4, 2])
        #expect(
            InklingNVFP4TensorMap.apply(.transpose, to: [1, 2, 3, 4, 5, 6], rows: 2, columns: 3)
                == [1, 4, 2, 5, 3, 6])
    }

    @Test func rolesMapToTheMeasuredCheckpointTensors() {
        let present: Set<String> = [
            "model.llm.layers.5.attn.k_sconv.static_w", "model.llm.layers.4.attn.k_sconv.weight",
            "model.llm.attn_res.attn_queries.8", "model.llm.layers.9.controllers.mlp.gain_weight",
            "model.llm.layers.0.attn.rel_logits_proj.proj", "model.llm.unembed.weight",
        ]
        let exists = { present.contains($0) }
        typealias Source = InklingNVFP4TensorMap.Source
        #expect(
            InklingNVFP4TensorMap.source(for: "blk.5.attn_k_conv.weight", exists: exists)
                == Source(
                    name: "model.llm.layers.5.attn.k_sconv.static_w",
                    transform: .transposeReversed, converted: true))
        #expect(
            InklingNVFP4TensorMap.source(for: "blk.4.attn_k_conv.weight", exists: exists)?.name
                == "model.llm.layers.4.attn.k_sconv.weight")
        #expect(
            InklingNVFP4TensorMap.source(for: "blk.9.attn_res_query.weight", exists: exists)?.name
                == "model.llm.attn_res.attn_queries.8")
        #expect(
            InklingNVFP4TensorMap.source(for: "blk.9.loop_mlp_gain.weight", exists: exists)?.name
                == "model.llm.layers.9.controllers.mlp.gain_weight")
        #expect(
            InklingNVFP4TensorMap.source(for: "blk.0.attn_rel_bias.weight", exists: exists)?
                .transform == .transpose)
        #expect(
            InklingNVFP4TensorMap.source(for: "output.weight", exists: exists)
                == Source(name: "model.llm.unembed.weight", transform: .identity, converted: false))
        #expect(
            InklingNVFP4TensorMap.source(for: "blk.0.attn_res_query.weight", exists: exists) == nil)
    }
}
