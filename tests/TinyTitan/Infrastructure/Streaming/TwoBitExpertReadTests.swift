import Foundation
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

/// The cold tier reads a 2-bit expert that is half a slot long, then expands it
/// to the 4-bit slot the existing kernels expect. This pins both halves on a
/// real file: the offset/byte-count read returns exactly the expert's bytes,
/// and the expansion reproduces the 4-bit values the 2-bit ones stood for.
@Suite struct TwoBitExpertReadTests {

    /// Writes one layer file: a 4-bit expert at 0..<stride and a 2-bit expert
    /// at the next stride. Returns the file URL and the two payloads.
    private static func writeLayerFile(
        stride: Int
    ) throws -> (url: URL, short: [UInt8], shortOffset: UInt64) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-two-bit-read-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("layer_00.bin")

        // The short expert is half a slot: 2-bit values, four per byte, derived
        // from a deterministic pattern so the expansion can be checked exactly.
        var short = [UInt8](repeating: 0, count: stride / 2)
        for i in short.indices { short[i] = UInt8(truncatingIfNeeded: i &* 29 &+ 7) }
        var payload = [UInt8](repeating: 0xAA, count: stride)  // hot expert filler
        payload.append(contentsOf: short)
        try Data(payload).write(to: url)
        return (url, short, UInt64(stride))
    }

    @Test func aShortExpertIsReadByOffsetAndCount() throws {
        let stride = 16_384
        let (url, short, offset) = try Self.writeLayerFile(stride: stride)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let reader = try ParallelExpertReader(
            path: url.path, expertStride: stride, threads: 2, bypassCache: true)
        var destination = [UInt8](repeating: 0, count: stride)
        destination.withUnsafeMutableBytes { raw in
            guard let pointer = raw.baseAddress else {
                Issue.record("destination buffer has no base address")
                return
            }
            #expect(throws: Never.self) {
                try reader.fetch(
                    offsets: [offset], byteCounts: [short.count], into: [pointer])
            }
        }
        #expect(destination.prefix(short.count).elementsEqual(short))
        // Bytes past the payload are untouched, not zeroed by the read.
        #expect(destination[short.count] == 0x00)
    }

    @Test func expansionReproducesTheFourBitValues() throws {
        let stride = 16_384
        let (url, short, offset) = try Self.writeLayerFile(stride: stride)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let reader = try ParallelExpertReader(
            path: url.path, expertStride: stride, threads: 2, bypassCache: true)
        var staged = [UInt8](repeating: 0, count: short.count)
        staged.withUnsafeMutableBytes { raw in
            guard let pointer = raw.baseAddress else {
                Issue.record("staging buffer has no base address")
                return
            }
            try? reader.fetch(offsets: [offset], byteCounts: [short.count], into: [pointer])
        }

        var expanded = [UInt8](repeating: 0, count: short.count * 2)
        staged.withUnsafeBufferPointer { src in
            expanded.withUnsafeMutableBufferPointer { dst in
                guard let source = src.baseAddress, let destination = dst.baseAddress
                else {
                    Issue.record("expansion buffers have no base address")
                    return
                }
                TwoBitExpand.expand(
                    source: source, destination: destination,
                    byteCount: short.count)
            }
        }
        // Every 2-bit value in the source becomes 5x in the 4-bit stream, so
        // decoding both back to values must agree.
        for index in 0..<short.count {
            let byte = short[index]
            let values = [
                (byte >> 6) & 0x3, (byte >> 4) & 0x3,
                (byte >> 2) & 0x3, byte & 0x3,
            ]
            let out0 = expanded[2 * index]
            let out1 = expanded[2 * index + 1]
            // 4-bit packs element 2k in the low nibble, 2k+1 in the high one.
            #expect((out0 & 0xF) == values[0] * 5)
            #expect((out0 >> 4) == values[1] * 5)
            #expect((out1 & 0xF) == values[2] * 5)
            #expect((out1 >> 4) == values[3] * 5)
        }
    }

    /// The expansion is only useful if the arithmetic survives it: an expanded
    /// 4-bit value against the divided scale must reproduce the 2-bit product,
    /// which is what lets the existing 4-bit kernels run a cold expert.
    @Test func expandedValuesReproduceTheTwoBitProduct() {
        let scales: [Float] = [0.03125, 0.1, 1.0, 7.5]
        let biases: [Float] = [0, -0.5, 2.25]
        for q2 in 0...3 {
            let q4 = Float(q2 * 5)
            for scale in scales {
                for bias in biases {
                    let original = Float(q2) * scale + bias
                    let expanded = q4 * TwoBitExpand.adjustedScale(scale) + bias
                    #expect(abs(original - expanded) <= max(1e-6, abs(original) * 1e-5))
                }
            }
        }
    }

    /// The in-place rewrite: a compact blob becomes the 4-bit slot layout
    /// without a second buffer. Processed backwards, so the scale move lands
    /// before the weight expansion overwrites the bytes it came from.
    @Test func compactBlobRewritesIntoTheFourBitLayout() {
        let slot = UnsafeMutableRawPointer.allocate(byteCount: 16, alignment: 1)
        defer { slot.deallocate() }
        let bytes = slot.assumingMemoryBound(to: UInt8.self)
        for i in 0..<16 { bytes[i] = 0 }
        // weight: 4 bytes of 2-bit values; scale: 5.0, 10.0; bias: 1.0, -2.0
        let weight: [UInt8] = [0x1B, 0xE4, 0x4E, 0xB1]
        for (i, b) in weight.enumerated() { bytes[i] = b }
        func putBF16(_ value: Float, at index: Int) {
            let bits = value.bitPattern >> 16
            bytes[index] = UInt8(truncatingIfNeeded: bits)
            bytes[index + 1] = UInt8(truncatingIfNeeded: bits >> 8)
        }
        putBF16(5.0, at: 4); putBF16(10.0, at: 6)
        putBF16(1.0, at: 8); putBF16(-2.0, at: 10)

        let tensors = [
            ExpertBlobTensor(
                sourceOffset: 0, sourceSize: 4,
                targetOffset: 0, targetSize: 8, bits: 2,
                expands: true, rescales: false),
            ExpertBlobTensor(
                sourceOffset: 4, sourceSize: 4,
                targetOffset: 8, targetSize: 4, bits: 0,
                expands: false, rescales: true),
            ExpertBlobTensor(
                sourceOffset: 8, sourceSize: 4,
                targetOffset: 12, targetSize: 4, bits: 0,
                expands: false, rescales: false),
        ]
        TwoBitExpand.expandExpertInPlace(slot: slot, tensors: tensors)

        var expected = [UInt8](repeating: 0, count: 8)
        weight.withUnsafeBufferPointer { src in
            expected.withUnsafeMutableBufferPointer { dst in
                guard let source = src.baseAddress, let destination = dst.baseAddress
                else {
                    Issue.record("expected buffers have no base address")
                    return
                }
                TwoBitExpand.expand(
                    source: source, destination: destination,
                    byteCount: weight.count)
            }
        }
        for i in 0..<8 { #expect(bytes[i] == expected[i]) }
        func bf16(at index: Int) -> Float {
            let bits = UInt32(bytes[index]) | (UInt32(bytes[index + 1]) << 8)
            return Float(bitPattern: bits << 16)
        }
        #expect(abs(bf16(at: 8) - 1.0) < 1e-3)
        #expect(abs(bf16(at: 10) - 2.0) < 1e-3)
        #expect(abs(bf16(at: 12) - 1.0) < 1e-3)
        #expect(abs(bf16(at: 14) + 2.0) < 1e-3)
    }

    /// The layout the in-place expansion trusts: a map whose write region
    /// reaches into a lower tensor's unread source is refused at load, not
    /// executed as a silent clobber.
    @Suite struct ExpansionLayoutValidationTests {
        private func tensor(
            sourceOffset: Int, sourceSize: Int, targetOffset: Int,
            targetSize: Int, bits: Int = 2, expands: Bool = true,
            rescales: Bool = false
        ) -> ExpertBlobTensor {
            ExpertBlobTensor(
                sourceOffset: sourceOffset, sourceSize: sourceSize,
                targetOffset: targetOffset, targetSize: targetSize,
                bits: bits, expands: expands, rescales: rescales)
        }

        /// The detail of the refusal a layout produces, or `nil` when the map is
        /// accepted. Asserting `throws: ModelError.self` cannot tell one guard
        /// from another, which is how a test can pass for the wrong reason: the
        /// fixture this replaced was refused by the slot bounds long before the
        /// overlap check it named.
        private func refusalDetail(
            _ tensors: [ExpertBlobTensor], expertStride: Int
        ) -> String? {
            do {
                try TwoBitExpand.validateExpansionLayout(
                    tensors, expertStride: expertStride)
                return nil
            } catch let error as ModelError {
                guard case .indexCorrupt(let detail) = error else {
                    return "unexpected ModelError: \(error)"
                }
                return detail
            } catch {
                return "unexpected error: \(error)"
            }
        }

        @Test func theConverterLayoutIsValid() throws {
            // The shape `compactBlobRewritesIntoTheFourBitLayout` runs with:
            // a 2-bit weight that doubles, a rescaled scale block, a bias.
            let tensors = [
                tensor(sourceOffset: 0, sourceSize: 4, targetOffset: 0, targetSize: 8),
                tensor(
                    sourceOffset: 4, sourceSize: 4, targetOffset: 8,
                    targetSize: 4, bits: 0, expands: false, rescales: true),
                tensor(
                    sourceOffset: 8, sourceSize: 4, targetOffset: 12,
                    targetSize: 4, bits: 0, expands: false),
            ]
            #expect(throws: Never.self) {
                try TwoBitExpand.validateExpansionLayout(
                    tensors, expertStride: 16)
            }
        }

        @Test func aTargetBeforeItsSourceIsRefused() {
            let tensors = [
                tensor(sourceOffset: 4, sourceSize: 4, targetOffset: 0, targetSize: 8)
            ]
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    tensors, expertStride: 16)
            }
        }

        @Test func aWriteIntoAnUnreadSourceIsRefused() {
            // The weight at source 4 is processed first (highest source), and
            // its write 4..<12 lands inside the scale's unread source 0..<10.
            let tensors = [
                tensor(sourceOffset: 4, sourceSize: 4, targetOffset: 4, targetSize: 8),
                tensor(
                    sourceOffset: 0, sourceSize: 10, targetOffset: 16,
                    targetSize: 10, bits: 0, expands: false, rescales: true),
            ]
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    tensors, expertStride: 32)
            }
        }

        @Test func tensorsThatShareATargetRegionAreRefused() throws {
            // Both targets sit at or beyond their own source, so the bounds
            // guard accepts this map and only the overlap of the two write
            // regions (8..<16 and 12..<16) refuses it. Processing from the
            // highest source down writes 12..<16 first, then 8..<16 clobbers
            // those bytes.
            let tensors = [
                tensor(sourceOffset: 0, sourceSize: 4, targetOffset: 8, targetSize: 8),
                tensor(sourceOffset: 4, sourceSize: 2, targetOffset: 12, targetSize: 4),
            ]
            let detail = try #require(refusalDetail(tensors, expertStride: 32))
            #expect(detail.contains("share a target region"))
        }

        /// `expand3BitReversed` decodes `sourceSize / 3` whole groups, so a
        /// leftover byte is simply not expanded: seven source bytes are two
        /// groups of four target bytes, which the size guard alone accepts. The
        /// whole-group guard is the only thing that refuses it.
        @Test func aThreeBitTensorWithALeftoverGroupIsRefused() throws {
            let tensors = [
                tensor(
                    sourceOffset: 0, sourceSize: 7, targetOffset: 0,
                    targetSize: 8, bits: 3)
            ]
            let detail = try #require(refusalDetail(tensors, expertStride: 16))
            #expect(detail.contains("not a whole number of groups"))
        }

        @Test func sizesThatDoNotMatchTheTransformAreRefused() {
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [tensor(sourceOffset: 0, sourceSize: 4, targetOffset: 0, targetSize: 4)],
                    expertStride: 16)
            }
            // Four source bytes are not a whole number of 3-bit groups.
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [
                        tensor(
                            sourceOffset: 0, sourceSize: 4, targetOffset: 0, targetSize: 6, bits: 3)
                    ],
                    expertStride: 16)
            }
            // 3 source bytes are one group of four target bytes, not five.
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [
                        tensor(
                            sourceOffset: 0, sourceSize: 3, targetOffset: 0, targetSize: 5, bits: 3)
                    ],
                    expertStride: 16)
            }
            // Six source bytes are exactly two groups of four.
            #expect(throws: Never.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [
                        tensor(
                            sourceOffset: 0, sourceSize: 6, targetOffset: 0, targetSize: 8, bits: 3)
                    ],
                    expertStride: 16)
            }
            // Three bytes are exactly one group of four.
            #expect(throws: Never.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [
                        tensor(
                            sourceOffset: 0, sourceSize: 3, targetOffset: 0, targetSize: 4, bits: 3)
                    ],
                    expertStride: 16)
            }
        }

        /// A rescaled tensor is bf16 data: `rescaleBF16InPlace` divides
        /// `bytes / 2` halves, so an odd size would silently leave the last
        /// byte unscaled in the slot. The guard refuses it, and nothing pinned
        /// that refusal before this test.
        @Test func anOddRescaledTensorIsRefused() {
            let tensors = [
                tensor(
                    sourceOffset: 0, sourceSize: 3, targetOffset: 4,
                    targetSize: 3, bits: 0, expands: false, rescales: true)
            ]
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    tensors, expertStride: 8)
            }
        }

        /// The bounds guard also rejects a map that is not merely outside the
        /// slot but negative: a `-1` source offset would index the slot before
        /// its start and the `targetOffset >= sourceOffset` test alone would
        /// accept it.
        @Test func aNegativeSourceOffsetOrSizeIsRefused() {
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [
                        tensor(
                            sourceOffset: -1, sourceSize: 4, targetOffset: 0,
                            targetSize: 8)
                    ],
                    expertStride: 16)
            }
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [
                        tensor(
                            sourceOffset: 0, sourceSize: -1, targetOffset: 0,
                            targetSize: 8)
                    ],
                    expertStride: 16)
            }
        }

        @Test func aRegionOutsideTheSlotIsRefused() {
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [tensor(sourceOffset: 0, sourceSize: 8, targetOffset: 0, targetSize: 16)],
                    expertStride: 12)
            }
            #expect(throws: ModelError.self) {
                try TwoBitExpand.validateExpansionLayout(
                    [tensor(sourceOffset: 8, sourceSize: 8, targetOffset: 8, targetSize: 16)],
                    expertStride: 16)
            }
        }
    }
}
