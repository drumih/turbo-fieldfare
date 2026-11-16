import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// `PLEBlock`'s host-side state: the scratch geometry the load path sizes from
/// the architecture, and the convolution window's reset and rewind.
///
/// The block is constructed only when `cfg.ple.enabled` -- a Qwen3.8-Flash-Next
/// install -- so its arithmetic is pinned at the kernel level by
/// `PLEBlockTests` and, against a real model, stage by stage by the activation
/// dump. What needs no model is the memory bookkeeping: buffer sizes, the clear
/// between completions, and the rewind a speculative pass owes the window. All
/// three are host-visible on shared buffers, and all three are silent when
/// wrong -- an undersized buffer is a host heap overwrite rather than a GPU
/// fault, and a window advanced two rows for one accepted token stays
/// desynchronized for the rest of the generation.
@Suite struct PLEBlockStateTests {

    /// dim 8 x streams 2 = 16 columns per row, embedDim 16, K 4, dil 3,
    /// so `history` is 9 rows and a row is 32 bytes.
    private static func makeBlock(maxRows: Int = 3) throws -> PLEBlock {
        let ctx = try MetalContext()
        return try PLEBlock(
            context: ctx, dim: 8, streams: 2, embedDim: 16,
            kernelSize: 4, dilation: 3, maxRows: maxRows)
    }

    private static func rowBytes(_ ple: PLEBlock) -> Int {
        ple.dim * ple.streams * MemoryLayout<Float16>.stride
    }

    private static func bytes(_ buffer: MTLBuffer) -> [UInt8] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(
                    to: UInt8.self, capacity: buffer.length),
                count: buffer.length))
    }

    private static func fill(_ buffer: MTLBuffer, _ byte: Int) {
        memset(buffer.contents(), Int32(byte), buffer.length)
    }

    /// Write each row of `buffer` with a byte naming its own row index, so any
    /// copy of it says where it came from.
    private static func labelRows(_ buffer: MTLBuffer, width: Int) {
        let base = buffer.contents()
        for row in 0..<(buffer.length / width) {
            memset(base.advanced(by: row * width), Int32(1 + row), width)
        }
    }

    private static func rows(_ buffer: MTLBuffer, width: Int) -> [[UInt8]] {
        let all = Self.bytes(buffer)
        return stride(from: 0, to: all.count, by: width).map {
            Array(all[$0..<$0 + width])
        }
    }

    private static func row(_ byte: Int, _ width: Int) -> [UInt8] {
        [UInt8](repeating: UInt8(byte), count: width)
    }

    /// Every buffer is sized from the architecture times the row budget. The
    /// gather width comes from the n-gram sidecar and the embedding buffer from
    /// `embedDim`, so a disagreement overruns the host allocation; this is the
    /// sizing that has to stay exactly as the load path describes it.
    @Test func scratchIsSizedFromTheGeometryAndRowBudget() throws {
        let ple = try Self.makeBlock(maxRows: 3)
        let f16 = MemoryLayout<Float16>.stride
        let wide = ple.dim * ple.streams
        #expect(ple.history == 9)
        #expect(ple.embedding.length == ple.embedDim * 3 * f16)
        #expect(ple.keyBuf.length == wide * 3 * f16)
        #expect(ple.valueBuf.length == ple.dim * 3 * f16)
        #expect(ple.keyNormed.length == wide * 3 * f16)
        #expect(ple.queryNormed.length == wide * 3 * f16)
        #expect(ple.scoreBuf.length == ple.streams * 3 * f16)
        #expect(ple.gateBuf.length == ple.streams * 3 * f16)
        #expect(ple.gatedBuf.length == wide * 3 * f16)
        #expect(ple.convOut.length == wide * 3 * f16)
        #expect(ple.xpad.count == 2)
        for buffer in ple.xpad {
            // `history` rows of carried state plus the rows of the pass.
            #expect(buffer.length == (ple.history + 3) * wide * f16)
        }
    }

    @Test func constructionClearsTheWindow() throws {
        let ple = try Self.makeBlock()
        for buffer in ple.xpad {
            #expect(Self.bytes(buffer).allSatisfy { $0 == 0 })
        }
    }

    /// A state left over from a previous prompt leaks that prompt's n-grams
    /// into the first tokens of the next one, so the clear has to cover both
    /// ping-pong buffers and their full length, not just the carried rows.
    @Test func resetStateClearsBothWindowBuffersCompletely() throws {
        let ple = try Self.makeBlock()
        Self.fill(ple.xpad[0], 0xA5)
        Self.fill(ple.xpad[1], 0x5A)
        ple.resetState()
        for buffer in ple.xpad {
            let cleared = Self.bytes(buffer)
            #expect(cleared.allSatisfy { $0 == 0 })
            #expect(cleared.count == buffer.length)
        }
        ple.resetState()
        #expect(Self.bytes(ple.xpad[0]).allSatisfy { $0 == 0 })
    }

    /// The rewind is exact and costs one copy: the buffer the discarded pass
    /// read from is still intact, so accepting fewer rows than it ran is a
    /// different source offset into the pair's partner, for `history` rows.
    @Test(arguments: [(0, 3), (1, 3), (2, 3)])
    func rewindCopiesThePartnersHistoryRowsFromTheAcceptedCount(
        accepted: Int, pass: Int
    ) throws {
        let ple = try Self.makeBlock(maxRows: pass)
        let width = Self.rowBytes(ple)
        #expect(width == 32)
        let partner = ple.xpad[1]
        Self.labelRows(partner, width: width)
        let partnerBefore = Self.bytes(partner)
        ple.rewindWindow(acceptedRows: accepted, passRows: pass)
        let window = Self.rows(ple.xpad[0], width: width)
        for i in 0..<ple.history {
            #expect(
                window[i] == Self.row(1 + accepted + i, width),
                "window row \(i) should be partner row \(accepted + i)")
        }
        // Only the carried history moves: the rows the next pass writes into
        // stay clear, which is what makes an over-long copy visible.
        for i in ple.history..<window.count {
            #expect(
                window[i].allSatisfy { $0 == 0 },
                "row \(i) was overwritten")
        }
        #expect(Self.bytes(partner) == partnerBefore, "the rewind wrote its source")
    }

    /// Accepting every row the pass ran is not a rewind: the window is already
    /// where the next token needs it, and copying would shift live state.
    @Test func aFullPassRewindsNothing() throws {
        let ple = try Self.makeBlock(maxRows: 2)
        Self.fill(ple.xpad[0], 0x77)
        Self.fill(ple.xpad[1], 0x22)
        ple.rewindWindow(acceptedRows: 2, passRows: 2)
        #expect(Self.bytes(ple.xpad[0]).allSatisfy { $0 == 0x77 })
        #expect(Self.bytes(ple.xpad[1]).allSatisfy { $0 == 0x22 })
    }
}
