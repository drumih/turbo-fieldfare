import Testing

@testable import TinyTitan
import TinyTitanCore

/// The multirow bf16 GEMV's dispatch, pinned without a Metal device.
///
/// The kernel's arithmetic is covered by the golden baseline -- the ornith-4
/// prompt prefills through this path and the stored text matches -- but a wrong
/// grid still produces *an* answer, so the grid needs its own tests. These are
/// the numbers the shader indexes with: row blocks of eight simdgroups, one
/// threadgroup per token, and the 64-column alignment it steps by.
@Suite struct BF16PrefillDispatchTests {
    @Test func theGridIsRowBlocksByTokens() throws {
        let grid = try #require(BF16PrefillDispatch.grid(rows: 4096, tokens: 1500))
        // 4096 rows / 8 per threadgroup = 512 blocks, one per token.
        #expect(grid.rowBlocks == 512)
        #expect(grid.tokens == 1500)
        #expect(grid.threadsPerThreadgroup == 256)
    }

    /// A ragged tail still gets a block: the shader returns for rows past the
    /// end, so rounding down would leave the last rows unwritten.
    @Test func aRaggedRowCountRoundsUp() throws {
        let grid = try #require(BF16PrefillDispatch.grid(rows: 513, tokens: 2))
        #expect(grid.rowBlocks == 65)  // ceil(513 / 8)
        #expect(grid.tokens == 2)
    }

    @Test func oneRowIsOneBlock() throws {
        let grid = try #require(BF16PrefillDispatch.grid(rows: 1, tokens: 7))
        #expect(grid.rowBlocks == 1)
        #expect(grid.tokens == 7)
    }

    /// Zero work is refused rather than encoded: a zero-sized grid would be a
    /// command with nothing in it, and a shape that reached here is a bug.
    @Test func noGridForAnEmptyDimension() {
        #expect(BF16PrefillDispatch.grid(rows: 0, tokens: 4) == nil)
        #expect(BF16PrefillDispatch.grid(rows: 4, tokens: 0) == nil)
        #expect(BF16PrefillDispatch.grid(rows: 0, tokens: 0) == nil)
    }

    /// The shader reads two adjacent columns per lane and steps by 64.
    @Test func columnsMustBeAPositiveMultipleOf64() {
        for columns in [64, 128, 2048, 4096] {
            #expect(BF16PrefillDispatch.isEncodable(columns: columns), "\(columns)")
        }
        for columns in [0, -64, 65, 2047] {
            #expect(!BF16PrefillDispatch.isEncodable(columns: columns), "\(columns)")
        }
    }

    /// Decode keeps the single-row GEMV; a chunk takes the multirow one, and
    /// the switch exists only to compare the two on one prompt.
    @Test func onlyAChunkTakesTheMultirowPath() {
        #expect(!BF16PrefillDispatch.usesMultiRow(tokenCount: 1, singleRowForced: false))
        #expect(BF16PrefillDispatch.usesMultiRow(tokenCount: 2, singleRowForced: false))
        #expect(BF16PrefillDispatch.usesMultiRow(tokenCount: 1500, singleRowForced: false))
        #expect(!BF16PrefillDispatch.usesMultiRow(tokenCount: 1500, singleRowForced: true))
        #expect(!BF16PrefillDispatch.usesMultiRow(tokenCount: 0, singleRowForced: false))
    }
}
