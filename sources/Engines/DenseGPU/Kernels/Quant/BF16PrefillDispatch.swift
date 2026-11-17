import Foundation

/// The grid the multirow bf16 GEMV dispatches, and when it is allowed.
///
/// The kernel's arithmetic is one row of a dot product, already pinned by the
/// golden baseline (whose prompt prefill goes through this path). What is not
/// covered there -- because a wrong grid still produces some answer, just not
/// this prompt's -- is the *dispatch*: how many row blocks, one threadgroup per
/// token, and the column alignment the shader requires. That is pure integer
/// arithmetic, so it is testable without a Metal device, and it is here rather
/// than inside the wrapper for exactly that reason.
package enum BF16PrefillDispatch {
    /// Rows one threadgroup covers: eight simdgroups of 32 lanes each.
    package static let rowsPerThreadgroup = 8
    package static let threadsPerThreadgroup = 32 * rowsPerThreadgroup

    package struct Grid: Equatable, Sendable {
        package let rowBlocks: Int
        package let tokens: Int
        package let threadsPerThreadgroup: Int
    }

    /// The dispatch for `rows` output rows over `tokens` inputs.
    ///
    /// Zero rows or zero tokens dispatch nothing; a caller that got here with
    /// either has a shape bug, and a zero-sized grid would encode a command
    /// with no work rather than say so.
    package static func grid(rows: Int, tokens: Int) -> Grid? {
        guard rows > 0, tokens > 0 else { return nil }
        return Grid(
            rowBlocks: (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            tokens: tokens,
            threadsPerThreadgroup: threadsPerThreadgroup)
    }

    /// The shader reads two adjacent columns per lane and steps by 64, so a
    /// column count that is not a multiple of 64 would read past the last row.
    package static func isEncodable(columns: Int) -> Bool {
        columns > 0 && columns % 64 == 0
    }

    /// Whether a chunk takes the multirow path.
    ///
    /// One token is the decode GEMV's job and is cheaper there; the environment
    /// switch exists so the two paths can be compared on one prompt, which is
    /// how the kernel's output was shown to be byte-identical to the loop it
    /// replaced.
    package static func usesMultiRow(
        tokenCount: Int,
        singleRowForced: Bool
    ) -> Bool {
        tokenCount > 1 && !singleRowForced
    }
}
