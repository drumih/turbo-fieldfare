import Foundation
import Testing

@testable import TinyTitan
import TinyTitanCore

/// A repack is a byte copy, so a dense `.ssdai` install must be *exactly* the
/// snapshot it came from. This is the gate that says so.
///
/// It is opt-in, and deliberately not part of the default suite: it loads two
/// real 2B-9B models and runs inference, which the unit tests otherwise never
/// do (`AGENTS.md`). Enable it with `TINYTITAN_DENSE_EQUIV=1` and point
/// `TINYTITAN_DENSE_EQUIV_PAIRS` at `<snapshot>:<install>,...`; `tools/repack_dense.sh`
/// does both for you.
///
/// The reason this is a permanent test rather than a one-off probe: the failure
/// it catches is *silent*. A wrong width, or a wrong per-tensor width, changes
/// the stride of the dequantize, divides evenly, passes every shape check, and
/// answers fluently and wrongly. Nothing downstream of the loader can see that.
@Suite("Dense .ssdai equivalence")
struct DenseSSDAIEquivalenceTests {

    /// `<snapshot dir>:<ssdai dir>` pairs, comma separated.
    private static var pairs: [(snapshot: URL, install: URL)] {
        guard let raw = ProcessInfo.processInfo.environment["TINYTITAN_DENSE_EQUIV_PAIRS"] else {
            return []
        }
        return raw.split(separator: ",").compactMap { pair in
            let parts = pair.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { return nil }
            return (URL(fileURLWithPath: parts[0]), URL(fileURLWithPath: parts[1]))
        }
    }

    /// Logits, not generated text. Greedy text can agree while two argmaxes
    /// differ by 1e-9, and the point here is that the two paths are the same
    /// arithmetic on the same bytes, not merely that they agree today.
    private static let tokens = [9707, 11, 1879, 13, 3838, 374]

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_EQUIV"] != nil))
    func denseInstallMatchesItsSnapshot() throws {
        let pairs = Self.pairs
        try #require(
            !pairs.isEmpty,
            "set TINYTITAN_DENSE_EQUIV_PAIRS to <snapshot>:<install>[,<snapshot>:<install>]")

        for pair in pairs {
            try #require(
                FileManager.default.fileExists(atPath: pair.snapshot.path),
                "snapshot is missing: \(pair.snapshot.path)")
            try #require(
                FileManager.default.fileExists(atPath: pair.install.path),
                "install is missing: \(pair.install.path)")

            let snapshot = try CPUQwen35(
                snapshot: try AffineSnapshot(directory: pair.snapshot), threads: 4)
            let install = try CPUQwen35(
                snapshot: try AffineSnapshot(ssdai: pair.install), threads: 4)

            var worst: Float = 0
            for token in Self.tokens {
                let fromSnapshot = try snapshot.step(token: token)
                let fromInstall = try install.step(token: token)
                #expect(
                    fromSnapshot.count == fromInstall.count,
                    "\(pair.install.lastPathComponent): vocabulary width changed")
                let difference =
                    zip(fromSnapshot, fromInstall)
                    .map { abs($0 - $1) }.max() ?? 0
                worst = max(worst, difference)
            }
            #expect(
                worst == 0,
                Comment(
                    rawValue:
                        "\(pair.install.lastPathComponent) is not byte-equivalent to "
                        + "\(pair.snapshot.lastPathComponent): largest logit difference "
                        + "\(worst). A repack is a byte copy, so this is a wrong "
                        + "width or a wrong offset, not rounding."))
        }
    }
}
