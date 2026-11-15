import Foundation
import TinyTitanCore

/// Which links get more than the profile's one speculative read.
///
/// `TINYTITAN_PREFETCH_LAYER_DEPTH="3=2,7=2"` deepens the link whose *source*
/// is layer 3 and the one whose source is layer 7; every other link keeps the
/// profile's depth. Links are keyed by their source layer because that is how
/// the trace screen keys them (`docs/qwen38-decode-12gb-measurements.md` §5al),
/// so a screened layer list transcribes without a shift; the ring is told the
/// layer it is reading *for*, so the table comes out keyed by `source + 1`.
///
/// Unset, the table is empty and the ring is one read a layer everywhere it
/// ships. The profile is an experiment switch, not a shipped default: §5al's
/// offline screen ranks the links but two price models disagree on the sign, so
/// only a live paired run can settle it.
enum PrefetchDepthProfile {
    static let environmentKey = "TINYTITAN_PREFETCH_LAYER_DEPTH"

    /// Target layer -> reads to issue for it. Throws on any item the ring could
    /// not apply, because an arm's layer list is transcribed from a document
    /// and a typo must stop the run instead of quietly measuring a different arm.
    static func parse(raw: String?, numLayers: Int, topKExperts: Int) throws -> [Int: Int] {
        guard let raw, !raw.isEmpty else { return [:] }
        var table: [Int: Int] = [:]
        // Empty subsequences are kept so a trailing or doubled comma is refused.
        for item in raw.split(separator: ",", omittingEmptySubsequences: false) {
            let parts = item.split(separator: "=", omittingEmptySubsequences: false)
            guard
                parts.count == 2,
                let source = Int(parts[0]),
                let depth = Int(parts[1])
            else {
                throw ModelError.internalInconsistency(
                    detail: "\(environmentKey) item \"\(item)\" must be layer=depth")
            }
            // A link needs a target, so the last layer cannot be a source.
            guard (0..<max(0, numLayers - 1)).contains(source) else {
                throw ModelError.internalInconsistency(
                    detail: "\(environmentKey) layer \(source) has no next layer "
                        + "of \(numLayers) and no link to deepen")
            }
            // Not `(1...topKExperts)`: a dense install has `topKExperts == 0`,
            // and `(1...0)` traps instead of refusing (BuildCore's own trap).
            guard depth >= 1, depth <= topKExperts else {
                throw ModelError.internalInconsistency(
                    detail: "\(environmentKey) depth \(depth) for layer \(source) "
                        + "must be 1...\(topKExperts)")
            }
            guard table[source + 1] == nil else {
                throw ModelError.internalInconsistency(
                    detail: "\(environmentKey) names layer \(source) twice")
            }
            table[source + 1] = depth
        }
        return table
    }
}
