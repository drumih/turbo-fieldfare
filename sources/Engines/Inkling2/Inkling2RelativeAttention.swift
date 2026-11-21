import Foundation

/// The relative-distance part of Inkling's attention: a per-head query vector
/// `r` times a learned `[d_rel, extent]` table, with the causal/sliding band.
public enum Inkling2RelativeAttention {
    public static let masked: Float = -Float.infinity

    /// Returns one additive score matrix per query position. `keyCount` is the
    /// full KV length; `newTokens` is how many of the final positions are the
    /// current queries (the rest are cached prefix).
    public static func biases(
        r: [[Float]],
        relProjection: [Float],
        relativeDimension: Int,
        extent: Int,
        sliding: Int,
        keyCount: Int,
        newTokens: Int
    ) -> [[Float]] {
        precondition(r.allSatisfy { $0.count == relativeDimension })
        precondition(relProjection.count == relativeDimension * extent)
        let queryCount = r.count
        let prefix = keyCount - newTokens
        precondition(prefix >= 0)
        var result: [[Float]] = []
        result.reserveCapacity(queryCount)
        for query in 0..<queryCount {
            let absolute = prefix + query
            var row = [Float](repeating: masked, count: keyCount)
            for key in 0..<keyCount {
                let distance = absolute - key
                guard distance >= 0 else { continue }
                if sliding > 0, distance >= sliding { continue }
                guard distance < extent else { continue }
                var dot: Float = 0
                for dimension in 0..<relativeDimension {
                    dot += r[query][dimension] * relProjection[dimension * extent + distance]
                }
                row[key] = dot
            }
            result.append(row)
        }
        return result
    }
}
