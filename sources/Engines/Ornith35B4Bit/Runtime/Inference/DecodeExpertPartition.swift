/// Allocation-free partitioning of routed top-k positions for decode.
///
/// The values are positions within the router's top-k result, not global expert
/// IDs. Reusing caller-owned arrays matters because this runs once per MoE layer
/// per generated token.
enum DecodeExpertPartition {
    static func populate(
        topK: Int,
        missIndices: [Int],
        hits: inout [UInt32],
        misses: inout [UInt32]
    ) {
        hits.removeAll(keepingCapacity: true)
        misses.removeAll(keepingCapacity: true)
        hits.reserveCapacity(topK)
        misses.reserveCapacity(missIndices.count)

        for index in 0..<topK {
            if missIndices.contains(index) {
                misses.append(UInt32(index))
            } else {
                hits.append(UInt32(index))
            }
        }
    }
}
