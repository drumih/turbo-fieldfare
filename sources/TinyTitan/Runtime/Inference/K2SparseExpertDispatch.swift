import Foundation
import Metal
import TinyTitanCore

/// The routed-expert half of one sparse layer's chunk pass.
///
/// The layer's expert cache is a ring of slots that a fetch reuses, so a fetch
/// must never run ahead of a dispatch that still reads a slot: an earlier
/// version committed all sixteen expert buffers and awaited them at the end,
/// and the same chunk size produced different tokens from run to run at 8-bit.
///
/// Within that constraint the fetch is amortised: consecutive tokens are
/// grouped while the union of their selected ids still fits the ring, the union
/// is fetched once, and every token in the group dispatches against it in one
/// command buffer. The group grows until its union would overflow the ring: on
/// the real 4-bit install sixteen tokens' top-8 union fits the 64 slots, so a
/// 16-token chunk costs one fetch per layer instead of sixteen (495 against
/// 7,335 for a 163-token prompt), and experts that several tokens picked are
/// read from disk once. The 8-bit ring holds 16, so its groups are smaller.
final class K2SparseExpertDispatch {
    private let model: Model
    private let layer: Int
    private let hiddenSize: Int
    private let stages: [K2MoEStage]
    private let normed2: MTLBuffer
    private let moeOut: MTLBuffer
    /// How many routed experts the layer's streaming cache holds at once.
    private let cacheSlots: Int

    init(
        model: Model, layer: Int, hiddenSize: Int, stages: [K2MoEStage],
        normed2: MTLBuffer, moeOut: MTLBuffer
    ) throws {
        self.model = model
        self.layer = layer
        self.hiddenSize = hiddenSize
        self.stages = stages
        self.normed2 = normed2
        self.moeOut = moeOut
        switch model.streamingMode {
        case .pread(let slots):
            self.cacheSlots = max(1, slots)
        }
    }

    /// Read the chunk's router ids, then fetch and run the routed experts.
    ///
    /// The layer's expert cache is a ring of `expertCacheSlots` slots that a
    /// fetch reuses, so a fetch must never run ahead of a dispatch that still
    /// reads a slot: an earlier version committed all sixteen buffers and
    /// awaited them at the end, and the same chunk size produced different
    /// tokens from run to run at 8-bit.
    ///
    /// Within that constraint the fetch is amortised: consecutive tokens are
    /// grouped while the union of their selected ids still fits the ring, the
    /// union is fetched once, and every token in the group dispatches against
    /// it in one command buffer. The group grows until its union would overflow
    /// the ring: on the real 4-bit install sixteen tokens' top-8 union fits the
    /// 64 slots, so a 16-token chunk costs one fetch per layer instead of
    /// sixteen (495 against 7,335 for a 163-token prompt), and experts that
    /// several tokens picked are read from disk once. The 8-bit ring holds 16,
    /// so its groups are smaller.
    func encode(queue: MTLCommandQueue, tokenCount: Int) async throws {
        let d = hiddenSize
        let half = MemoryLayout<UInt16>.stride
        let idsPerToken = (0..<tokenCount).map {
            stages[$0].selectedExpertIDs()
        }
        var start = 0
        while start < tokenCount {
            var end = start
            var union = Set<Int>()
            while end < tokenCount {
                let next = union.union(idsPerToken[end])
                if next.count > cacheSlots { break }
                union = next
                end += 1
            }
            if end == start {
                // A single token selected more experts than the ring holds;
                // the token path has the same limit, so fetch its set and let
                // the streamer decide.
                union = Set(idsPerToken[start])
                end = start + 1
            }
            var views: [Int: K2MoEStage.ExpertWeights] = [:]
            let ordered = union.sorted()
            let blobs = try await model.fetchRoutedExperts(
                layer: layer, experts: ordered)
            for (index, id) in ordered.enumerated() {
                let entry = try model.packedExpertsLayout.expert(
                    layer: layer, expert: id)
                views[id] = try K2MoEStage.expertWeights(
                    blob: blobs[index], entry: entry)
            }
            guard let cb = queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            for token in start..<end {
                var experts: [K2MoEStage.ExpertWeights] = []
                for id in idsPerToken[token] {
                    guard let weights = views[id] else {
                        throw ModelError.internalInconsistency(
                            detail:
                                "K2 sparse batch lost expert \(id) from the fetched union")
                    }
                    experts.append(weights)
                }
                try stages[token].encodeFusedExperts(
                    commandBuffer: cb, hidden: normed2,
                    hiddenOffset: token * d * half, experts: experts,
                    shared: nil, out: moeOut, outOffset: token * d * half)
            }
            cb.commit()
            await cb.completed()
            if let error = cb.error {
                throw ModelError.internalInconsistency(
                    detail: "K2 sparse batch expert phase failed: \(error)")
            }
            start = end
        }
    }

}
