import Foundation
import Metal

/// Counts for the next-layer early read, summed over decode forwards.
public struct EarlyExpertReadStats: Sendable, Equatable {
    /// Decode forwards that ran with the early read.
    public var forwards: UInt64 = 0
    /// Routed experts those forwards' cache plans missed, so each had to come
    /// from storage. An early read that is used supplies one of them.
    public var misses: UInt64 = 0
    /// Early reads issued.
    public var reads: UInt64 = 0
    /// Early reads whose expert the next layer then missed. `reads - used`
    /// were wasted.
    public var used: UInt64 = 0

    public init() {}

    /// The share of early reads that were used.
    public var precision: Double? {
        reads > 0 ? Double(used) / Double(reads) : nil
    }
}

/// Decode control: read the next layer's likeliest routed experts while
/// storage would otherwise sit idle. On by default;
/// `RuntimeConfiguration.earlyExpertRead` turns it off.
///
/// Per layer L < last: cb1(L) also scores layer L+1's experts over L's router
/// input, with the fitted guess the runtime ships for this model
/// (`NextLayerGuessWeights`) or else with L+1's own router. After L's own reads
/// finish, the best-scoring experts that L+1's cache does not hold are read,
/// best first, into staging buffers outside every slot. Each one L+1's plan
/// then misses has its staging buffer swapped into the slot the plan assigned
/// to it and its read skipped; the rest are wasted. The cache plans exactly as
/// it would without this, so hit rate and generated text are unchanged; only
/// where some misses' bytes come from differs.
///
/// Two reads per layer by default; `TURBO_FIELDFARE_EARLY_EXPERT_READS` (1–8)
/// sets another count, each read with its own 3.36 MB staging buffer.
final class NextLayerExpertPrefetcher: @unchecked Sendable {
    static let readsEnvironmentKey = "TURBO_FIELDFARE_EARLY_EXPERT_READS"
    static let defaultReadsPerLayer = 2
    static let maxReadsPerLayer = 8

    /// Written by the read before `done` is left, read only after waiting.
    private final class Outcome: @unchecked Sendable {
        var error: Error?
    }

    private struct Pending {
        let expert: Int
        let staging: Int
        let done: DispatchGroup
        let outcome: Outcome
        var adopted = false
    }

    private let staging: [ExpertStagingBuffer]
    private let queue = DispatchQueue(label: "turbo-fieldfare.early-expert-read",
                                      qos: .userInitiated)
    private var pendingLayer: Int?
    private var pending: [Pending] = []
    private(set) var stats = EarlyExpertReadStats()
    /// The fitted guess, or nil to score with the next layer's own router.
    let guessWeights: NextLayerGuessWeights?

    /// Early reads per layer.
    var readsPerLayer: Int { staging.count }

    init(model: Model, guess: RuntimeEarlyExpertRead, device: MTLDevice,
         environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        let count = Self.readsPerLayer(environment: environment)
        let streamer = try model.routedExpertStreamer(layer: 0)
        staging = try (0..<count).map { _ in try streamer.makeStagingBuffer() }
        let config = model.config
        guessWeights = guess == .fitted
            ? try NextLayerGuessWeights.bundled(
                sourceSnapshotHash: model.sourceSnapshotHash, numLayers: config.numLayers,
                numExperts: config.numExperts, hiddenSize: config.hiddenSize, device: device)
            : nil
    }

    /// Which guess scores the next layer, `fitted` or `router`.
    var guess: RuntimeEarlyExpertRead { guessWeights == nil ? .router : .fitted }

    deinit {
        // A read may still be filling a staging buffer, which is freed with
        // this object.
        for read in pending { read.done.wait() }
    }

    /// `TURBO_FIELDFARE_EARLY_EXPERT_READS` when it is a whole number from 1
    /// to `maxReadsPerLayer`, otherwise `defaultReadsPerLayer`.
    static func readsPerLayer(environment: [String: String]) -> Int {
        guard let value = environment[readsEnvironmentKey].flatMap({ Int($0) }),
              (1...maxReadsPerLayer).contains(value)
        else { return defaultReadsPerLayer }
        return value
    }

    /// Up to `limit` highest-scoring experts not in `resident`, best first;
    /// ties go to the lower ID.
    static func choose(scores: UnsafePointer<Float>, count: Int,
                       resident: Set<Int>, limit: Int = 1) -> [Int] {
        var best: [Int] = []
        for expert in 0..<count where !resident.contains(expert) {
            let position = best.firstIndex { scores[expert] > scores[$0] } ?? best.count
            if position < limit {
                best.insert(expert, at: position)
                if best.count > limit { best.removeLast() }
            }
        }
        return best
    }

    /// Drops whatever an earlier forward left pending, so an early read never
    /// crosses a forward. Prefill may change the cache in between.
    func beginForward() throws {
        try waitForPending()
        pending = []
        pendingLayer = nil
        stats.forwards += 1
    }

    /// Starts reading `experts` of `layer`, in order, one into each staging
    /// buffer. Waits first for the previous early reads, whose buffers may
    /// still be filling.
    func issue(layer: Int, experts: [Int], streamer: PreadExpertStreamer) throws {
        precondition(experts.count <= staging.count, "more early reads than staging buffers")
        try waitForPending()
        pending = []
        pendingLayer = layer
        for (index, expert) in experts.enumerated() {
            let done = DispatchGroup()
            let outcome = Outcome()
            // Captured now: an adoption swaps `staging[index].pointer`, but this
            // read's destination must stay the memory it started into.
            nonisolated(unsafe) let destination = staging[index].pointer
            done.enter()
            queue.async {
                do {
                    try streamer.readEarly(expert: expert, into: destination)
                } catch {
                    outcome.error = error
                }
                done.leave()
            }
            pending.append(Pending(expert: expert, staging: index, done: done, outcome: outcome))
        }
        stats.reads += UInt64(experts.count)
    }

    /// Counts `plan`'s misses. For each pending expert the plan misses, swaps
    /// its staging buffer into that miss's slot. Returns those misses' indices,
    /// which the plan's execution must skip and wait for with
    /// `waitForAdopted`. Call before the plan's buffers are bound.
    func adopt(into plan: RoutedExpertFetchPlan, streamer: PreadExpertStreamer) -> Set<Int> {
        stats.misses += UInt64(plan.misses.count)
        guard pendingLayer == plan.layer else { return [] }
        var adopted: Set<Int> = []
        for i in pending.indices where !pending[i].adopted {
            guard let index = plan.misses.first(where: { plan.experts[$0] == pending[i].expert })
            else { continue }
            streamer.adoptStagedExpert(staging[pending[i].staging], plan: plan.cachePlan,
                                       index: index)
            pending[i].adopted = true
            adopted.insert(index)
        }
        stats.used += UInt64(adopted.count)
        return adopted
    }

    /// Blocks until every adopted early read has finished, and rethrows the
    /// first error.
    func waitForAdopted() throws {
        try wait(for: pending.filter(\.adopted))
    }

    /// Blocks until every pending early read has finished, and rethrows the
    /// first error.
    func waitForPending() throws {
        try wait(for: pending)
    }

    private func wait(for reads: [Pending]) throws {
        for read in reads { read.done.wait() }
        if let error = reads.lazy.compactMap({ $0.outcome.error }).first { throw error }
    }
}
