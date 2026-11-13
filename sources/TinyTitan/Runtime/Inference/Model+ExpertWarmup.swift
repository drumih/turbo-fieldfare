import Foundation

// The startup warm-up: load the experts the persisted routing profile ranks
// hottest into the slot cache before the first request needs them.
//
// Ported from K2 (`K2ProbeRuntime.warmExpertCache`, c8487f1d). The profile
// (`Model+ExpertProfile.swift`) already seeds the aging-LFU order, but the
// cache itself starts empty, so the first request after a restart misses on
// every expert of every layer. Warming loads the top N per layer through the
// same planner decode uses, so the slots they land in and the residency table
// they publish are exactly what decode would have produced. It is residency
// only: which experts a token computes is the router's choice and does not
// read the cache, so the output is the same with or without it.
//
// The planner is called with `recordsUse: false`: no token routed to these
// experts, so the LFU score stays the profile's seed and the hit/miss
// statistics a request reports stay that request's own.

/// What one warm-up did, for the log line and for tests.
public struct ExpertWarmupReport: Sendable, Equatable {
    /// Experts per layer the warm-up asked for.
    public let perLayer: Int
    /// Layers whose top experts were placed.
    public let layersWarmed: Int
    /// Layers the profile ranked (the warm-up's work list).
    public let layersRanked: Int
    /// Experts actually read from storage (already resident ones are free).
    public let expertsRead: Int
    public let bytesRead: UInt64
    public let seconds: Double
    /// True when a request (or teardown) stopped it before the last layer.
    public let stoppedEarly: Bool

    public init(
        perLayer: Int, layersWarmed: Int, layersRanked: Int, expertsRead: Int,
        bytesRead: UInt64, seconds: Double, stoppedEarly: Bool
    ) {
        self.perLayer = perLayer
        self.layersWarmed = layersWarmed
        self.layersRanked = layersRanked
        self.expertsRead = expertsRead
        self.bytesRead = bytesRead
        self.seconds = seconds
        self.stoppedEarly = stoppedEarly
    }

    /// `TinyTitan warm experts: N/layer, X GB read in Y s`, plus where it
    /// stopped when a request cut it short.
    public var logLine: String {
        let gigabytes = Double(bytesRead) / 1_000_000_000
        var line = String(
            format: "TinyTitan warm experts: %d/layer, %.2f GB read in %.2f s",
            perLayer, gigabytes, seconds)
        if stoppedEarly {
            line += " (stopped by a request after \(layersWarmed)/\(layersRanked) layers)"
        }
        return line
    }
}

/// The `TINYTITAN_WARM_EXPERTS` policy: how many experts per layer to warm.
public enum ExpertWarmup {
    public static let environmentKey = "TINYTITAN_WARM_EXPERTS"

    /// Slots left empty by the default. The routing profile's tail is flat
    /// (on the 4-bit Qwen3.8 install ranks 38-48 each carry 0.53-0.61 % of a
    /// layer's routings), so its order at the boundary is mostly noise; the
    /// last few slots are better filled by the session's own first misses,
    /// which then land in an empty slot instead of evicting a warmed expert.
    /// One sixteenth of the cache, at least one slot: 3 of 48 at `--ram 10`.
    public static func headroom(slotCount: Int) -> Int {
        max(1, slotCount / 16)
    }

    /// The default: the whole cache minus `headroom`. 45 of 48 slots at
    /// `--ram 10` on 4-bit Qwen3.8, about 6 GB of background reads.
    public static func defaultPerLayer(slotCount: Int) -> Int {
        max(0, slotCount - headroom(slotCount: slotCount))
    }

    /// The operator's value: `nil` when unset or not an integer (the default
    /// applies), otherwise the number, where 0 or less disables warming.
    public static func requestedPerLayer(environment: [String: String]) -> Int? {
        guard let raw = environment[environmentKey] else { return nil }
        guard let value = Int(raw.trimmingCharacters(in: .whitespaces)) else {
            FileHandle.standardError.write(
                Data(
                    ("[cache] ignoring \(environmentKey) '\(raw)'; "
                        + "expected an integer\n").utf8))
            return nil
        }
        return value
    }

    /// The count in force, capped by the slot count (a warm-up cannot hold
    /// more than the cache) and the layer's experts. 0 means off.
    ///
    /// Off unless requested. Measured on 4-bit Qwen3.8 at `--ram 10` with the
    /// install's profile (2026-11-13, three Swift/git prompts): warming 45 of
    /// 48 slots moved the hit rate by +0.009, -0.011 and -0.035 -- the profile
    /// is diffuse (its top 48 cover about 46 % of a layer's routings), so the
    /// warmed experts outlast the prompt's own working set. `defaultPerLayer`
    /// stays as the suggested value for an operator who opts in.
    public static func perLayer(
        requested: Int?, slotCount: Int, expertsPerLayer: Int
    ) -> Int {
        let wanted = requested ?? 0
        guard wanted > 0 else { return 0 }
        return max(0, min(wanted, slotCount, expertsPerLayer))
    }
}

/// A warm-up that can run on its own task: the model it fills, and nothing
/// else. It holds the model rather than whoever started it, so the owner can be
/// released (and cancel the task) while one layer's reads finish.
///
/// unchecked-invariant: `model` is immutable here and the warm-up reaches it
/// only through the routed-expert streamer API, whose cache state each
/// streamer guards with its own lock (decode calls the same API from the I/O
/// threads). The caller keeps requests out of the planner until the warm-up
/// has stopped (`ServerModelSession.stopExpertWarmup`).
public final class ExpertWarmupJob: @unchecked Sendable {
    private let model: Model

    public init(model: Model) {
        self.model = model
    }

    public func run(perLayer: Int) async -> ExpertWarmupReport? {
        await model.warmRoutedExpertCache(perLayer: perLayer)
    }
}

extension Model {
    /// The profile's hottest experts per layer, hottest first, at most
    /// `perLayer` each. Layers with no routed experts or no history are
    /// absent; experts never routed are never warmed.
    public func expertWarmupRanking(perLayer: Int) -> [(layer: Int, experts: [Int])] {
        guard perLayer > 0, let profile = expertUsageProfile, !profile.isEmpty
        else { return [] }
        let layout = packedExpertsLayout
        return (0..<min(profile.layers, layout.numLayers)).compactMap { layer in
            let available = min(profile.expertsPerLayer, layout.layers[layer].experts.count)
            let counts: [UInt32] = (0..<available).map {
                profile.count(layer: layer, expert: $0)
            }
            // Hottest first; equal counts in expert order, so the ranking
            // (and with it the slot each expert lands in) is deterministic.
            let routed: [Int] = counts.indices.filter { counts[$0] > 0 }
            let ordered: [Int] = routed.sorted { lhs, rhs in
                counts[lhs] != counts[rhs] ? counts[lhs] > counts[rhs] : lhs < rhs
            }
            let ranked = Array(ordered.prefix(perLayer))
            return ranked.isEmpty ? nil : (layer, ranked)
        }
    }

    /// Place `experts` on `layer` through the decode planner and wait for the
    /// reads. Returns how many were read (resident ones cost nothing).
    ///
    /// The plan is pinned until its reads complete, exactly as a decode plan
    /// is, and the completion is awaited even if the calling task is
    /// cancelled: a slot left `loading` must never be visible to a request.
    public func warmRoutedExperts(layer: Int, experts: [Int]) async throws -> Int {
        let streamer = try openStreamer(for: layer)
        guard let cachePlan = streamer.planWarmupExperts(experts) else {
            return 0
        }
        let plan = RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
        let lease = try pinRoutedExperts(for: plan)
        defer { lease.release() }
        _ = try await beginFetchRoutedExperts(plan: plan).completion()
        return plan.misses.count
    }

    /// Warm every ranked layer in turn, stopping between layers once the
    /// calling task is cancelled. Nil when there is nothing to warm (no slot
    /// cache, no profile, or `perLayer` 0). A layer that fails to read is
    /// skipped: warming is best effort and the request reads on demand.
    public func warmRoutedExpertCache(perLayer: Int) async -> ExpertWarmupReport? {
        guard let slotCount = routedExpertCacheSlotCount(), perLayer > 0 else {
            return nil
        }
        let capped = min(perLayer, slotCount)
        let ranking = expertWarmupRanking(perLayer: capped)
        guard !ranking.isEmpty else { return nil }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var warmed = 0
        var read = 0
        var stopped = false
        for (layer, experts) in ranking {
            if Task.isCancelled {
                stopped = true
                break
            }
            guard let count = try? await warmRoutedExperts(layer: layer, experts: experts)
            else { continue }
            warmed += 1
            read += count
        }
        let elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
        return ExpertWarmupReport(
            perLayer: capped, layersWarmed: warmed, layersRanked: ranking.count,
            expertsRead: read,
            bytesRead: UInt64(read) * packedExpertsLayout.expertStride,
            seconds: Double(elapsed) / 1e9, stoppedEarly: stopped)
    }
}
