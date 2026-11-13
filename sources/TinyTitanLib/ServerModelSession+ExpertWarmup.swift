import Foundation
import TinyTitan

// The session's side of the startup expert warm-up (`Model+ExpertWarmup`).
//
// `load` starts it in the background and returns, so a server is ready to
// accept requests at once. It must never plan the slot cache at the same time
// as a request: a warm-up slot still `loading` is neither a hit nor evictable
// for a decode plan, so a decode needing that expert would place a second copy
// and the per-expert residency table would end up naming two slots. A request
// therefore stops it first (`stopExpertWarmup`): the warm-up finishes the layer
// whose reads are in flight -- one layer's batch, ~0.1 s on the 4-bit Qwen3.8
// install -- and does not start another. What it loaded stays loaded.
//
// A one-shot caller with nothing else to do first, like the CLI, can instead
// wait for the whole warm-up (`awaitExpertWarmup`).
extension ServerModelSession {
    /// Start warming `TINYTITAN_WARM_EXPERTS` experts per layer (default: the
    /// slot count minus `ExpertWarmup.headroom`). Returns immediately; the
    /// `TinyTitan warm experts:` line is logged when it ends. A no-op without
    /// a routed slot cache, with 0, or while a warm-up is still running.
    package func startExpertWarmup(
        requestedPerLayer: Int? = ExpertWarmup.requestedPerLayer(
            environment: ProcessInfo.processInfo.environment)
    ) {
        guard expertWarmupTask == nil,
            let slotCount = model.routedExpertCacheSlotCount()
        else { return }
        let perLayer = ExpertWarmup.perLayer(
            requested: requestedPerLayer, slotCount: slotCount,
            expertsPerLayer: model.config.numExperts)
        guard perLayer > 0 else { return }
        // Detached and holding only the model: a session dropped mid-warm
        // cancels it from `deinit`, and it stops after the layer in flight.
        let job = ExpertWarmupJob(model: model)
        expertWarmupTask = Task.detached {
            let report = await job.run(perLayer: perLayer)
            ServerLog.diagnostic(
                report?.logLine
                    ?? "TinyTitan warm experts: \(perLayer)/layer, none ranked "
                    + "(no expert profile yet)")
            return report
        }
    }

    /// Stop the warm-up, if one is running, and wait for its in-flight layer.
    /// Every request calls this before it touches the runner.
    @discardableResult
    package func stopExpertWarmup() async -> ExpertWarmupReport? {
        guard let task = expertWarmupTask else { return nil }
        task.cancel()
        return await settle(task)
    }

    /// Wait for the whole warm-up without stopping it.
    @discardableResult
    package func awaitExpertWarmup() async -> ExpertWarmupReport? {
        guard let task = expertWarmupTask else { return nil }
        return await settle(task)
    }

    private func settle(
        _ task: Task<ExpertWarmupReport?, Never>
    ) async -> ExpertWarmupReport? {
        let report = await task.value
        // Another request may have settled it while this one waited.
        if expertWarmupTask == task { expertWarmupTask = nil }
        return report
    }
}
