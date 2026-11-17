import TinyTitanCore

extension ServerModelSession {
    package func startExpertWarmup(requestedPerLayer: Int? = nil) {
        guard expertWarmupTask == nil, let warmup = makeWarmup(model, requestedPerLayer) else {
            return
        }
        expertWarmupTask = Task.detached { await warmup.run() }
    }

    @discardableResult
    package func stopExpertWarmup() async -> (any Sendable)? {
        guard let task = expertWarmupTask else { return nil }
        task.cancel()
        return await settleWarmup(task)
    }

    @discardableResult
    package func awaitExpertWarmup() async -> (any Sendable)? {
        guard let task = expertWarmupTask else { return nil }
        return await settleWarmup(task)
    }

    private func settleWarmup(
        _ task: Task<(any Sendable)?, Never>
    ) async -> (any Sendable)? {
        let report = await task.value
        if expertWarmupTask == task { expertWarmupTask = nil }
        return report
    }
}
