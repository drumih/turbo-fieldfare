import Foundation

/// unchecked-invariant: every access to the mutable load result is under `lock`.
private final class Inkling2ExpertLoadBox: @unchecked Sendable {
    let lock = NSLock()
    var values: [Inkling2PackedExpert?]
    var error: Error?
    init(count: Int) { self.values = Array(repeating: nil, count: count) }
}

/// One token through a routed-expert MLP: route top-8, stream those experts,
/// run the softcapped SwiGLU and combine by router weight.
public struct Inkling2MLP: Sendable {
    private let router: Inkling2Router
    private let store: Inkling2ExpertStore
    private let layer: Int
    private let gpu: Inkling2GPUExpert?

    public init(
        tensors: Inkling2DenseTensors, store: Inkling2ExpertStore, layer: Int,
        gpu: Inkling2GPUExpert? = nil
    ) throws {
        self.router = try Inkling2Router(tensors: tensors, layer: layer)
        self.store = store
        self.layer = layer
        self.gpu = gpu
    }

    public func forward(_ x: [Float]) throws -> [Float] {
        let route = router.route(x)
        var output = [Float](repeating: 0, count: x.count)
        if let gpu {
            let loadStart = Date()
            let box = Inkling2ExpertLoadBox(count: route.indices.count)
            DispatchQueue.concurrentPerform(iterations: route.indices.count) {
                index in
                do {
                    let value = try store.packed(
                        layer: layer, expert: route.indices[index])
                    box.lock.withLock { box.values[index] = value }
                } catch {
                    box.lock.withLock { box.error = error }
                }
            }
            if let loadError = box.error { throw loadError }
            Inkling2Profiler.record(
                "expertDiskLoad", Date().timeIntervalSince(loadStart))
            let packed = box.values.compactMap { $0 }
            precondition(packed.count == route.indices.count)
            let results = try gpu.expertForwardBatch(
                packed, x: x, hidden: packed[0].rows,
                intermediate: packed[0].w2Columns,
                gateCap: Inkling2MoE.gateCap, upCap: Inkling2MoE.upCap)
            for (result, weight) in zip(results, route.weights) {
                for dimension in output.indices {
                    output[dimension] += result[dimension] * weight
                }
            }
            return output
        }
        for (index, weight) in zip(route.indices, route.weights) {
            let expert = try store.load(layer: layer, expert: index)
            let result = Inkling2MoE.expert(x, weights: expert)
            for dimension in output.indices {
                output[dimension] += result[dimension] * weight
            }
        }
        return output
    }
}
