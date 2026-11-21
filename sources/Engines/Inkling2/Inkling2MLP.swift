import Foundation

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
        for (index, weight) in zip(route.indices, route.weights) {
            let result: [Float]
            if let gpu {
                let packed = try store.packed(layer: layer, expert: index)
                let gateUp = try gpu.gemv(
                    packed, x: x, rows: 2 * packed.w2Columns,
                    columns: packed.w13Columns, w13: true)
                let activated = Inkling2MoE.activate(
                    gateUp, hidden: packed.rows, intermediate: packed.w2Columns)
                result = try gpu.gemv(
                    packed, x: activated, rows: packed.rows,
                    columns: packed.w2Columns, w13: false)
            } else {
                let expert = try store.load(layer: layer, expert: index)
                result = Inkling2MoE.expert(x, weights: expert)
            }
            for dimension in output.indices {
                output[dimension] += result[dimension] * weight
            }
        }
        return output
    }
}
