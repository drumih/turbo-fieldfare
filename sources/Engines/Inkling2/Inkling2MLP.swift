import Foundation

/// One token through a routed-expert MLP: route top-8, stream those experts,
/// run the softcapped SwiGLU and combine by router weight.
public struct Inkling2MLP: Sendable {
    private let router: Inkling2Router
    private let store: Inkling2ExpertStore
    private let layer: Int

    public init(tensors: Inkling2DenseTensors, store: Inkling2ExpertStore, layer: Int)
        throws
    {
        self.router = try Inkling2Router(tensors: tensors, layer: layer)
        self.store = store
        self.layer = layer
    }

    public func forward(_ x: [Float]) throws -> [Float] {
        let route = router.route(x)
        var output = [Float](repeating: 0, count: x.count)
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
