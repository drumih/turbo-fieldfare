import Foundation
import Metal
import TurboFieldfareFormat

public struct QwenMoEWeights {
    public let router: TensorView
    public let sharedExpertGate: TensorView
    public let sharedExpertUp: TensorView
    public let sharedExpertDown: TensorView
    public let sharedRouterGate: TensorView

    public init(router: TensorView,
                sharedExpertGate: TensorView,
                sharedExpertUp: TensorView,
                sharedExpertDown: TensorView,
                sharedRouterGate: TensorView) {
        self.router = router
        self.sharedExpertGate = sharedExpertGate
        self.sharedExpertUp = sharedExpertUp
        self.sharedExpertDown = sharedExpertDown
        self.sharedRouterGate = sharedRouterGate
    }
}

public struct QwenDeltaNetWeights {
    public let qkv: TensorView
    public let z: TensorView
    public let b: TensorView
    public let a: TensorView
    public let convolution: TensorView
    public let aLog: TensorView
    public let dtBias: TensorView
    public let norm: TensorView
    public let out: TensorView

    public init(qkv: TensorView,
                z: TensorView,
                b: TensorView,
                a: TensorView,
                convolution: TensorView,
                aLog: TensorView,
                dtBias: TensorView,
                norm: TensorView,
                out: TensorView) {
        self.qkv = qkv
        self.z = z
        self.b = b
        self.a = a
        self.norm = norm
        self.out = out
        self.convolution = convolution
        self.aLog = aLog
        self.dtBias = dtBias
    }
}

public struct QwenFullAttentionWeights {
    public let q: TensorView
    public let k: TensorView
    public let v: TensorView
    public let o: TensorView
    public let qNorm: TensorView
    public let kNorm: TensorView

    public init(q: TensorView,
                k: TensorView,
                v: TensorView,
                o: TensorView,
                qNorm: TensorView,
                kNorm: TensorView) {
        self.q = q
        self.k = k
        self.v = v
        self.o = o
        self.qNorm = qNorm
        self.kNorm = kNorm
    }
}

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

struct RoutedExpertFetchResult {
    let views: [TensorView]
    let readDiagnostics: ExpertReadDiagnostics
}

extension Model {
    public func qwenDeltaNetWeights(layer L: Int) throws -> QwenDeltaNetWeights {
        let prefix = "language_model.model.layers.\(L).linear_attn"
        return QwenDeltaNetWeights(
            qkv: try resident(name: "\(prefix).in_proj_qkv.weight"),
            z: try resident(name: "\(prefix).in_proj_z.weight"),
            b: try resident(name: "\(prefix).in_proj_b.weight"),
            a: try resident(name: "\(prefix).in_proj_a.weight"),
            convolution: try resident(name: "\(prefix).conv1d.weight"),
            aLog: try resident(name: "\(prefix).A_log"),
            dtBias: try resident(name: "\(prefix).dt_bias"),
            norm: try resident(name: "\(prefix).norm.weight"),
            out: try resident(name: "\(prefix).out_proj.weight"))
    }

    public func qwenFullAttentionWeights(layer L: Int) throws -> QwenFullAttentionWeights {
        let prefix = "language_model.model.layers.\(L).self_attn"
        return QwenFullAttentionWeights(
            q: try resident(name: "\(prefix).q_proj.weight"),
            k: try resident(name: "\(prefix).k_proj.weight"),
            v: try resident(name: "\(prefix).v_proj.weight"),
            o: try resident(name: "\(prefix).o_proj.weight"),
            qNorm: try resident(name: "\(prefix).q_norm.weight"),
            kNorm: try resident(name: "\(prefix).k_norm.weight"))
    }

    public func qwenMoEWeights(layer L: Int) throws -> QwenMoEWeights {
        let prefix = "language_model.model.layers.\(L)"
        return QwenMoEWeights(
            router: try resident(name: "\(prefix).mlp.gate.weight"),
            sharedExpertGate: try resident(
                name: "\(prefix).mlp.shared_expert.gate_proj.weight"),
            sharedExpertUp: try resident(
                name: "\(prefix).mlp.shared_expert.up_proj.weight"),
            sharedExpertDown: try resident(
                name: "\(prefix).mlp.shared_expert.down_proj.weight"),
            sharedRouterGate: try resident(
                name: "\(prefix).mlp.shared_expert_gate.weight"))
    }

    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            guard let tensor = expert.subTensors[role],
                  let offset = UInt32(exactly: tensor.offset) else {
                preconditionFailure("invalid routed expert metadata for role \(role)")
            }
            return offset
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots))
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: plan.layer,
                        experts: plan.experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func fetchRoutedExpertsWithDiagnostics(plan: RoutedExpertFetchPlan) async throws
        -> RoutedExpertFetchResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let execution = try streamer.executeExpertCachePlanWithDiagnostics(
                        plan.cachePlan)
                    continuation.resume(returning: RoutedExpertFetchResult(
                        views: Self.makeExpertViews(
                            execution.buffers,
                            layer: plan.layer,
                            experts: plan.experts),
                        readDiagnostics: execution.readDiagnostics))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.loadExpertsCached(experts: experts)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }
    }
}
