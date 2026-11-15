import Darwin
import Foundation
import Metal
import TinyTitanCore

public struct PrefillStreamedTileBinding: Sendable, Equatable {
    public let expertIDs: [Int]
    public let views: [TensorView]

    public init(expertIDs: [Int], views: [TensorView]) throws {
        guard !expertIDs.isEmpty else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile binding must include at least one expert")
        }
        guard expertIDs.count <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile binding has \(expertIDs.count) experts; maximum is 16")
        }
        guard expertIDs.count == views.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "expertIDs.count \(expertIDs.count) != views.count \(views.count)")
        }
        var seen = Set<Int>()
        for expert in expertIDs {
            guard expert >= 0 else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "expert id \(expert) must be non-negative")
            }
            guard seen.insert(expert).inserted else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "duplicate expert id \(expert) in tile binding")
            }
        }
        self.expertIDs = expertIDs
        self.views = views
    }

    public func localSlot(for expert: UInt32) -> Int? {
        expertIDs.firstIndex(of: Int(expert))
    }

    public static func expertIDs(
        forTile tileIndex: Int,
        routes: PrefillMoEGroupedRoutes
    ) throws -> [Int] {
        guard routes.tiles.indices.contains(tileIndex) else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile index \(tileIndex) is out of range")
        }
        let tile = routes.tiles[tileIndex]
        let groupStart = Int(tile.groupStart)
        let groupCount = Int(tile.groupCount)
        guard groupCount > 0, groupCount <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile has \(groupCount) live experts; expected 1...16")
        }
        guard groupStart >= 0, groupStart + groupCount <= routes.groups.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile group range \(groupStart)..<\(groupStart + groupCount) exceeds \(routes.groups.count)"
            )
        }
        return routes.groups[groupStart..<(groupStart + groupCount)].map { Int($0.expert) }
    }

    public static func fetchBindingForTile(
        model: Model,
        layer: Int,
        tileIndex: Int,
        routes: PrefillMoEGroupedRoutes,
        plannedFetch: RoutedExpertFetchPlan? = nil,
        avoidingSlots: Set<Int> = []
    ) async throws
        -> PrefillStreamedTileFetchResult
    {
        let expertIDs = try expertIDs(forTile: tileIndex, routes: routes)
        let plan =
            try plannedFetch
            ?? model.planRoutedExperts(
                layer: layer,
                experts: expertIDs,
                avoidingSlots: avoidingSlots)
        let views: [TensorView]
        let usedPlannedFetch: Bool
        let plannedHits: Int
        let plannedMissIndices: [Int]
        let plannedAssignedSlots: [Int]
        let plannedMissSlots: [Int]
        if let plan {
            guard plan.layer == layer, plan.experts == expertIDs else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "preplanned fetch does not match tile \(tileIndex)")
            }
            views = try await model.fetchRoutedExperts(plan: plan)
            usedPlannedFetch = true
            plannedHits = plan.hits
            plannedMissIndices = plan.misses
            plannedAssignedSlots = plan.assignedSlots
            plannedMissSlots = plan.misses.map { plan.assignedSlots[$0] }
        } else {
            views = try await model.fetchRoutedExperts(layer: layer, experts: expertIDs)
            usedPlannedFetch = false
            plannedHits = 0
            plannedMissIndices = []
            plannedAssignedSlots = []
            plannedMissSlots = []
        }
        let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
        return PrefillStreamedTileFetchResult(
            expertIDs: expertIDs,
            binding: binding,
            usedPlannedFetch: usedPlannedFetch,
            plannedHits: plannedHits,
            plannedMissIndices: plannedMissIndices,
            plannedAssignedSlots: plannedAssignedSlots,
            plannedMissSlots: plannedMissSlots)
    }

    public func validateCoversPairs(
        _ pairs: [PrefillTokenExpertPair],
        pairStart: Int,
        pairCount: Int
    ) throws {
        guard pairStart >= 0, pairCount >= 0, pairStart + pairCount <= pairs.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "pair range \(pairStart)..<\(pairStart + pairCount) exceeds \(pairs.count)")
        }
        for pair in pairs[pairStart..<(pairStart + pairCount)] {
            guard localSlot(for: pair.expert) != nil else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "route expert \(pair.expert) is not bound in tile")
            }
        }
    }

    public static func == (
        lhs: PrefillStreamedTileBinding,
        rhs: PrefillStreamedTileBinding
    ) -> Bool {
        guard lhs.expertIDs == rhs.expertIDs, lhs.views.count == rhs.views.count else {
            return false
        }
        for index in lhs.views.indices {
            let l = lhs.views[index]
            let r = rhs.views[index]
            guard l.buffer === r.buffer,
                l.offset == r.offset,
                l.length == r.length,
                l.scaleOffset == r.scaleOffset,
                l.scaleLength == r.scaleLength,
                l.biasOffset == r.biasOffset,
                l.biasLength == r.biasLength,
                l.shape == r.shape,
                l.dtype == r.dtype
            else {
                return false
            }
        }
        return true
    }
}

enum PrefillGroupedRoutedMoEError: Error, Equatable, CustomStringConvertible {
    case invalidStreamedTileBinding(String)
    case allocationFailed(String)

    public var description: String {
        switch self {
        case .invalidStreamedTileBinding(let reason):
            return "invalid streamed tile binding: \(reason)"
        case .allocationFailed(let label):
            return "failed to allocate \(label)"
        }
    }
}

final class PrefillGroupedRoutedMoE {
    private let batchedPhase1PSO: MTLComputePipelineState
    private let batchedDownPSO: MTLComputePipelineState
    private let streamedArgEncoder: MTLArgumentEncoder

    func makeStreamedArgumentBuffer(
        device: MTLDevice,
        binding: PrefillStreamedTileBinding
    ) throws -> PrefillStreamedTileArgumentBuffer {
        guard
            let buffer = device.makeBuffer(
                length: streamedArgEncoder.encodedLength,
                options: .storageModeShared)
        else {
            throw PrefillGroupedRoutedMoEError.allocationFailed(
                "prefill streamed expert argument buffer")
        }
        buffer.label = "prefill.groupedMoe.streamedArgumentBuffer"

        streamedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for index in binding.views.indices {
            let view = binding.views[index]
            streamedArgEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }

        return PrefillStreamedTileArgumentBuffer(buffer: buffer)
    }

    init(
        context: MetalContext,
        siluActivation: Bool = false,
        weightBits: Int = 4
    ) throws {
        precondition([2, 4, 8].contains(weightBits))
        var activationConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(
                index: 78,
                value: .uint32(UInt32(weightBits)))
        ]
        if siluActivation {
            activationConstants.append(
                MetalFunctionConstant(index: 77, value: .bool(true)))
        }
        self.batchedPhase1PSO = try context.pipeline(
            "prefill_grouped_routed_moe_batched_phase1",
            constants: activationConstants)
        self.batchedDownPSO = try context.pipeline(
            "prefill_grouped_routed_moe_batched_down",
            constants: [
                MetalFunctionConstant(
                    index: 78,
                    value: .uint32(UInt32(weightBits)))
            ])
        guard
            let streamedFn = context.library.makeFunction(
                name: "prefill_grouped_routed_moe_batched_phase1")
        else {
            throw MetalError.missingFunction("prefill_grouped_routed_moe_batched_phase1")
        }
        self.streamedArgEncoder = streamedFn.makeArgumentEncoder(
            bufferIndex: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
    }

    func makeStreamedMetadataBuffers(
        device: MTLDevice,
        routes: PrefillMoEGroupedRoutes
    ) throws -> PrefillGroupedRoutedMoEStreamedMetadataBuffers {
        let bytes = routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride
        // K13: sortedPairs can be empty (zero routed pairs for the chunk).
        // withUnsafeBufferPointer on an empty array yields a nil baseAddress,
        // so guard the empty case and hand out a zero-length buffer instead of
        // force-unwrapping. The batched kernels guard on pair_count == 0.
        guard !routes.sortedPairs.isEmpty else {
            guard let empty = device.makeBuffer(length: 0, options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("prefill sorted route pairs")
            }
            return PrefillGroupedRoutedMoEStreamedMetadataBuffers(sortedPairs: empty)
        }
        guard
            let sortedPairs = routes.sortedPairs.withUnsafeBufferPointer({ ptr -> MTLBuffer? in
                guard let base = ptr.baseAddress else { return nil }
                return device.makeBuffer(bytes: base, length: bytes, options: .storageModeShared)
            })
        else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill sorted route pairs")
        }
        return PrefillGroupedRoutedMoEStreamedMetadataBuffers(sortedPairs: sortedPairs)
    }

    @discardableResult
    func encodeStreamedBatched(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        hiddenOffset: Int = 0,
        sortedPairs: MTLBuffer,
        sortedPairsOffset: Int = 0,
        routePartials: MTLBuffer,
        routePartialsOffset: Int = 0,
        gateUpActScratch: MTLBuffer,
        gateUpActScratchOffset: Int = 0,
        downScratch: MTLBuffer,
        downScratchOffset: Int = 0,
        argumentBuffer: PrefillStreamedTileArgumentBuffer,
        binding: PrefillStreamedTileBinding,
        params: PrefillGroupedRoutedMoEStreamedParams,
        pairMicrobatchRows: Int = 32
    ) throws -> Int {
        guard params.pairCount > 0,
            params.liveExpertCount == UInt32(binding.views.count),
            pairMicrobatchRows > 0
        else { return 0 }
        var consumed: UInt32 = 0
        var microbatchCount = 0
        while consumed < params.pairCount {
            var p = params
            p.pairStart = params.pairStart + consumed
            p.pairCount = min(UInt32(pairMicrobatchRows), params.pairCount - consumed)

            guard let enc1 = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc1.setComputePipelineState(batchedPhase1PSO)
            enc1.setBuffer(
                hidden, offset: hiddenOffset, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
            enc1.setBuffer(
                sortedPairs, offset: sortedPairsOffset,
                index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            enc1.setBuffer(
                gateUpActScratch, offset: gateUpActScratchOffset,
                index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
            enc1.setBuffer(
                argumentBuffer.buffer, offset: 0,
                index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            enc1.setBytes(
                &p,
                length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                index: PrefillGroupedRoutedMoEBufferIndex.params)
            for view in binding.views {
                enc1.useResource(view.buffer, usage: .read)
            }
            enc1.dispatchThreads(
                MTLSize(
                    width: Int(p.routedIntermediate),
                    height: Int(p.pairCount),
                    depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc1.endEncoding()

            guard let enc2 = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc2.setComputePipelineState(batchedDownPSO)
            enc2.setBuffer(
                sortedPairs, offset: sortedPairsOffset,
                index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            enc2.setBuffer(
                routePartials, offset: routePartialsOffset,
                index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
            enc2.setBuffer(
                gateUpActScratch, offset: gateUpActScratchOffset,
                index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
            enc2.setBuffer(
                downScratch, offset: downScratchOffset,
                index: PrefillGroupedRoutedMoEBufferIndex.downScratch)
            enc2.setBuffer(
                argumentBuffer.buffer, offset: 0,
                index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            enc2.setBytes(
                &p,
                length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                index: PrefillGroupedRoutedMoEBufferIndex.params)
            for view in binding.views {
                enc2.useResource(view.buffer, usage: .read)
            }
            enc2.dispatchThreads(
                MTLSize(
                    width: Int(p.d),
                    height: Int(p.pairCount),
                    depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc2.endEncoding()

            consumed += p.pairCount
            microbatchCount += 1
        }
        return microbatchCount
    }

}
