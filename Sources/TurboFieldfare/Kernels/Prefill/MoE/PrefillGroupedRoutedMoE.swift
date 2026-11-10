import Foundation
import Metal
import Darwin

enum PrefillGroupedRoutedMoEBufferIndex {
    static let hidden = 0
    static let sortedPairs = 1
    static let routePartials = 5
    static let gateUpActScratch = 7
    static let downScratch = 8
    static let expertArgumentState = 9
    static let params = 10
}

struct PrefillGroupedRoutedMoEStreamedMetadataBuffers {
    let sortedPairs: MTLBuffer
}

struct PrefillStreamedTileArgumentBuffer {
    let buffer: MTLBuffer
}

public struct PrefillStreamedTileFetchResult {
    public let expertIDs: [Int]
    public let binding: PrefillStreamedTileBinding
    public let usedPlannedFetch: Bool
    public let plannedHits: Int
    public let plannedMissIndices: [Int]
    public let plannedAssignedSlots: [Int]
    public let plannedMissSlots: [Int]

    public init(expertIDs: [Int],
                binding: PrefillStreamedTileBinding,
                usedPlannedFetch: Bool,
                plannedHits: Int,
                plannedMissIndices: [Int],
                plannedAssignedSlots: [Int],
                plannedMissSlots: [Int]) {
        self.expertIDs = expertIDs
        self.binding = binding
        self.usedPlannedFetch = usedPlannedFetch
        self.plannedHits = plannedHits
        self.plannedMissIndices = plannedMissIndices
        self.plannedAssignedSlots = plannedAssignedSlots
        self.plannedMissSlots = plannedMissSlots
    }
}

enum PrefillStreamedTileLifetimeError: Error, Equatable, CustomStringConvertible {
    case duplicateSlots(tileIndex: Int, slots: [Int])
    case slotReuseBeforeCompletion(tileIndex: Int, conflictingTileIndex: Int, slots: [Int])
    case completeWithoutInFlightTile(tileIndex: Int)

    public var description: String {
        switch self {
        case .duplicateSlots(let tileIndex, let slots):
            return "prefill streamed tile \(tileIndex) has duplicate planned slots \(slots)"
        case .slotReuseBeforeCompletion(let tileIndex, let conflictingTileIndex, let slots):
            return "prefill streamed tile \(tileIndex) would reuse planned slots \(slots) while tile \(conflictingTileIndex) is in flight"
        case .completeWithoutInFlightTile(let tileIndex):
            return "prefill streamed tile \(tileIndex) completed without a matching in-flight tile"
        }
    }
}

struct PrefillStreamedTileSlotLifetime: Sendable, Equatable {
    private var inFlightSlotsByTile: [Int: Set<Int>] = [:]

    init() {}

    mutating func begin(tileIndex: Int, plannedSlots: [Int]) throws {
        let slots = try normalizedSlots(tileIndex: tileIndex, plannedSlots: plannedSlots)
        for (otherTile, otherSlots) in inFlightSlotsByTile {
            let overlap = slots.intersection(otherSlots)
            if !overlap.isEmpty {
                throw PrefillStreamedTileLifetimeError.slotReuseBeforeCompletion(
                    tileIndex: tileIndex,
                    conflictingTileIndex: otherTile,
                    slots: overlap.sorted())
            }
        }
        inFlightSlotsByTile[tileIndex] = slots
    }

    mutating func complete(tileIndex: Int) throws {
        guard inFlightSlotsByTile.removeValue(forKey: tileIndex) != nil else {
            throw PrefillStreamedTileLifetimeError.completeWithoutInFlightTile(tileIndex: tileIndex)
        }
    }

    private func normalizedSlots(tileIndex: Int, plannedSlots: [Int]) throws -> Set<Int> {
        var slots = Set<Int>()
        for slot in plannedSlots {
            guard slots.insert(slot).inserted else {
                throw PrefillStreamedTileLifetimeError.duplicateSlots(
                    tileIndex: tileIndex,
                    slots: plannedSlots.sorted())
            }
        }
        return slots
    }
}

struct PrefillGroupedRoutedMoEStreamedParams: Equatable, Sendable {
    var pairStart: UInt32
    var pairCount: UInt32
    var d: UInt32
    var routedIntermediate: UInt32
    var topK: UInt32
    var hiddenStrideElements: UInt32
    var liveExpertCount: UInt32
    var localExpert0: UInt32
    var localExpert1: UInt32
    var localExpert2: UInt32
    var localExpert3: UInt32
    var localExpert4: UInt32
    var localExpert5: UInt32
    var localExpert6: UInt32
    var localExpert7: UInt32
    var localExpert8: UInt32
    var localExpert9: UInt32
    var localExpert10: UInt32
    var localExpert11: UInt32
    var localExpert12: UInt32
    var localExpert13: UInt32
    var localExpert14: UInt32
    var localExpert15: UInt32
    var gateWOff: UInt32
    var gateSOff: UInt32
    var gateBOff: UInt32
    var upWOff: UInt32
    var upSOff: UInt32
    var upBOff: UInt32
    var downWOff: UInt32
    var downSOff: UInt32
    var downBOff: UInt32

    init(pairStart: UInt32,
                pairCount: UInt32,
                d: UInt32,
                routedIntermediate: UInt32,
                topK: UInt32,
                hiddenStrideElements: UInt32,
                binding: PrefillStreamedTileBinding,
                offsets: MoEExpertOffsets) {
        var ids = Array(repeating: UInt32.max, count: 16)
        for (index, expert) in binding.expertIDs.enumerated() {
            ids[index] = UInt32(expert)
        }
        self.pairStart = pairStart
        self.pairCount = pairCount
        self.d = d
        self.routedIntermediate = routedIntermediate
        self.topK = topK
        self.hiddenStrideElements = hiddenStrideElements
        self.liveExpertCount = UInt32(binding.expertIDs.count)
        self.localExpert0 = ids[0]
        self.localExpert1 = ids[1]
        self.localExpert2 = ids[2]
        self.localExpert3 = ids[3]
        self.localExpert4 = ids[4]
        self.localExpert5 = ids[5]
        self.localExpert6 = ids[6]
        self.localExpert7 = ids[7]
        self.localExpert8 = ids[8]
        self.localExpert9 = ids[9]
        self.localExpert10 = ids[10]
        self.localExpert11 = ids[11]
        self.localExpert12 = ids[12]
        self.localExpert13 = ids[13]
        self.localExpert14 = ids[14]
        self.localExpert15 = ids[15]
        self.gateWOff = offsets.gateWOff
        self.gateSOff = offsets.gateSOff
        self.gateBOff = offsets.gateBOff
        self.upWOff = offsets.upWOff
        self.upSOff = offsets.upSOff
        self.upBOff = offsets.upBOff
        self.downWOff = offsets.downWOff
        self.downSOff = offsets.downSOff
        self.downBOff = offsets.downBOff
    }
}

public struct PrefillStreamedTileBinding: Sendable, Equatable {
    public let expertIDs: [Int]
    public let views: [TensorView]

    public init(expertIDs: [Int], views: [TensorView]) throws {
        guard !expertIDs.isEmpty else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding("tile binding must include at least one expert")
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

    public static func expertIDs(forTile tileIndex: Int,
                                 routes: PrefillMoEGroupedRoutes) throws -> [Int] {
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
                "tile group range \(groupStart)..<\(groupStart + groupCount) exceeds \(routes.groups.count)")
        }
        return routes.groups[groupStart..<(groupStart + groupCount)].map { Int($0.expert) }
    }

    public static func fetchBindingForTile(model: Model,
                                           layer: Int,
                                           tileIndex: Int,
                                           routes: PrefillMoEGroupedRoutes,
                                           plannedFetch: RoutedExpertFetchPlan? = nil,
                                           avoidingSlots: Set<Int> = []) async throws
        -> PrefillStreamedTileFetchResult {
        let expertIDs = try expertIDs(forTile: tileIndex, routes: routes)
        let plan = try plannedFetch ?? model.planRoutedExperts(layer: layer,
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
        return PrefillStreamedTileFetchResult(expertIDs: expertIDs,
                                             binding: binding,
                                             usedPlannedFetch: usedPlannedFetch,
                                             plannedHits: plannedHits,
                                             plannedMissIndices: plannedMissIndices,
                                             plannedAssignedSlots: plannedAssignedSlots,
                                             plannedMissSlots: plannedMissSlots)
    }

    public func validateCoversPairs(_ pairs: [PrefillTokenExpertPair],
                                    pairStart: Int,
                                    pairCount: Int) throws {
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

    public static func == (lhs: PrefillStreamedTileBinding,
                           rhs: PrefillStreamedTileBinding) -> Bool {
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
                  l.dtype == r.dtype else {
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
    private let groupedMPP: (gather: MTLComputePipelineState, scatter: MTLComputePipelineState,
                             gemm: MTLComputePipelineState, gelu: MTLComputePipelineState)?
    private var groupedScratch: (gathered: MTLBuffer, gate: MTLBuffer, up: MTLBuffer,
                                 act: MTLBuffer, down: MTLBuffer, rows: Int)?

    var groupedMPPAvailable: Bool { groupedMPP != nil }

    /// Each live slot's contiguous run of rows within the tile's sorted pairs,
    /// or nil when the pairs of one expert are not contiguous.
    static func expertSegments(sortedPairs: [PrefillTokenExpertPair],
                               pairStart: Int,
                               pairCount: Int,
                               expertIDs: [Int]) -> (start: [UInt32], count: [UInt32])? {
        var starts = [UInt32](repeating: 0, count: 16)
        var counts = [UInt32](repeating: 0, count: 16)
        var slotByExpert: [Int: Int] = [:]
        for (slot, expert) in expertIDs.enumerated() { slotByExpert[expert] = slot }
        var previousSlot = -1
        var seen = Set<Int>()
        for row in 0..<pairCount {
            guard let slot = slotByExpert[Int(sortedPairs[pairStart + row].expert)] else { return nil }
            if slot != previousSlot {
                guard seen.insert(slot).inserted else { return nil }
                starts[slot] = UInt32(row)
                previousSlot = slot
            }
            counts[slot] += 1
        }
        return (starts, counts)
    }

    private func ensureGroupedScratch(device: MTLDevice, rows: Int, d: Int, f: Int) throws
        -> (gathered: MTLBuffer, gate: MTLBuffer, up: MTLBuffer, act: MTLBuffer, down: MTLBuffer) {
        if let groupedScratch, groupedScratch.rows >= rows {
            return (groupedScratch.gathered, groupedScratch.gate, groupedScratch.up,
                    groupedScratch.act, groupedScratch.down)
        }
        let half = MemoryLayout<Float16>.stride
        guard let gathered = device.makeBuffer(length: rows * d * half, options: .storageModePrivate),
              let gate = device.makeBuffer(length: rows * f * half, options: .storageModePrivate),
              let up = device.makeBuffer(length: rows * f * half, options: .storageModePrivate),
              let act = device.makeBuffer(length: rows * f * half, options: .storageModePrivate),
              let down = device.makeBuffer(length: rows * d * half, options: .storageModePrivate) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill grouped routed scratch for \(rows) rows")
        }
        groupedScratch = (gathered, gate, up, act, down, rows)
        return (gathered, gate, up, act, down)
    }

    /// Gathers the tile's rows, runs gate, up and down as grouped INT4 matrix
    /// multiplies over every live expert, and scatters the result into the
    /// route partials the token-major reduce reads.
    func encodeStreamedGroupedMPP(commandBuffer: MTLCommandBuffer,
                                  device: MTLDevice,
                                  hidden: MTLBuffer,
                                  sortedPairs: MTLBuffer,
                                  routePartials: MTLBuffer,
                                  argumentBuffer: PrefillStreamedTileArgumentBuffer,
                                  binding: PrefillStreamedTileBinding,
                                  params: PrefillGroupedRoutedMoEStreamedParams,
                                  segments: (start: [UInt32], count: [UInt32])) throws {
        guard let groupedMPP, params.pairCount > 0 else { return }
        let rows = Int(params.pairCount)
        let d = Int(params.d)
        let f = Int(params.routedIntermediate)
        let scratch = try ensureGroupedScratch(device: device, rows: rows, d: d, f: f)
        var p = params
        var segmentBytes = segments.start + segments.count
        let maxRows = Int(segments.count.max() ?? 0)

        func encodeRows(_ pipeline: MTLComputePipelineState, _ buffers: [MTLBuffer]) {
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pipeline)
            for (index, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: index) }
            enc.setBytes(&p, length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride, index: buffers.count)
            enc.dispatchThreads(MTLSize(width: d, height: rows, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
            enc.endEncoding()
        }

        func encodeGEMM(input: MTLBuffer, output: MTLBuffer, n: Int, k: Int,
                        weightOff: UInt32, scaleOff: UInt32, biasOff: UInt32) {
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(groupedMPP.gemm)
            enc.setBuffer(argumentBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(input, offset: 0, index: 1)
            enc.setBuffer(output, offset: 0, index: 2)
            enc.setBytes(&segmentBytes, length: segmentBytes.count * MemoryLayout<UInt32>.stride, index: 3)
            var matrix: [UInt32] = [UInt32(n), UInt32(k), weightOff, scaleOff, biasOff]
            enc.setBytes(&matrix, length: matrix.count * MemoryLayout<UInt32>.stride, index: 4)
            for view in binding.views { enc.useResource(view.buffer, usage: .read) }
            enc.dispatchThreadgroups(
                MTLSize(width: (n + 31) / 32, height: (maxRows + 63) / 64, depth: binding.views.count),
                threadsPerThreadgroup: MTLSize(width: groupedMPP.gemm.threadExecutionWidth * 4, height: 1, depth: 1))
            enc.endEncoding()
        }

        encodeRows(groupedMPP.gather, [hidden, sortedPairs, scratch.gathered])
        encodeGEMM(input: scratch.gathered, output: scratch.gate, n: f, k: d,
                   weightOff: params.gateWOff, scaleOff: params.gateSOff, biasOff: params.gateBOff)
        encodeGEMM(input: scratch.gathered, output: scratch.up, n: f, k: d,
                   weightOff: params.upWOff, scaleOff: params.upSOff, biasOff: params.upBOff)
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(groupedMPP.gelu)
            enc.setBuffer(scratch.gate, offset: 0, index: 0)
            enc.setBuffer(scratch.up, offset: 0, index: 1)
            enc.setBuffer(scratch.act, offset: 0, index: 2)
            var count = UInt32(rows * f)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreads(MTLSize(width: rows * f, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        encodeGEMM(input: scratch.act, output: scratch.down, n: d, k: f,
                   weightOff: params.downWOff, scaleOff: params.downSOff, biasOff: params.downBOff)
        encodeRows(groupedMPP.scatter, [scratch.down, sortedPairs, routePartials])
    }

    func makeStreamedArgumentBuffer(device: MTLDevice,
                                           binding: PrefillStreamedTileBinding) throws -> PrefillStreamedTileArgumentBuffer {
        guard let buffer = device.makeBuffer(length: streamedArgEncoder.encodedLength,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill streamed expert argument buffer")
        }
        buffer.label = "prefill.groupedMoe.streamedArgumentBuffer"

        streamedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for index in binding.views.indices {
            let view = binding.views[index]
            streamedArgEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }

        return PrefillStreamedTileArgumentBuffer(buffer: buffer)
    }

    init(context: MetalContext) throws {
        self.batchedPhase1PSO = try context.pipeline("prefill_grouped_routed_moe_batched_phase1")
        self.batchedDownPSO = try context.pipeline("prefill_grouped_routed_moe_batched_down")
        guard let streamedFn = context.library.makeFunction(name: "prefill_grouped_routed_moe_batched_phase1") else {
            throw MetalError.missingFunction("prefill_grouped_routed_moe_batched_phase1")
        }
        self.streamedArgEncoder = streamedFn.makeArgumentEncoder(
            bufferIndex: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
        if ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PREFILL_ROUTED_MPP"] != "0" {
            do {
                self.groupedMPP = (try context.pipeline("prefill_routed_gather_rows"),
                                   try context.pipeline("prefill_routed_scatter_rows"),
                                   try context.pipeline("prefill_routed_grouped_int4_mpp"),
                                   try context.pipeline("prefill_gelu_mul_rows"))
            } catch {
                self.groupedMPP = nil
                FileHandle.standardError.write(Data(
                    "PrefillGroupedRoutedMoE: grouped MPP path unavailable: \(error)\n".utf8))
            }
        } else {
            self.groupedMPP = nil
        }
    }

    func makeStreamedMetadataBuffers(
        device: MTLDevice,
        routes: PrefillMoEGroupedRoutes
    ) throws -> PrefillGroupedRoutedMoEStreamedMetadataBuffers {
        let bytes = routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride
        guard let sortedPairs = routes.sortedPairs.withUnsafeBufferPointer({ ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: bytes,
                              options: .storageModeShared)
        }) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill sorted route pairs")
        }
        return PrefillGroupedRoutedMoEStreamedMetadataBuffers(sortedPairs: sortedPairs)
    }

    @discardableResult
    func encodeStreamedBatched(commandBuffer: MTLCommandBuffer,
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
                                      pairMicrobatchRows: Int = 32) -> Int {
        guard params.pairCount > 0,
              params.liveExpertCount == UInt32(binding.views.count),
              pairMicrobatchRows > 0 else { return 0 }
        var consumed: UInt32 = 0
        var microbatchCount = 0
        while consumed < params.pairCount {
            var p = params
            p.pairStart = params.pairStart + consumed
            p.pairCount = min(UInt32(pairMicrobatchRows), params.pairCount - consumed)

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(batchedPhase1PSO)
                enc.setBuffer(hidden, offset: hiddenOffset, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
                enc.setBytes(&p,
                             length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                             index: PrefillGroupedRoutedMoEBufferIndex.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                enc.dispatchThreads(MTLSize(width: Int(p.routedIntermediate),
                                            height: Int(p.pairCount),
                                            depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                enc.endEncoding()
            }

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(batchedDownPSO)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
                enc.setBuffer(routePartials, offset: routePartialsOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
                enc.setBuffer(downScratch, offset: downScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.downScratch)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
                enc.setBytes(&p,
                             length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                             index: PrefillGroupedRoutedMoEBufferIndex.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                enc.dispatchThreads(MTLSize(width: Int(p.d),
                                            height: Int(p.pairCount),
                                            depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                enc.endEncoding()
            }

            consumed += p.pairCount
            microbatchCount += 1
        }
        return microbatchCount
    }

}
