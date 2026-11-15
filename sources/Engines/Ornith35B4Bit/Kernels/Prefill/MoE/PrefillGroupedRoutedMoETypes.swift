import Darwin
import Foundation
import Metal

// The prefill grouped-MoE value types: buffer indices, streamed metadata and
// tile parameter blocks, fetch results, the per-tile lifetime tracker and the
// streamed-tile error.
//
// Split out of `PrefillGroupedRoutedMoE.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
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

    public init(
        expertIDs: [Int],
        binding: PrefillStreamedTileBinding,
        usedPlannedFetch: Bool,
        plannedHits: Int,
        plannedMissIndices: [Int],
        plannedAssignedSlots: [Int],
        plannedMissSlots: [Int]
    ) {
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
            return
                "prefill streamed tile \(tileIndex) would reuse planned slots \(slots) while tile \(conflictingTileIndex) is in flight"
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

    init(
        pairStart: UInt32,
        pairCount: UInt32,
        d: UInt32,
        routedIntermediate: UInt32,
        topK: UInt32,
        hiddenStrideElements: UInt32,
        binding: PrefillStreamedTileBinding,
        offsets: MoEExpertOffsets
    ) {
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
