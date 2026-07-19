import Metal
import Testing

@testable import TurboFieldfare

extension PrefillGroupedRoutedMoETests {
  @Test func streamedTileBindingMapsGlobalExpertsToLocalSlots() throws {
    let ctx = try MetalContext()
    let expertIDs = [3, 9, 17]
    let views = try Self.fakeTensorViews(device: ctx.device, count: expertIDs.count)
    let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)

    #expect(binding.localSlot(for: 3) == 0)
    #expect(binding.localSlot(for: 9) == 1)
    #expect(binding.localSlot(for: 3) == 0)
    #expect(binding.localSlot(for: 17) == 2)
    #expect(binding.localSlot(for: 5) == nil)

    let pairs = [
      Self.pair(token: 0, expert: 3, rank: 0),
      Self.pair(token: 0, expert: 9, rank: 1),
      Self.pair(token: 1, expert: 3, rank: 0),
      Self.pair(token: 1, expert: 17, rank: 1),
    ]
    try binding.validateCoversPairs(pairs, pairStart: 0, pairCount: pairs.count)
  }

  @Test func streamedTileBindingRejectsUnboundRouteExpert() throws {
    let ctx = try MetalContext()
    let binding = try PrefillStreamedTileBinding(
      expertIDs: [3, 9],
      views: Self.fakeTensorViews(device: ctx.device, count: 2))
    let pairs = [
      Self.pair(token: 0, expert: 3, rank: 0),
      Self.pair(token: 0, expert: 17, rank: 1),
    ]

    #expect {
      try binding.validateCoversPairs(pairs, pairStart: 0, pairCount: pairs.count)
    } throws: { error in
      guard case PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(let reason) = error else {
        return false
      }
      return reason.contains("17")
    }

    #expect {
      _ = try PrefillStreamedTileBinding(
        expertIDs: [3, 3],
        views: Self.fakeTensorViews(device: ctx.device, count: 2))
    } throws: { error in
      guard case PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(let reason) = error else {
        return false
      }
      return reason.contains("duplicate")
    }
  }

  @Test func streamedTilePlanningNeverExceedsDefaultSlotCount() throws {
    let pairs = (0..<24).map { index in
      Self.pair(
        token: UInt32(index / 8),
        expert: UInt32(min(index, 19)),
        rank: UInt32(index % 8))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs,
      queryCount: 3,
      topK: 8,
      numExperts: 128,
      tileExpertCount: 16)

    let firstTile = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    let secondTile = try PrefillStreamedTileBinding.expertIDs(forTile: 1, routes: routes)

    #expect(firstTile == Array(0..<16))
    #expect(secondTile == Array(16..<20))
    #expect(firstTile.count <= 16)
    #expect(secondTile.count <= 16)
    #expect(firstTile != Array(0..<20))
    #expect(secondTile != Array(0..<20))
  }

  @Test func streamedTileSlotLifetimeRejectsPlannedCacheSlotReuseBeforeCompletion() throws {
    let firstPlan = ExpertCachePlan(
      experts: [0, 1, 2, 3],
      assignedSlots: [0, 1, 2, 3],
      misses: [0, 1, 2, 3],
      hits: 0)
    let secondPlan = ExpertCachePlan(
      experts: [4, 5, 6, 7],
      assignedSlots: [0, 1, 2, 3],
      misses: [0, 1, 2, 3],
      hits: 0)
    var lifetime = PrefillStreamedTileSlotLifetime()

    try lifetime.begin(tileIndex: 0, plannedSlots: firstPlan.assignedSlots)
    #expect {
      try lifetime.begin(tileIndex: 1, plannedSlots: secondPlan.assignedSlots)
    } throws: { error in
      guard
        case PrefillStreamedTileLifetimeError.slotReuseBeforeCompletion(
          tileIndex: 1,
          conflictingTileIndex: 0,
          slots: [0, 1, 2, 3]) = error
      else {
        return false
      }
      return true
    }

    try lifetime.complete(tileIndex: 0)
    try lifetime.begin(tileIndex: 1, plannedSlots: secondPlan.assignedSlots)
    try lifetime.complete(tileIndex: 1)
  }

  @Test func streamedTileSlotLifetimeAllowsPinnedNonOverlappingSlots() throws {
    var lifetime = PrefillStreamedTileSlotLifetime()

    try lifetime.begin(tileIndex: 0, plannedSlots: [0, 1])
    try lifetime.begin(tileIndex: 1, plannedSlots: [2, 3])

    try lifetime.complete(tileIndex: 0)
    try lifetime.complete(tileIndex: 1)

    #expect {
      try lifetime.begin(tileIndex: 2, plannedSlots: [4, 4])
    } throws: { error in
      guard
        case PrefillStreamedTileLifetimeError.duplicateSlots(
          tileIndex: 2,
          slots: [4, 4]) = error
      else {
        return false
      }
      return true
    }
  }

  @Test func streamedTileSlotLifetimeAllowsSequentialReuseAfterCompletion() throws {
    var lifetime = PrefillStreamedTileSlotLifetime()
    let reusedSlots = [0, 1, 2, 3, 4, 5, 6, 7]

    for tile in 0..<4 {
      try lifetime.begin(tileIndex: tile, plannedSlots: reusedSlots)
      try lifetime.complete(tileIndex: tile)
    }
  }

  @Test func streamedTileFetchBindingUsesPreadPlanAndCacheHits() async throws {
    let dir = try ModelLoaderTests.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      streamingMode: .pread(slotCount: 16))
    let routes = try Self.tileFetchRoutes()

    let first = try await PrefillStreamedTileBinding.fetchBindingForTile(
      model: model,
      layer: 1,
      tileIndex: 0,
      routes: routes)
    let second = try await PrefillStreamedTileBinding.fetchBindingForTile(
      model: model,
      layer: 1,
      tileIndex: 0,
      routes: routes)

    #expect(first.expertIDs == [1, 3, 5])
    #expect(first.usedPlannedFetch)
    #expect(first.plannedHits == 0)
    #expect(first.plannedMissIndices == [0, 1, 2])
    #expect(first.plannedMissSlots == [0, 1, 2])
    #expect(second.usedPlannedFetch)
    #expect(second.plannedHits == 3)
    #expect(second.plannedMissIndices.isEmpty)
    #expect(second.plannedMissSlots.isEmpty)
    for result in [first, second] {
      #expect(result.binding.views.allSatisfy { $0.offset == 0 })
      try result.binding.validateCoversPairs(
        routes.sortedPairs,
        pairStart: 0,
        pairCount: routes.sortedPairs.count)
      for (index, expert) in result.expertIDs.enumerated() {
        let view = result.binding.views[index]
        #expect(Self.byte(view, at: 0) == 1)
        #expect(Self.byte(view, at: 1) == UInt8(expert))
      }
    }
  }

  @Test func streamedTileFetchBindingAvoidsInFlightPlannedSlots() async throws {
    let dir = try ModelLoaderTests.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      streamingMode: .pread(slotCount: 16))
    let routes = try Self.tileFetchRoutes()

    let first = try await PrefillStreamedTileBinding.fetchBindingForTile(
      model: model,
      layer: 1,
      tileIndex: 0,
      routes: routes)
    let second = try await PrefillStreamedTileBinding.fetchBindingForTile(
      model: model,
      layer: 1,
      tileIndex: 0,
      routes: routes,
      avoidingSlots: Set(first.plannedAssignedSlots))

    #expect(first.usedPlannedFetch)
    #expect(first.plannedAssignedSlots == [0, 1, 2])
    #expect(first.plannedMissSlots == [0, 1, 2])
    #expect(second.usedPlannedFetch)
    #expect(second.plannedAssignedSlots == [0, 1, 2])
    #expect(second.plannedHits == 3)
    #expect(second.plannedMissIndices.isEmpty)
    #expect(second.plannedMissSlots.isEmpty)
    try second.binding.validateCoversPairs(
      routes.sortedPairs,
      pairStart: 0,
      pairCount: routes.sortedPairs.count)
  }

}
