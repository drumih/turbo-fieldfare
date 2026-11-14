import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

extension PrefillGroupedRoutedMoETests {
  @Test func streamedBatchedMatchesReferenceAcrossPartialMicrobatch() throws {
    let d = 64
    let f = 64
    let rows = 3
    let topK = 2
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [
        Self.pair(token: 0, expert: 2, rank: 0),
        Self.pair(token: 0, expert: 0, rank: 1),
        Self.pair(token: 1, expert: 1, rank: 0),
        Self.pair(token: 1, expert: 2, rank: 1),
        Self.pair(token: 2, expert: 0, rank: 0),
        Self.pair(token: 2, expert: 1, rank: 1),
      ],
      queryCount: rows,
      topK: topK,
      numExperts: 16,
      tileExpertCount: 16)
    let pool = Self.makeSyntheticExpertPool(numExperts: 16, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in
      Float16(Float((i % 17) - 8) * 0.01)
    }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes,
      hidden: hidden,
      hiddenStride: d,
      pool: pool,
      topK: topK,
      d: d,
      f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 4 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 4 * d * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }

    let expertIDs = Array(0..<16)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(
      device: ctx.device,
      binding: binding)
    let microbatches = grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: activationScratch,
      downScratch: downScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      params: params,
      pairMicrobatchRows: 4)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(microbatches == 2)
    #expect(maxAbsoluteError <= 0.0015, "maxAbsoluteError=\(maxAbsoluteError)")
    #expect(binding.views.allSatisfy { $0.offset > 0 })
  }


  /// The grouped MPP path against the CPU reference, with one expert holding
  /// more rows than a 64-row tile and the others a partial tile each.
  @Test func streamedGroupedMPPMatchesReference() throws {
    let d = 128
    let f = 128
    let rows = 90
    let topK = 2
    let numExperts = 4
    var pairs: [PrefillTokenExpertPair] = []
    for token in 0..<rows {
      pairs.append(Self.pair(token: UInt32(token), expert: 0, rank: 0))
      pairs.append(Self.pair(token: UInt32(token), expert: UInt32(1 + token % 3), rank: 1))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs, queryCount: rows, topK: topK, numExperts: numExperts, tileExpertCount: numExperts)
    let pool = Self.makeSyntheticExpertPool(numExperts: numExperts, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in Float16(Float((i % 23) - 11) * 0.01) }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes, hidden: hidden, hiddenStride: d, pool: pool, topK: topK, d: d, f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    guard grouped.groupedMPPAvailable else { return }
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device, halves: [Float16](repeating: -77, count: rows * topK * d)),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }
    let expertIDs = Array(0..<numExperts)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(device: ctx.device, pool: pool, expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let segments = try #require(PrefillGroupedRoutedMoE.expertSegments(
      sortedPairs: routes.sortedPairs,
      pairStart: 0,
      pairCount: routes.sortedPairs.count,
      expertIDs: expertIDs))
    #expect(segments.count[0] == UInt32(rows))
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: ctx.device, binding: binding)
    try grouped.encodeStreamedGroupedMPP(
      commandBuffer: commandBuffer,
      device: ctx.device,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      argumentBuffer: argumentBuffer,
      binding: binding,
      params: params,
      segments: segments)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(maxAbsoluteError <= 0.005, "maxAbsoluteError=\(maxAbsoluteError)")
  }

  @Test func expertSegmentsRefuseNonContiguousExperts() {
    let pairs = [
      Self.pair(token: 0, expert: 1, rank: 0),
      Self.pair(token: 1, expert: 2, rank: 0),
      Self.pair(token: 2, expert: 1, rank: 0),
    ]
    #expect(PrefillGroupedRoutedMoE.expertSegments(
      sortedPairs: pairs, pairStart: 0, pairCount: 3, expertIDs: [1, 2]) == nil)
    let segments = PrefillGroupedRoutedMoE.expertSegments(
      sortedPairs: pairs, pairStart: 1, pairCount: 1, expertIDs: [1, 2])
    #expect(segments?.start[1] == 0)
    #expect(segments?.count[1] == 1)
    #expect(segments?.count[0] == 0)
  }
}
