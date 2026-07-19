import Metal
import Testing

@testable import TurboFieldfare

extension PrefillGroupedRoutedMoETests {
  @Test func streamedParamsLayoutIsStable() {
    #expect(MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.size == 128)
    #expect(MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride == 128)
    #expect(MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.pairStart) == .some(0))
    #expect(MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.pairCount) == .some(4))
    #expect(MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.d) == .some(8))
    #expect(
      MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.routedIntermediate)
        == .some(12))
    #expect(MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.topK) == .some(16))
    #expect(
      MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.hiddenStrideElements)
        == .some(20))
    #expect(
      MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.liveExpertCount) == .some(24))
    #expect(
      MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.localExpert0) == .some(28))
    #expect(
      MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.localExpert15) == .some(88))
    #expect(MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.gateWOff) == .some(92))
    #expect(MemoryLayout.offset(of: \PrefillGroupedRoutedMoEStreamedParams.downBOff) == .some(124))
  }

  @Test func streamedMetadataAllocatesOnlySortedPairs() throws {
    let routes = try Self.measuredPressureRoutes()
    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    let buffers = try grouped.makeStreamedMetadataBuffers(device: ctx.device, routes: routes)
    let pairBytes = routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride
    let omittedBytes =
      routes.groups.count * MemoryLayout<PrefillMoEGroup>.stride
      + routes.tiles.count * MemoryLayout<PrefillMoETile>.stride

    #expect(buffers.sortedPairs.storageMode == .shared)
    #expect(buffers.sortedPairs.length == pairBytes)
    #expect(omittedBytes > 0)

    let pairPtr = buffers.sortedPairs.contents()
      .bindMemory(to: PrefillTokenExpertPair.self, capacity: routes.sortedPairs.count)
    #expect(pairPtr[0] == routes.sortedPairs[0])
    #expect(pairPtr[routes.sortedPairs.count - 1] == routes.sortedPairs.last)
  }
}
