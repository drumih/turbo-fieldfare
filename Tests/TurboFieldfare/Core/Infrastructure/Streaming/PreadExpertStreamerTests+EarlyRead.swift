import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

extension PreadExpertStreamerTests {
  /// An adopted early read serves the planned miss from the staging buffer,
  /// skips that read, and hands the slot's previous buffer back as staging.
  @Test func adoptedEarlyReadServesMissWithoutRereading() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    _ = try streamer.loadExpertsCached(experts: [0, 1])

    // A sentinel instead of expert 3's bytes: if the plan read expert 3 again,
    // the sentinel would be overwritten.
    let sentinel: UInt8 = 0x55
    let staging = try streamer.makeStagingBuffer()
    let stagedBuffer = staging.buffer
    memset(staging.pointer, Int32(sentinel), Self.expertStride)

    let plan = streamer.planExpertsCached(experts: [0, 3, 2])
    let index = try #require(plan.experts.firstIndex(of: 3))
    #expect(plan.misses.contains(index))
    let evictedBuffer = streamer.expertCachePlanBuffers(plan)[index].buffer
    streamer.adoptStagedExpert(staging, plan: plan, index: index)
    var waited = false
    let results = try streamer.executeExpertCachePlan(
      plan, skippingMisses: [index], beforeAdmitting: { waited = true })

    #expect(waited)
    #expect(results[index].buffer === stagedBuffer)
    #expect(staging.buffer === evictedBuffer)
    for (offset, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      let expected = offset == index ? sentinel : Self.tagByte(plan.experts[offset])
      #expect(got.allSatisfy { $0 == expected })
    }
    #expect(streamer.residentExperts() == [0, 1, 2, 3])  // 4 slots: nothing evicted

    // The next plan finds 3 resident in the adopted buffer.
    let again = streamer.planExpertsCached(experts: [3])
    #expect(again.misses.isEmpty)
    #expect(streamer.expertCachePlanBuffers(again)[0].buffer === stagedBuffer)
  }

  /// An early read fills the staging buffer with the expert's bytes; leaving it
  /// unused changes nothing in the cache.
  @Test func unusedEarlyReadLeavesCacheUnchanged() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    _ = try streamer.loadExpertsCached(experts: [0, 1])

    let staging = try streamer.makeStagingBuffer()
    try streamer.readEarly(expert: 3, into: staging.pointer)
    #expect(streamer.residentExperts() == [0, 1])
    let staged = Self.bytes(of: staging.buffer, offset: 0, count: Self.expertStride)
    #expect(staged.allSatisfy { $0 == Self.tagByte(3) })
    let plan = streamer.planExpertsCached(experts: [0, 1])
    #expect(plan.misses.isEmpty)
  }

  @Test func earlyGuessesAreBestNonResidentExperts() {
    let scores: [Float] = [0.5, 2.0, 1.5, 3.0, 2.0]
    scores.withUnsafeBufferPointer { p in
      let s = p.baseAddress!
      #expect(NextLayerExpertPrefetcher.choose(scores: s, count: 5, resident: [3]) == [1])
      #expect(NextLayerExpertPrefetcher.choose(scores: s, count: 5, resident: []) == [3])
      #expect(NextLayerExpertPrefetcher.choose(
        scores: s, count: 5, resident: [3], limit: 3) == [1, 4, 2])  // tie: lower ID first
      #expect(NextLayerExpertPrefetcher.choose(
        scores: s, count: 5, resident: [1, 2, 3], limit: 4) == [4, 0])
      #expect(NextLayerExpertPrefetcher.choose(
        scores: s, count: 5, resident: [0, 1, 2, 3, 4]).isEmpty)
    }
  }

  @Test func earlyReadsPerLayerDefaultsToOne() {
    let key = NextLayerExpertPrefetcher.readsEnvironmentKey
    #expect(NextLayerExpertPrefetcher.readsPerLayer(environment: [:]) == 1)
    #expect(NextLayerExpertPrefetcher.readsPerLayer(environment: [key: "2"]) == 2)
    #expect(NextLayerExpertPrefetcher.readsPerLayer(environment: [key: "8"]) == 8)
    for invalid in ["0", "9", "-1", "two", ""] {
      #expect(NextLayerExpertPrefetcher.readsPerLayer(environment: [key: invalid]) == 1)
    }
  }
}
