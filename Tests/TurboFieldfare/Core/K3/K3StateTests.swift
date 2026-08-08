import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// `K3State` allocation math, per-layer offset views, and `reset()` zeroing.
/// Uses the tiny K3 config (2 KDA + 1 MLA layers, hidden 64) plus one
/// page-exceeding config for the MADV_DONTNEED path.
@Suite struct K3StateTests {

    static func tinyConfig() -> K3ArchConfig {
        K3ArchConfig(
            hiddenSize: 64, vocabSize: 256, numLayers: 3,
            denseMLPIntermediateSize: 128,
            rmsNormEpsilon: 1e-5, tieWordEmbeddings: false,
            hiddenActivation: "situ_glu", bosTokenID: 250, eosTokenID: 251,
            denseLayers: [1], kdaLayers: [1, 3], fullAttnLayers: [2],
            kdaNumHeads: 2, kdaHeadDim: 32, kdaConvWidth: 4,
            kdaDecayLowRankSize: 64, kdaDecayProjectionSize: 64,
            kdaGateLowerBound: -5.0, kdaFullRankOutputGate: true,
            mlaNumHeads: 2, mlaQLoraRank: 64, mlaKVLoraRank: 64,
            mlaQKNopeHeadDim: 64, mlaQKRopeHeadDim: 64, mlaVHeadDim: 64,
            mlaOutputGate: true,
            attnResBlockSize: 12,
            moeNumExperts: 2, moeTopKExperts: 2,
            moeLatentBottleneckSize: 64, moeExpertIntermediateSize: 64,
            moeNumSharedExperts: 1, moeSharedExpertIntermediateSize: 64,
            situGLUGateBeta: 4.0, situGLUUpBeta: 25.0,
            routerRenormalize: true, routerCorrectionBias: true,
            expertsPerLayer: 2, expertStride: 16_384)
    }

    @Test func allocationSizesMatchMaxContext() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let config = Self.tinyConfig()
        let maxContext = 16
        let state = try K3State(device: device, config: config, maxContext: maxContext)

        // KDA: 2 layers x 2 heads x 32 x 32 fp32.
        let kdaStateBytes = 2 * 2 * 32 * 32 * 4
        // Conv: 2 layers x 3 projections x 64 channels x 3 taps fp32.
        let kdaConvBytes = 2 * 3 * 64 * 3 * 4
        // MLA: 1 layer x 16 tokens x (64 + 64) fp16.
        let mlaBytes = 1 * maxContext * 128 * 2
        // AttnRes: 9 blocks x 64 fp16 + 64 fp16 prefix.
        let attnResBytes = 9 * 64 * 2 + 64 * 2

        let report = state.memoryReport()
        #expect(report.kdaStateBytes == kdaStateBytes)
        #expect(report.kdaConvBytes == kdaConvBytes)
        #expect(report.mlaCacheBytes == mlaBytes)
        #expect(report.attnResBytes == attnResBytes)
        #expect(report.totalBytes
                == kdaStateBytes + kdaConvBytes + mlaBytes + attnResBytes)

        // maxContext scaling: doubling capacity doubles only the MLA cache.
        let bigger = try K3State(device: device, config: config, maxContext: 32)
        #expect(bigger.memoryReport().mlaCacheBytes == 2 * mlaBytes)
        #expect(bigger.memoryReport().kdaStateBytes == kdaStateBytes)
    }

    @Test func perLayerOffsetsAreOrdinalMajor() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let config = Self.tinyConfig()
        let state = try K3State(device: device, config: config, maxContext: 8)

        // 0-based layers 0 and 2 are KDA (ordinals 0, 1); layer 1 is MLA.
        let kda0 = state.kdaRecurrentState(layer0: 0)
        let kda2 = state.kdaRecurrentState(layer0: 2)
        #expect(kda0.offset == 0)
        #expect(kda2.offset == 2 * 32 * 32 * 4)
        #expect(kda0.buffer === kda2.buffer)

        let conv0 = state.kdaConvState(layer0: 0)
        let conv2 = state.kdaConvState(layer0: 2)
        #expect(conv0.offset == 0)
        #expect(conv2.offset == 3 * 64 * 3 * 4)

        let mla1 = state.mlaCache(layer0: 1)
        #expect(mla1.offset == 0)
        #expect(state.mlaCacheRowBytes == 128 * 2)

        #expect(state.attnResBlocks.length == 9 * 64 * 2)
        #expect(state.attnResPrefix.length == 64 * 2)
    }

    @Test func resetZeroesAndReusesBuffers() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let config = Self.tinyConfig()
        let state = try K3State(device: device, config: config, maxContext: 16)

        // Dirty the state slabs (KDA recurrent + conv) — the slabs whose
        // contents must read zero after reset().
        let kdaState = state.kdaRecurrentState(layer0: 0)
        let kdaConv = state.kdaConvState(layer0: 0)
        let zeroedBuffers: [(MTLBuffer, Int)] = [
            (kdaState.buffer, kdaState.offset),
            (kdaConv.buffer, kdaConv.offset),
        ]
        for (buffer, offset) in zeroedBuffers {
            memset(buffer.contents().advanced(by: offset), 0xAB,
                   min(buffer.length - offset, 4 * 1024))
        }
        // The counter-guarded slabs (MLA cache, AttnRes) follow the house
        // KV-cache rule: stale bytes are never read past position/blockCount.
        state.advance(by: 7)
        state.beginToken()
        state.recordAttnResBoundary()
        #expect(state.position == 7)
        #expect(state.attnResBlockCount == 1)

        state.reset()
        #expect(state.position == 0)
        #expect(state.attnResBlockCount == 0)

        // Same buffer objects, zeroed state contents.
        let after = state.kdaRecurrentState(layer0: 0)
        #expect(after.buffer === kdaState.buffer)
        #expect(state.mlaCache(layer0: 1).buffer.length > 0)
        #expect(state.attnResBlocks.length == 9 * 64 * 2)
        for (buffer, offset) in zeroedBuffers {
            let probe = min(buffer.length - offset, 4 * 1024)
            let bytes = UnsafeRawBufferPointer(
                start: buffer.contents().advanced(by: offset), count: probe)
            #expect(bytes.allSatisfy { $0 == 0 },
                    "state buffer not zeroed after reset()")
        }
    }

    @Test func positionAndBoundaryAccounting() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let config = Self.tinyConfig()
        let state = try K3State(device: device, config: config, maxContext: 4)
        #expect(state.position == 0)
        state.advance()
        state.advance(by: 2)
        #expect(state.position == 3)
        state.beginToken()
        state.recordAttnResBoundary()
        #expect(state.attnResBlockCount == 1)
        state.beginToken()
        #expect(state.attnResBlockCount == 0)
    }

    @Test func snapshotRestoresAllPersistentAndScratchState() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let config = Self.tinyConfig()
        let state = try K3State(device: device, config: config, maxContext: 8)

        func fill(_ buffer: MTLBuffer, offset: Int = 0,
                  count: Int? = nil, byte: UInt8) {
            memset(buffer.contents().advanced(by: offset), Int32(byte),
                   count ?? (buffer.length - offset))
        }
        let kda0 = state.kdaRecurrentState(layer0: 0)
        let conv2 = state.kdaConvState(layer0: 2)
        fill(kda0.buffer, byte: 0x11)
        fill(conv2.buffer, byte: 0x22)
        state.advance(by: 3)
        let mla = state.mlaCache(layer0: 1)
        let activeMLABytes = state.position * state.mlaCacheRowBytes
        fill(mla.buffer, offset: mla.offset, count: activeMLABytes, byte: 0x33)
        fill(state.attnResBlocks, byte: 0x44)
        fill(state.attnResPrefix, byte: 0x55)
        state.beginToken()
        state.recordAttnResBoundary()

        let snapshot = try state.captureSnapshot()
        #expect(snapshot.position == 3)
        #expect(snapshot.attnResBlockCount == 1)
        #expect(snapshot.mlaActiveBytesPerLayer == activeMLABytes)

        state.reset()
        fill(mla.buffer, offset: mla.offset, count: activeMLABytes, byte: 0xEE)
        fill(state.attnResBlocks, byte: 0xEE)
        fill(state.attnResPrefix, byte: 0xEE)
        state.restoreSnapshot(snapshot)

        #expect(state.position == 3)
        #expect(state.attnResBlockCount == 1)
        #expect(UnsafeRawBufferPointer(start: kda0.buffer.contents(),
                                      count: kda0.buffer.length)
            .allSatisfy { $0 == 0x11 })
        #expect(UnsafeRawBufferPointer(start: conv2.buffer.contents(),
                                      count: conv2.buffer.length)
            .allSatisfy { $0 == 0x22 })
        #expect(UnsafeRawBufferPointer(
            start: mla.buffer.contents().advanced(by: mla.offset),
            count: activeMLABytes).allSatisfy { $0 == 0x33 })
        #expect(UnsafeRawBufferPointer(start: state.attnResBlocks.contents(),
                                      count: state.attnResBlocks.length)
            .allSatisfy { $0 == 0x44 })
        #expect(UnsafeRawBufferPointer(start: state.attnResPrefix.contents(),
                                      count: state.attnResPrefix.length)
            .allSatisfy { $0 == 0x55 })
    }

    @Test func canonicalSlabSizes() throws {
        // Pin the canonical K3 state footprint without allocating it:
        // 69 KDA layers x 96x128x128 fp32, 24 MLA layers x 576 fp16/token.
        let c = K3ArchConfig.kimiK3
        #expect(c.kdaLayers0.count == 69)
        #expect(c.mlaLayers0.count == 24)
        #expect(c.kdaStateElementsPerLayer == 96 * 128 * 128)
        #expect(c.kdaConvStateElementsPerLayer == 3 * 12_288 * 3)
        #expect(c.mlaCacheRowElements == 576)
        #expect(c.attnResBoundaryCount == 8)
        #expect(c.mlaLayers0.contains(3) && c.mlaLayers0.contains(91)
                && c.mlaLayers0.contains(92) && !c.mlaLayers0.contains(0))
        #expect(c.isDense(layer0: 0) && c.isKDA(layer0: 0))
        let stateBytes = 69 * 96 * 128 * 128 * 4
        #expect(stateBytes == 434_110_464)
        let kvPerToken = 24 * 576 * 2
        #expect(kvPerToken == 27_648)  // 27.6 KB/token per the evaluation doc
    }
}
