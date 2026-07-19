import Testing

@testable import TurboFieldfare

extension TurboQuantTests {
    @Test func fusedBulkWriteMatchesRepeatedScalarWrites() throws {
        try Self.runBulkWriteMatchesRepeatedScalarWrites()
    }

    @Test func fusedBulkWriteUsesDestinationTokenBase() throws {
        try Self.runBulkWriteMatchesRepeatedScalarWrites(dstTokenBase: 2)
    }

    @Test func bulkWriteRejectsOutOfRangeTokenCount() throws {
        let d = 256
        let numHeads = 2
        let layout = TurboQuantKVLayout.role(mode: .k4v4NormCorrected,
                                             headDim: d,
                                             numKVHeads: numHeads)
        let context = try Self.makeContext()
        let quant = try TurboQuantQuant(context: context)
        let commandBuffer = context.queue.makeCommandBuffer()!
        let input = Self.makeBuffer(context, count: numHeads * d, type: Float16.self)
        let cache = Self.makeBuffer(context, count: layout.bytesPerToken, type: UInt8.self)
        let write = TurboQuantKVWriteParams(d: UInt32(d),
                                            numHeads: UInt32(numHeads),
                                            roleLayout: layout)
        let params = TurboQuantKVBulkWriteParams(kv: write,
                                                 tokenCount: 2,
                                                 dstTokenBase: 0,
                                                 cacheTokenCapacity: 1,
                                                 sourceTokenStrideElements: numHeads * d)

        #expect(throws: TurboQuantKVBulkWriteError.self) {
            try quant.encodeKVWriteWHTBulk(commandBuffer: commandBuffer,
                                           x: input,
                                           cache: cache,
                                           params: params,
                                           whtParams: TurboQuantWHTParams())
        }
    }
}
