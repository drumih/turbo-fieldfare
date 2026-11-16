import Foundation
import Metal
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

@Suite struct BatchedDecodeScratchTests {
    private let config = ArchConfig.qwen36_35B_A3B

    @Test(arguments: [2, 4])
    func allocatesOneBufferSetPerWidth(rows: Int) throws {
        let context = try MetalContext()
        let scratch = try BatchedDecodeScratch.allocate(
            device: context.device, config: config, rows: rows)
        #expect(scratch.rows == rows)
        let buffers = [
            scratch.hidden, scratch.normed, scratch.q, scratch.k, scratch.v,
            scratch.attn, scratch.o, scratch.routedX, scratch.h1, scratch.h2,
            scratch.routeIDs, scratch.routeWeights, scratch.nextIDs,
            scratch.nextWeights,
        ]
        #expect(Set(buffers.map(ObjectIdentifier.init)).count == buffers.count)
        let d = config.hiddenSize
        #expect(scratch.hidden.length == rows * d * 2)
        #expect(scratch.routeIDs.length == rows * config.topKExperts * 4)
        #expect(scratch.routeWeights.length == rows * config.topKExperts * 2)
    }

    @Test func rejectsWidthOne() throws {
        let context = try MetalContext()
        #expect(throws: (any Error).self) {
            _ = try BatchedDecodeScratch.allocate(
                device: context.device, config: config, rows: 1)
        }
    }
}
