import Foundation
import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

@Suite struct TurboQuantTests {
    static func makeContext() throws -> MetalContext {
        try MetalContext()
    }

    static func makeBuffer<T>(_ context: MetalContext, count: Int, type: T.Type) -> MTLBuffer {
        context.device.makeBuffer(length: count * MemoryLayout<T>.stride,
                                  options: .storageModeShared)!
    }

    static func bufferWith<T>(_ context: MetalContext, _ values: [T]) -> MTLBuffer {
        let buffer = makeBuffer(context, count: values.count, type: T.self)
        values.withUnsafeBytes { source in
            if let base = source.baseAddress {
                memcpy(buffer.contents(), base, source.count)
            }
        }
        return buffer
    }

    static func runPackedRoundTrip(headDim: Int) throws {
        let numHeads = headDim == 256 ? 8 : 2
        let tokenCount = 2
        let layer: UInt32 = 3
        let rotationSeed: UInt32 = 0x51A7_1010
        var random = SeedTree(0x51A7_1010).key("packed-round-trip-d\(headDim)")
        let input = (0..<(tokenCount * numHeads * headDim)).map {
            _ in Float16(random.uniform(-1, 1))
        }
        let layout = TurboQuantKVLayout.role(mode: .k4v4NormCorrected,
                                             headDim: headDim,
                                             numKVHeads: numHeads)
        let context = try makeContext()
        let quant = try TurboQuantQuant(context: context)
        let inputBuffer = bufferWith(context, input)
        let cache = makeBuffer(context,
                               count: tokenCount * layout.bytesPerToken,
                               type: UInt8.self)
        let params = TurboQuantKVWriteParams(d: UInt32(headDim),
                                             numHeads: UInt32(numHeads),
                                             roleLayout: layout)
        let transform = TurboQuantWHTParams(numHeads: UInt32(numHeads),
                                            layer: layer,
                                            rotationSeed: rotationSeed,
                                            applyRotation: true)

        let commandBuffer = context.queue.makeCommandBuffer()!
        quant.encodeKVWriteWHT(commandBuffer: commandBuffer,
                               x: inputBuffer,
                               cache: cache,
                               params: params,
                               whtParams: transform,
                               pairs: tokenCount * numHeads)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = cache.contents().bindMemory(to: UInt8.self, capacity: cache.length)
        let cacheBytes = Array(UnsafeBufferPointer(start: pointer, count: cache.length))
        let actual = TurboQuantRef.materializePackedCache(
            cacheBytes,
            tokenCount: tokenCount,
            headDim: headDim,
            numHeads: numHeads,
            bytesPerHead: layout.bytesPerHead,
            bytesPerToken: layout.bytesPerToken,
            packedOffset: layout.packedOffsetPerHead,
            scaleOffset: layout.scaleOffsetPerHead,
            layer: layer,
            rotationSeed: rotationSeed,
            applyRotation: true)
        let expected = input.map(Float.init)
        let squaredError = zip(actual, expected).reduce(Float.zero) { result, pair in
            let error = pair.0 - pair.1
            return result + error * error
        }
        let squaredReference = expected.reduce(Float.zero) { $0 + $1 * $1 }
        #expect((squaredError / squaredReference).squareRoot() < 0.16)
    }

    static func runBulkWriteMatchesRepeatedScalarWrites(dstTokenBase: Int = 0) throws {
        let d = 256
        let numHeads = 2
        let tokenCount = 3
        let capacity = dstTokenBase + tokenCount + 1
        let layout = TurboQuantKVLayout.role(mode: .k4v4NormCorrected,
                                             headDim: d,
                                             numKVHeads: numHeads)
        var random = SeedTree(0xB117).key("tq-bulk-k4-base-\(dstTokenBase)")
        let input = (0..<(tokenCount * numHeads * d)).map {
            _ in Float16(random.uniform(-1, 1))
        }
        let context = try makeContext()
        let quant = try TurboQuantQuant(context: context)
        let inputBuffer = bufferWith(context, input)
        let scalar = makeBuffer(context,
                                count: capacity * layout.bytesPerToken,
                                type: UInt8.self)
        let bulk = makeBuffer(context,
                              count: capacity * layout.bytesPerToken,
                              type: UInt8.self)
        memset(scalar.contents(), 0xA5, scalar.length)
        memset(bulk.contents(), 0xA5, bulk.length)
        let params = TurboQuantKVWriteParams(d: UInt32(d),
                                             numHeads: UInt32(numHeads),
                                             roleLayout: layout)
        let bulkParams = TurboQuantKVBulkWriteParams(
            kv: params,
            tokenCount: tokenCount,
            dstTokenBase: dstTokenBase,
            cacheTokenCapacity: capacity,
            sourceTokenStrideElements: numHeads * d)

        let commandBuffer = context.queue.makeCommandBuffer()!
        for token in 0..<tokenCount {
            var tokenParams = params
            tokenParams.tokenBase = UInt32(dstTokenBase + token)
            quant.encodeKVWriteWHT(
                commandBuffer: commandBuffer,
                x: inputBuffer,
                xOffset: token * numHeads * d * MemoryLayout<Float16>.stride,
                cache: scalar,
                cacheOffset: (dstTokenBase + token) * layout.bytesPerToken,
                params: tokenParams,
                whtParams: TurboQuantWHTParams(),
                pairs: numHeads)
        }
        try quant.encodeKVWriteWHTBulk(commandBuffer: commandBuffer,
                                       x: inputBuffer,
                                       cache: bulk,
                                       params: bulkParams,
                                       whtParams: TurboQuantWHTParams())
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let lhs = scalar.contents().bindMemory(to: UInt8.self, capacity: scalar.length)
        let rhs = bulk.contents().bindMemory(to: UInt8.self, capacity: bulk.length)
        for index in 0..<scalar.length {
            #expect(lhs[index] == rhs[index], "bulk cache byte \(index)")
        }
    }
}
