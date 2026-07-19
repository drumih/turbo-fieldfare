import Foundation
import Metal

struct TurboQuantKVWriteParams {
    var d: UInt32
    var numHeads: UInt32
    var bytesPerHead: UInt32
    var packedOffset: UInt32
    var scaleOffset: UInt32
    var tokenBase: UInt32
    var bytesPerToken: UInt32

    init(d: UInt32,
         numHeads: UInt32,
         roleLayout: TurboQuantKVRoleLayout,
         tokenBase: UInt32 = 0) {
        self.d = d
        self.numHeads = numHeads
        self.bytesPerHead = UInt32(roleLayout.bytesPerHead)
        self.packedOffset = UInt32(roleLayout.packedOffsetPerHead)
        self.scaleOffset = UInt32(roleLayout.scaleOffsetPerHead)
        self.tokenBase = tokenBase
        self.bytesPerToken = UInt32(roleLayout.bytesPerToken)
    }
}

enum TurboQuantKVBulkWriteError: Error, Equatable, CustomStringConvertible {
    case invalidTokenCount(Int)
    case invalidDestinationTokenBase(Int)
    case invalidCacheCapacity(Int)
    case rangeExceedsCache(start: Int, count: Int, capacity: Int)
    case unsupportedSourceTokenStride(actual: Int, expected: Int)

    public var description: String {
        switch self {
        case .invalidTokenCount(let count):
            return "invalid TurboQuant bulk token count \(count)"
        case .invalidDestinationTokenBase(let base):
            return "invalid TurboQuant bulk destination token base \(base)"
        case .invalidCacheCapacity(let capacity):
            return "invalid TurboQuant bulk cache capacity \(capacity)"
        case let .rangeExceedsCache(start, count, capacity):
            return "TurboQuant bulk write range start=\(start) count=\(count) exceeds capacity=\(capacity)"
        case let .unsupportedSourceTokenStride(actual, expected):
            return "TurboQuant bulk source token stride \(actual) does not match contiguous stride \(expected)"
        }
    }
}

struct TurboQuantKVBulkWriteParams {
    var kv: TurboQuantKVWriteParams
    var tokenCount: Int
    var dstTokenBase: Int
    var cacheTokenCapacity: Int
    var sourceTokenStrideElements: Int

    init(kv: TurboQuantKVWriteParams,
         tokenCount: Int,
         dstTokenBase: Int,
         cacheTokenCapacity: Int,
         sourceTokenStrideElements: Int) {
        self.kv = kv
        self.tokenCount = tokenCount
        self.dstTokenBase = dstTokenBase
        self.cacheTokenCapacity = cacheTokenCapacity
        self.sourceTokenStrideElements = sourceTokenStrideElements
    }
}

struct TurboQuantWHTParams {
    var numHeads: UInt32
    var layer: UInt32
    var rotationSeed: UInt32
    var applyRotation: UInt32

    init(numHeads: UInt32 = 1,
         layer: UInt32 = 0,
         rotationSeed: UInt32 = 0,
         applyRotation: Bool = false) {
        self.numHeads = numHeads
        self.layer = layer
        self.rotationSeed = rotationSeed
        self.applyRotation = applyRotation ? 1 : 0
    }
}

/// Packs pre-WHT FP16 K/V rows directly into the production TurboQuant cache
/// layout. One threadgroup handles each flattened `(token, head)` pair.
final class TurboQuantQuant {
    private struct Shape: Hashable {
        var d: UInt32
        var numHeads: UInt32
    }

    private let kvWriteWHTPSO: MTLComputePipelineState
    private let kvWriteWHTSpecializedPSOs: [Shape: MTLComputePipelineState]
    private static let realDecodeShapes: [Shape] = [
        Shape(d: 256, numHeads: 8),
        Shape(d: 512, numHeads: 2),
    ]

    init(context: MetalContext) throws {
        self.kvWriteWHTPSO = try context.pipeline("turboquant_quant_kv_write_wht")
        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "turboquant_quant_kv_write_wht",
                constants: [
                    MetalFunctionConstant(index: 100, value: .uint32(shape.d)),
                    MetalFunctionConstant(index: 101, value: .uint32(shape.numHeads)),
                    MetalFunctionConstant(index: 103, value: .bool(true)),
                ])
        }
        self.kvWriteWHTSpecializedPSOs = variants
    }

    func encodeKVWriteWHT(commandBuffer: MTLCommandBuffer,
                          x: MTLBuffer,
                          xOffset: Int = 0,
                          cache: MTLBuffer,
                          cacheOffset: Int = 0,
                          params: TurboQuantKVWriteParams,
                          whtParams: TurboQuantWHTParams,
                          pairs: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            kvWriteWHTSpecializedPSOs[Shape(d: params.d,
                                            numHeads: params.numHeads)] ?? kvWriteWHTPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(cache, offset: cacheOffset, index: 1)
        var writeParams = params
        var transformParams = whtParams
        encoder.setBytes(&writeParams,
                         length: MemoryLayout<TurboQuantKVWriteParams>.size,
                         index: 2)
        encoder.setBytes(&transformParams,
                         length: MemoryLayout<TurboQuantWHTParams>.size,
                         index: 3)
        let threads = Int(params.d)
        encoder.dispatchThreadgroups(
            MTLSize(width: pairs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeKVWriteWHTBulk(commandBuffer: MTLCommandBuffer,
                              x: MTLBuffer,
                              xOffset: Int = 0,
                              cache: MTLBuffer,
                              cacheOffset: Int = 0,
                              params: TurboQuantKVBulkWriteParams,
                              whtParams: TurboQuantWHTParams) throws {
        let writeParams = try validateBulkWriteParams(params)
        let destinationOffset = cacheOffset
            + params.dstTokenBase * Int(writeParams.bytesPerToken)
        encodeKVWriteWHT(commandBuffer: commandBuffer,
                         x: x,
                         xOffset: xOffset,
                         cache: cache,
                         cacheOffset: destinationOffset,
                         params: writeParams,
                         whtParams: whtParams,
                         pairs: params.tokenCount * Int(writeParams.numHeads))
    }

    private func validateBulkWriteParams(
        _ params: TurboQuantKVBulkWriteParams
    ) throws -> TurboQuantKVWriteParams {
        guard params.tokenCount > 0 else {
            throw TurboQuantKVBulkWriteError.invalidTokenCount(params.tokenCount)
        }
        guard params.dstTokenBase >= 0 else {
            throw TurboQuantKVBulkWriteError.invalidDestinationTokenBase(params.dstTokenBase)
        }
        guard params.cacheTokenCapacity >= 0 else {
            throw TurboQuantKVBulkWriteError.invalidCacheCapacity(params.cacheTokenCapacity)
        }
        guard params.dstTokenBase + params.tokenCount <= params.cacheTokenCapacity else {
            throw TurboQuantKVBulkWriteError.rangeExceedsCache(
                start: params.dstTokenBase,
                count: params.tokenCount,
                capacity: params.cacheTokenCapacity)
        }

        let expectedStride = Int(params.kv.numHeads) * Int(params.kv.d)
        guard params.sourceTokenStrideElements == expectedStride else {
            throw TurboQuantKVBulkWriteError.unsupportedSourceTokenStride(
                actual: params.sourceTokenStrideElements,
                expected: expectedStride)
        }

        var writeParams = params.kv
        writeParams.tokenBase = UInt32(params.dstTokenBase)
        return writeParams
    }
}
