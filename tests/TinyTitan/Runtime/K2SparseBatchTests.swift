import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
@testable import TinyTitanCore

@testable import TinyTitan

/// The chunked sparse layer against the token path: same weights, same K/V
/// cache positions, same MoVA and MoE per token. What is batched is the norm,
/// the q/k/gate/o projections, the softplus gate and the residuals; the parity
/// test is what says those batched stages compute the token path's numbers.
@Suite struct K2SparseBatchTests {
    private static func halves(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float16] {
        let base = buffer.contents()
        return (0..<count).map {
            base.load(
                fromByteOffset: offset + $0 * MemoryLayout<Float16>.stride,
                as: Float16.self)
        }
    }

    private static func copy(_ values: [Float16], into buffer: MTLBuffer, offset: Int = 0) {
        values.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                memcpy(
                    buffer.contents() + offset, base,
                    values.count * MemoryLayout<Float16>.stride)
            }
        }
    }

    private static func values(_ count: Int, salt: Int) -> [Float16] {
        (0..<count).map { index in
            Float16(Float((index * 31 + salt * 7) % 23) / 11 - 1)
        }
    }

    @Test func chunkMatchesTheTokenPath() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let d = config.hiddenSize
        let chunk = 4
        let layer = config.numLayers - 1
        let probe = try K2SparseLayerProbe(
            model: model, context: ctx, layer: layer, maxSequence: 8)
        let batch = try K2SparseBatch(
            model: model, context: ctx, layer: layer,
            maxSequence: 8, maxTokens: chunk)

        guard
            let chunkIn = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared),
            let chunkOut = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared)
        else {
            Issue.record("could not allocate the chunk buffers")
            return
        }
        let input = Self.values(chunk * d, salt: 3)
        Self.copy(input, into: chunkIn)

        var expected: [Float16] = []
        for index in 0..<chunk {
            let row = Array(input[index * d..<(index + 1) * d])
            let out = try await probe.run(
                queue: ctx.queue, hidden: row, position: index)
            expected += out.map { Float16($0) }
        }

        try await batch.encode(
            queue: ctx.queue, hidden: chunkIn, out: chunkOut,
            basePosition: 0, tokenCount: chunk)

        let actual = Self.halves(chunkOut, offset: 0, count: chunk * d)
            .map { Float($0) }
        let reference = expected.map { Float($0) }
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }
}
