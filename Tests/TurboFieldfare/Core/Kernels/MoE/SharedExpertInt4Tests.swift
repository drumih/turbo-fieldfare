import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct SharedExpertInt4Tests {
    private static let d = 128
    private static let f = 64

    @Test func sharedExpertInt4MatchesAffineReference() throws {
        var rng = SeedTree(0x604).key("shared-expert-int4")
        let x = (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<Self.f).map { _ in (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) } }
        let up = (0..<Self.f).map { _ in (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<Self.d).map { _ in (0..<Self.f).map { _ in rng.uniform(-0.4, 0.4) } }
        let gatePack = Self.pack(gate)
        let upPack = Self.pack(up)
        let downPack = Self.pack(down)
        let x16 = x.map { Float(Float16($0)) }
        let gateOut = DequantInt4GemvRef.apply(weightRows: gatePack.rows, x: x16, n: Self.d)
        let upOut = DequantInt4GemvRef.apply(weightRows: upPack.rows, x: x16, n: Self.d)
        let act = zip(gateOut, upOut).map { gateValue, upValue in
            let cube = gateValue * gateValue * gateValue
            let inner = 0.7978845608028654 * Double(gateValue + 0.044715 * cube)
            return Float(Float16(Float(0.5 * Double(gateValue) * (1 + tanh(inner))) * upValue))
        }
        let reference = DequantInt4GemvRef.apply(weightRows: downPack.rows, x: act, n: Self.f)

        let context = try MetalContext()
        let runtime = try SharedExpertInt4(context: context)
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let yBuffer = try #require(Fp16Buffer.make(context.device, count: Self.d))
        let gateScratch = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let upScratch = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let actScratch = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try runtime.encode(commandBuffer: commandBuffer,
                           x: xBuffer,
                           gate: Self.projection(context, gatePack, rows: Self.f, cols: Self.d),
                           up: Self.projection(context, upPack, rows: Self.f, cols: Self.d),
                           down: Self.projection(context, downPack, rows: Self.d, cols: Self.f),
                           y: yBuffer,
                           scratchGate: gateScratch,
                           scratchUp: upScratch,
                           scratchAct: actScratch)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let actual = Fp16Buffer.read(yBuffer, count: Self.d)
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.quantInt4 * 4, "shared-expert int4 rel=\(error)")
    }

    private static func pack(_ values: [[Float]]) ->
        (rows: [Quantization.Int4AffineRow], packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let rows = values.map(Quantization.quantizeInt4Affine)
        return (rows,
                rows.flatMap(\.packed),
                rows.flatMap(\.scales),
                rows.flatMap(\.biases))
    }

    private static func projection(
        _ context: MetalContext,
        _ packed: (rows: [Quantization.Int4AffineRow], packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: context.device.makeBuffer(bytes: packed.packed,
                                                length: packed.packed.count,
                                                options: .storageModeShared)!,
            scales: context.device.makeBuffer(bytes: packed.scales,
                                               length: packed.scales.count * 2,
                                               options: .storageModeShared)!,
            biases: context.device.makeBuffer(bytes: packed.biases,
                                               length: packed.biases.count * 2,
                                               options: .storageModeShared)!,
            rows: UInt32(rows), cols: UInt32(cols))
    }
}
