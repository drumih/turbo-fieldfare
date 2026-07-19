import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Validates the standalone 8-bit dense MLP wrapper used by Gemma 4 as the
/// "shared expert" parallel branch of its MoE:
///
///     y = down(gelu_pytorch_tanh(gate(x)) * up(x))
///
/// Reference composes DequantInt8GemvRef + gelu_pytorch_tanh + vDSP_vmul,
/// matching the math but with a different staging (bulk dequant rows, then
/// dotpr) so a bug in either side won't replicate.
@Suite struct SharedExpertInt8Tests {

    private struct Sizes {
        static let D = 128
        static let F = 64
    }

    private static func packMatrix(_ rows: [[Float]])
        -> (rows: [Quantization.Int8AffineRow],
            packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let M = rows.count
        let N = rows[0].count
        let gpr = N / Quantization.groupSize
        var packed = [UInt8]( repeating: 0, count: M * N)
        var scales = [UInt16](repeating: 0, count: M * gpr)
        var biases = [UInt16](repeating: 0, count: M * gpr)
        var rowsOut: [Quantization.Int8AffineRow] = []
        rowsOut.reserveCapacity(M)
        for m in 0..<M {
            let q = Quantization.quantizeInt8Affine(rows[m])
            for i in 0..<N { packed[m * N + i] = q.packed[i] }
            for g in 0..<gpr {
                scales[m * gpr + g] = q.scales[g]
                biases[m * gpr + g] = q.biases[g]
            }
            rowsOut.append(q)
        }
        return (rowsOut, packed, scales, biases)
    }

    @Test func sharedExpertInt8_matchesReference_swaShape() throws {
        try Self.runSharedExpertInt8MatchesReference()
    }

    private static func runSharedExpertInt8MatchesReference() throws {
        var rng = SeedTree(0x601).key("shared-expert-int8")
        let xFp32 = (0..<Sizes.D).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<Sizes.F).map { _ in (0..<Sizes.D).map { _ in rng.uniform(-0.4, 0.4) } }
        let up   = (0..<Sizes.F).map { _ in (0..<Sizes.D).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<Sizes.D).map { _ in (0..<Sizes.F).map { _ in rng.uniform(-0.4, 0.4) } }

        // Reference: dense MLP applied to x.
        let xFp16 = xFp32.map { Float(Float16($0)) }
        let gatePack = Self.packMatrix(gate)
        let upPack   = Self.packMatrix(up)
        let downPack = Self.packMatrix(down)
        let gateOut = DequantInt8GemvRef.apply(weightRows: gatePack.rows, x: xFp16, n: Sizes.D)
        let upOut   = DequantInt8GemvRef.apply(weightRows: upPack.rows,   x: xFp16, n: Sizes.D)
        // gelu_pytorch_tanh + multiply (round through FP16 once to match the kernel's storage).
        let act: [Float] = zip(gateOut, upOut).map { g, u in
            let x3 = g * g * g
            let inner = 0.7978845608028654 * Double(g + 0.044715 * x3)
            let gelu = 0.5 * Double(g) * (1.0 + tanh(inner))
            return Float(Float16(Float(gelu) * u))
        }
        let yRef = DequantInt8GemvRef.apply(weightRows: downPack.rows, x: act, n: Sizes.F)

        // Kernel
        let ctx = try MetalContext()
        let wrapper = try SharedExpertInt8(context: ctx)

        guard let xBuf = Fp16Buffer.make(ctx.device, values: xFp32),
              let yBuf = Fp16Buffer.make(ctx.device, count: Sizes.D),
              let sa   = Fp16Buffer.make(ctx.device, count: Sizes.F) else {
            Issue.record("alloc failed"); return
        }

        func pack(_ p: (rows: [Quantization.Int8AffineRow],
                        packed: [UInt8], scales: [UInt16], biases: [UInt16]),
                  rows: UInt32, cols: UInt32) -> SharedExpertInt8Proj {
            let wBuf = ctx.device.makeBuffer(bytes: p.packed,
                                             length: p.packed.count,
                                             options: .storageModeShared)!
            let sBuf = ctx.device.makeBuffer(bytes: p.scales,
                                             length: p.scales.count * 2,
                                             options: .storageModeShared)!
            let bBuf = ctx.device.makeBuffer(bytes: p.biases,
                                             length: p.biases.count * 2,
                                             options: .storageModeShared)!
            return SharedExpertInt8Proj(weights: wBuf, scales: sBuf, biases: bBuf,
                                        rows: rows, cols: cols)
        }
        let gateProj = pack(gatePack, rows: UInt32(Sizes.F), cols: UInt32(Sizes.D))
        let upProj   = pack(upPack,   rows: UInt32(Sizes.F), cols: UInt32(Sizes.D))
        let downProj = pack(downPack, rows: UInt32(Sizes.D), cols: UInt32(Sizes.F))

        let cb = ctx.queue.makeCommandBuffer()!
        try wrapper.encode(commandBuffer: cb,
                           x: xBuf, gate: gateProj, up: upProj, down: downProj,
                           y: yBuf,
                           scratchAct: sa)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(yBuf, count: Sizes.D)
        let rel = RelError.compute(actual: actual, reference: yRef)
        #expect(rel < Tolerance.quantInt8 * 4, "shared-expert int8 rel=\(rel)")
    }

}
