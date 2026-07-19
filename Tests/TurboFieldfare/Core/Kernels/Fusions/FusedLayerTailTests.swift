import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct FusedLayerTailTests {
    private static let d = ArchConfig.gemma4_26B_A4B.hiddenSize
    private static let eps: Float = 1e-6

    @Test func fusedLayerTail_matchesCpuReferenceTolerance_realShape() throws {
        var rng = SplitMix64(seed: 0xC0DE_7A11)
        let h2 = (0..<Self.d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let h1 = (0..<Self.d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let hidden = (0..<Self.d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let wPostFFN2 = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPostFFN = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let layerScalar = rng.uniform(0.75, 1.25)

        let ctx = try MetalContext()
        let fused = try FusedLayerTail(context: ctx)
        guard
            let h2Buf = ctx.device.makeBuffer(bytes: h2, length: Self.bytes(h2.count), options: .storageModeShared),
            let h1Buf = ctx.device.makeBuffer(bytes: h1, length: Self.bytes(h1.count), options: .storageModeShared),
            let hiddenBuf = ctx.device.makeBuffer(bytes: hidden, length: Self.bytes(hidden.count), options: .storageModeShared),
            let w2 = ctx.device.makeBuffer(bytes: wPostFFN2,
                                           length: wPostFFN2.count * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared),
            let w = ctx.device.makeBuffer(bytes: wPostFFN,
                                          length: wPostFFN.count * MemoryLayout<UInt16>.size,
                                          options: .storageModeShared)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }

        guard let commandBuffer = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to create command buffer")
            return
        }
        fused.encode(commandBuffer: commandBuffer,
                     h2: h2Buf,
                     h1: h1Buf,
                     hidden: hiddenBuf,
                     postFFN2Weight: w2,
                     postFFNWeight: w,
                     d: UInt32(Self.d),
                     eps: Self.eps,
                     layerScalar: layerScalar)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let ref = Self.cpuReference(h2: h2,
                               h1: h1,
                               hidden: hidden,
                               wPostFFN2: wPostFFN2,
                               wPostFFN: wPostFFN,
                               layerScalar: layerScalar)
        var maxAbs: Float = 0
        var refNorm: Float = 0
        for i in 0..<Self.d {
            let got = Float(hiddenBuf.contents().load(fromByteOffset: i * MemoryLayout<Float16>.size,
                                                      as: Float16.self))
            maxAbs = max(maxAbs, abs(got - ref[i]))
            refNorm = max(refNorm, abs(ref[i]))
        }
        let rel = maxAbs / max(refNorm, 1e-6)
        #expect(rel < 4e-3, "rel=\(rel) maxAbs=\(maxAbs) refNorm=\(refNorm)")
    }

    private static func cpuReference(h2: [Float16],
                                     h1: [Float16],
                                     hidden: [Float16],
                                     wPostFFN2: [UInt16],
                                     wPostFFN: [UInt16],
                                     layerScalar: Float) -> [Float] {
        let w2 = wPostFFN2.map { Quantization.bf16ToFloat($0) }
        let w = wPostFFN.map { Quantization.bf16ToFloat($0) }
        let h2Norm = RmsNormRef.apply(x: h2.map(Float.init), weight: w2, eps: eps)
            .map { Float(Float16($0)) }
        let h12 = zip(h1, h2Norm).map { Float(Float16(Float($0.0) + $0.1)) }
        let h12Norm = RmsNormRef.apply(x: h12, weight: w, eps: eps)
            .map { Float(Float16($0)) }
        let hScale = Float(Float16(layerScalar))
        return zip(hidden, h12Norm).map { Float(Float16(Float(Float16(Float($0.0) + $0.1)) * hScale)) }
    }

    private static func bytes(_ count: Int) -> Int {
        count * MemoryLayout<Float16>.size
    }

}
