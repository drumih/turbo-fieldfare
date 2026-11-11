import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The F32 router: the selection rule on explicit logits, and the GEMV that
/// produces them without rounding the F32 weights or the logits.
@Suite struct GGUFRouterSelectionTests {
    @Test func selectsTheLargestProbabilitiesAndRenormalises() {
        let logits: [Float] = [0.1, 2.0, -1.0, 1.5, 2.0, 0.0]
        let selection = GGUFRouter.select(logits: logits, topK: 3)
        // 1 and 4 tie; the lower index ranks first.
        #expect(selection.experts == [1, 4, 3])
        #expect(abs(selection.weights.reduce(0, +) - 1) < 1e-6)
        #expect(selection.weights[0] == selection.weights[1])
        #expect(selection.weights[2] < selection.weights[0])
    }

    @Test func weightsAreTheRenormalisedSoftmax() {
        let logits: [Float] = [1, 2, 3]
        let selection = GGUFRouter.select(logits: logits, topK: 2)
        let e3 = exp(Float(3)), e2 = exp(Float(2))
        #expect(selection.experts == [2, 1])
        #expect(abs(selection.weights[0] - e3 / (e3 + e2)) < 1e-6)
        #expect(abs(selection.weights[1] - e2 / (e3 + e2)) < 1e-6)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func f32GEMVMatchesADoublePrecisionDot() throws {
        let context = try MetalContext()
        let pipeline = try context.pipeline("gguf_f32_gemv_rows")
        let rows = 64, columns = 2_560
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Double(state >> 11) / Double(1 << 53)) * 2 - 1
        }
        let weights = (0..<(rows * columns)).map { _ in next() * 0.05 }
        let x = (0..<columns).map { _ in Float16(next()) }
        let device = context.device
        let w = try #require(
            device.makeBuffer(
                bytes: weights, length: weights.count * 4, options: .storageModeShared))
        let xb = try #require(
            device.makeBuffer(
                bytes: x.map(\.bitPattern), length: x.count * 2, options: .storageModeShared))
        let y = try #require(device.makeBuffer(length: rows * 4, options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(w, offset: 0, index: 0)
        encoder.setBuffer(xb, offset: 0, index: 1)
        encoder.setBuffer(y, offset: 0, index: 2)
        var c = UInt32(columns), r = UInt32(rows)
        encoder.setBytes(&c, length: 4, index: 3)
        encoder.setBytes(&r, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let out = y.contents().assumingMemoryBound(to: Float.self)
        var worst = 0.0
        for row in 0..<rows {
            var reference = 0.0
            for column in 0..<columns {
                reference += Double(weights[row * columns + column]) * Double(Float(x[column]))
            }
            worst = max(worst, abs(Double(out[row]) - reference))
        }
        // F32 accumulation over 2,560 terms: far below the half rounding
        // (~1e-3 at this magnitude) the old router's logits carried.
        #expect(worst < 1e-4, "worst absolute error \(worst)")
    }
}
