import Foundation
import Metal
import Testing

@testable import TinyTitan

/// What a bare command-buffer round trip costs on this machine. The decoder
/// makes 96 of them a token, so if this is tens of milliseconds the waits - not
/// the kernels - are the token time.
@Suite struct GGUFCommandBufferOverheadTests {
    @Test func measuresTheEmptyRoundTrip() throws {
        let context = try MetalContext()
        guard let buffer = context.device.makeBuffer(length: 4096, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("overhead buffer") }
        let iterations = 96
        // Warm up so the first submission's driver setup is not counted.
        for _ in 0..<8 {
            guard let cb = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            cb.commit()
            cb.waitUntilCompleted()
        }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard let cb = context.queue.makeCommandBuffer(),
                let encoder = cb.makeComputeCommandEncoder()
            else { throw MetalError.commandEncoderFailed }
            // One trivial dispatch so the buffer is real work, not only a fence.
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let perBufferMs = seconds * 1000 / Double(iterations)
        print(
            "empty command buffer round trip: \(String(format: "%.2f", perBufferMs)) ms "
                + "(x96 = \(String(format: "%.0f", perBufferMs * 96)) ms/token)")
        #expect(perBufferMs > 0)
    }
}
