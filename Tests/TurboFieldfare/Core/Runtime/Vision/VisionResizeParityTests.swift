import Foundation
import Testing
@testable import TurboFieldfare

/// The GPU resize exists to make the reference filter affordable, so it has to
/// be the same filter — not a close one. Every case here asserts byte equality
/// with `TorchBicubicResize`, which is the path validated against the pinned
/// image processor.
@Suite struct VisionResizeParityTests {
    private func pattern(width: Int, height: Int, seed: UInt64) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        var state = seed &* 6_364_136_223_846_793_005 &+ 1
        for index in pixels.indices {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            // High-frequency content: a smooth ramp hides exactly the
            // differences this test is looking for.
            pixels[index] = UInt8(truncatingIfNeeded: state >> 33)
        }
        return pixels
    }

    private func cpuResize(_ source: [UInt8], _ sw: Int, _ sh: Int,
                           _ dw: Int, _ dh: Int) -> [UInt8] {
        var destination = [UInt8](repeating: 0, count: dw * dh * 4)
        source.withUnsafeBufferPointer { input in
            destination.withUnsafeMutableBufferPointer { output in
                _ = TorchBicubicResize.resize(
                    source: input.baseAddress!, sourceWidth: sw, sourceHeight: sh,
                    sourceRowBytes: sw * 4,
                    destination: output.baseAddress!,
                    destinationWidth: dw, destinationHeight: dh,
                    destinationRowBytes: dw * 4)
            }
        }
        return destination
    }

    private func gpuResize(_ resizer: VisionResize, _ source: [UInt8],
                           _ sw: Int, _ sh: Int, _ dw: Int, _ dh: Int) throws -> [UInt8] {
        var destination = [UInt8](repeating: 0, count: dw * dh * 4)
        try source.withUnsafeBufferPointer { input in
            try destination.withUnsafeMutableBufferPointer { output in
                try resizer.resize(
                    source: input.baseAddress!, sourceWidth: sw, sourceHeight: sh,
                    sourceRowBytes: sw * 4,
                    destination: output.baseAddress!,
                    destinationWidth: dw, destinationHeight: dh,
                    destinationRowBytes: dw * 4)
            }
        }
        return destination
    }

    @Test func theGPUFilterIsByteIdenticalToTheCPUOracle() throws {
        let context = try MetalContext()
        let resizer = try VisionResize(context: context)

        // Downscales of several ratios, an upscale, a one-axis change, and the
        // identity — the shapes a real corpus produces.
        let cases: [(Int, Int, Int, Int)] = [
            (640, 480, 320, 240),
            (1_920, 1_080, 720, 405),
            (2_484, 3_000, 720, 864),
            (333, 217, 111, 73),
            (64, 64, 192, 192),
            (500, 300, 500, 120),
            (256, 256, 256, 256),
        ]
        for (sw, sh, dw, dh) in cases {
            let source = pattern(width: sw, height: sh, seed: UInt64(sw &* 31 &+ sh))
            let expected = cpuResize(source, sw, sh, dw, dh)
            let actual = try gpuResize(resizer, source, sw, sh, dw, dh)
            #expect(actual == expected,
                    "\(sw)x\(sh) -> \(dw)x\(dh) differs from the CPU filter")
            if actual != expected {
                let differing = zip(actual, expected).filter { $0 != $1 }.count
                let worst = zip(actual, expected)
                    .map { abs(Int($0) - Int($1)) }.max() ?? 0
                Issue.record("\(differing) bytes differ, worst by \(worst)")
            }
        }
    }

    /// A tall, thin reduction exercises the vertical pass alone, where the CPU
    /// walks bytes rather than pixels; an off-by-one in that indexing would
    /// otherwise show up only as a colour shift on real images.
    @Test func singleAxisReductionsMatch() throws {
        let context = try MetalContext()
        let resizer = try VisionResize(context: context)
        for (sw, sh, dw, dh) in [(97, 1_000, 97, 137), (1_000, 97, 137, 97)] {
            let source = pattern(width: sw, height: sh, seed: 7)
            #expect(try gpuResize(resizer, source, sw, sh, dw, dh)
                    == cpuResize(source, sw, sh, dw, dh),
                    "\(sw)x\(sh) -> \(dw)x\(dh) differs")
        }
    }
}
