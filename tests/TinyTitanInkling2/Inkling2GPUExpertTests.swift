import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2GPUExpertTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func nvfp4ExpertMatchesCPU() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let store = try Inkling2ExpertStore(directory: directory)
        let packed = try store.packed(layer: 0, expert: 0)
        let cpu = try store.load(layer: 0, expert: 0)
        let x = (0..<2304).map { Float(($0 % 17) - 8) * 0.01 }
        let gpu = try Inkling2GPUExpert().gemv(
            packed, x: x, rows: 2304, columns: 2304, w13: true)
        for row in 0..<8 {
            var dot: Float = 0
            for column in 0..<2304 {
                dot += cpu.w13[row * 2304 + column] * x[column]
            }
            #expect(abs(gpu[row] - dot) < 1e-3)
        }
    }
}
