import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// How expert-read throughput scales with concurrency: three reads at a time
/// (today) against all thirty of a layer's matrices at once.
@Suite struct GGUFReadScalingTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func scalesTheReads() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        var ranges: [GGUFTensorRange] = []
        for expert in 0..<10 {
            let weights = try GGUFExpertLayout.weights(layer: 0, expert: expert, in: model)
            ranges.append(weights.gate)
            ranges.append(weights.up)
            ranges.append(weights.down)
        }
        let bytes = ranges.reduce(0) { $0 + Int($1.byteCount) }
        let url: URL = model.shards[ranges[0].shardIndex].url
        guard let buffer = context.device.makeBuffer(
            length: 1 << 20, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("read scaling") }

        func read(_ range: GGUFTensorRange) {
            let fd = open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { return }
            defer { close(fd) }
            var total = 0
            while total < Int(range.byteCount) {
                let count = min(Int(range.byteCount) - total, 1 << 20)
                let got = pread(
                    fd, buffer.contents(), count, off_t(range.fileOffset) + off_t(total))
                if got <= 0 { break }
                total += got
            }
        }

        func measure(concurrency: Int, rounds: Int) -> Double {
            var best = Double.infinity
            for _ in 0..<rounds {
                let start = DispatchTime.now().uptimeNanoseconds
                var index = 0
                let lock = NSLock()
                DispatchQueue.concurrentPerform(iterations: concurrency) { _ in
                    while true {
                        lock.lock()
                        let next = index
                        index += 1
                        lock.unlock()
                        if next >= ranges.count { break }
                        read(ranges[next])
                    }
                }
                let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                best = min(best, seconds)
            }
            return best
        }

        for concurrency in [3, 10, 30] {
            _ = measure(concurrency: concurrency, rounds: 1)
            let seconds = measure(concurrency: concurrency, rounds: 4)
            let rate = Double(bytes) / seconds / 1e9
            print(
                "\(concurrency) readers: \(String(format: "%.2f", seconds * 1000)) ms, "
                    + "\(String(format: "%.2f", rate)) GB/s")
        }
        #expect(bytes > 0)
    }
}
