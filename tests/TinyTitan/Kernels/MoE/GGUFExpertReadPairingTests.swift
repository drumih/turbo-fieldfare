import Foundation
import Metal
import Testing

@testable import TinyTitan

/// Paired, interleaved comparison of the two ways to get an expert's bytes in
/// front of the GPU: map the slice and let the pages fault in, or pread it into
/// a reused buffer. Same ranges, same order, alternating within each round, so
/// machine drift cancels in the paired difference.
@Suite struct GGUFExpertReadPairingTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func pairsMappedAgainstPreadReads() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let mapped = try GGUFMappedShards(model: model, device: context.device)

        // Forty (layer, expert) pairs spread over the stack, each read twice per
        // round: once mapped, once pread.
        var pairs: [(Int, Int)] = []
        for layer in stride(from: 0, to: 48, by: 12) {
            for expert in [7, 311] { pairs.append((layer, expert)) }
        }
        var ranges: [GGUFTensorRange] = []
        for (layer, expert) in pairs {
            let weights = try GGUFExpertLayout.weights(layer: layer, expert: expert, in: model)
            ranges.append(weights.gate)
            ranges.append(weights.up)
            ranges.append(weights.down)
        }

        func checksum(_ bytes: UnsafeRawPointer, _ count: Int) -> UInt64 {
            var hash: UInt64 = 1469598103934665603
            let pointer = bytes.assumingMemoryBound(to: UInt8.self)
            var index = 0
            while index < count {
                hash = (hash ^ UInt64(pointer[index])) &* 1099511628211
                index += 4096
            }
            return hash
        }

        func readMapped() throws -> (UInt64, Int) {
            var hash: UInt64 = 0
            var bytes = 0
            for range in ranges {
                let url = model.shards[range.shardIndex].url
                let weight = try mapped.weight(
                    key: "pair.\(range.fileOffset)", url: url,
                    fileOffset: range.fileOffset, byteCount: range.byteCount,
                    type: range.type, rows: 1, columns: 1)
                // Touch every page: the fault-in the GPU would pay.
                var offset = 0
                var local: UInt64 = 0
                while offset < Int(range.byteCount) {
                    local ^= UInt64(
                        weight.buffer
                            .contents().advanced(by: weight.offset + offset)
                            .assumingMemoryBound(to: UInt8.self).pointee)
                    offset += 4096
                }
                hash ^= local
                bytes += Int(range.byteCount)
            }
            return (hash, bytes)
        }

        let fileBuffer = try #require(
            context.device.makeBuffer(length: 1 << 20, options: .storageModeShared))
        func readPread() throws -> (UInt64, Int) {
            var hash: UInt64 = 0
            var bytes = 0
            for range in ranges {
                let fd = open(model.shards[range.shardIndex].url.path, O_RDONLY | O_CLOEXEC)
                guard fd >= 0 else { throw MetalError.invalidState("open failed") }
                defer { close(fd) }
                // Each chunk is read into the same 1 MB window; a range larger
                // than the window must not advance the destination pointer past
                // it (that was a memory overrun when this first ran).
                var total = 0
                while total < Int(range.byteCount) {
                    let count = min(Int(range.byteCount) - total, 1 << 20)
                    let read = pread(
                        fd, fileBuffer.contents(), count,
                        off_t(range.fileOffset) + off_t(total))
                    if read <= 0 { break }
                    total += read
                    hash ^= checksum(fileBuffer.contents(), read)
                }
                bytes += total
            }
            return (hash, bytes)
        }

        // Noise floor: the same arm twice.
        _ = try readMapped()
        _ = try readPread()
        var mappedTimes: [Double] = []
        var preadTimes: [Double] = []
        var mappedHashes = Set<UInt64>()
        var preadHashes = Set<UInt64>()
        var bytes = 0
        for round in 0..<3 {
            let startA = DispatchTime.now().uptimeNanoseconds
            let (hashA, countA) = try readMapped()
            mappedTimes.append(Double(DispatchTime.now().uptimeNanoseconds - startA) / 1e9)
            mappedHashes.insert(hashA)
            bytes = countA
            let startB = DispatchTime.now().uptimeNanoseconds
            let (hashB, _) = try readPread()
            preadTimes.append(Double(DispatchTime.now().uptimeNanoseconds - startB) / 1e9)
            preadHashes.insert(hashB)
            print(
                "round \(round): mapped \(String(format: "%.0f", mappedTimes[round] * 1000)) ms, "
                    + "pread \(String(format: "%.0f", preadTimes[round] * 1000)) ms, "
                    + "paired delta \(String(format: "%+.0f", (mappedTimes[round] - preadTimes[round]) * 1000)) ms"
            )
        }
        // Output invariance: the two arms must have touched the same bytes.
        #expect(mappedHashes.count == 1)
        #expect(preadHashes.count == 1)
        print("bytes per pass: \(bytes)")
        let mappedRate = Double(bytes * 3) / mappedTimes.reduce(0, +) / 1e9
        let preadRate = Double(bytes * 3) / preadTimes.reduce(0, +) / 1e9
        print(
            "mapped \(String(format: "%.2f", mappedRate)) GB/s, "
                + "pread \(String(format: "%.2f", preadRate)) GB/s")
    }
}
