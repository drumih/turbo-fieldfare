import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The reader replaces v3.x's one-at-a-time fetch, so what has to be pinned is
/// that parallelism never reorders or mixes up destinations: expert `i` must land
/// in destination `i` whatever order the workers happen to complete in. A
/// throughput regression is visible in benchmarks; a crossed destination would
/// silently compute one expert's weights against another's slot.
@Suite struct ParallelExpertReaderTests {
    /// Small stride so the fixtures stay cheap; the reader does not care.
    static let stride = 4096

    /// Builds a file of `count` blocks where block `e` is filled with a byte
    /// pattern derived from `e`, so a misdirected read is detectable.
    private static func makeFixture(count: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-io-\(UUID().uuidString).bin")
        var data = Data(capacity: count * stride)
        for expert in 0..<count {
            data.append(Data(repeating: UInt8(expert % 251), count: stride))
        }
        try data.write(to: url)
        return url
    }

    private static func withDestinations<T>(
        _ n: Int, _ body: ([UnsafeMutableRawPointer]) throws -> T
    ) rethrows -> T {
        let bufs = (0..<n).map { _ in
            UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 16384)
        }
        defer {
            for buffer in bufs { buffer.deallocate() }
        }
        return try body(bufs)
    }

    @Test func readsEachExpertIntoItsOwnDestination() throws {
        let count = 64
        let url = try Self.makeFixture(count: count)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 4)
        // Deliberately out of order and with a repeat, which is what routing
        // produces and what a naive index/offset mix-up would break.
        let wanted: [UInt32] = [63, 0, 17, 17, 40, 5, 62, 1]
        try Self.withDestinations(wanted.count) { bufs in
            try reader.fetch(experts: wanted, into: bufs)
            for (i, expert) in wanted.enumerated() {
                let expected = UInt8(Int(expert) % 251)
                let bytes = bufs[i].assumingMemoryBound(to: UInt8.self)
                #expect(
                    bytes[0] == expected,
                    "destination \(i) wanted expert \(expert)")
                #expect(
                    bytes[Self.stride - 1] == expected,
                    "destination \(i) tail mismatch for expert \(expert)")
            }
        }
    }

    @Test func singleThreadedMatchesMultiThreaded() throws {
        let count = 32
        let url = try Self.makeFixture(count: count)
        defer { try? FileManager.default.removeItem(at: url) }
        let ids = (0..<UInt32(count)).reversed().map { $0 }
        func firstBytes(threads: Int) throws -> [UInt8] {
            let reader = try ParallelExpertReader(
                path: url.path,
                expertStride: Self.stride,
                threads: threads)
            return try Self.withDestinations(ids.count) { bufs in
                try reader.fetch(experts: ids, into: bufs)
                return bufs.map { $0.assumingMemoryBound(to: UInt8.self)[0] }
            }
        }
        #expect(try firstBytes(threads: 1) == firstBytes(threads: 8))
    }

    @Test func repeatedBatchesReuseThePool() throws {
        let url = try Self.makeFixture(count: 16)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 4)
        // Workers park between batches; a lost wake-up would hang here rather
        // than fail, so this also guards the condition-variable handshake.
        try Self.withDestinations(4) { bufs in
            for round in 0..<8 {
                let ids = (0..<4).map { UInt32((round * 4 + $0) % 16) }
                try reader.fetch(experts: ids, into: bufs)
                for (i, expert) in ids.enumerated() {
                    #expect(
                        bufs[i].assumingMemoryBound(to: UInt8.self)[0]
                            == UInt8(Int(expert) % 251))
                }
            }
        }
    }

    @Test func concurrentCallersCannotOverwritePublishedBatch() async throws {
        let url = try Self.makeFixture(count: 32)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 4)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for worker in 0..<2 {
                group.addTask {
                    try Self.withDestinations(4) { buffers in
                        for round in 0..<50 {
                            let ids = (0..<4).map {
                                UInt32((worker * 13 + round * 4 + $0) % 32)
                            }
                            try reader.fetch(experts: ids, into: buffers)
                            for (index, expert) in ids.enumerated() {
                                let byte = buffers[index]
                                    .assumingMemoryBound(to: UInt8.self)[0]
                                guard byte == UInt8(Int(expert) % 251) else {
                                    throw ParallelExpertReader.Failure.readFailed(errno: EIO)
                                }
                            }
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
    }

    @Test func emptyFetchIsANoOp() throws {
        let url = try Self.makeFixture(count: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride)
        try reader.fetch(experts: [], into: [])
        #expect(reader.threadCount >= 1)
    }

    @Test func missingFileFailsWithTheOpenErrno() {
        #expect(throws: ParallelExpertReader.Failure.self) {
            _ = try ParallelExpertReader(
                path: "/nonexistent/tinytitan/experts.bin", expertStride: 4096)
        }
    }

    /// A read past the end must fail rather than hand back a half-filled slot.
    @Test func readBeyondEndOfFileFails() throws {
        let url = try Self.makeFixture(count: 2)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 2)
        _ = Self.withDestinations(1) { bufs in
            #expect(throws: ParallelExpertReader.Failure.self) {
                try reader.fetch(experts: [99], into: bufs)
            }
        }
    }

    @Test func threadCountIsClampedToTheSupportedRange() throws {
        let url = try Self.makeFixture(count: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        let low = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride, threads: 0)
        #expect(low.threadCount == 1)
        let high = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride, threads: 999)
        #expect(high.threadCount == 16)
    }

    // MARK: - Split fetch (plan §5 / B3)

    /// The blob is gate|up|down contiguous, so the split fetches the front
    /// `firstBytes` and then the tail. Whatever the split, the assembled slot
    /// must be byte-identical to a whole-region fetch.
    @Test func splitFetchMatchesWholeRegionByteForByte() throws {
        let count = 8
        let url = try Self.makeFixture(count: count)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 4)
        let wanted: [UInt32] = [7, 2, 5, 0]
        let firstBytes = Self.stride * 2 / 3
        try Self.withDestinations(wanted.count) { whole in
            try Self.withDestinations(wanted.count) { split in
                try reader.fetch(experts: wanted, into: whole)
                let fronts = split.map { $0 }
                let tails = split.map { $0.advanced(by: firstBytes) }
                try reader.fetch(
                    offsets: wanted.map { UInt64($0) * UInt64(Self.stride) },
                    byteCounts: [Int](repeating: firstBytes, count: wanted.count),
                    into: fronts)
                try reader.fetch(
                    offsets: wanted.map {
                        UInt64($0) * UInt64(Self.stride) + UInt64(firstBytes)
                    },
                    byteCounts: [Int](repeating: Self.stride - firstBytes, count: wanted.count),
                    into: tails)
                for i in 0..<wanted.count {
                    let expected = Data(
                        bytes: whole[i],
                        count: Self.stride)
                    let actual = Data(bytes: split[i], count: Self.stride)
                    #expect(expected == actual, "destination \(i) diverged")
                }
            }
        }
    }

    /// A first range far smaller than the stride (the gate+up share is ~2/3,
    /// but nothing in the reader should depend on that shape).
    @Test func splitFetchHandlesAShortFirstRange() throws {
        let url = try Self.makeFixture(count: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 2)
        let firstBytes = Self.stride / 8
        try Self.withDestinations(2) { bufs in
            try reader.fetch(
                offsets: [0, 3 * UInt64(Self.stride)],
                byteCounts: [firstBytes, firstBytes],
                into: bufs)
            for (i, expert) in [0, 3].enumerated() {
                let bytes = bufs[i].assumingMemoryBound(to: UInt8.self)
                #expect(bytes[0] == UInt8(expert))
                #expect(bytes[firstBytes - 1] == UInt8(expert))
            }
        }
    }

    /// Partial progress is the point of the split: the first call must write
    /// exactly the bytes it was asked for and leave the rest of the slot alone,
    /// so the phase-1 compute can start while the tail is still in flight.
    @Test func firstRangeFetchLeavesTheTailUnwritten() throws {
        let url = try Self.makeFixture(count: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 2)
        let firstBytes = Self.stride * 2 / 3
        try Self.withDestinations(1) { bufs in
            bufs[0].assumingMemoryBound(to: UInt8.self)
                .update(repeating: 0xFF, count: Self.stride)
            try reader.fetch(
                offsets: [2 * UInt64(Self.stride)],
                byteCounts: [firstBytes],
                into: [bufs[0]])
            let bytes = bufs[0].assumingMemoryBound(to: UInt8.self)
            #expect(bytes[0] == 2)
            #expect(bytes[firstBytes - 1] == 2)
            #expect(
                bytes[firstBytes] == 0xFF,
                "the split fetch wrote past the range it was asked for")
            #expect(bytes[Self.stride - 1] == 0xFF)
        }
    }

    @Test func splitReadBeyondEndOfFileFails() throws {
        let url = try Self.makeFixture(count: 2)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(
            path: url.path,
            expertStride: Self.stride,
            threads: 2)
        _ = Self.withDestinations(1) { bufs in
            #expect(throws: ParallelExpertReader.Failure.self) {
                try reader.fetch(
                    offsets: [UInt64(Self.stride)],
                    byteCounts: [Self.stride + 1],
                    into: bufs)
            }
        }
    }
}
