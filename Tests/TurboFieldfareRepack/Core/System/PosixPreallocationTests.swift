import Foundation
import Darwin
import Testing

@testable import TurboFieldfareRepackCore

@Suite
struct PosixPreallocationTests {
    @Test func allocatesPhysicalBlocksAndPreservesWritableBounds() throws {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-preallocate-\(UUID().uuidString)")
        let path = (root as NSString).appendingPathComponent("weights.bin")
        defer { try? FileManager.default.removeItem(atPath: root) }
        try Posix.mkdirP(root)
        let descriptor = try Posix.openCreateRW(path)
        defer { close(descriptor) }
        let size = UInt64(4 * 1024 * 1024)

        try Posix.preallocate(descriptor, path: path, size: size)

        var info = stat()
        #expect(fstat(descriptor, &info) == 0)
        #expect(UInt64(info.st_size) == size)
        #expect(UInt64(info.st_blocks) * 512 >= size)

        var first: UInt8 = 0xA5
        var last: UInt8 = 0x5A
        try withUnsafeBytes(of: &first) {
            try Posix.pwriteAll(
                fd: descriptor, path: path, buf: $0.baseAddress!, count: 1, offset: 0)
        }
        try withUnsafeBytes(of: &last) {
            try Posix.pwriteAll(
                fd: descriptor, path: path, buf: $0.baseAddress!, count: 1, offset: size - 1)
        }

        var actualFirst: UInt8 = 0
        var actualLast: UInt8 = 0
        try withUnsafeMutableBytes(of: &actualFirst) {
            try Posix.preadAll(
                fd: descriptor, path: path, buf: $0.baseAddress!, count: 1, offset: 0)
        }
        try withUnsafeMutableBytes(of: &actualLast) {
            try Posix.preadAll(
                fd: descriptor, path: path, buf: $0.baseAddress!, count: 1, offset: size - 1)
        }
        #expect(actualFirst == first)
        #expect(actualLast == last)
    }
}