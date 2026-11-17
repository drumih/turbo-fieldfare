import Darwin
import Foundation
import Metal
import TinyTitanCore

/// `mmap`'d view of `model_weights.bin`'s tensor data region, wrapped in
/// one shared `MTLBuffer`. All resident `TensorView`s alias byte offsets
/// inside `buffer`.
///
/// unchecked-invariant: the mapping and the wrapped `MTLBuffer` are `let`,
/// created once in init and never mutated; Metal buffers are safe to read
/// from any thread, so concurrent holders of one `ResidentBuffer` are safe.
/// Needed so `Model` can witness the Sendable-bounded TT-097 engine contract.
final class ResidentBuffer: @unchecked Sendable {
    let buffer: MTLBuffer

    /// `mmap` the page-aligned window covering `[fileOffset, fileOffset + residentSize)`
    /// inside the file at `fileURL`. The wrapped `MTLBuffer` starts at the
    /// (sub-page) offset within the mapping so the resident bytes start at
    /// byte 0 of the buffer.
    init(
        fileURL: URL,
        fileOffset: UInt64,
        residentSize: UInt64,
        device: MTLDevice,
        fileDescriptor: Int32? = nil
    ) throws {
        let pageSize = Int(getpagesize())

        let fd =
            fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(fileURL.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw ModelError.posixFailed(call: "fstat(\(fileURL.path))", errno: errno)
        }
        guard (info.st_mode & S_IFMT) == S_IFREG, info.st_size >= 0 else {
            throw ModelError.indexCorrupt(detail: "resident weights are not a regular file")
        }
        let (fileEnd, endOverflow) = fileOffset.addingReportingOverflow(residentSize)
        guard residentSize > 0, !endOverflow, fileEnd <= UInt64(info.st_size),
            fileOffset <= UInt64(Int64.max)
        else {
            throw ModelError.indexCorrupt(detail: "resident mapping exceeds the weights file")
        }

        let alignedOffset = (fileOffset / UInt64(pageSize)) * UInt64(pageSize)
        let sliceShift = Int(fileOffset - alignedOffset)
        guard residentSize <= UInt64(Int.max - sliceShift) else {
            throw ModelError.indexCorrupt(detail: "resident mapping exceeds addressable memory")
        }
        let mappedLen = sliceShift + Int(residentSize)
        let mapped = mmap(
            nil, mappedLen, PROT_READ, MAP_PRIVATE,
            fd, off_t(alignedOffset))
        if mapped == MAP_FAILED {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }
        guard let base = mapped else {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }

        _ = posix_madvise(base, mappedLen, POSIX_MADV_RANDOM)

        // Pin the resident range so the expert slot cache cannot evict it
        // from the page cache. Measured on the 4-bit M3: at 64 slots/layer an
        // unpinned resident range loses the cache's io savings to page
        // faults (net +1.4%); with the pin the decode gains ~10% interleaved.
        // An 8 GB allocation (128 slots) still hurts even pinned (memory
        // pressure), so the pin alone is not enough there. Best-effort: a
        // low RLIMIT_MEMLOCK must not fail the load. Set TINYTITAN_NO_PIN=1 to
        // opt out.
        if ProcessInfo.processInfo.environment["TINYTITAN_NO_PIN"] == nil {
            if mlock(base, mappedLen) != 0 {
                // ignore — the pin is best-effort
            }
        }

        let sliceStart = base.advanced(by: sliceShift)

        // Capture pointer + length for the deallocator. Do NOT capture self
        // here — that would create a retain cycle through the MTLBuffer.
        nonisolated(unsafe) let captureBase = base
        let captureLen = mappedLen
        guard
            let buf = device.makeBuffer(
                bytesNoCopy: sliceStart,
                length: Int(residentSize),
                options: .storageModeShared,
                deallocator: { _, _ in
                    munmap(captureBase, captureLen)
                }
            )
        else {
            munmap(base, mappedLen)
            throw ModelError.residentBufferWrapFailed
        }

        self.buffer = buf
    }
}
