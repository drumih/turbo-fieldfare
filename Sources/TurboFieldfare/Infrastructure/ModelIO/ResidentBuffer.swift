import Foundation
import Darwin
import Metal

/// `mmap`'d view of `model_weights.bin`'s tensor data region, wrapped in
/// one shared `MTLBuffer`. All resident `TensorView`s alias byte offsets
/// inside `buffer`.
final class ResidentBuffer {
    let buffer: MTLBuffer

    /// `mmap` the page-aligned window covering `[fileOffset, fileOffset + residentSize)`
    /// inside the file at `fileURL`. The wrapped `MTLBuffer` starts at the
    /// (sub-page) offset within the mapping so the resident bytes start at
    /// byte 0 of the buffer.
    init(fileURL: URL,
                fileOffset: UInt64,
                residentSize: UInt64,
                device: MTLDevice) throws {
        let pageSize = Int(getpagesize())

        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        let alignedOffset = (fileOffset / UInt64(pageSize)) * UInt64(pageSize)
        let sliceShift = Int(fileOffset - alignedOffset)
        let mappedLen = sliceShift + Int(residentSize)
        let mapped = mmap(nil, mappedLen, PROT_READ, MAP_PRIVATE,
                          fd, off_t(alignedOffset))
        if mapped == MAP_FAILED {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }
        let base = mapped!

        _ = posix_madvise(base, mappedLen, POSIX_MADV_RANDOM)

        let sliceStart = base.advanced(by: sliceShift)

        // Capture pointer + length for the deallocator. Do NOT capture self
        // here — that would create a retain cycle through the MTLBuffer.
        nonisolated(unsafe) let captureBase = base
        let captureLen = mappedLen
        guard let buf = device.makeBuffer(
            bytesNoCopy: sliceStart,
            length: Int(residentSize),
            options: .storageModeShared,
            deallocator: { _, _ in
                munmap(captureBase, captureLen)
            }
        ) else {
            munmap(base, mappedLen)
            throw ModelError.residentBufferWrapFailed
        }

        self.buffer = buf
    }
}
