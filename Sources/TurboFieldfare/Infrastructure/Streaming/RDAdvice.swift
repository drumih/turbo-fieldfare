import Darwin
import Foundation

struct RDAdviceCallResult: Sendable, Equatable {
    let requestedBytes: UInt64
    let errnoValue: CInt
    let elapsedNanos: UInt64

    init(requestedBytes: UInt64, errnoValue: CInt, elapsedNanos: UInt64) {
        self.requestedBytes = requestedBytes
        self.errnoValue = errnoValue
        self.elapsedNanos = elapsedNanos
    }

    var succeeded: Bool { errnoValue == 0 }
}

enum RDAdvice {
    static func clippedByteCount(_ byteCount: UInt64) -> UInt64 {
        min(byteCount, UInt64(Int32.max))
    }

    static func call(fd: CInt,
                            offset: UInt64,
                            byteCount: UInt64) -> RDAdviceCallResult {
        let clippedCount = clippedByteCount(byteCount)
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var errnoValue: CInt = 0

        #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
        var advice = radvisory(
            ra_offset: off_t(offset),
            ra_count: Int32(clippedCount))
        if fcntl(fd, F_RDADVISE, &advice) != 0 {
            errnoValue = errno
        }
        #else
        errnoValue = ENOTSUP
        #endif

        return RDAdviceCallResult(
            requestedBytes: clippedCount,
            errnoValue: errnoValue,
            elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
    }
}
