import Darwin
import Testing
@testable import TurboFieldfare

@Suite struct RDAdviceProbeTests {
    @Test func clipsByteCountToRadvisoryIntLimit() {
        #expect(RDAdvice.clippedByteCount(0) == 0)
        #expect(RDAdvice.clippedByteCount(64) == 64)
        #expect(RDAdvice.clippedByteCount(UInt64(Int32.max)) == UInt64(Int32.max))
        #expect(RDAdvice.clippedByteCount(UInt64(Int32.max) + 1) == UInt64(Int32.max))
    }

    @Test func invalidNonVnodeDescriptorReportsFailure() {
        var fds = [CInt](repeating: -1, count: 2)
        #expect(pipe(&fds) == 0)
        defer {
            close(fds[0])
            close(fds[1])
        }

        let result = RDAdvice.call(fd: fds[0], offset: 0, byteCount: 1)
        #expect(!result.succeeded)
        #expect(result.requestedBytes == 1)
        #expect(result.errnoValue != 0)
    }
}
