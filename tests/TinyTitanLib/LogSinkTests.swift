import Foundation
import Testing

@testable import TinyTitanLib
import TinyTitanCore

/// The library's log destination belongs to whoever embedded it.
///
/// Serialized because the sink is process-wide by design (see `ServerLog`): two
/// tests installing different sinks at the same time would be testing the race
/// rather than the contract.
@Suite(.serialized)
struct LibraryLogSinkTests {
    @Test func diagnosticsGoToTheInstalledSinkInsteadOfStderr() {
        let box = LineBox()
        ServerLog.useSink { line in box.append(line) }
        defer { ServerLog.useSink(nil) }

        ServerLog.diagnostic("TinyTitan test line")
        ServerLog.accepted(id: "abc", streaming: false)

        let lines = box.lines
        #expect(lines.count == 2)
        // A diagnostic carries no timestamp prefix: `benchmark/*.py` parses
        // these lines out of a merged log, so their text is a contract.
        #expect(lines.first == "TinyTitan test line\n")
        #expect(lines.last?.contains("request abc accepted") == true)
    }

    @Test func anEmptySinkSilencesTheLibrary() {
        let box = LineBox()
        ServerLog.useSink { _ in }
        defer { ServerLog.useSink(nil) }

        ServerLog.diagnostic("this line must not reach the capturing sink")
        #expect(box.lines.isEmpty)

        // Restoring the default must not keep the silencing closure: a line
        // emitted after `nil` goes to stderr, which this test does not capture.
        ServerLog.useSink { line in box.append(line) }
        ServerLog.diagnostic("captured again")
        #expect(box.lines == ["captured again\n"])
    }
}

/// Collects the lines a sink is handed. The sink is called with the log lock
/// held and may be reached from any thread, so the box is locked too.
final class LineBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func append(_ line: String) {
        lock.withLock { storage.append(line) }
    }

    var lines: [String] {
        lock.withLock { storage }
    }
}
