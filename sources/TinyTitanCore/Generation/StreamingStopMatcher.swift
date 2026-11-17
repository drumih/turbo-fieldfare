import Foundation

public struct StreamingStopMatcher: Sendable {
    private let stops: [String]
    private var pending = ""
    public private(set) var isStopped = false
    /// The stop string that ended the stream, once one has matched. The
    /// Anthropic Messages API reports it as `stop_sequence`.
    public private(set) var matchedStop: String?

    public init(stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
    }

    public mutating func push(_ text: String) -> String {
        guard !isStopped else { return "" }
        pending += text
        if let match = earliestMatch(in: pending) {
            let output = String(pending[..<match.index])
            pending = ""
            isStopped = true
            matchedStop = match.stop
            return output
        }
        let retained = longestPossibleSuffix(in: pending)
        let boundary = utf8Boundary(pending, retainingLastBytes: retained)
        let output = String(pending[..<boundary])
        pending = String(pending[boundary...])
        return output
    }

    public mutating func finish() -> String {
        guard !isStopped else { return "" }
        defer { pending = "" }
        return pending
    }

    private func earliestMatch(in text: String) -> (index: String.Index, stop: String)? {
        stops.compactMap { stop in text.range(of: stop).map { (index: $0.lowerBound, stop: stop) } }
            .min { $0.index < $1.index }
    }

    /// Longest byte-length suffix of `text` that could be the byte prefix of a
    /// stop string (R20). Matching on UTF-8 code units — not Characters — so a
    /// stop prefix that shares only part of a multibyte sequence across a
    /// token boundary is still retained instead of being emitted as output.
    private func longestPossibleSuffix(in text: String) -> Int {
        var best = 0
        let textUTF8 = text.utf8
        for stop in stops {
            let stopUTF8 = stop.utf8
            let maximum = min(textUTF8.count, max(stopUTF8.count - 1, 0))
            for length in stride(from: maximum, through: 1, by: -1)
            where textUTF8.suffix(length).elementsEqual(stopUTF8.prefix(length)) {
                best = max(best, length)
                break
            }
        }
        return best
    }

    /// The Character boundary at or after `text.utf8.count - retainingLastBytes`.
    /// Rounding up (never splitting a multibyte sequence) keeps the retained
    /// pending suffix valid UTF-8; the few extra bytes simply stay pending for
    /// the next push. With nothing retained the boundary is the end of the
    /// text, so the whole pending buffer is emitted and reset.
    private func utf8Boundary(_ text: String, retainingLastBytes: Int) -> String.Index {
        let utf8 = text.utf8
        guard retainingLastBytes > 0 else { return text.endIndex }
        let target = utf8.index(
            utf8.startIndex,
            offsetBy: utf8.count - retainingLastBytes)
        // A UTF-8 continuation byte has top bits 10xxxxxx. Walk forward to the
        // next lead byte so the retained suffix never splits a codepoint.
        var boundary = target
        while boundary != utf8.endIndex {
            if utf8[boundary] & 0xC0 != 0x80 { break }
            utf8.formIndex(after: &boundary)
        }
        return boundary
    }
}
