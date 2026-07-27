import Foundation

public struct StreamingStopMatcher: Sendable {
    private let stops: [String]
    private var pending = ""
    public private(set) var isStopped = false

    public init(stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
    }

    public mutating func push(_ text: String) -> String {
        guard !isStopped else { return "" }
        pending += text
        if let match = earliestMatch(in: pending) {
            let output = String(pending[..<match])
            pending = ""
            isStopped = true
            return output
        }
        let retained = longestPossibleSuffix(in: pending)
        let boundary = pending.index(pending.endIndex, offsetBy: -retained)
        let output = String(pending[..<boundary])
        pending = String(pending[boundary...])
        return output
    }

    public mutating func finish() -> String {
        guard !isStopped else { return "" }
        defer { pending = "" }
        return pending
    }

    private func earliestMatch(in text: String) -> String.Index? {
        stops.compactMap { text.range(of: $0)?.lowerBound }.min()
    }

    private func longestPossibleSuffix(in text: String) -> Int {
        var best = 0
        for stop in stops {
            let maximum = min(text.count, max(stop.count - 1, 0))
            for length in stride(from: maximum, through: 1, by: -1) {
                if text.suffix(length) == stop.prefix(length) {
                    best = max(best, length)
                    break
                }
            }
        }
        return best
    }
}
