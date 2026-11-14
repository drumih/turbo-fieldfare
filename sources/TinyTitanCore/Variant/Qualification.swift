import Foundation

/// A variant's qualification record (7.0 spec §1.5, §10): the evidence that the
/// engine path is qualified, not merely compiled. Committed per
/// (model, quant, hardware) triple and rendered into the README/wiki tables by
/// `tinytitan catalog`. A variant is shipped only when this record is present.
public struct Qualification: Codable, Sendable, Equatable {
    /// SHA-256 of the greedy golden output over the fixed prompt set.
    public let goldenSHA256: String
    public let decodeTokS: Double
    public let prefillTokS: Double
    public let ttftS: Double
    public let memoryCeilingBytes: Int
    public let terminalBench: TerminalBench?
    public let hardware: String
    public let date: String

    public init(
        goldenSHA256: String, decodeTokS: Double, prefillTokS: Double, ttftS: Double,
        memoryCeilingBytes: Int, terminalBench: TerminalBench?,
        hardware: String, date: String
    ) {
        self.goldenSHA256 = goldenSHA256
        self.decodeTokS = decodeTokS
        self.prefillTokS = prefillTokS
        self.ttftS = ttftS
        self.memoryCeilingBytes = memoryCeilingBytes
        self.terminalBench = terminalBench
        self.hardware = hardware
        self.date = date
    }

    /// The published Terminal-Bench suite/score for the model, with the harness
    /// version and date.
    public struct TerminalBench: Codable, Sendable, Equatable {
        public let suite: String
        public let score: Double
        public let harnessVersion: String
        public let date: String

        public init(suite: String, score: Double, harnessVersion: String, date: String) {
            self.suite = suite
            self.score = score
            self.harnessVersion = harnessVersion
            self.date = date
        }
    }
}
