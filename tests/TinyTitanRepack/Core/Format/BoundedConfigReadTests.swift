import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

/// The converter's own ceiling on the checkpoint's `config.json` (AUD-142 group 2).
///
/// This is a different boundary from the engine's: the operator chose the
/// directory, so a cap here risks refusing a *legitimate* checkpoint rather than an
/// attacker's. What that changes is the number's provenance, not whether it
/// exists — the read is `Posix.readBoundedData`, which refuses on the `fstat` size
/// before allocating, so the ceiling bounds memory instead of reporting it
/// afterwards.
@Suite("Bounded config read")
struct BoundedConfigReadTests {

    /// Valid JSON that is not an object, so the parser has something to say about
    /// it: the refusal the tests compare against has to be a *content* complaint
    /// one and the same file makes only once the size bound has passed it.
    private static func nonObjectDocument() throws -> Data {
        try JSONSerialization.data(withJSONObject: [[Int]](repeating: [], count: 64))
    }

    private static func scratchFile(_ bytes: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("archinfo-bound-\(UUID().uuidString).json")
        try bytes.write(to: url)
        return url
    }

    /// The bound fires before the parser does, and the two refusals stay
    /// distinguishable. A file is refused for its size under a small cap and for
    /// its content under a large one, so dropping the bound cannot hide behind the
    /// other error — it flips the message from "exceeds … cap" to the parse one.
    @Test func theConfigBoundFiresBeforeTheParser() throws {
        let document = try Self.nonObjectDocument()
        let url = try Self.scratchFile(document)
        defer { try? FileManager.default.removeItem(at: url) }
        let size = document.count
        #expect(size > 64, "the fixture has to cross a 64-byte bound")

        var overBound = ""
        do {
            _ = try ArchInfo.load(configPath: url.path, maxBytes: 64)
            Issue.record("expected a refusal")
        } catch {
            overBound = String(describing: error)
        }
        #expect(
            overBound.contains("\(size)") && overBound.contains("exceeds")
                && overBound.contains("64-byte cap"),
            "the size refusal was not legible: \(overBound)")
        #expect(
            !overBound.contains("not a JSON object"),
            "the parser answered before the bound: \(overBound)")

        var misparsed = ""
        do {
            _ = try ArchInfo.load(configPath: url.path)
            Issue.record("expected a parse failure")
        } catch {
            misparsed = String(describing: error)
        }
        #expect(
            misparsed.contains("not a JSON object"),
            "under the ceiling the file should reach the parser: \(misparsed)")
    }

    /// One document, one ceiling: `IndexLoader` reads the same `config.json` and
    /// now names this constant rather than a second literal that can drift below
    /// it. Pinned because a bound here has already refused a legitimate file once
    /// (`IndexLoader.maximumIndexBytes` — 4 MiB against a real 9.7 MiB index), and
    /// because the installed `config.json` measures 12,935 bytes, so the margin is
    /// load-bearing rather than decorative.
    @Test func theConfigCeilingIsStatedWhereTheOperatorCanSeeIt() {
        #expect(ArchInfo.maxConfigBytes == 8 * 1024 * 1024)
        #expect(
            ArchInfo.maxConfigBytes > 12_935,
            "the ceiling has to clear the installed document it must never refuse")
    }
}
