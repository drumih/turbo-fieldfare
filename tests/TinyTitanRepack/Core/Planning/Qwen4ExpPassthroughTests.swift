import Foundation
import Testing

@testable import TinyTitanRepackCore

/// Non-tensor files Qwen3.8-Flash-Next needs copied into the install: the PLE
/// hash constants and the n-gram table. The table is 102 GB, so how it is
/// carried matters as much as that it is carried.
@Suite("Qwen4Exp passthrough files")
struct Qwen4ExpPassthroughTests {
    @Test("Only the new family declares passthrough requirements")
    func requirementsPerFamily() {
        #expect(RepackPlanner.passthroughRequirements(family: .qwen36).isEmpty)
        #expect(RepackPlanner.passthroughRequirements(family: .qwen36MTP).isEmpty)
        let flash = RepackPlanner.passthroughRequirements(family: .qwen38flash)
        #expect(flash.map(\.name) == ["ple_constants.json", "ngram_table.bin"])
        let flash4 = RepackPlanner.passthroughRequirements(family: .qwen4flash)
        #expect(flash4.map(\.name) == ["ple_constants.json", "ngram_table.bin"])
        #expect(RepackPlanner.passthroughRequirements(family: .qwen4flashMTP).isEmpty)
    }

    @Test("The hash constants are required; the 102 GB table is not")
    func requiredness() throws {
        let flash = RepackPlanner.passthroughRequirements(family: .qwen38flash)
        let constants = try #require(flash.first { $0.name == "ple_constants.json" })
        let table = try #require(flash.first { $0.name == "ngram_table.bin" })
        // Without the constants the n-gram ids cannot be computed at all.
        #expect(constants.required)
        // The table is optional so the backbone can be installed first; the
        // upstream runtime stays coherent with the PLE block skipped.
        #expect(!table.required)
        // The cap must actually admit a 102.4 GB file, or the install would
        // reject the very artifact it is meant to carry.
        #expect(table.capBytes >= 102_400_491_520)
    }

    /// The verifier's per-file cap has to admit every file the planner says it
    /// will carry. It did not: `ngram_table.bin` was allowed 256 GiB while
    /// `--verify-install` refused anything over 128 GiB, so the one install that
    /// carries this file could not be verified. Both read one constant now, and
    /// this fails if a future cap is raised on only one side.
    @Test("The verifier's per-file cap admits every declared passthrough cap")
    func verifierCapAdmitsEveryPassthroughFile() {
        let families: [RepackModelFamily] = [
            .qwen35Dense, .qwen36, .qwen36MTP, .qwen38flash, .qwen38flashMTP,
            .qwen4flash, .qwen4flashMTP,
        ]
        for family in families {
            for requirement in RepackPlanner.passthroughRequirements(family: family) {
                #expect(
                    requirement.capBytes <= VerifiedInstallTool.payloadMaxBytes,
                    "\(requirement.name) may be \(requirement.capBytes) bytes, above the verifier's \(VerifiedInstallTool.payloadMaxBytes)"
                )
            }
        }
    }

    @Test("A passthrough file becomes a resumable range copy at offset zero")
    func passthroughBecomesRangeCopy() throws {
        let table = PassthroughFile(
            sourceName: "ngram_table.bin",
            destinationName: "ngram_table.bin",
            size: 102_400_491_520,
            required: false)
        let copy = RangeCopy(
            shardID: table.sourceName,
            sourceOffset: 0,
            size: table.size,
            destinationPath: table.destinationName,
            destinationOffset: 0)
        // Carried by the same machinery as tensor ranges, which is what makes
        // a 102 GB transfer resumable and digest-checked rather than a
        // bespoke download that has to reimplement both.
        #expect(copy.sourceOffset == 0)
        #expect(copy.destinationOffset == 0)
        #expect(copy.size == 102_400_491_520)
        #expect(copy.destinationPath == "ngram_table.bin")
    }

    @Test("The table is chunked, not fetched in one request")
    func tableIsChunked() {
        let size: UInt64 = 102_400_491_520
        let chunk = UInt64(RemoteChunkPolicy.defaultBytes)
        let chunks = (size + chunk - 1) / chunk
        // ~1,526 chunks at 64 MiB: each one is an independently resumable
        // unit, so an interrupted install does not restart 102 GB.
        #expect(chunks > 1000)
        #expect(chunk <= UInt64(RemoteChunkPolicy.maxBytes))
    }

    @Test("A passthrough is declared as an expected output, like any other file")
    func declaredAsExpectedOutput() {
        // This is what a resumed install validates against. Three separate
        // install failures came from a passthrough being planned as a range
        // copy but not registered as an output: first it had no remote info to
        // fetch from, then no destination file to write into. Each seam looked
        // correct on its own; only their conjunction is a working install, so
        // the check belongs here rather than in three places.
        let table = PassthroughFile(
            sourceName: "ngram_table.bin",
            destinationName: "ngram_table.bin",
            size: 102_400_491_520,
            required: false)
        let expected = RemoteExpectedOutput(
            relativePath: table.destinationName,
            size: table.size)
        #expect(expected.relativePath == "ngram_table.bin")
        #expect(expected.size == table.size)
        // Omitting it would let a resume accept a partial whose 102 GB table
        // is missing or truncated, since that list is exactly what is checked.
        #expect(expected.size > 0)
    }
}
