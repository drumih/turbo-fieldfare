import Foundation
import Testing

@testable import TinyTitan

/// The PLE constants come from the checkpoint, and every row the hash produces
/// must land inside the table's per-head range.
@Suite struct GGUFPLESetupTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func derivesThePLEConstants() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let setup = try GGUFPLESetup.read(from: model)
        #expect(setup.layerIndices == [1])
        #expect(setup.ngramSize == 3)
        #expect(setup.headsPerNgram == 8)
        #expect(setup.convKernelSize == 4)
        #expect(setup.headCount == 16)
        #expect(setup.embedDim == 16 * 160)
        #expect(setup.tableRows == 320_001_536)
        #expect(
            setup.hash.multipliers == [
                UInt64(23_703_573_157_769), UInt64(20_109_073_645_365),
                UInt64(8_052_911_324_071),
            ])
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func everyRowLandsInsideItsHeadRange() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let setup = try GGUFPLESetup.read(from: model)
        let eos = Int32(truncatingIfNeeded: 248_044)
        let contexts: [[Int32]] = [
            [1],
            [1, 2],
            [100, 200, 300],
            [eos],
            [eos, 5, 6],
            [248_319, 0, 248_043],
            [7, 8],
        ]
        for context in contexts {
            let rows = setup.rows(context: context)
            #expect(rows.count == setup.headCount)
            for (index, row) in rows.enumerated() {
                let offset = setup.hash.offsets[index]
                let size = setup.hash.vocabSizes[index]
                #expect(row >= UInt32(truncatingIfNeeded: offset))
                #expect(UInt64(row) < offset + size)
                #expect(UInt64(row) < setup.tableRows)
            }
        }
        // A longer context than the n-gram window must still resolve.
        let long = setup.rows(context: [11, 12, 13, 14])
        #expect(long.count == setup.headCount)
    }
}
