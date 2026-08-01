import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct AppResidentWeightProbeTests {
    private func makeModelDirectory(_ name: String, manifest: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("resident-probe-\(name)-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(manifest.utf8).write(
            to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    @Test func sumsEverythingExceptRoutedExperts() throws {
        let directory = try makeModelDirectory("basic", manifest: """
        {
          "files": {
            "embedding.bin": { "size": 100 },
            "attention.bin": { "size": 200 },
            "shared_experts.bin": { "size": 50 },
            "packed_experts/layer_00.bin": { "size": 100000 },
            "packed_experts/layer_01.bin": { "size": 100000 }
          }
        }
        """)
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(AppResidentWeightProbe.residentBytes(atModelDirectory: directory) == 350)
    }

    @Test func handlesUnpaddedExpertFileNames() throws {
        let directory = try makeModelDirectory("unpadded", manifest: """
        {
          "files": {
            "embedding.bin": { "size": 10 },
            "packed_experts/layer_7.bin": { "size": 99999 }
          }
        }
        """)
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(AppResidentWeightProbe.residentBytes(atModelDirectory: directory) == 10)
    }

    @Test func returnsNilWhenManifestIsMissing() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("resident-probe-absent-\(UUID().uuidString)",
                                    isDirectory: true)
        #expect(AppResidentWeightProbe.residentBytes(atModelDirectory: directory) == nil)
    }

    @Test func returnsNilWhenManifestIsUnreadable() throws {
        let directory = try makeModelDirectory("corrupt", manifest: "{ not json")
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(AppResidentWeightProbe.residentBytes(atModelDirectory: directory) == nil)
    }

    /// A nil probe must fall back to the estimate rather than reporting zero —
    /// zero would make every context length look affordable.
    @Test func fallbackIsTheEstimateNotZero() throws {
        let directory = try makeModelDirectory("fallback", manifest: "{ not json")
        defer { try? FileManager.default.removeItem(at: directory) }

        let bytes = AppResidentWeightProbe.residentBytesOrEstimate(atModelDirectory: directory)
        #expect(bytes == AppMemoryBudget.residentWeightEstimateBytes)
    }
}
