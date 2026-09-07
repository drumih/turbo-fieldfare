import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelInstallationProbeTests {
    @Test func storageMetricsUseMetadataAndExcludeSymlinks() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("storage-metrics-\(UUID())", isDirectory: true)
        let nested = root.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("abc".utf8).write(to: root.appendingPathComponent("a"))
        try Data("12345".utf8).write(to: nested.appendingPathComponent("b"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("alias"),
            withDestinationURL: nested.appendingPathComponent("b"))

        let metrics = try AppModelStorageMetrics.measure(at: root)
        #expect(metrics.logicalBytes == 8)
        #expect(metrics.allocatedBytes >= metrics.logicalBytes)
        #expect(metrics.availableBytes != nil)
    }

    @Test func missingDirectoryIsMissing() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-missing-\(UUID().uuidString).gturbo")
        #expect(AppModelInstallationProbe.status(at: url) == .missing)
    }

    @Test func manifestWithoutFinalMetadataIsPartial() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-partial-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{}".utf8).write(to: url.appendingPathComponent("manifest.json"))
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    @Test func validBoundedMetadataIsComplete() throws {
        let url = try makeCompleteModelInstall("probe")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }

    @Test func loadFailureFixturePassesProbeButOmitsResidentWeights() throws {
        let source = try makeCompleteModelInstall("load-failure-source")
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("load-failure-output-\(UUID()).gturbo", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        var repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<4 { repository.deleteLastPathComponent() }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ruby")
        process.arguments = [
            repository.appendingPathComponent(
                "Scripts/make_app_load_failure_fixture.rb").path,
            source.path,
            output.path,
        ]
        try process.run()
        process.waitUntilExit()

        #expect(process.terminationStatus == 0)
        #expect(AppModelInstallationProbe.status(at: output) == .complete)
        #expect(!FileManager.default.fileExists(
            atPath: output.appendingPathComponent("model_weights.bin").path))
    }

    @Test func receiptBoundToDifferentPathIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-path")
        defer { try? FileManager.default.removeItem(at: url) }
        let receiptURL = url.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/different/model.gturbo"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    @Test func differentCheckpointIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-checkpoint")
        defer { try? FileManager.default.removeItem(at: url) }
        let descriptor = AppModelInstallDescriptor(
            displayName: "different",
            repoID: "example/different",
            revision: "revision",
            sourceIndexSHA256: String(repeating: "f", count: 64),
            approximateDownloadBytes: 1,
            installedBytes: 1,
            rangeStagingBytes: 1,
            reserveBytes: 1)
        guard case .partial = AppModelInstallationProbe.status(at: url, descriptor: descriptor) else {
            Issue.record("expected checkpoint mismatch to be partial")
            return
        }
    }
}
