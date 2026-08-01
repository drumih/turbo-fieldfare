import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct LegacyInstallMigrationTests {
    private func makeRoot(_ name: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("legacy-migrate-\(name)-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func seedInstall(at directory: URL, marker: String) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(marker.utf8).write(
            to: directory.appendingPathComponent("manifest.json"))
    }

    @Test func movesALegacyInstallToTheSlugPath() throws {
        let root = try makeRoot("move")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let destination = root.appendingPathComponent("models/slug/model.gturbo",
                                                      isDirectory: true)
        try seedInstall(at: legacy, marker: "original")

        #expect(AppModelLocation.migrateLegacyInstall(legacy: legacy, destination: destination))

        let moved = destination.appendingPathComponent("manifest.json")
        #expect(FileManager.default.fileExists(atPath: moved.path))
        #expect(try String(data: Data(contentsOf: moved), encoding: .utf8) == "original")
        #expect(FileManager.default.fileExists(atPath: legacy.path) == false)
    }

    @Test func isANoOpWhenThereIsNoLegacyInstall() throws {
        let root = try makeRoot("absent")
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(AppModelLocation.migrateLegacyInstall(
            legacy: root.appendingPathComponent("gemma4.gturbo", isDirectory: true),
            destination: root.appendingPathComponent("models/slug/model.gturbo",
                                                     isDirectory: true)) == false)
    }

    /// Never clobber a real install. If the user already has weights at the new
    /// path, the legacy directory is left alone for them to delete by hand
    /// rather than silently overwritten.
    @Test func refusesToOverwriteAnExistingInstall() throws {
        let root = try makeRoot("occupied")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let destination = root.appendingPathComponent("models/slug/model.gturbo",
                                                      isDirectory: true)
        try seedInstall(at: legacy, marker: "legacy")
        try seedInstall(at: destination, marker: "current")

        #expect(AppModelLocation.migrateLegacyInstall(legacy: legacy,
                                                      destination: destination) == false)
        let kept = try Data(contentsOf: destination.appendingPathComponent("manifest.json"))
        #expect(String(data: kept, encoding: .utf8) == "current")
        #expect(FileManager.default.fileExists(atPath: legacy.path))
    }

    @Test func isIdempotent() throws {
        let root = try makeRoot("idempotent")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let destination = root.appendingPathComponent("models/slug/model.gturbo",
                                                      isDirectory: true)
        try seedInstall(at: legacy, marker: "original")

        #expect(AppModelLocation.migrateLegacyInstall(legacy: legacy, destination: destination))
        #expect(AppModelLocation.migrateLegacyInstall(legacy: legacy,
                                                      destination: destination) == false)
        #expect(FileManager.default.fileExists(
            atPath: destination.appendingPathComponent("manifest.json").path))
    }

    /// `VerifiedInstallReceipt.validate` compares the recorded
    /// `modelDirectoryPath` against the directory it is loading from, so a move
    /// that leaves the receipt untouched produces a model that verifies on disk
    /// but refuses to load.
    @Test func rewritesTheInstallReceiptToTheNewPath() throws {
        let root = try makeRoot("receipt")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let destination = root.appendingPathComponent("models/slug/model.gturbo",
                                                      isDirectory: true)
        try seedInstall(at: legacy, marker: "original")
        let receipt: [String: Any] = [
            "schemaVersion": 1,
            "manifestSha256": "abc",
            "modelDirectoryPath": legacy.standardizedFileURL.path,
            "files": ["manifest.json": ["size": 8, "sha256": "abc"]],
        ]
        try JSONSerialization.data(withJSONObject: receipt).write(
            to: legacy.appendingPathComponent("verified-install.json"))

        #expect(AppModelLocation.migrateLegacyInstall(legacy: legacy, destination: destination))

        let moved = destination.appendingPathComponent("verified-install.json")
        let decoded = try JSONSerialization.jsonObject(with: Data(contentsOf: moved))
        let path = (decoded as? [String: Any])?["modelDirectoryPath"] as? String
        #expect(path == destination.standardizedFileURL.path)
        // Everything else in the receipt must survive untouched.
        #expect((decoded as? [String: Any])?["manifestSha256"] as? String == "abc")
    }

    @Test func migratesEvenWhenThereIsNoReceipt() throws {
        let root = try makeRoot("no-receipt")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let destination = root.appendingPathComponent("models/slug/model.gturbo",
                                                      isDirectory: true)
        try seedInstall(at: legacy, marker: "original")

        #expect(AppModelLocation.migrateLegacyInstall(legacy: legacy, destination: destination))
        #expect(FileManager.default.fileExists(
            atPath: destination.appendingPathComponent("manifest.json").path))
    }

    @Test func derivesTheLegacyPathInsideAPackageCheckout() {
        let legacy = AppModelLocation.legacyInstallURL(
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { path in
                path == "/repo/Package.swift" || path == "/repo/Sources/TurboFieldfareApp/Mac"
            })
        #expect(legacy.path == "/repo/scratch/gemma4.gturbo")
    }

    @Test func derivesTheLegacyPathForAStandaloneApp() {
        let legacy = AppModelLocation.legacyInstallURL(
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { _ in false })
        #expect(legacy.path == "/support/TurboFieldfare/gemma4.gturbo")
    }
}
