import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelLocationTests {
    @Test func explicitURLWins() {
        let result = AppModelLocation.resolve(
            explicitURL: URL(fileURLWithPath: "/models/explicit.gturbo"),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/models/explicit.gturbo")
    }

    @Test func executableAncestorFindsPackageRootOutsideCWD() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.gturbo")
    }

    @Test func currentDirectoryCanBePackageRoot() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.gturbo")
    }

    @Test func standaloneAppFallsBackToApplicationSupport() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/support/TurboFieldfare/gemma4.gturbo")
    }
}

@Suite struct SelectedModelLocationValidationTests {
    @MainActor @Test func parentSelectionPointsToCompletedTextModelChild() throws {
        let fixture = try makeCompleteModelInstall("parent-selection-guidance")
        defer { try? FileManager.default.removeItem(at: fixture) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("parent-selection-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("gemma4.gturbo")
        try FileManager.default.moveItem(at: fixture, to: directory)
        // The fixture receipt binds its original location; rebind only this synthetic pack.
        let receiptURL = directory.appendingPathComponent("verified-install.json")
        var receipt = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as? [String: Any])
        receipt["modelDirectoryPath"] = directory.standardizedFileURL.path
        try JSONSerialization.data(withJSONObject: receipt).write(to: receiptURL)
        #expect(AppModelInstallationProbe.status(at: directory) == .complete)
        // A parent folder is not necessarily source checkpoint files.
        do {
            try AppModelLocation.validate(directory.deletingLastPathComponent(), intent: .existing)
            Issue.record("Parent selection must require choosing the text model itself")
        } catch {
            #expect(String(describing: error).contains("Choose the gemma4.gturbo folder inside this folder"))
        }
    }

    @Test(arguments: [false, true]) func unrelatedFolderGuidanceDoesNotClaimItContainsAModel(hasIncompleteChild: Bool) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("empty-selection-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        if hasIncompleteChild {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent("gemma4.gturbo"), withIntermediateDirectories: true)
        }
        do {
            try AppModelLocation.validate(directory, intent: .existing)
            Issue.record("Unrelated folder must be rejected")
        } catch {
            #expect(String(describing: error).contains("Choose a completed text model folder ending in .gturbo"))
        }
    }

    @MainActor @Test func existingSelectionAcceptsReadOnlyInstallAndRejectsSourceOrPartialDirectories() throws {
        let directory = try makeCompleteModelInstall("existing-selection")
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            try? FileManager.default.removeItem(at: directory)
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: directory.path)
        try AppModelLocation.validate(directory, intent: .existing)
        #expect(throws: (any Error).self) { try AppModelLocation.validate(directory.deletingLastPathComponent(), intent: .existing) }
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        try FileManager.default.removeItem(at: directory.appendingPathComponent("manifest.json"))
        #expect(throws: (any Error).self) { try AppModelLocation.validate(directory, intent: .existing) }
    }

    @MainActor @Test func downloadSelectionRejectsUnrelatedDirectoryAndMissingVolume() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("destination-validation-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let candidate = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: candidate, withIntermediateDirectories: true)
        #expect(throws: (any Error).self) { try AppModelLocation.validate(candidate, intent: .download) }
        #expect(throws: (any Error).self) { try AppModelLocation.validate(URL(fileURLWithPath: "/Volumes/missing-\(UUID())/gemma4.gturbo"), intent: .download) }
        try FileManager.default.removeItem(at: candidate)
        try AppModelLocation.validate(candidate, intent: .download)
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: root.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path) }
        #expect(throws: (any Error).self) { try AppModelLocation.validate(candidate, intent: .download) }
    }
}
