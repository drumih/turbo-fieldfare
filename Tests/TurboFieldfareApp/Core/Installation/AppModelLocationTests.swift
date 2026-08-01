import Foundation
import Testing
@testable import TurboFieldfareAppCore

private let gemmaRepoID = "mlx-community/gemma-4-26b-a4b-it-4bit"
private let gemmaSlug = "mlx-community--gemma-4-26b-a4b-it-4bit"

@Suite struct AppModelLocationTests {
    @Test func explicitURLWins() throws {
        let result = try AppModelLocation.resolve(
            repoID: gemmaRepoID,
            explicitURL: URL(fileURLWithPath: "/models/explicit.gturbo"),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/models/explicit.gturbo")
    }

    @Test func executableAncestorFindsPackageRootOutsideCWD() throws {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let result = try AppModelLocation.resolve(
            repoID: gemmaRepoID,
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/models/\(gemmaSlug)/model.gturbo")
    }

    @Test func currentDirectoryCanBePackageRoot() throws {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let result = try AppModelLocation.resolve(
            repoID: gemmaRepoID,
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/models/\(gemmaSlug)/model.gturbo")
    }

    @Test func standaloneAppFallsBackToApplicationSupport() throws {
        let result = try AppModelLocation.resolve(
            repoID: gemmaRepoID,
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/support/TurboFieldfare/models/\(gemmaSlug)/model.gturbo")
    }
}

@Suite struct AppModelLocationSlugTests {
    @Test func packageRootInstallsUnderScratchModelsSlug() throws {
        let root = URL(fileURLWithPath: "/repo", isDirectory: true)
        let resolved = try AppModelLocation.resolve(
            repoID: gemmaRepoID,
            explicitURL: nil,
            executableURL: root.appendingPathComponent(".build/debug/TurboFieldfareMac"),
            currentDirectoryURL: root,
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { path in
                path == "/repo/Package.swift" || path == "/repo/Sources/TurboFieldfareApp/Mac"
            })
        #expect(resolved.path == "/repo/scratch/models/\(gemmaSlug)/model.gturbo")
    }

    @Test func applicationSupportFallbackUsesModelsSlug() throws {
        let resolved = try AppModelLocation.resolve(
            repoID: gemmaRepoID,
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { _ in false })
        #expect(resolved.path == "/support/TurboFieldfare/models/\(gemmaSlug)/model.gturbo")
    }

    @Test func distinctRepositoriesGetDistinctDirectories() throws {
        let support = URL(fileURLWithPath: "/support", isDirectory: true)
        let elsewhere = URL(fileURLWithPath: "/elsewhere", isDirectory: true)
        let first = try AppModelLocation.resolve(
            repoID: gemmaRepoID, explicitURL: nil, executableURL: nil,
            currentDirectoryURL: elsewhere,
            applicationSupportURL: support, fileExists: { _ in false })
        let second = try AppModelLocation.resolve(
            repoID: "someone/gemma-4-26b-a4b-uncensored", explicitURL: nil, executableURL: nil,
            currentDirectoryURL: elsewhere,
            applicationSupportURL: support, fileExists: { _ in false })
        #expect(first != second)
    }

    @Test func explicitURLStillWins() throws {
        let resolved = try AppModelLocation.resolve(
            repoID: gemmaRepoID,
            explicitURL: URL(fileURLWithPath: "/custom/path", isDirectory: true),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { _ in false })
        #expect(resolved.path == "/custom/path")
    }

    @Test func hostileRepositoryIDCannotEscapeModelsDirectory() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try AppModelLocation.resolve(
                repoID: "../../etc",
                explicitURL: nil,
                executableURL: nil,
                currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
                applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
                fileExists: { _ in false })
        }
    }

    @Test func supportDirectoryIsTheModelsParent() {
        let support = AppModelLocation.supportDirectory(
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true))
        #expect(support.path == "/support/TurboFieldfare")
    }
}
