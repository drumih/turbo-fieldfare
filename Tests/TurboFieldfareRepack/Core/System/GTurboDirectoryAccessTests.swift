import Foundation
import Testing

@testable import TurboFieldfareRepackCore

@Suite struct GTurboDirectoryAccessTests {
    // readdir returns a buffered snapshot, so an entry can be unlinked before
    // the scan stats it. That is ordinary concurrent-directory behavior, not a
    // damaged install, and treating it as an error made concurrent verification
    // fail with a spurious ENOENT.
    @Test func enumerationToleratesEntriesRemovedDuringTheScan() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-\(UUID().uuidString)-scan")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var files: [URL] = []
        for index in 0..<600 {
            let file = root.appendingPathComponent(String(format: "entry-%04d", index))
            try Data("x".utf8).write(to: file)
            files.append(file)
        }

        let access = try GTurboDirectoryAccess(rootPath: root.path)
        async let removals: Void = Task.detached(priority: .userInitiated) {
            for file in files { try? FileManager.default.removeItem(at: file) }
        }.value

        _ = try access.relativeEntries(maxDepth: 4)
        await removals
    }

    @Test func enumerationStillReportsSurvivingEntries() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-\(UUID().uuidString)-scan-stable")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("x".utf8).write(to: root.appendingPathComponent("kept.bin"))

        let access = try GTurboDirectoryAccess(rootPath: root.path)
        #expect(try access.relativeEntries(maxDepth: 4) == ["kept.bin"])
    }
}
