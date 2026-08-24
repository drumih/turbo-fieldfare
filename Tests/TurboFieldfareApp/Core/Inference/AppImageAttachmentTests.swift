import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppImageAttachmentTests {
    @Test func stagesThroughBoundedImmutableCopy() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("source.png")
        try Data("fixture".utf8).write(to: source)
        let store = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true))

        let attachment = try store.stage(source)

        #expect(attachment.displayName == "source.png")
        #expect(attachment.encodedBytes == 7)
        #expect(try Data(contentsOf: attachment.fileURL) == Data("fixture".utf8))
        let attributes = try FileManager.default.attributesOfItem(
            atPath: attachment.fileURL.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o400)
    }
}
