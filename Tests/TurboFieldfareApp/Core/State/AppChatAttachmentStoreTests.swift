import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppChatAttachmentStoreTests {
    @Test func imageImportCreatesABoundedManagedReference() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: model,
            withIntermediateDirectories: true)
        let source = root.appendingPathComponent("selected.png")
        try Self.onePixelPNG.write(to: source)

        let attachment = try AppChatAttachmentStore.importImage(
            from: source,
            chatID: UUID(),
            forModelDirectory: model)
        let managedURL = try AppChatAttachmentStore.fileURL(
            for: attachment,
            modelDirectory: model)

        #expect(attachment.originalFilename == "selected.png")
        #expect(attachment.mediaTypeIdentifier == "public.png")
        #expect(attachment.pixelWidth == 1)
        #expect(attachment.pixelHeight == 1)
        #expect(attachment.byteCount == UInt64(Self.onePixelPNG.count))
        #expect(attachment.sha256.count == 64)
        #expect(FileManager.default.fileExists(atPath: managedURL.path))

        try FileManager.default.removeItem(at: source)
        try AppChatAttachmentStore.validateStoredImage(
            attachment,
            forModelDirectory: model)
    }

    @Test func unsupportedFileIsRejectedAndLeavesNoManagedImage() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: model,
            withIntermediateDirectories: true)
        let source = root.appendingPathComponent("notes.txt")
        try Data("not an image".utf8).write(to: source)

        #expect(throws: AppChatAttachmentStoreError.self) {
            _ = try AppChatAttachmentStore.importImage(
                from: source,
                chatID: UUID(),
                forModelDirectory: model)
        }

        let attachmentRoot = AppChatAttachmentStore.rootURL(
            forModelDirectory: model)
        let files = try? FileManager.default.subpathsOfDirectory(
            atPath: attachmentRoot.path)
        #expect(files?.contains(where: { path in
            ["png", "jpg", "heic", "webp"].contains(
                URL(fileURLWithPath: path).pathExtension)
        }) != true)
    }

    @Test func traversalReferenceIsRejected() {
        let attachment = AppImageAttachment(
            relativePath: "../outside.png",
            originalFilename: "outside.png",
            mediaTypeIdentifier: "public.png",
            pixelWidth: 1,
            pixelHeight: 1,
            byteCount: 1,
            sha256: String(repeating: "a", count: 64))

        #expect(throws: AppChatAttachmentStoreError.self) {
            _ = try AppChatAttachmentStore.fileURL(
                for: attachment,
                modelDirectory: FileManager.default.temporaryDirectory)
        }
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "AppChatAttachmentStoreTests-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        return root
    }

    static let onePixelPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")!
}
