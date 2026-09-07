import Foundation
import Testing
import TurboFieldfareRepackCore
@testable import TurboFieldfareAppCore

@Suite struct ConversationImageWriterTests {
    private enum EncodingFailure: Error { case interrupted }

    /// A failed later turn must not damage a file already referenced by an
    /// earlier turn, or publish a partial file for a new image.
    @Test(arguments: [false, true])
    func failedEncodingDoesNotPublishPartialBytes(existing: Bool) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("image-atomic-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("image.png")
        let original = Data("previous complete image".utf8)
        if existing { try original.write(to: destination) }

        #expect(throws: EncodingFailure.self) {
            try ConversationImageWriter.writeAtomically(to: destination) { temporary in
                try Data("interrupted image".utf8).write(to: temporary)
                throw EncodingFailure.interrupted
            }
        }
        if existing {
            #expect(try Data(contentsOf: destination) == original)
        } else {
            #expect(!FileManager.default.fileExists(atPath: destination.path))
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path)
            == (existing ? ["image.png"] : []))
    }

    @Test(arguments: [false, true])
    func successfulEncodingPublishesOnlyTheCompletedFile(existing: Bool) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("image-atomic-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("image.thumb.jpg")
        let original = Data("previous thumbnail".utf8)
        let completed = Data("completed thumbnail".utf8)
        if existing { try original.write(to: destination) }

        try ConversationImageWriter.writeAtomically(to: destination) { temporary in
            try Data("partial thumbnail".utf8).write(to: temporary)
            if existing {
                #expect(try Data(contentsOf: destination) == original)
            } else {
                #expect(!FileManager.default.fileExists(atPath: destination.path))
            }
            try completed.write(to: temporary)
        }
        #expect(try Data(contentsOf: destination) == completed)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path)
            == ["image.thumb.jpg"])
    }

    @Test func failedPromotionPreservesTheDestinationAndRemovesTheTemporary() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("image-promotion-\(UUID())")
        let destination = root.appendingPathComponent("image.png")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let sentinel = destination.appendingPathComponent("keep")
        try Data("untouched".utf8).write(to: sentinel)
        #expect(throws: RepackError.self) {
            try ConversationImageWriter.writeAtomically(to: destination) {
                try Data("complete image".utf8).write(to: $0)
            }
        }
        #expect(try Data(contentsOf: sentinel) == Data("untouched".utf8))
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["image.png"])
    }
}
