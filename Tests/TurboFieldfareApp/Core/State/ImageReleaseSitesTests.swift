import Foundation
import Testing

/// Where a staged picture may be deleted, and nowhere else.
///
/// The compiler already refuses a `StoredImage`, which is the half that used to
/// cost a stored conversation its pictures. The half it cannot check is *how
/// many* places delete a staged one: several call sites, each with its own idea of
/// when it applied, is how a release ran on a path nobody meant it to and how a
/// turn's copies were left behind on a path that forgot. This is a grep, and it
/// fails until a new call site is written down here with the reason it exists.
@Suite struct ImageReleaseSitesTests {
    /// Every file in the app that calls `AppImageAttachmentStore.remove`, and
    /// how many times.
    ///
    /// `AppModel.swift`, eight, all of them the composer's own lifecycle or the
    /// send's hand-off:
    ///
    /// 1. `addImages` — a staging batch is all-or-nothing, so the copies made
    ///    before a failure are referenced by nothing.
    /// 2. `removeImage` — the user took one picture off the composer.
    /// 3. `clearImages` — the user emptied the composer.
    /// 4. `finishAddingImages` — a batch that landed after a run started, which
    ///    has already snapshotted its own images.
    /// 5. `finishAddingImages` — the copies past the capacity two overlapping
    ///    adds pushed it to.
    /// 6. `deliver`, stage 4 — the links made before a retain failed.
    /// 7. `deliver`, the hand-off — the composer's own copies, now that the turn
    ///    is carried by its retained links.
    /// 8. `restoreComposer` — the turn's copies when the composer was used again
    ///    while it was in flight, so the draft's are the ones that survive.
    ///
    /// `AppModelHistory.swift`, five. Three inside the one effect handler
    /// `releaseImagesOfHeldConversation`: the out-of-context turns, the live
    /// conversation's turns, and — only when the transition says so — the
    /// pictures the live fields are still drawing. Two for the writer's own
    /// links: `beginStoringTurnImages` releasing the ones made before a
    /// retain failed, and `writeStoredImages` releasing them all once the
    /// stored copies are written.
    private static let allowed: [String: Int] = [
        "Core/State/AppModel.swift": 8,
        "Core/State/AppModelHistory.swift": 5,
    ]

    private static let needle = "attachmentStore.remove("

    @Test func stagedPicturesAreDeletedOnlyWhereThisListSaysTheyAre() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // State
            .deletingLastPathComponent()   // Core
            .deletingLastPathComponent()   // TurboFieldfareApp
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // the tree's root
            .appendingPathComponent("Sources/TurboFieldfareApp", isDirectory: true)
        var isDirectory: ObjCBool = false
        // Otherwise a scan that found nothing would read as a clean tree.
        #expect(FileManager.default.fileExists(
            atPath: sources.path, isDirectory: &isDirectory) && isDirectory.boolValue,
            "the app's sources are not at \(sources.path)")

        var found: [String: Int] = [:]
        let enumerator = try #require(FileManager.default.enumerator(
            at: sources, includingPropertiesForKeys: nil))
        var scanned = 0
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            scanned += 1
            let text = try String(contentsOf: url, encoding: .utf8)
            let count = text.components(separatedBy: Self.needle).count - 1
            guard count > 0 else { continue }
            let relative = url.standardizedFileURL.path
                .replacingOccurrences(of: sources.standardizedFileURL.path + "/",
                                      with: "")
            found[relative] = count
        }
        #expect(scanned > 0, "no Swift files were scanned")

        for (file, count) in found.sorted(by: { $0.key < $1.key }) {
            guard let expected = Self.allowed[file] else {
                Issue.record(Comment(rawValue:
                    "\(file) deletes staged pictures and is not on the "
                        + "allow-list: add it with the reason it exists"))
                continue
            }
            #expect(count == expected,
                    "\(file) has \(count) release site(s), \(expected) listed")
        }
        for file in Self.allowed.keys.sorted() where found[file] == nil {
            Issue.record(Comment(rawValue:
                "\(file) is on the allow-list and releases nothing: take it off"))
        }
    }
}
