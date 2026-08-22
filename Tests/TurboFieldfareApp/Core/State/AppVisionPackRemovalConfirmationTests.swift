import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Deleting the companion pack throws away 1.14 GB, and the only way back is
/// downloading it again — so it asks first.
///
/// The Inspector hides its image-support block once the pack is installed, which
/// left its confirmation-guarded Remove button on an unreachable branch: the
/// Model menu item was the only path, and it called `removeVisionPack` straight
/// through. The menu now goes through `requestVisionPackRemoval`, and the window
/// presents the dialog.
@Suite struct AppVisionPackRemovalConfirmationTests {
    @MainActor
    @Test func requestingRemovalAsksRatherThanDeleting() throws {
        let directory = try makeVisionReadyModelInstall("removal-confirm")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)

        try #require(model.canRemoveVisionPack)
        #expect(!model.isConfirmingVisionPackRemoval)

        model.requestVisionPackRemoval()

        #expect(model.isConfirmingVisionPackRemoval,
                "the menu item deleted the pack without asking")
        #expect(model.isVisionPackInstalled,
                "the pack was removed before the confirmation was answered")
    }

    /// Answering the dialog is what performs the removal, and the flag has to
    /// drop with it or the dialog returns on the next state change.
    @MainActor
    @Test func confirmingClearsTheRequestAndRemoves() throws {
        let directory = try makeVisionReadyModelInstall("removal-confirmed")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)

        model.requestVisionPackRemoval()
        try #require(model.isConfirmingVisionPackRemoval)

        model.removeVisionPack()

        #expect(!model.isConfirmingVisionPackRemoval)
    }

    /// A removal that cannot run must not raise a dialog that would do nothing:
    /// a loaded session blocks every companion operation.
    @MainActor
    @Test func norequestIsRaisedWhenRemovalIsRefused() throws {
        // No companion beside the model, so there is nothing to remove.
        let directory = try makeCompleteModelInstall("removal-refused")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)

        try #require(!model.canRemoveVisionPack)
        model.requestVisionPackRemoval()

        #expect(!model.isConfirmingVisionPackRemoval,
                "a dialog was raised for a removal that would be refused anyway")
    }
}
