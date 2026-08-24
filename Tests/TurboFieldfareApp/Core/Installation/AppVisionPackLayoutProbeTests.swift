import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// "This model cannot host a companion pack" and "the pack on disk is damaged"
/// are different answers and need different states.
///
/// `VisionPackLocation.companionURL` throws for exactly one reason — a directory
/// name that does not end in `.gturbo` — and reporting that as `.partial`
/// printed a raw `invalidTextModelPath(...)` under "Needs repair" and offered a
/// Repair button for a pack that could never be located.
@Suite struct AppVisionPackLayoutProbeTests {
    @Test func adirectoryThatCannotHostAPackIsNotReportedAsDamaged() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4-model", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        let status = AppVisionPackInstallationProbe.status(at: model)

        #expect(status == .unsupportedLayout)
        if case .partial(let message) = status {
            Issue.record("a non-.gturbo layout was reported as damaged: \(message)")
        }
    }

    /// The `.gturbo` layout with no companion beside it is the ordinary
    /// "not installed yet" case, and has to stay distinguishable from the one
    /// above — that is the whole point of the new state.
    @Test func agturboDirectoryWithNoCompanionIsMissingRatherThanUnsupported() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        #expect(AppVisionPackInstallationProbe.status(at: model) == .missing)
    }

    /// A layout with nowhere to put a pack cannot be repaired by downloading
    /// one, so the install action must not be offered.
    @MainActor
    @Test func alayoutThatCannotHostAPackIsNotOfferedAnInstall() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4-model", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        let app = AppModel(modelDirectory: model)

        #expect(app.visionInstallationStatus == .unsupportedLayout)
        #expect(!app.canInstallVisionPack,
                "a pack was offered for a directory that cannot hold one")
        #expect(!app.isVisionPackInstalled)
    }

    @MainActor
    @Test func unsupportedHardwareIsNotOfferedAVisionInstall() throws {
        let model = try makeCompleteModelInstall("unsupported-vision-hardware")
        defer { try? FileManager.default.removeItem(at: model) }
        let installer = MockVisionPackInstallerClient(events: [.checking])
        let supported = AppModel(
            modelDirectory: model,
            visionInstaller: installer,
            visionRuntimeSupported: true)
        let unsupported = AppModel(
            modelDirectory: model,
            visionInstaller: installer,
            visionRuntimeSupported: false)

        #expect(supported.canInstallVisionPack,
                "the fixture did not otherwise qualify for installation")
        #expect(!unsupported.canInstallVisionPack)
        unsupported.installVisionPack()
        #expect(unsupported.visionInstallState == .idle,
                "unsupported hardware started a companion download")

        unsupported.visionInstallState = .readyToActivate(model)
        #expect(!unsupported.canActivateVisionPack)
        unsupported.activateVisionPack()
        #expect(installer.activationCount == 0,
                "unsupported hardware started companion activation")
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-layout-probe-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }
}
