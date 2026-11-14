import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct ExistingModelSelectionTests {
    @MainActor @Test(arguments: ["not-a-pack", "other.vision.gturbo"])
    func invalidAliasTargetDoesNotHideAValidSiblingModel(targetName: String) throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let text = try fixture(in: root, name: "ready")
        let target = root.appendingPathComponent(targetName)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("invalid.gturbo"),
                                                  withDestinationURL: target)
        // Validate the resolved target, not just the alias's .gturbo suffix.
        let review = try ExistingModelSelection.inspect(root)
        #expect(review.candidates.map(\.textURL.path) == [text.path])
        #expect(review.initialSelection == text.path)
    }

    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("model-review-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @MainActor private func fixture(in root: URL, name: String, vision: Bool = false) throws -> URL {
        let original = try makeCompleteModelInstall("folder-review")
        defer { try? FileManager.default.removeItem(at: original) }
        let text = root.appendingPathComponent(name + ".gturbo")
        try FileManager.default.moveItem(at: original, to: text)
        let receiptURL = text.appendingPathComponent("verified-install.json")
        var receipt = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as? [String: Any])
        receipt["modelDirectoryPath"] = text.standardizedFileURL.path
        try JSONSerialization.data(withJSONObject: receipt).write(to: receiptURL)
        if vision { try installVisionCompanion(forTextModel: text) }
        return text
    }

    @MainActor @Test func parentAndDirectSelectionFindSamePairWithoutIncludingVisionAsText() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let text = try fixture(in: root, name: "custom тест", vision: true)
        let parent = try ExistingModelSelection.inspect(root)
        let direct = try ExistingModelSelection.inspect(text)
        #expect(parent.candidates == direct.candidates)
        #expect(parent.candidates.count == 1)
        #expect(parent.initialSelection == text.path)
        #expect(parent.candidates[0].visionStatus == .complete)
        #expect(parent.candidates[0].visionURL.lastPathComponent == "custom тест.vision.gturbo")
        #expect(throws: (any Error).self) { try ExistingModelSelection.inspect(parent.candidates[0].visionURL) }
    }

    @MainActor @Test func multipleCandidatesRequireChoiceAndNeverBorrowAnotherModelsVisionPack() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try fixture(in: root, name: "a", vision: true)
        let second = try fixture(in: root, name: "b")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("incomplete.gturbo"), withIntermediateDirectories: true)
        let review = try ExistingModelSelection.inspect(root)
        #expect(review.initialSelection == nil)
        #expect(review.candidates.count == 3)
        #expect(review.candidates.first { $0.textURL.path == first.path }?.visionStatus == .complete)
        #expect(review.candidates.first { $0.textURL.path == second.path }?.visionStatus == .missing)
        #expect(review.candidates.last?.canSelect == false)
    }

    @MainActor @Test func brokenAliasDoesNotHideAValidSiblingModel() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let text = try fixture(in: root, name: "ready")
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("disconnected.gturbo"),
                                                  withDestinationURL: root.appendingPathComponent("missing/model.gturbo"))
        let review = try ExistingModelSelection.inspect(root)
        #expect(review.candidates.filter(\.canSelect).map(\.textURL.path) == [text.path])
    }

    @Test func emptyMissingAndNestedOnlyFoldersAreRejected() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: (any Error).self) { try ExistingModelSelection.inspect(root) }
        #expect(throws: (any Error).self) { try ExistingModelSelection.inspect(root.appendingPathComponent("missing")) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("nested/model.gturbo"), withIntermediateDirectories: true)
        #expect(throws: (any Error).self) { try ExistingModelSelection.inspect(root) }
    }

    @MainActor @Test func incompatibleCompanionIsReportedWithoutRejectingValidText() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let text = try fixture(in: root, name: "a", vision: true)
        let vision = try VisionPackLocation.companionURL(forTextModel: text)
        let manifest = vision.appendingPathComponent("manifest.json")
        var json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any])
        json["compatibleTextManifestSha256"] = String(repeating: "0", count: 64)
        try JSONSerialization.data(withJSONObject: json).write(to: manifest)
        let candidate = try #require(ExistingModelSelection.inspect(root).candidates.first)
        #expect(candidate.canSelect)
        if case .partial = candidate.visionStatus {} else { Issue.record("Incompatible vision must explain the failure") }
        #expect(candidate.visionSummary.contains("invalid or incompatible"))
    }

    @MainActor @Test(arguments: [false, true], [false, true])
    func singleValidModelIsSelectedWithoutReview(vision: Bool, direct: Bool) async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let packs = root.appendingPathComponent("packs")
        try FileManager.default.createDirectory(at: packs, withIntermediateDirectories: true)
        let text = try fixture(in: packs, name: "chosen", vision: vision)
        try FileManager.default.createDirectory(at: packs.appendingPathComponent("partial.gturbo"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: packs.appendingPathComponent("alias.gturbo"), withDestinationURL: text)
        let data = root.appendingPathComponent("app-data")
        let model = AppModel(appDataDirectory: data,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": data.path],
                             client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        defer { model.shutdownForTermination() }
        // Partial packs and aliases must not force confirmation of the only usable model.
        await model.reviewExistingModelLocation(direct ? text : packs)
        #expect(model.existingModelReview == nil)
        #expect(model.modelPathText == text.path)
        #expect(model.modelLocationDiagnostic == nil)
        #expect(model.isVisionPackInstalled == vision)
        #expect(try MacAppSettingsFileStore.load(at: data.appendingPathComponent("mac-app-settings.json"))?.selectedModelPath == text.path)
    }

    @MainActor @Test func reviewCancelAndChangedFilesDoNotCommitSelection() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let packs = root.appendingPathComponent("packs")
        try FileManager.default.createDirectory(at: packs, withIntermediateDirectories: true)
        let text = try fixture(in: packs, name: "chosen", vision: true)
        _ = try fixture(in: packs, name: "other")
        let data = root.appendingPathComponent("app-data")
        let model = AppModel(appDataDirectory: data,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": data.path],
                             client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        defer { model.shutdownForTermination() }
        let previous = model.modelPathText
        let settings = data.appendingPathComponent("mac-app-settings.json")
        let saved = try Data(contentsOf: settings)
        await model.reviewExistingModelLocation(packs)
        #expect(model.modelPathText == previous)
        #expect(try Data(contentsOf: settings) == saved)
        model.existingModelReview = nil
        #expect(try Data(contentsOf: settings) == saved)
        await model.reviewExistingModelLocation(packs)
        let candidate = try #require(model.existingModelReview?.candidates.first)
        try FileManager.default.removeItem(at: candidate.visionURL)
        #expect(await !model.confirmExistingModelSelection(candidate))
        #expect(model.modelPathText == previous)
        #expect(try Data(contentsOf: settings) == saved)
        let refreshed = try #require(model.existingModelReview?.candidates.first)
        #expect(refreshed.visionStatus == .missing)
        #expect(await model.confirmExistingModelSelection(refreshed))
        #expect(model.modelPathText == text.path)
        #expect(try MacAppSettingsFileStore.load(at: settings)?.selectedModelPath == text.path)
        #expect(model.existingModelReview == nil)
        #expect(!model.isVisionPackInstalled)
        model.shutdownForTermination()
        let reopened = AppModel(appDataDirectory: data,
                                environment: ["TURBO_FIELDFARE_APP_DATA_PATH": data.path],
                                client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        defer { reopened.shutdownForTermination() }
        #expect(reopened.modelPathText == text.path)
    }

    @MainActor @Test func aliasesAreDeduplicatedAndBusyConfirmationDoesNotChangeSelection() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let text = try fixture(in: root, name: "original")
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("alias.gturbo"), withDestinationURL: text)
        #expect(try ExistingModelSelection.inspect(root).candidates.count == 1)
        let aliases = root.appendingPathComponent("aliases")
        try FileManager.default.createDirectory(at: aliases, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: aliases.appendingPathComponent("external.gturbo"), withDestinationURL: text)
        #expect(try ExistingModelSelection.inspect(aliases).candidates.first?.id == text.path)
        let data = root.appendingPathComponent("settings")
        let model = AppModel(appDataDirectory: data, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": data.path], client: MockLifecycleInferenceClient())
        defer { model.shutdownForTermination() }
        let previous = model.modelPathText
        _ = try fixture(in: root, name: "second")
        await model.reviewExistingModelLocation(root)
        let candidate = try #require(model.existingModelReview?.candidates.first)
        model.loadState = .loading(.tokenizer)
        #expect(await !model.confirmExistingModelSelection(candidate))
        #expect(model.modelPathText == previous)
    }

    @MainActor @Test func invalidTextCandidateCannotBeConfirmed() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("partial.gturbo"), withIntermediateDirectories: true)
        let review = try ExistingModelSelection.inspect(root)
        #expect(review.initialSelection == nil)
        #expect(review.candidates[0].canSelect == false)
        let model = AppModel(appDataDirectory: root, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path], client: MockLifecycleInferenceClient())
        defer { model.shutdownForTermination() }
        let previous = model.modelPathText
        await model.reviewExistingModelLocation(root)
        #expect(model.existingModelReview == nil)
        #expect(model.modelLocationDiagnostic?.contains("manifest.json is missing") == true)
        #expect(await !model.confirmExistingModelSelection(review.candidates[0]))
        #expect(model.modelPathText == previous)
    }
}
