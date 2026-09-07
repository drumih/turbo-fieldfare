import Foundation
private struct PreparedVisionAppFixture {
  let text: URL
  let output: URL
  let partial: URL
  let checkpoint: URL

  func remove() {
    try? FileManager.default.removeItem(at: text)
    try? FileManager.default.removeItem(at: partial)
    try? FileManager.default.removeItem(at: checkpoint)
  }
}

private func makePreparedVisionAppFixture(_ tag: String) throws
  -> PreparedVisionAppFixture {
  let text = try makeCompleteModelInstall(tag)
  let output = text.deletingLastPathComponent().appendingPathComponent(
    text.deletingPathExtension().lastPathComponent + ".vision.gturbo")
  let paths = try RemoteInstallPaths(outputDirectory: output.path)
  try FileManager.default.createDirectory(
    atPath: paths.partialDirectory,
    withIntermediateDirectories: true)
  try Data("checkpoint".utf8).write(to: URL(fileURLWithPath: paths.checkpointFile))
  return PreparedVisionAppFixture(
    text: text,
    output: output.standardizedFileURL,
    partial: URL(fileURLWithPath: paths.partialDirectory),
    checkpoint: URL(fileURLWithPath: paths.checkpointFile))
}
import Testing
import TurboFieldfareRepackCore

@testable import TurboFieldfareAppCore

@Suite struct AppModelInstallTests {

  @MainActor
  @Test func completedInstallRebindsHistoryWithoutRelaunch() async throws {
    let fixture = try makeCompleteModelInstall("history-rebind")
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("history-rebind-\(UUID())", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer {
      try? FileManager.default.removeItem(at: fixture)
      try? FileManager.default.removeItem(at: root)
    }
    let installed = root.appendingPathComponent("model.gturbo", isDirectory: true)
    try FileManager.default.moveItem(at: fixture, to: installed)
    let receiptURL = installed.appendingPathComponent("verified-install.json")
    var receipt = try #require(
      JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as? [String: Any])
    receipt["modelDirectoryPath"] = installed.standardizedFileURL.path
    try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
      .write(to: receiptURL)
    let missing = root
      .appendingPathComponent("missing-\(UUID()).gturbo", isDirectory: true)
    let client = FakeInferenceClient(eventDelay: .milliseconds(1))
    let model = AppModel(
      modelDirectory: missing,
      client: client,
      installer: MockModelInstallerClient(events: [.installed(installed)]),
      settingsPersistenceEnabled: true)

    model.installModel()
    try await waitUntil {
      if case .installed = model.installState { return true }
      return false
    }
    #expect(model.conversationStore != nil)
    #expect(model.conversationIdentity != nil)

    try await client.ensureLoaded(
      modelDirectory: installed, maxContextTokens: model.maxContextTokens,
      options: model.runtimeOptions, forceLogitsHead: true) { _ in }
    model.loadState = .ready(modelDirectory: installed, loadSeconds: 0)
    model.promptText = "persist after install"
    model.send()
    await SendWaiting.turnEnds(model)
    await model.persistenceTail?.value

    let id = try #require(model.storedConversationID)
    let opened = try await #require(model.conversationStore).open(id: id)
    #expect(opened.meta.turnCount == 2)
  }

  @MainActor
  @Test func missingModelCanInstall() {
    let installer = MockModelInstallerClient()
    let directory = temporaryInstallPath("missing")
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(!model.isModelInstalled)
    #expect(model.requiresModelInstallation)
    #expect(model.canInstallModel)
  }

  @MainActor
  @Test func installedModelShowsLoadNotInstall() throws {
    let directory = try makeCompleteModelInstall("installed")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(!model.canInstallModel)
    #expect(model.canLoadModel)
    #expect(model.canInstallVisionPack)
  }

  @MainActor
  @Test func visionInstallProgressCanCancelAndResumeWithoutBlockingText() async throws {
    let directory = try makeCompleteModelInstall("vision-progress")
    defer { try? FileManager.default.removeItem(at: directory) }
    let visionInstaller = MockVisionPackInstallerClient(
      events: [.copyingPayload(
        reusedBytes: 4,
        downloadedThisRunBytes: 6,
        totalBytes: 20)],
      holdOpen: true)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(),
      visionInstaller: visionInstaller)

    model.installVisionPack()
    try await waitUntil {
      model.visionInstallState == .copyingPayload(
        reusedBytes: 4,
        downloadedThisRunBytes: 6,
        totalBytes: 20)
    }
    #expect(model.visionInstallProgressFraction == 0.5)
    #expect(model.isModelInstalled)
    model.cancelVisionInstall()
    try await waitUntil { model.visionInstallState == .cancelled }
    #expect(model.canInstallVisionPack)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func visionPrepareWaitsForExplicitActivation() async throws {
    let directory = try makeCompleteModelInstall("vision-ready")
    defer { try? FileManager.default.removeItem(at: directory) }
    let companion = directory.deletingLastPathComponent()
      .appendingPathComponent(
        directory.deletingPathExtension().lastPathComponent + ".vision.gturbo")
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(),
      visionInstaller: MockVisionPackInstallerClient(
        events: [.readyToActivate(companion)]))

    model.installVisionPack()
    try await waitUntil {
      model.visionInstallState == .readyToActivate(companion)
    }

    #expect(!model.isVisionPackInstalled)
    #expect(!model.canInstallVisionPack)
    #expect(model.canActivateVisionPack)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func busyVisionActivationRemainsReadyWithoutRedownload() async throws {
    let fixture = try makePreparedVisionAppFixture("vision-activation-busy")
    defer { fixture.remove() }
    let model = AppModel(
      modelDirectory: fixture.text,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(),
      visionInstaller: MockVisionPackInstallerClient(
        preparedValid: true,
        activationError: .installBusy(path: fixture.output.path)))

    #expect(model.canActivateVisionPack)
    model.activateVisionPack()
    try await waitUntil {
      if case .readyToActivate = model.visionInstallState { return true }
      return false
    }
    #expect(model.canActivateVisionPack)
  }

  @MainActor
  @Test func corruptVisionActivationBecomesRecoverable() async throws {
    let fixture = try makePreparedVisionAppFixture("vision-activation-corrupt")
    defer { fixture.remove() }
    let model = AppModel(
      modelDirectory: fixture.text,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(),
      visionInstaller: MockVisionPackInstallerClient(
        preparedValid: true,
        activationError: .configurationInvalid(detail: "weights hash mismatch")))

    model.activateVisionPack()
    try await waitUntil {
      if case .recoverable = model.visionInstallState { return true }
      return false
    }
    #expect(model.canInstallVisionPack)
  }

  @MainActor
  @Test func loadedModelAllowsVisionPayloadDownload() throws {
    let directory = try makeCompleteModelInstall("vision-loaded-prepare")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(),
      visionInstaller: MockVisionPackInstallerClient(holdOpen: true))
    model.loadState = .ready(modelDirectory: directory, loadSeconds: 0.5)

    #expect(!model.canBeginVisionCompanionOperation)
    #expect(model.canInstallVisionPack)
    model.installVisionPack()
    #expect(model.visionInstallState != .idle)
  }

  @MainActor
  @Test func loadedModelBlocksVisionActivationAndDiscard() throws {
    let fixture = try makePreparedVisionAppFixture("vision-loaded-activate")
    defer { fixture.remove() }
    let model = AppModel(
      modelDirectory: fixture.text,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(),
      visionInstaller: MockVisionPackInstallerClient(preparedValid: true))
    model.loadState = .ready(modelDirectory: fixture.text, loadSeconds: 0.5)

    #expect(!model.canActivateVisionPack)
    #expect(!model.canDiscardVisionPackDownload)
    model.activateVisionPack()
    guard case .readyToActivate = model.visionInstallState else {
      Issue.record("blocked activation left \(model.visionInstallState)")
      return
    }

    model.loadState = .notLoaded
    #expect(model.canActivateVisionPack)
    #expect(model.canDiscardVisionPackDownload)
  }

  @MainActor
  @Test func visionPayloadDownloadKeepsTextRuntimeAvailableAndDraftIntact() async throws {
    let directory = try makeCompleteModelInstall("vision-blocking")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(),
      visionInstaller: MockVisionPackInstallerClient(
        events: [.copyingPayload(
          reusedBytes: 4,
          downloadedThisRunBytes: 6,
          totalBytes: 20)],
        holdOpen: true))
    model.promptText = "describe this"
    model.outputText = "earlier answer"
    model.loadState = .ready(modelDirectory: directory, loadSeconds: 0.5)

    model.installVisionPack()
    try await waitUntil { model.isVisionCompanionOperationInProgress }

    #expect(model.canUnloadModel)
    #expect(!model.canInstallModel)
    #expect(model.canRun)
    #expect(!model.canInstallVisionPack)

    model.cancelVisionInstall()
    try await waitUntil { model.visionInstallState == .cancelled }

    #expect(model.promptText == "describe this")
    #expect(model.outputText == "earlier answer")
    #expect(model.canUnloadModel)
    #expect(model.canInstallVisionPack)
  }

  @MainActor
  @Test func checkAgainDetectsModelInstalledAfterLaunch() throws {
    let directory = try makeCompleteModelInstall("external-install")
    let stagedDirectory = directory.deletingLastPathComponent()
      .appendingPathComponent("staged-\(UUID().uuidString).gturbo")
    try FileManager.default.moveItem(at: directory, to: stagedDirectory)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.requiresModelInstallation)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.moveItem(at: stagedDirectory, to: directory)

    model.refreshInstallReadiness()

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func checkAgainUsesCurrentModelLocation() throws {
    let initialDirectory = temporaryInstallPath("initial-location")
    let currentDirectory = try makeCompleteModelInstall("current-location")
    defer { try? FileManager.default.removeItem(at: currentDirectory) }
    let model = AppModel(
      modelDirectory: initialDirectory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    model.modelPathText = currentDirectory.path
    model.recheckModelAtCurrentLocation()

    #expect(model.modelPathText == currentDirectory.standardizedFileURL.path)
    #expect(model.isModelInstalled)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func defaultInstallDescriptorMatchesPinnedAudit() {
    let descriptor = AppModelInstallDescriptor.default
    #expect(descriptor.displayName == "Gemma 4 26B-A4B IT 4-bit")
    #expect(descriptor.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
    #expect(descriptor.revision == "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
    #expect(descriptor.sourceIndexSHA256 == "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
    #expect(descriptor.approximateDownloadBytes == 14_620_479_420)
    #expect(descriptor.installedBytes == 14_291_921_884)
    #expect(descriptor.requiredFreeBytes == 15_432_772_572)
  }

  @MainActor
  @Test func insufficientSpaceDisablesInstallAndExposesShortfall() {
    let requirement = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 100,
      availableBytes: 40)
    let installer = MockModelInstallerClient(requirement: requirement)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("space"),
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(model.installReadiness == .insufficientSpace(requirement))
    #expect(model.installRequirement?.shortfallBytes == 60)
    #expect(!model.canInstallModel)
  }

  @MainActor
  @Test func installProgressUpdatesStatusAndByteCounts() async throws {
    let directory = temporaryInstallPath("progress")
    let installer = MockModelInstallerClient(
      events: [
        .checking,
        .copyingPayload(
          reusedBytes: 1,
          downloadedThisRunBytes: 3,
          totalBytes: 10),
      ], holdOpen: true)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      model.installState == .copyingPayload(
        reusedBytes: 1,
        downloadedThisRunBytes: 3,
        totalBytes: 10)
    }
    #expect(model.installDownloadedBytes == 4)
    #expect(model.installTotalBytes == 10)
    #expect(model.installProgressFraction == 0.4)
    #expect(model.presentation.label == "Downloading model")
    model.cancelInstall()
    try await waitUntil { model.installState == .cancelled }
  }

  @MainActor
  @Test func installCompletionStopsUnloaded() async throws {
    let requestedDirectory = temporaryInstallPath("requested")
    let completedDirectory = try makeCompleteModelInstall("complete")
    defer { try? FileManager.default.removeItem(at: completedDirectory) }
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(events: [.installed(completedDirectory)])
    let model = AppModel(
      modelDirectory: requestedDirectory,
      client: client,
      installer: installer)

    model.installModel()
    try await waitUntil {
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL)
    }
    #expect(
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL))
    #expect(model.modelPathText == completedDirectory.standardizedFileURL.path)
    #expect(model.loadState == .notLoaded)
    #expect(model.canLoadModel)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func installFailureDoesNotAttemptLoad() async throws {
    struct SyntheticError: Error {}
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(failure: SyntheticError())
    let model = AppModel(
      modelDirectory: temporaryInstallPath("failure"),
      client: client,
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }
    #expect(model.loadState == .notLoaded)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func networkFailureWithSavedProgressLeavesResumeEnabled() async throws {
    struct NetworkFailure: Error {}
    let directory = temporaryInstallPath("network-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: NetworkFailure()))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func diskFailureCanResumeAfterSpaceRecheck() async throws {
    let directory = temporaryInstallPath("disk-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: error))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }
    #expect(!model.canInstallModel)

    model.recheckModelAtCurrentLocation()

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func invalidSavedDownloadsRemainDiscardOnly() throws {
    let descriptor = AppModelInstallDescriptor.default
    for incompatible in [false, true] {
      let directory = temporaryInstallPath(
        incompatible ? "incompatible-checkpoint" : "corrupt-checkpoint")
      let paths = try makeSavedDownload(at: directory)
      defer { cleanUpSavedDownload(paths) }
      if incompatible {
        try RemoteInstallCheckpoint(
          repoID: "other/model",
          requestedRevision: descriptor.revision,
          resolvedCommit: String(repeating: "a", count: 40),
          sourceIndexSHA256: String(repeating: "b", count: 64),
          planFingerprint: String(repeating: "c", count: 64),
          totalSourceBytes: 1
        ).write(
          to: paths.checkpointFile,
          parentDirectory: paths.parentDirectory)
      } else {
        try Data("{}".utf8).write(
          to: URL(fileURLWithPath: paths.checkpointFile))
      }
      let model = AppModel(
        modelDirectory: directory,
        client: MockLifecycleInferenceClient(),
        installer: RepackModelInstallerClient(descriptor: descriptor))

      #expect(!model.canInstallModel)
      #expect(model.canDiscardModelDownload)
      guard case .failed = model.installReadiness else {
        Issue.record("invalid checkpoint did not fail readiness")
        continue
      }
    }
  }

  @MainActor
  @Test func diskFailureKeepsExactRequirementAndShortfall() async throws {
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let installer = MockModelInstallerClient(failure: error)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("disk-failure"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }

    let expected = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 120,
      availableBytes: 45)
    #expect(model.installReadiness == .insufficientSpace(expected))
    #expect(model.installRequirement?.shortfallBytes == 75)
  }

  @MainActor
  @Test func cancelInstallWaitsForAcknowledgementAndAllowsRetry() async throws {
    let installer = MockModelInstallerClient(
      events: [.downloadingMetadata],
      holdOpen: true,
      delayCancellationAcknowledgement: true)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("cancel"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()
    try await waitUntil { model.installState == .downloadingMetadata }

    model.cancelInstall()
    #expect(installer.cancelCalled)
    try await waitUntil { installer.cancellationAcknowledgementPending }
    #expect(model.installState == .cancelling)
    #expect(!model.canInstallModel)

    await installer.releaseCancellationAcknowledgement()
    try await waitUntil { model.installState == .cancelled }

    #expect(model.loadState == .notLoaded)
    #expect(model.canInstallModel)
  }

  private func temporaryInstallPath(_ tag: String) -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("turbofieldfare-app-install-\(tag)-\(UUID().uuidString).gturbo")
  }

  private func makeSavedDownload(at directory: URL) throws -> RemoteInstallPaths {
    let paths = try RemoteInstallPaths(outputDirectory: directory.path)
    try FileManager.default.createDirectory(
      atPath: paths.partialDirectory,
      withIntermediateDirectories: true)
    return paths
  }

  private func cleanUpSavedDownload(_ paths: RemoteInstallPaths) {
    for path in [
      paths.finalDirectory,
      paths.partialDirectory,
      paths.checkpointFile,
      paths.lockFile,
    ] {
      try? FileManager.default.removeItem(atPath: path)
    }
  }

  @MainActor
  private func waitUntil(_ predicate: @escaping @MainActor () -> Bool) async throws {
    for _ in 0..<200 {
      if predicate() { return }
      try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for condition")
  }

}
