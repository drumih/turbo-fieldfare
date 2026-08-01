import Foundation
import TurboFieldfareDecodeProtocol
import TurboFieldfareRepackCore
import Observation

@MainActor
@Observable
public final class AppModel {
    public enum RunState: Equatable {
        case idle
        case running
    }

    public var modelPathText: String
    public var promptText: String = ""
    /// The in-flight exchange. Committed turns live in `activeConversation`;
    /// these two hold the turn being generated right now, so the transcript's
    /// committed prefix stays immutable while tokens stream in.
    public private(set) var outputPromptText: String = ""
    public var outputText: String = ""
    public private(set) var conversations: [Conversation] = []
    public private(set) var activeConversationID: UUID?
    /// Set when a send was refused because the conversation no longer leaves
    /// room for a reply. Distinct from `error`, which means a run actually failed.
    public private(set) var isContextOverflowNoticeVisible = false
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens: Int = 4096
    public var temperature: Double = 0.2
    public var topKEnabled: Bool = true
    public var topK: Int = 64
    public var topPEnabled: Bool = true
    public var topP: Double = 0.95
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    /// Mirrored into `installStates` on every write so the picker's per-row
    /// progress stays in step with the single-model state the existing install
    /// UI reads, without duplicating the assignment at each transition.
    public var installState: AppModelInstallState = .idle {
        didSet { installStates.setState(installState, for: selectedRepoID) }
    }
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus

    /// Curated entries merged with whatever the user has added.
    public private(set) var catalog: ModelCatalog = ModelCatalog(custom: [])
    /// Install progress for every known model, so the picker can render each
    /// row independently of the single-model `installState` above.
    public private(set) var installStates = ModelInstallStates()
    /// Repository ID of the selected model, loaded or not.
    public private(set) var selectedRepoID: String = ModelCatalog.curated.first?.repoID ?? ""

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public private(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    public private(set) var livePrefillDone: Int = 0
    public private(set) var livePrefillTotal: Int = 0
    public private(set) var liveMemoryBytes: UInt64?
    public private(set) var isCancellationPending: Bool = false

    private let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private var runTask: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var unloadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    private var hasHandledTerminalEvent = false
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    private let conversationsPersistenceEnabled: Bool
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                settingsPersistenceEnabled: Bool = false,
                conversationsPersistenceEnabled: Bool = false,
                migratesLegacyInstall: Bool = false) {
        // Adopt a pre-multi-model install before probing, so an existing model
        // is found at the new slug path instead of appearing missing.
        //
        // Off by default and opted into only by the real app: this moves a
        // multi-gigabyte directory on the user's disk, and a test run that
        // constructs an AppModel must never do that.
        if migratesLegacyInstall, modelDirectory == nil,
           let curated = ModelCatalog.curated.first {
            AppModelLocation.migrateLegacyInstallIfNeeded(repoID: curated.repoID)
        }
        let directory = (modelDirectory ?? AppModelLocation.curatedDefaultURL()).standardizedFileURL
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
            : MacAppSettings()
        self.modelPathText = directory.path
        self.runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled)
        self.maxContextTokens = settings.contextTokens
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.memorySampler = memorySampler
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.conversationsPersistenceEnabled = conversationsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        refreshInstallReadiness()
        loadCatalog()
        loadConversations()
    }

    /// Support directory holding `catalog.json` and `conversations.json`.
    /// Derived from the model directory's ancestry so a dev build writing into
    /// `scratch/models/<slug>/model.gturbo` keeps its state beside the repo
    /// rather than in Application Support.
    private var supportDirectory: URL {
        URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func loadCatalog() {
        let custom = conversationsPersistenceEnabled
            ? ModelCatalogStore.load(inSupportDirectory: supportDirectory).customEntries
            : []
        catalog = ModelCatalog(custom: custom)
        if case .complete = installationStatus {
            installStates.setState(
                .installed(modelDirectory: URL(fileURLWithPath: modelPathText, isDirectory: true)),
                for: selectedRepoID)
        }
    }

    public var isRunning: Bool { runState == .running }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        isModelInstalled && !isRunning && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady
    }

    public var isModelInstalled: Bool { installationStatus == .complete }

    public var requiresModelInstallation: Bool { !isModelInstalled }

    public var installDescriptor: AppModelInstallDescriptor { installer.descriptor }

    public var installRequirement: AppModelInstallRequirement? {
        installReadiness.requirement
    }

    public var isInstallingModel: Bool { installState.isInstalling }

    public var canInstallModel: Bool {
        guard case .ready = installReadiness else { return false }
        return !isRunning && !loadState.isLoading && !isInstallingModel
            && requiresModelInstallation
    }

    public var canCancelInstall: Bool { installState.canCancel }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        return min(addition.overflow ? UInt64.max : addition.partialValue, total)
    }

    public var installTotalBytes: UInt64? {
        guard case .copyingPayload(_, _, let total) = installState else {
            return nil
        }
        return total
    }

    public var installReusedBytes: UInt64? {
        guard case .copyingPayload(let reused, _, _) = installState else {
            return nil
        }
        return reused
    }

    public var installDownloadedThisRunBytes: UInt64? {
        guard case .copyingPayload(_, let downloaded, _) = installState else {
            return nil
        }
        return downloaded
    }

    public var installProgressFraction: Double? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState,
              total > 0 else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var installPhaseLabel: String {
        switch installState {
        case .idle: return "Model required"
        case .checking: return "Checking installation"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning installation"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading model"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing installation"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding download"
        case .cancelled: return "Download paused"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Model installed"
        case .failed: return "Installation failed"
        }
    }

    public var canRun: Bool {
        !isRunning && isModelAvailable && !loadState.isLoading
            && !hasStaleLoadedRuntime
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    public var hasOutputTranscript: Bool {
        !committedTurns.isEmpty || !outputPromptText.isEmpty || !outputText.isEmpty
    }

    public var outputResponsePlainText: String {
        generationTranscriptMailbox?.completeText ?? outputText
    }

    public var outputConversationPlainText: String {
        var sections: [String] = []
        for turn in committedTurns {
            sections.append(turn.role == .user
                ? "You:\n\(turn.content)"
                : "Answer:\n\(turn.content)")
        }
        if !outputPromptText.isEmpty { sections.append("You:\n\(outputPromptText)") }
        let response = outputResponsePlainText
        if !response.isEmpty { sections.append("Answer:\n\(response)") }
        return sections.joined(separator: "\n\n")
    }

    // MARK: - Conversations

    public var activeConversation: Conversation? {
        guard let activeConversationID else { return nil }
        return conversations.first { $0.id == activeConversationID }
    }

    /// Turns already committed to the active conversation. Immutable, and the
    /// stable prefix the transcript renderer appends to.
    public var committedTurns: [ChatTurn] { activeConversation?.turns ?? [] }

    /// Newest first, so the sidebar shows recent chats at the top.
    public var conversationsByRecency: [Conversation] {
        conversations.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// A reply needs room too: blocking exactly at the context limit would
    /// permit sends that can only produce a truncated answer, because
    /// `effectiveMaxNewTokens` clamps the response to whatever is left.
    public static let responseHeadroomTokens = 256

    /// Character-based, deliberately conservative. Real Gemma ratios run about
    /// 4 chars/token for prose and 3 for code, so 3.5 errs toward over-counting:
    /// blocking slightly early is recoverable by raising the context length,
    /// while under-counting fails mid-send inside the decode service.
    static func estimateTokens(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        return max(1, Int((Double(text.count) / 3.5).rounded(.up)))
    }

    /// Anchored on the exact prompt count the service reported for the previous
    /// generation, so only the turns added since then are estimated.
    public var estimatedNextPromptTokens: Int {
        let pending = Self.estimateTokens(
            promptText.trimmingCharacters(in: .whitespacesAndNewlines))
        guard let conversation = activeConversation else { return pending }
        guard let anchor = conversation.lastPromptTokenCount else {
            return conversation.turns.reduce(0) {
                $0 + Self.estimateTokens($1.content)
            } + pending
        }
        let sinceAnchor = conversation.turns
            .last(where: { $0.role == .assistant })
            .map { Self.estimateTokens($0.content) } ?? 0
        return anchor + sinceAnchor + pending
    }

    public var contextCapacityTokens: Int { maxContextTokens }

    /// What the meter shows: the conversation as it stands, excluding whatever
    /// is still being typed.
    public var contextUsedTokens: Int {
        guard let conversation = activeConversation else { return 0 }
        guard let anchor = conversation.lastPromptTokenCount else {
            return conversation.turns.reduce(0) {
                $0 + Self.estimateTokens($1.content)
            }
        }
        let sinceAnchor = conversation.turns
            .last(where: { $0.role == .assistant })
            .map { Self.estimateTokens($0.content) } ?? 0
        return anchor + sinceAnchor
    }

    /// True when the context count is a guess rather than a service-reported
    /// figure, so the UI can mark it.
    public var isContextUsageEstimated: Bool {
        activeConversation?.lastPromptTokenCount == nil
    }

    public var isConversationOverflowing: Bool {
        estimatedNextPromptTokens + Self.responseHeadroomTokens > maxContextTokens
    }

    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    public var presentation: AppPresentationState {
        AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: requiresModelInstallation,
            installState: installState,
            installReadiness: installReadiness,
            loadState: loadState,
            hasStaleRuntime: hasStaleLoadedRuntime,
            isRunning: isRunning,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            lastStopReason: diagnostics?.stopReason))
    }

    public var currentProcessMemoryBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        if let reporter = client as? any AppInferenceMemoryReporting,
           let bytes = reporter.currentInferenceMemoryBytes {
            return bytes
        }
        return memorySampler.sample()
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox? {
        (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
    }

    private var currentRuntimeKey: AppLoadedRuntimeKey {
        AppLoadedRuntimeKey(modelDirectory: URL(fileURLWithPath: modelPathText),
                            maxContextTokens: maxContextTokens,
                            options: runtimeOptions,
                            forceLogitsHead: currentForceLogitsHead)
    }

    private var currentForceLogitsHead: Bool {
        temperature != 0
    }

    public func setModelURL(_ url: URL) {
        guard !isRunning else { return }
        let path = url.standardizedFileURL.path
        guard path != modelPathText else { return }

        modelPathText = path
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        loadConversations()
        resetLiveTurn()
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        installGeneration &+= 1
        installTask?.cancel()
        installer.cancel()
        installTask = nil
        resetInstallETA()
        installState = .idle
        pendingExplicitLoadRuntimeKey = nil
        activeRunRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        diagnostics = nil
        error = nil
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(at: URL(fileURLWithPath: path))
        refreshInstallReadiness()

        if let lifecycle = client as? AppModelLifecycleClient {
            unloadGeneration &+= 1
            let generation = unloadGeneration
            let task = Task { [weak self, lifecycle] in
                await lifecycle.unload()
                self?.clearUnloadTask(generation: generation)
            }
            unloadTask = task
        }
    }

    public func loadModel() {
        guard canLoadModel else { return }
        beginLoad()
    }

    public func perform(_ action: AppModelAction) {
        switch action {
        case .install: installModel()
        case .cancelInstall: cancelInstall()
        case .load, .retryLoad: loadModel()
        case .cancelLoad: cancelLoad()
        case .reload: reloadModel()
        case .unload: unloadModel()
        }
    }

    public func reloadModel() {
        guard canReloadModel else { return }
        beginLoad()
    }

    private func beginLoad() {
        guard let lifecycle = client as? AppModelLifecycleClient else {
            loadState = .failed(.modelLoadFailed("This client has no model load lifecycle."))
            return
        }
        let directory = URL(fileURLWithPath: modelPathText)
        let maxContext = maxContextTokens
        let options = runtimeOptions
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: options,
                                             forceLogitsHead: forceLogitsHead)
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        error = nil
        loadState = .loading(.validatingDirectory)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation)
                    }
                }
            } catch is CancellationError {
            } catch let appError as AppInferenceError {
                await self?.applyLoadState(.failed(appError), generation: generation)
            } catch {
                await self?.applyLoadState(
                    .failed(.modelLoadFailed("\(error)")),
                    generation: generation)
            }
            await self?.clearLoadTask(generation: generation)
        }
    }

    public func cancelLoad() {
        guard canCancelLoad, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .cancelling
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func installModel() {
        guard !isRunning, !loadState.isLoading, !isInstallingModel,
              requiresModelInstallation else {
            return
        }
        refreshInstallReadiness()
        guard canInstallModel else { return }
        installTask?.cancel()
        installer.cancel()
        resetInstallETA()
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .checking
        let entry = selectedEntry
            ?? ModelCatalogEntry(descriptor: installer.descriptor, trustTier: .curated)
        installTask = Task { [weak self, installer] in
            do {
                for try await event in installer.install(entry: entry,
                                                         outputDirectory: outputDirectory) {
                    guard let self else { return }
                    self.applyInstallEvent(event, generation: generation)
                }
                self?.finishInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishInstallCancellation(generation: generation)
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelInstall() {
        guard canCancelInstall else { return }
        installState = .cancelling
        installer.cancel()
    }

    // MARK: - Model selection

    public var selectedEntry: ModelCatalogEntry? {
        catalog.entry(forRepoID: selectedRepoID)
    }

    /// Resident bytes of the selected model, measured from its manifest when
    /// installed. Drives the context picker's memory maths.
    public var residentWeightBytes: UInt64 {
        AppResidentWeightProbe.residentBytesOrEstimate(
            atModelDirectory: URL(fileURLWithPath: modelPathText, isDirectory: true))
    }

    public func switchVerdict(for entry: ModelCatalogEntry) -> ModelSwitchVerdict {
        ModelSwitchGuard.evaluate(
            target: entry,
            currentRepoID: selectedRepoID,
            loadState: loadState,
            isGenerating: isRunning,
            installStates: installStates)
    }

    /// Switches the loaded model.
    ///
    /// Prompt-cache correctness rides on `AppPromptCacheDomain`, which already
    /// keys on `modelID` and `sourceSnapshotHash`, so an entry cached under the
    /// previous model cannot match after the switch. The cache itself is
    /// private to `RealInferenceClient` and is not reachable from here.
    public func switchModel(to entry: ModelCatalogEntry) {
        switch switchVerdict(for: entry) {
        case .alreadyLoaded:
            return
        case .blockedByGeneration:
            error = .generationInFlight
            return
        case .busy:
            return
        case .notInstalled:
            selectModel(entry)
            return
        case .allowed:
            break
        }
        selectModel(entry)
        loadModel()
    }

    /// Repoints state at a model without loading it, so the picker can select
    /// an uninstalled entry and then download it.
    private func selectModel(_ entry: ModelCatalogEntry) {
        guard let directory = try? AppModelLocation.defaultURL(forRepoID: entry.repoID) else {
            error = .invalidRequest("Invalid repository ID: \(entry.repoID)")
            return
        }
        if loadState.isReady { unloadModel() }
        selectedRepoID = entry.repoID
        modelPathText = directory.standardizedFileURL.path
        installationStatus = AppModelInstallationProbe.status(at: directory)
        if case .complete = installationStatus {
            installStates.setState(.installed(modelDirectory: directory), for: entry.repoID)
        }
        installState = installStates.state(for: entry.repoID)
        refreshInstallReadiness()
    }

    public func startInstall(for entry: ModelCatalogEntry) {
        selectModel(entry)
        installModel()
    }

    /// Removes installed weights but keeps the catalog entry, so the model can
    /// be re-downloaded without retyping the repository.
    public func deleteInstall(for entry: ModelCatalogEntry) {
        guard !(selectedRepoID == entry.repoID && loadState.isReady) else {
            error = .invalidRequest("Unload \(entry.displayName) before deleting it.")
            return
        }
        guard let directory = try? AppModelLocation.defaultURL(forRepoID: entry.repoID) else {
            return
        }
        // Removes the slug directory, not just model.gturbo, so a partial
        // install's staging files go with it.
        try? FileManager.default.removeItem(at: directory.deletingLastPathComponent())
        installStates.setState(.idle, for: entry.repoID)
        if selectedRepoID == entry.repoID {
            installState = .idle
            installationStatus = AppModelInstallationProbe.status(at: directory)
            refreshInstallReadiness()
        }
    }

    /// Adds a user-supplied repository after the caller has run the
    /// architecture pre-flight and obtained consent.
    @discardableResult
    public func addCustomModel(_ entry: ModelCatalogEntry) -> Bool {
        guard let updated = try? catalog.addingCustom(entry) else { return false }
        catalog = updated
        if conversationsPersistenceEnabled {
            try? ModelCatalogStore.save(
                ModelCatalogFile(customEntries: updated.customEntries),
                inSupportDirectory: supportDirectory)
        }
        return true
    }

    public var hasPartialModelDownload: Bool {
        guard let paths = try? RemoteInstallPaths(outputDirectory: modelPathText) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardModelDownload: Bool {
        hasPartialModelDownload && !isInstallingModel && !isRunning
    }

    public func discardModelDownload() {
        guard canDiscardModelDownload else { return }
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .discarding
        installTask = Task { [weak self, installer] in
            do {
                try await installer.discardPartialInstall(
                    outputDirectory: outputDirectory)
                guard let self, generation == self.installGeneration else { return }
                self.installTask = nil
                self.installState = .idle
                self.refreshInstallReadiness()
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func refreshInstallReadiness() {
        refreshInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true).standardizedFileURL)
    }

    public func recheckModelAtCurrentLocation() {
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        modelPathText = directory.path
        refreshInstallReadiness(at: directory)
    }

    private func refreshInstallReadiness(at outputDirectory: URL) {
        installationStatus = AppModelInstallationProbe.status(
            at: outputDirectory,
            descriptor: installer.descriptor)
        guard !isModelInstalled else { return }
        installReadiness = .checking
        do {
            let requirement = try installer.checkInstallRequirement(
                outputDirectory: outputDirectory)
            installReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            installReadiness = .failed("\(error)")
        }
    }

    private func applyInstallEvent(_ event: AppModelInstallEvent, generation: UInt64) {
        guard generation == installGeneration else { return }
        switch event {
        case .checking:
            resetInstallETA()
            installState = .checking
        case .downloadingMetadata:
            resetInstallETA()
            installState = .downloadingMetadata
        case .planning:
            resetInstallETA()
            installState = .planning
        case .reservingOutput:
            resetInstallETA()
            installState = .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            installState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
            updateInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            resetInstallETA()
            installState = .hashingOutput(file)
        case .finalizing:
            resetInstallETA()
            installState = .finalizing
        case .installed(let directory):
            resetInstallETA()
            let directory = directory.standardizedFileURL
            installationStatus = AppModelInstallationProbe.status(
                at: directory,
                descriptor: installer.descriptor)
            guard installationStatus == .complete else {
                finishInstallFailure(
                    RepackError.configurationInvalid(detail: "completed install did not pass metadata validation"),
                    generation: generation)
                return
            }
            installState = .installed(modelDirectory: directory)
            installTask = nil
            modelPathText = directory.path
            loadState = .notLoaded
        }
    }

    private func finishInstallStream(generation: UInt64) {
        guard generation == installGeneration, installTask != nil else { return }
        if installState == .cancelling {
            finishInstallCancellation(generation: generation)
        } else if !isModelInstalled {
            finishInstallFailure(
                RepackError.configurationInvalid(detail: "installer ended before completion"),
                generation: generation)
        }
    }

    private func finishInstallCancellation(generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        installState = .cancelled
        resetInstallETA()
        refreshInstallReadiness()
    }

    private func updateInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let timestamp = installETATimestamp
        setInstallETAPresentation(
            installETAEstimator.update(observation, timestamp: timestamp))
    }

    private var installETATimestamp: Double {
        let components = installETAOrigin.duration(to: installETAClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func resetInstallETA() {
        installETAEstimator.reset()
        installETAPresentation = .hidden
        installETAText = nil
    }

    private func setInstallETAPresentation(
        _ presentation: DownloadETAPresentation
    ) {
        installETAPresentation = presentation
        installETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func applyPersistedSettings(forModelDirectory modelDirectory: URL) {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled)
        maxContextTokens = settings.contextTokens
        temperature = settings.temperature
        topKEnabled = settings.topKEnabled
        topK = settings.topK
        topPEnabled = settings.topPEnabled
        topP = settings.topP
    }

    private func persistSettings() {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        try? MacAppSettingsFileStore.save(
            settings,
            forModelDirectory: modelDirectory)
    }

    private func finishInstallFailure(_ error: Error, generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        resetInstallETA()
        let hasSavedDownload = hasPartialModelDownload
        installState = hasSavedDownload ? .recoverable("\(error)") : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            let requirement = AppModelInstallRequirement(probePath: path,
                                                          requiredBytes: required,
                                                          availableBytes: available)
            installReadiness = .insufficientSpace(requirement)
        } else {
            refreshInstallReadiness()
            if hasSavedDownload {
                installState = .recoverable("\(error)")
            }
        }
    }

    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    private func applyLoadState(_ state: AppModelLoadState, generation: UInt64) {
        guard generation == loadGeneration else { return }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        switch state {
        case .notLoaded:
            loadedRuntimeKey = nil
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            loadedRuntimeKey = pendingExplicitLoadRuntimeKey
                ?? activeRunRuntimeKey
                ?? currentRuntimeKey
            pendingExplicitLoadRuntimeKey = nil
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            error = loadError
        }
    }

    /// Kept for the existing "clear" affordances: starting a fresh chat is the
    /// least surprising reading of clearing the transcript now that history
    /// persists.
    public func clearOutput() {
        newConversation()
    }

    public func newConversation() {
        guard !isRunning else { return }
        // Never accumulate empty chats: reuse the active one if it is untouched.
        if let active = activeConversation, active.isEmpty {
            resetLiveTurn()
            return
        }
        let conversation = Conversation()
        conversations.append(conversation)
        activeConversationID = conversation.id
        resetLiveTurn()
        persistConversations()
    }

    public func selectConversation(_ id: UUID) {
        guard !isRunning, id != activeConversationID,
              conversations.contains(where: { $0.id == id }) else { return }
        activeConversationID = id
        resetLiveTurn()
    }

    public func deleteConversation(_ id: UUID) {
        guard !isRunning, let index = conversations.firstIndex(where: { $0.id == id }) else {
            return
        }
        conversations.remove(at: index)
        if activeConversationID == id {
            activeConversationID = conversationsByRecency.first?.id
            resetLiveTurn()
        }
        if conversations.isEmpty {
            let conversation = Conversation()
            conversations.append(conversation)
            activeConversationID = conversation.id
        }
        persistConversations()
    }

    public func renameConversation(_ id: UUID, to title: String) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            // Clearing a custom title hands naming back to the first user turn.
            conversations[index].titleIsCustom = false
            conversations[index].retitleIfNeeded()
        } else {
            conversations[index].title = trimmed
            conversations[index].titleIsCustom = true
        }
        persistConversations()
    }

    private func resetLiveTurn() {
        outputPromptText = ""
        outputText = ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
        isContextOverflowNoticeVisible = false
    }

    public func dismissContextOverflowNotice() {
        isContextOverflowNoticeVisible = false
    }

    /// Conversations live in one global store rather than inside a model
    /// directory: history has to survive a model switch, and deleting a model
    /// must not delete the chats made with it.
    private func loadConversations() {
        let store = conversationsPersistenceEnabled
            ? ConversationFileStore.loadGlobal(inSupportDirectory: supportDirectory)
            : ConversationStoreFile()
        conversations = store.conversations
        if conversations.isEmpty {
            conversations = [Conversation()]
        }
        activeConversationID = conversationsByRecency.first?.id
    }

    private func persistConversations() {
        guard conversationsPersistenceEnabled else { return }
        let store = ConversationStoreFile(conversations: conversations)
        try? ConversationFileStore.saveGlobal(store, inSupportDirectory: supportDirectory)
    }

    /// Commits the finished exchange. Both turns land together so the committed
    /// list is never half-written, and only here does the store get touched —
    /// writing per token would thrash the disk at decode speed.
    private func commitLiveTurn() {
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID })
        else { return }
        let response = outputResponsePlainText
        guard !outputPromptText.isEmpty || !response.isEmpty else { return }
        let now = Date()
        if !outputPromptText.isEmpty {
            conversations[index].turns.append(
                ChatTurn(role: .user, content: outputPromptText, timestamp: now))
        }
        if !response.isEmpty {
            conversations[index].turns.append(
                ChatTurn(role: .assistant, content: response, timestamp: now))
        }
        conversations[index].lastPromptTokenCount = diagnostics?.promptTokenCount
        conversations[index].updatedAt = now
        conversations[index].retitleIfNeeded()
        outputPromptText = ""
        outputText = ""
        generationTranscriptMailbox?.reset()
        persistConversations()
    }

    public func run() {
        guard canRun else { return }
        // Refuse before the request crosses the IPC boundary: the decode service
        // would reject it anyway, and blocking here keeps an oversized prompt
        // from being framed at all.
        guard !isConversationOverflowing else {
            isContextOverflowNoticeVisible = true
            return
        }
        isContextOverflowNoticeVisible = false
        let request: AppGenerationRequest
        do {
            request = try makeRequest()
        } catch let appError as AppInferenceError {
            error = appError
            return
        } catch {
            let appError = AppInferenceError.unknown("\(error)")
            self.error = appError
            return
        }
        persistSettings()

        generationTranscriptMailbox?.reset()
        outputPromptText = request.latestUserContent ?? ""
        outputText = ""
        diagnostics = nil
        error = nil
        hasHandledTerminalEvent = false
        activeRunRuntimeKey = AppLoadedRuntimeKey(
            modelDirectory: request.modelDirectory,
            maxContextTokens: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: !request.isPureGreedy)
        isCancellationPending = false
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
        liveMemoryBytes = nil
        phase = .prefill
        runState = .running

        runTask = Task.detached { [weak self, client, request] in
            guard let self else { return }
            do {
                for try await event in client.generate(request) {
                    await self.apply(event)
                }
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"))
            }
        }
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        client.cancel()
    }

    public func makeRequest() throws -> AppGenerationRequest {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            messages: committedTurns.map(\.decodeMessage)
                + [DecodeChatMessage(role: .user, content: promptText)],
            maxNewTokens: maxNewTokensOverride ?? maxContextTokens,
            maxContextTokens: maxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: runtimeOptions)
        try request.validate(requireModelDirectory: true)
        return request
    }

    func apply(_ event: AppInferenceEvent) {
        switch event {
        case .prefillProgress(let done, let total):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
        case .token(let token):
            phase = .decode
            liveTokenCount = token.index + 1
            liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
            if let reporter = client as? any AppInferenceMemoryReporting {
                liveMemoryBytes = reporter.currentInferenceMemoryBytes
            } else {
                liveMemoryBytes = memorySampler.sample()
            }
            if !token.textDelta.isEmpty {
                outputText += token.textDelta
            }
        case .finished(let diagnostics):
            finishSuccessfully(diagnostics)
        case .cancelled(let diagnostics):
            finishCancelled(diagnostics)
        case .failed(let appError, let partial):
            diagnostics = partial
            materializeServiceTranscript()
            finishWithError(appError)
        }
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        // Only a completed exchange is committed. A cancelled or failed run
        // stays in the live turn: it is still on screen, but it never becomes
        // history the model is asked to continue from, and the next run clears
        // it, so the transcript and the conversation never disagree.
        commitLiveTurn()
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        error = .cancelled
        finishTerminalRun()
    }

    private func materializeServiceTranscript() {
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        outputText = reporter.generationTranscriptMailbox.completeText
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        error = appError
        finishTerminalRun()
    }

    private func finishStreamFailure(_ appError: AppInferenceError) {
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        phase = .idle
        runState = .idle
        isCancellationPending = false
        activeRunRuntimeKey = nil
        runTask = nil
    }

    private func clearLoadTask(generation: UInt64) {
        guard generation == loadGeneration else { return }
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
    }

    private func clearUnloadTask(generation: UInt64) {
        guard generation == unloadGeneration else { return }
        unloadTask = nil
    }
}
