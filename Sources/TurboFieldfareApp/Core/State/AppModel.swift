import Foundation
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
    /// One app-managed local image queued for the next user turn.
    public private(set) var pendingImage: AppImageAttachment?
    public private(set) var attachmentError: String?
    public private(set) var isImportingImage = false
    /// Optional guidance that is prepended to each request in the current chat.
    public var systemPromptText: String = ""
    /// Locally saved conversations, ordered by most recently updated.
    public private(set) var chats: [AppChatThread] = []
    public private(set) var selectedChatID = UUID()
    /// Completed turns in the current in-memory chat.
    public private(set) var conversation: [AppChatMessage] = []
    /// The conversation snapshot currently being generated or displayed.
    public private(set) var outputMessages: [AppChatMessage] = []
    public private(set) var outputPromptText: String = ""
    public var outputText: String = ""
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens: Int = 4096
    public var temperature: Double = 0.2
    public var topKEnabled: Bool = true
    public var topK: Int = 64
    public var topPEnabled: Bool = true
    public var topP: Double = 0.95
    public private(set) var newlineShortcut: AppNewlineShortcut = .return
    public private(set) var showPromptExamples: Bool = true
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var installState: AppModelInstallState = .idle
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus

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
    private var runGeneration: UInt64 = 0
    private var loadGeneration: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    private var hasHandledTerminalEvent = false
    /// The original completed turn while its answer is being regenerated or
    /// its prompt is being edited. It stays in memory and on disk until the
    /// replacement succeeds.
    private var regenerationBackup: [AppChatMessage]?
    /// While a response is streaming, periodically checkpoint the displayed
    /// turn. This keeps a stopped or unexpectedly interrupted reply from
    /// disappearing from the selected chat without writing on every token.
    private var lastChatPersistenceDate: Date?
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                settingsPersistenceEnabled: Bool = false) {
        let directory = (modelDirectory ?? AppModelLocation.defaultURL()).standardizedFileURL
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
            : MacAppSettings()
        let chatHistory = settingsPersistenceEnabled
            ? AppChatHistoryFileStore.load(forModelDirectory: directory)
            : .fresh()
        let orderedChats = chatHistory.chats.sorted(by: Self.isMoreRecent)
        let normalizedChats = Self.normalizedChatList(
            orderedChats,
            preserving: chatHistory.selectedChatID)
        let selectedChat = normalizedChats.first(where: {
            $0.id == chatHistory.selectedChatID
        }) ?? normalizedChats[0]
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
        self.newlineShortcut = settings.newlineShortcut
        self.showPromptExamples = settings.showPromptExamples
        self.chats = normalizedChats
        self.selectedChatID = selectedChat.id
        self.conversation = selectedChat.messages
        self.systemPromptText = selectedChat.systemPrompt
        if let response = selectedChat.messages.last, response.role == .assistant {
            self.outputMessages = Array(selectedChat.messages.dropLast())
            self.outputPromptText = selectedChat.messages.dropLast()
                .last(where: { $0.role == .user })?.content ?? ""
            self.outputText = response.content
        } else {
            self.outputMessages = selectedChat.messages
        }
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.memorySampler = memorySampler
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        refreshInstallReadiness()
        if normalizedChats.count != orderedChats.count {
            persistChatHistory()
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
        !isRunning && !isImportingImage
            && isModelAvailable && !loadState.isLoading
            && !hasStaleLoadedRuntime
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canAttachImage: Bool {
        !isRunning
            && !isImportingImage
            && pendingImage == nil
            && conversation.reduce(0) { $0 + $1.images.count }
                < AppGenerationRequest.maximumImageAttachments
    }

    public var chatAttachmentRootURL: URL {
        AppChatAttachmentStore.rootURL(
            forModelDirectory: URL(
                fileURLWithPath: modelPathText,
                isDirectory: true))
    }

    public func imageURL(for attachment: AppImageAttachment) -> URL? {
        try? AppChatAttachmentStore.fileURL(
            for: attachment,
            modelDirectory: URL(
                fileURLWithPath: modelPathText,
                isDirectory: true))
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    public var canManageChats: Bool { !isRunning && !isImportingImage }

    public var completedTurnCount: Int {
        conversation.reduce(into: 0) { count, message in
            if message.role == .user { count += 1 }
        }
    }

    public var hasSystemPrompt: Bool {
        !systemPromptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canRegenerate: Bool {
        !isRunning && !isImportingImage
            && isModelAvailable && !loadState.isLoading && !hasStaleLoadedRuntime
            && hasCompletedLatestTurn
    }

    /// Editing is intentionally limited to the user message belonging to the
    /// latest completed assistant response. Earlier context stays immutable.
    public var canEditLastPrompt: Bool {
        !isRunning && !isImportingImage && hasCompletedLatestTurn
    }

    /// A completed prompt remains editable while the model is unloaded, but
    /// its replacement can only be submitted once generation is available.
    public var canSubmitEditedLastPrompt: Bool { canRegenerate }

    public var selectedChatTitle: String {
        chats.first(where: { $0.id == selectedChatID })?.title ?? "New chat"
    }

    public var hasOutputTranscript: Bool {
        !outputMessages.isEmpty || !outputText.isEmpty
    }

    public var outputResponsePlainText: String {
        guard let mailbox = generationTranscriptMailbox else { return outputText }
        let canonicalText = mailbox.completeText
        return canonicalText.isEmpty ? outputText : canonicalText
    }

    public var outputConversationPlainText: String {
        var messages = outputMessages
        let response = outputResponsePlainText
        if !response.isEmpty {
            messages.append(AppChatMessage(role: .assistant, content: response))
        }
        return messages.map { message in
            let label: String = switch message.role {
            case .system: "Instructions"
            case .user: "You"
            case .assistant: "Answer"
            }
            let imageLines = message.images.map {
                "[Image: \($0.originalFilename)]"
            }
            return (["\(label):"] + imageLines + [message.content])
                .joined(separator: "\n")
        }.joined(separator: "\n\n")
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

    private var hasCompletedLatestTurn: Bool {
        conversation.count >= 2
            && conversation[conversation.count - 2].role == .user
            && conversation.last?.role == .assistant
    }

    public func setModelURL(_ url: URL) {
        guard !isRunning else { return }
        let path = url.standardizedFileURL.path
        guard path != modelPathText else { return }

        saveActiveChat()
        discardPendingImage()
        attachmentError = nil
        modelPathText = path
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        loadChatHistory(forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
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

    public func setNewlineShortcut(_ shortcut: AppNewlineShortcut) {
        guard newlineShortcut != shortcut else { return }
        newlineShortcut = shortcut
        persistSettings()
    }

    public func setShowPromptExamples(_ show: Bool) {
        guard showPromptExamples != show else { return }
        showPromptExamples = show
        persistSettings()
    }

    @discardableResult
    public func attachImage(at sourceURL: URL) async -> Bool {
        guard canAttachImage else { return false }
        let chatID = selectedChatID
        let modelDirectory = URL(
            fileURLWithPath: modelPathText,
            isDirectory: true).standardizedFileURL
        attachmentError = nil
        isImportingImage = true
        defer { isImportingImage = false }
        do {
            let attachment = try await Task.detached(priority: .userInitiated) {
                let accessedSecurityScope = sourceURL.startAccessingSecurityScopedResource()
                defer {
                    if accessedSecurityScope {
                        sourceURL.stopAccessingSecurityScopedResource()
                    }
                }
                return try AppChatAttachmentStore.importImage(
                    from: sourceURL,
                    chatID: chatID,
                    forModelDirectory: modelDirectory)
            }.value
            let activeDirectory = URL(
                fileURLWithPath: modelPathText,
                isDirectory: true).standardizedFileURL
            guard selectedChatID == chatID,
                  activeDirectory == modelDirectory else {
                AppChatAttachmentStore.remove(
                    attachment,
                    forModelDirectory: modelDirectory)
                return false
            }
            pendingImage = attachment
            attachmentError = nil
            return true
        } catch {
            let activeDirectory = URL(
                fileURLWithPath: modelPathText,
                isDirectory: true).standardizedFileURL
            if selectedChatID == chatID, activeDirectory == modelDirectory {
                attachmentError = (error as? AppChatAttachmentStoreError)?.description
                    ?? "The image could not be added: \(error)"
            }
            return false
        }
    }

    public func removePendingImage() {
        guard !isRunning else { return }
        discardPendingImage()
        attachmentError = nil
    }

    public func dismissAttachmentError() {
        attachmentError = nil
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
        installTask = Task { [weak self, installer] in
            do {
                for try await event in installer.installDefaultModel(outputDirectory: outputDirectory) {
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
        newlineShortcut = settings.newlineShortcut
        showPromptExamples = settings.showPromptExamples
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
            prefillEnabled: runtimeOptions.prefillEnabled,
            newlineShortcut: newlineShortcut,
            showPromptExamples: showPromptExamples)
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

    public func clearOutput() {
        guard !isRunning, !isImportingImage else { return }
        discardPendingImage()
        AppChatAttachmentStore.removeAll(
            forChatID: selectedChatID,
            modelDirectory: URL(
                fileURLWithPath: modelPathText,
                isDirectory: true))
        attachmentError = nil
        conversation = []
        outputMessages = []
        systemPromptText = ""
        outputPromptText = ""
        outputText = ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
        saveActiveChat()
        normalizeNewChatPlaceholders()
        persistChatHistory()
    }

    public func newChat() {
        guard canManageChats else { return }
        saveActiveChat()

        normalizeNewChatPlaceholders()
        if let existing = chats.first(where: Self.isPristineNewChat) {
            // Reuse the existing blank page and reset any unsent draft so
            // repeated taps always land on a genuinely new-chat surface.
            apply(chat: existing)
            persistChatHistory()
            return
        }

        let chat = AppChatThread()
        chats.append(chat)
        orderChatsByRecency()
        selectedChatID = chat.id
        apply(chat: chat)
        persistChatHistory()
    }

    public func selectChat(_ id: UUID) {
        guard canManageChats, id != selectedChatID else { return }
        saveActiveChat()
        guard let chat = chats.first(where: { $0.id == id }) else { return }
        selectedChatID = chat.id
        apply(chat: chat)
        persistChatHistory()
    }

    public func deleteChat(_ id: UUID) {
        // Instruction edits do not otherwise have a dedicated Save button;
        // synchronize the selected chat before an unrelated sidebar action.
        saveActiveChat()
        guard canManageChats,
              let index = chats.firstIndex(where: { $0.id == id }) else { return }
        if selectedChatID == id {
            discardPendingImage()
            attachmentError = nil
        }
        chats.remove(at: index)
        AppChatAttachmentStore.removeAll(
            forChatID: id,
            modelDirectory: URL(
                fileURLWithPath: modelPathText,
                isDirectory: true))
        if chats.isEmpty {
            let chat = AppChatThread()
            chats = [chat]
        }
        if selectedChatID == id {
            let replacement = chats[0]
            selectedChatID = replacement.id
            apply(chat: replacement)
        }
        normalizeNewChatPlaceholders()
        persistChatHistory()
    }

    public func renameChat(_ id: UUID, to title: String) {
        saveActiveChat()
        guard canManageChats,
              let index = chats.firstIndex(where: { $0.id == id }) else { return }
        let cleaned = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return }
        chats[index].title = String(cleaned.prefix(80))
        chats[index].updatedAt = Date()
        orderChatsByRecency()
        persistChatHistory()
    }

    public func run() {
        guard canRun else { return }
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

        // The request retains the submitted prompt; clear the composer so the
        // user can prepare the next turn while this response streams.
        promptText = ""
        pendingImage = nil
        attachmentError = nil
        generationTranscriptMailbox?.reset()
        outputMessages = conversation + [request.messages.last!]
        outputPromptText = request.prompt
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
        runGeneration &+= 1
        let generation = runGeneration
        lastChatPersistenceDate = Date()
        // Store the submitted turn before prefill starts. In particular, this
        // preserves a prompt when the app is closed or generation is stopped
        // before the first token arrives.
        saveActiveChat(titleForFirstPrompt: request.prompt)

        runTask = Task.detached { [weak self, client, request, generation] in
            guard let self else { return }
            guard await self.canStartRun(generation: generation) else {
                await self.finishStreamFailure(.cancelled, generation: generation)
                return
            }

            let stream = client.generate(request)
            // `cancel()` can arrive before the client task has registered its
            // generation. Check again after creating the stream so Stop is
            // never lost in that narrow startup window.
            guard await self.canStartRun(generation: generation) else {
                client.cancel()
                await self.finishStreamFailure(.cancelled, generation: generation)
                return
            }
            do {
                for try await event in stream {
                    await self.apply(event, generation: generation)
                }
                if await self.hasActiveRun(generation: generation) {
                    let wasCancelled = await self.isCancellationPending
                    let error: AppInferenceError = wasCancelled
                        ? .cancelled
                        : .unknown("Generation ended without a completion event.")
                    await self.finishStreamFailure(error, generation: generation)
                }
            } catch is CancellationError {
                await self.finishStreamFailure(.cancelled, generation: generation)
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError, generation: generation)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"), generation: generation)
            }
        }
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        client.cancel()
    }

    public func makeRequest() throws -> AppGenerationRequest {
        var messages: [AppChatMessage] = []
        if hasSystemPrompt {
            messages.append(AppChatMessage(role: .system, content: systemPromptText))
        }
        messages.append(contentsOf: conversation)
        messages.append(AppChatMessage(
            role: .user,
            content: promptText,
            images: pendingImage.map { [$0] } ?? []))
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            prompt: promptText,
            maxNewTokens: maxNewTokensOverride ?? maxContextTokens,
            maxContextTokens: maxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: runtimeOptions,
            messages: messages)
        try request.validate(requireModelDirectory: true)
        return request
    }

    public func regenerateLastResponse() {
        guard canRegenerate, conversation.count >= 2 else { return }
        _ = replaceLastCompletedTurn(
            with: conversation[conversation.count - 2].content)
    }

    /// Replaces only the latest user prompt and regenerates its answer using
    /// every earlier turn as context. The old prompt and answer remain the
    /// persisted source of truth until the replacement finishes successfully.
    @discardableResult
    public func submitEditedLastPrompt(_ editedPrompt: String) -> Bool {
        replaceLastCompletedTurn(with: editedPrompt)
    }

    @discardableResult
    private func replaceLastCompletedTurn(with replacementPrompt: String) -> Bool {
        guard canSubmitEditedLastPrompt,
              conversation.count >= 2,
              !replacementPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }

        let previousConversation = conversation
        let previousUserMessage = conversation[conversation.count - 2]
        conversation.removeLast(2)
        regenerationBackup = previousConversation
        promptText = replacementPrompt
        // Editing changes only the latest query text. Its managed image stays
        // attached to both edit and regenerate requests.
        pendingImage = previousUserMessage.images.first
        run()

        guard isRunning else {
            promptText = ""
            pendingImage = nil
            restoreRegenerationBackupIfNeeded()
            return false
        }
        return true
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
            persistActiveRunIfNeeded()
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

    private func apply(_ event: AppInferenceEvent, generation: UInt64) {
        guard generation == runGeneration, isRunning else { return }
        apply(event)
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        commitDisplayedTurnToConversation()
        self.diagnostics = diagnostics
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        if !restoreRegenerationBackupIfNeeded() {
            commitDisplayedTurnToConversation()
        }
        self.diagnostics = diagnostics
        error = .cancelled
        finishTerminalRun()
    }

    private func materializeServiceTranscript() {
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        let canonicalText = reporter.generationTranscriptMailbox.completeText
        if !canonicalText.isEmpty || outputText.isEmpty {
            outputText = canonicalText
        }
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        if !restoreRegenerationBackupIfNeeded() {
            commitDisplayedTurnToConversation()
        }
        error = appError
        finishTerminalRun()
    }

    private func finishStreamFailure(_ appError: AppInferenceError) {
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishStreamFailure(_ appError: AppInferenceError, generation: UInt64) {
        guard generation == runGeneration, isRunning else { return }
        finishStreamFailure(appError)
    }

    private func finishTerminalRun() {
        phase = .idle
        runState = .idle
        isCancellationPending = false
        activeRunRuntimeKey = nil
        runTask = nil
        lastChatPersistenceDate = nil
    }

    /// Makes the transcript currently visible in the chat authoritative. A
    /// completed answer, a stopped partial answer, and a failed partial answer
    /// all remain available when the user changes chats or relaunches the app.
    private func commitDisplayedTurnToConversation() {
        conversation = outputMessages
        if !outputText.isEmpty {
            conversation.append(AppChatMessage(role: .assistant, content: outputText))
        }
        regenerationBackup = nil
        saveActiveChat()
    }

    @discardableResult
    private func restoreRegenerationBackupIfNeeded() -> Bool {
        guard let regenerationBackup else { return false }
        self.regenerationBackup = nil
        conversation = regenerationBackup
        generationTranscriptMailbox?.reset()
        if let response = regenerationBackup.last, response.role == .assistant {
            outputMessages = Array(regenerationBackup.dropLast())
            outputPromptText = regenerationBackup.dropLast()
                .last(where: { $0.role == .user })?.content ?? ""
            outputText = response.content
        } else {
            outputMessages = regenerationBackup
            outputPromptText = regenerationBackup.last(where: { $0.role == .user })?.content ?? ""
            outputText = ""
        }
        saveActiveChat()
        return true
    }

    private func apply(chat: AppChatThread) {
        discardPendingImage()
        attachmentError = nil
        selectedChatID = chat.id
        conversation = chat.messages
        systemPromptText = chat.systemPrompt
        promptText = ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
        if let response = chat.messages.last, response.role == .assistant {
            outputMessages = Array(chat.messages.dropLast())
            outputPromptText = chat.messages.dropLast()
                .last(where: { $0.role == .user })?.content ?? ""
            outputText = response.content
        } else {
            outputMessages = chat.messages
            outputPromptText = chat.messages.last(where: { $0.role == .user })?.content ?? ""
            outputText = ""
        }
    }

    private func loadChatHistory(forModelDirectory modelDirectory: URL) {
        let history = settingsPersistenceEnabled
            ? AppChatHistoryFileStore.load(forModelDirectory: modelDirectory)
            : .fresh()
        chats = Self.normalizedChatList(
            history.chats,
            preserving: history.selectedChatID)
        orderChatsByRecency()
        let chat = chats.first(where: { $0.id == history.selectedChatID }) ?? chats[0]
        apply(chat: chat)
        persistChatHistory()
    }

    private func saveActiveChat(titleForFirstPrompt prompt: String? = nil) {
        guard let index = chats.firstIndex(where: { $0.id == selectedChatID }) else { return }
        let messages = messagesForPersistence()
        var didChange = chats[index].systemPrompt != systemPromptText
            || chats[index].messages != messages
        chats[index].systemPrompt = systemPromptText
        chats[index].messages = messages
        if chats[index].title == "New chat",
           let prompt,
           !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chats[index].title = Self.chatTitle(for: prompt)
            didChange = true
        }
        if didChange {
            chats[index].updatedAt = Date()
            orderChatsByRecency()
        }
        persistChatHistory()
    }

    private func persistActiveRunIfNeeded(now: Date = Date()) {
        guard settingsPersistenceEnabled, isRunning else { return }
        if let lastChatPersistenceDate,
           now.timeIntervalSince(lastChatPersistenceDate) < 1 {
            return
        }
        lastChatPersistenceDate = now
        saveActiveChat()
    }

    /// Uses the displayed transcript while a generation is finishing so a
    /// chat switch cannot lose the submitted turn or its latest response.
    private func messagesForPersistence() -> [AppChatMessage] {
        guard isRunning else { return conversation }
        if let regenerationBackup { return regenerationBackup }
        var messages = outputMessages
        let response = outputResponsePlainText
        if !response.isEmpty {
            messages.append(AppChatMessage(role: .assistant, content: response))
        }
        return messages
    }

    private func persistChatHistory() {
        guard settingsPersistenceEnabled else { return }
        let history = AppChatHistoryDocument(selectedChatID: selectedChatID, chats: chats)
        try? AppChatHistoryFileStore.save(
            history,
            forModelDirectory: URL(fileURLWithPath: modelPathText, isDirectory: true))
    }

    private func discardPendingImage() {
        guard let pendingImage else { return }
        AppChatAttachmentStore.remove(
            pendingImage,
            forModelDirectory: URL(
                fileURLWithPath: modelPathText,
                isDirectory: true))
        self.pendingImage = nil
    }

    private func orderChatsByRecency() {
        chats.sort(by: Self.isMoreRecent)
    }

    private func normalizeNewChatPlaceholders() {
        let blankChats = chats.filter(Self.isPristineNewChat)
        guard blankChats.count > 1 else { return }

        let keepID = blankChats.first(where: { $0.id == selectedChatID })?.id
            ?? blankChats[0].id
        chats.removeAll { chat in
            Self.isPristineNewChat(chat) && chat.id != keepID
        }
        orderChatsByRecency()
    }

    private func canStartRun(generation: UInt64) -> Bool {
        generation == runGeneration && isRunning && !isCancellationPending
    }

    private func hasActiveRun(generation: UInt64) -> Bool {
        generation == runGeneration && isRunning
    }

    private static func isMoreRecent(_ lhs: AppChatThread, _ rhs: AppChatThread) -> Bool {
        if lhs.updatedAt == rhs.updatedAt { return lhs.createdAt > rhs.createdAt }
        return lhs.updatedAt > rhs.updatedAt
    }

    private static func normalizedChatList(
        _ chats: [AppChatThread],
        preserving preferredID: UUID?
    ) -> [AppChatThread] {
        let blankChats = chats.filter(Self.isPristineNewChat)
        guard blankChats.count > 1 else { return chats }

        let keepID = blankChats.first(where: { $0.id == preferredID })?.id
            ?? blankChats[0].id
        return chats.filter { chat in
            !Self.isPristineNewChat(chat) || chat.id == keepID
        }
    }

    private static func isPristineNewChat(_ chat: AppChatThread) -> Bool {
        chat.title.trimmingCharacters(in: .whitespacesAndNewlines) == "New chat"
            && chat.messages.isEmpty
            && chat.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func chatTitle(for prompt: String) -> String {
        let title = prompt
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        return title.isEmpty ? "New chat" : String(title.prefix(80))
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
