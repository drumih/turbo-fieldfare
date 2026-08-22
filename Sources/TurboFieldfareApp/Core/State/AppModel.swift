import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareRepackCore
import TurboFieldfare
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
    public private(set) var imageAttachments: [AppImageAttachment] = []
    public private(set) var imageAttachmentError: String?
    /// A count, not a flag. The picker and a drop can both be staging at once,
    /// and whichever finished first cleared a shared Bool — reopening `canRun`
    /// while the other was still copying, so Generate ran against a partial set
    /// and the remaining images were appended after the run had snapshotted its
    /// own, where `removeImage` and `clearImages` are no-ops.
    private var addingImagesCount = 0
    public var isAddingImages: Bool { addingImagesCount > 0 }
    /// Set by the Model menu's Remove Image Support item; the window presents
    /// the confirmation.
    public var isConfirmingVisionPackRemoval = false
    public private(set) var outputPromptText: String = ""
    public private(set) var outputImageAttachments: [AppImageAttachment] = []
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
    public private(set) var sentPromptBehavior: AppSentPromptBehavior = .keep
    public private(set) var loadModelOnLaunch: Bool = false
    /// Whether launching the app should load the model straight away. Off by
    /// default, because loading takes minutes and holds gigabytes.
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var installState: AppModelInstallState = .idle
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    /// The companion download is 1.5 GB and deserves the same answer to "how
    /// long is this going to take" as the model download. Kept separate because
    /// both can be in flight in principle and an estimator holds per-download
    /// rate state.
    public private(set) var visionInstallETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var visionInstallETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus
    public var visionInstallState: AppModelInstallState = .idle
    /// How far activation's hash of the companion weights has got, 0 to 1.
    /// Activation reads about 1.5 GB, which was a bare spinner with no way to
    /// tell a slow verify from a stuck one.
    public private(set) var visionActivationProgress: Double?
    public private(set) var visionInstallReadiness: AppModelInstallReadiness = .checking
    public private(set) var visionInstallationStatus: AppVisionPackInstallationStatus

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public private(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    public private(set) var livePrefillDone: Int = 0
    public private(set) var livePrefillTotal: Int = 0
    public private(set) var liveMemoryBytes: UInt64?
    /// Resident bytes of the inference process. The footprint above is what
    /// the system counts against the process; this is what it actually holds,
    /// including the mapped weights the footprint omits. A 26B model reports
    /// about 160 MB of footprint right after loading, which is true and reads
    /// as nonsense without this beside it.
    public private(set) var liveResidentBytes: UInt64?
    /// Tower weights the inference process is holding mapped, reported
    /// separately because no per-process counter attributes them.
    public private(set) var visionTowerMappedBytes: UInt64?
    public private(set) var isCancellationPending: Bool = false
    /// Increments when a generation starts. The transcript watches it to put
    /// the newest turn on screen: with several images attached, the prompt and
    /// its thumbnails are tall enough to push the answer out of view, so
    /// scrolling only when the reader was already at the bottom left them
    /// looking at their own attachments while the model worked.
    public private(set) var runIdentity: Int = 0

    private let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private let visionInstaller: any AppVisionPackInstallerClient
    private var runTask: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var visionInstallTask: Task<Void, Never>?
    private var unloadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    /// The highest load-phase sequence already applied. Each `onState` callback
    /// hops to the main actor in its own task, and ordering between separately
    /// created tasks is not guaranteed, so `.ready` could be applied before the
    /// `.loading(.preparingRunner)` that preceded it and leave the UI showing a
    /// phase the runtime had already left.
    private var appliedLoadSequence: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var visionInstallGeneration: UInt64 = 0
    private var visionInstallCancellationRequested = false
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    private var hasHandledTerminalEvent = false
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()
    private var visionInstallETAEstimator = DownloadETAEstimator()
    private let attachmentStore: AppImageAttachmentStore
    public let isVisionRuntimeSupported: Bool

    public static var currentDeviceSupportsVisionRuntime: Bool {
        VisionRuntime.isSupportedOnDefaultDevice
    }

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(),
                visionInstaller: any AppVisionPackInstallerClient = RepackVisionPackInstallerClient(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                attachmentStore: AppImageAttachmentStore = AppImageAttachmentStore(),
                visionRuntimeSupported: Bool = true,
                settingsPersistenceEnabled: Bool = false) {
        let directory = (modelDirectory ?? AppModelLocation.defaultURL()).standardizedFileURL
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
            : MacAppSettings()
        self.modelPathText = directory.path
        // The app always releases the image tower after each image. Keeping it
        // resident saves a few hundred milliseconds on a run of images and
        // holds about 1 GB of page cache to do it — a trade worth exposing to
        // a CLI or server operator, not to someone using the app, where it was
        // one more setting whose effect no figure on screen could show.
        // `keepReady` remains available through AppRuntimeOptions for those.
        self.runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled,
            rdadvisePolicy: settings.rdadvisePolicy,
            visionResidencyPolicy: .onDemand)
        self.maxContextTokens = settings.contextTokens
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.newlineShortcut = settings.newlineShortcut
        self.showPromptExamples = settings.showPromptExamples
        self.sentPromptBehavior = settings.sentPromptBehavior
        self.loadModelOnLaunch = settings.loadModelOnLaunch
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.visionInstallationStatus = AppVisionPackInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.visionInstaller = visionInstaller
        self.memorySampler = memorySampler
        self.attachmentStore = attachmentStore
        self.isVisionRuntimeSupported = visionRuntimeSupported
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        // Staged images of runs that were killed before they could clean up;
        // nothing else ever removes them.
        AppImageAttachmentStore.sweepAbandoned()
        refreshInstallReadiness()
        refreshVisionInstallReadiness()
    }

    public var isRunning: Bool { runState == .running }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && loadState.isReady
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
            && !isVisionCompanionOperationInProgress
            && requiresModelInstallation
    }

    public var canCancelInstall: Bool { installState.canCancel }

    public var isVisionPackInstalled: Bool { visionInstallationStatus == .complete }

    public var isInstallingVisionPack: Bool { visionInstallState.isInstalling }

    public var visionInstallDescriptor: AppModelInstallDescriptor {
        visionInstaller.descriptor
    }

    /// Every companion Download, Resume, Verify, Activate, Repair, and Remove
    /// operation is one app-blocking state: model actions stay disabled until it
    /// reaches a resting state, so a companion transaction never overlaps a
    /// loaded session or another companion operation.
    public var isVisionCompanionOperationInProgress: Bool {
        visionInstallState.isInstalling
    }

    /// A companion operation may only begin against an unloaded model session
    /// with no other transfer in flight; the draft, transcript, and attachments
    /// are untouched by the gate.
    public var canBeginVisionCompanionOperation: Bool {
        !isRunning && !loadState.isLoading && !loadState.isReady
            && !isInstallingModel && !isVisionCompanionOperationInProgress
    }

    public var canInstallVisionPack: Bool {
        guard isVisionRuntimeSupported else { return false }
        // A layout with nowhere to put a companion cannot be repaired by
        // downloading one, so do not offer to.
        guard visionInstallationStatus != .unsupportedLayout else { return false }
        guard isModelInstalled, !isVisionPackInstalled,
              case .ready = visionInstallReadiness else { return false }
        if case .readyToActivate = visionInstallState { return false }
        return canBeginVisionCompanionOperation
    }

    public var canActivateVisionPack: Bool {
        guard isVisionRuntimeSupported else { return false }
        guard case .readyToActivate = visionInstallState else { return false }
        return canBeginVisionCompanionOperation
    }

    public var canCancelVisionInstall: Bool { visionInstallState.canCancel }

    public var visionInstallProgressFraction: Double? {
        // Activation hashes about 1.5 GB, so it gets a bar of its own rather
        // than an indeterminate spinner for minutes.
        if case .activating = visionInstallState { return visionActivationProgress }
        guard case .copyingPayload(let reused, let downloaded, let total) = visionInstallState,
              total > 0 else { return nil }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var visionInstallPhaseLabel: String {
        switch visionInstallState {
        case .idle: return isVisionPackInstalled ? "Installed" : "Not installed"
        case .checking: return "Checking image support"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning image support"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading image support"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing download"
        case .activating:
            guard let fraction = visionActivationProgress else {
                return "Activating image support"
            }
            return "Verifying image support \(Int(fraction * 100))%"
        case .cancelling: return "Cancelling"
        case .discarding: return "Cleaning up"
        case .cancelled: return "Download paused"
        case .readyToActivate: return "Ready to activate"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Installed"
        case .failed: return "Installation failed"
        }
    }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        return min(reused.addingReportingOverflow(downloaded).partialValue, total)
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
        case .activating:
            guard let fraction = visionActivationProgress else {
                return "Activating image support"
            }
            return "Verifying image support \(Int(fraction * 100))%"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding download"
        case .cancelled: return "Download paused"
        case .readyToActivate: return "Ready to activate"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Model installed"
        case .failed: return "Installation failed"
        }
    }

    public var canRun: Bool {
        // Staging copies the files a request will carry. Starting a run while
        // it is in flight sent a request without those images and then landed
        // them on the next message instead.
        !isRunning && !isAddingImages && isModelAvailable && !loadState.isLoading
            && !isVisionCompanionOperationInProgress
            && !hasStaleLoadedRuntime
            && (!promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !imageAttachments.isEmpty)
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    public var hasOutputTranscript: Bool {
        !outputPromptText.isEmpty || !outputImageAttachments.isEmpty || !outputText.isEmpty
    }

    public var outputResponsePlainText: String {
        generationTranscriptMailbox?.completeText ?? outputText
    }

    public var outputConversationPlainText: String {
        let response = outputResponsePlainText
        switch (outputPromptText.isEmpty, response.isEmpty) {
        case (true, true):
            return ""
        case (false, true):
            return "You:\n\(outputPromptText)"
        case (true, false):
            return "Answer:\n\(response)"
        case (false, false):
            return "You:\n\(outputPromptText)\n\nAnswer:\n\(response)"
        }
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
            lastStopReason: diagnostics?.stopReason,
            isVisionCompanionOperationInProgress: isVisionCompanionOperationInProgress))
    }

    public var currentProcessMemoryBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        // `liveMemoryBytes` first because it is a tracked property: reading the
        // reporter alone told the truth but was invisible to observation, so
        // the figure only refreshed when something else — a generated token —
        // happened to redraw the view. Through prefill, nothing did.
        if let liveMemoryBytes { return liveMemoryBytes }
        // When inference runs in another process, its memory is the only
        // memory worth showing. Falling back to this app's own sampler put the
        // UI's footprint in a row labelled as the model's.
        if let reporter = client as? any AppInferenceMemoryReporting {
            return reporter.currentInferenceMemoryBytes
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
        clearImages()
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        installGeneration &+= 1
        installTask?.cancel()
        installer.cancel()
        installTask = nil
        visionInstallGeneration &+= 1
        visionInstallCancellationRequested = false
        visionInstallTask?.cancel()
        visionInstaller.cancel()
        visionInstallTask = nil
        resetInstallETA()
        installState = .idle
        visionInstallState = .idle
        pendingExplicitLoadRuntimeKey = nil
        activeRunRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        diagnostics = nil
        error = nil
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(at: URL(fileURLWithPath: path))
        visionInstallationStatus = AppVisionPackInstallationProbe.status(
            at: URL(fileURLWithPath: path))
        refreshInstallReadiness()
        refreshVisionInstallReadiness()

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

    public func setSentPromptBehavior(_ behavior: AppSentPromptBehavior) {
        guard sentPromptBehavior != behavior else { return }
        sentPromptBehavior = behavior
        persistSettings()
    }

    public func setLoadModelOnLaunch(_ enabled: Bool) {
        guard loadModelOnLaunch != enabled else { return }
        loadModelOnLaunch = enabled
        persistSettings()
    }

    /// Starts the launch load if it is switched on and the model can be loaded.
    /// Called once, when the window first appears; a model that is missing,
    /// already loading, or busy with a companion operation is left alone.
    public func loadModelAtLaunchIfEnabled() {
        guard loadModelOnLaunch, canLoadModel else { return }
        loadModel()
    }

    /// Whether an image can be attached at all.
    ///
    /// The runtime flag only says this build *can* use images; the companion
    /// pack is what makes it possible for this model. Gating on the flag alone
    /// offered an Add-images button with no tower behind it, and the failure
    /// only surfaced when the user pressed Generate.
    public var isImageInputAvailable: Bool {
        isVisionRuntimeSupported && isVisionPackInstalled
    }

    /// Image support is part of this build. Hardware support and companion-pack
    /// availability are separate so the inspector can explain either absence.
    public var visionRuntimeEnabled: Bool { true }

    /// Room left for the prompt when working out how many images fit. The
    /// runtime still rejects a combination that does not fit, so this only has
    /// to be a defensible reserve rather than an exact prompt measurement.
    static let reservedPromptTokens = 1_024

    /// How many images this conversation can hold, derived from the context
    /// exactly as the server derives its budget. It used to be a fixed four,
    /// which meant the same set of images was accepted over the API and refused
    /// in the app.
    public var maximumImageAttachments: Int {
        // The context a run will actually use, which is the loaded session's
        // until it is reloaded. Capping on the pending setting instead let the
        // composer accept images the request then refused.
        max(1, VisionImageTokenBudget.capacity(
            maxContext: effectiveMaxContextTokens,
            reservedTextTokens: Self.reservedPromptTokens))
    }

    /// The context a generation would run with right now.
    public var effectiveMaxContextTokens: Int {
        (loadedRuntimeKey ?? currentRuntimeKey).maxContextTokens
    }

    /// `discardingSourceDirectory` is the temp directory a file-promise drop
    /// wrote into. It is ours, it holds nothing but those copies, and staging
    /// takes its own copy — so it must not outlive the staging that consumed
    /// it, which is exactly how it leaked.
    public func addImages(_ urls: [URL], discardingSourceDirectory: URL? = nil) {
        // Every early return has to discard the promise directory itself. The
        // staging task's `defer` below owns it only once that task exists, so a
        // return above it strands the full-size copies with nothing left to
        // delete them: the attachment sweep only covers the staging root.
        func discardSource() {
            if let discardingSourceDirectory {
                try? FileManager.default.removeItem(at: discardingSourceDirectory)
            }
        }
        guard isImageInputAvailable, !urls.isEmpty else {
            discardSource()
            return
        }
        guard !isRunning else {
            // A promise drop admitted before the run started can be delivered
            // after it. Returning silently made the images look as though they
            // had simply vanished.
            imageAttachmentError =
                "Wait for the current run to finish before attaching images."
            discardSource()
            return
        }
        let capacity = maximumImageAttachments
        let available = max(0, capacity - imageAttachments.count)
        guard available > 0 else {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
            discardSource()
            return
        }
        addingImagesCount += 1
        imageAttachmentError = nil
        // Dropping the rest silently left the user believing every image they
        // chose was attached.
        let selected = Array(urls.prefix(available))
        if selected.count < urls.count {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
        }
        let store = attachmentStore
        Task.detached(priority: .userInitiated) { [weak self] in
            var staged: [AppImageAttachment] = []
            defer {
                if let discardingSourceDirectory {
                    try? FileManager.default.removeItem(at: discardingSourceDirectory)
                }
            }
            do {
                for url in selected {
                    staged.append(try store.stage(url))
                }
                await self?.finishAddingImages(staged)
            } catch {
                // The batch is all-or-nothing, so the copies made before the
                // failure are referenced by nothing and would never be deleted.
                for attachment in staged { store.remove(attachment) }
                await self?.finishAddingImages(error: error)
            }
        }
    }

    /// Attaches image bytes that have no file behind them: an image copied out
    /// of another app arrives on the pasteboard as data, and a drag from an app
    /// that has not written the file yet arrives as a promise.
    public func addImageData(_ data: Data, displayName: String) {
        guard isImageInputAvailable, !isRunning else { return }
        let capacity = maximumImageAttachments
        guard imageAttachments.count < capacity else {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
            return
        }
        addingImagesCount += 1
        imageAttachmentError = nil
        let store = attachmentStore
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let staged = try store.stage(data: data, displayName: displayName)
                await self?.finishAddingImages([staged])
            } catch {
                await self?.finishAddingImages(error: error)
            }
        }
    }

    static func imageCapacityMessage(capacity: Int, context: Int) -> String {
        "At most \(capacity) image\(capacity == 1 ? "" : "s") fit in the "
            + "\(context / 1_024)K context this session is running with. Raise "
            + "Context in Memory and reload the model to send more."
    }

    public func reportImageAttachmentError(_ error: Error) {
        imageAttachmentError = String(describing: error)
    }

    public func reportImageAttachmentError(_ message: String) {
        imageAttachmentError = message
    }

    public func removeImage(id: UUID) {
        guard !isRunning,
              let index = imageAttachments.firstIndex(where: { $0.id == id }) else { return }
        let attachment = imageAttachments.remove(at: index)
        attachmentStore.remove(attachment)
        imageAttachmentError = nil
    }

    public func clearImages() {
        guard !isRunning else { return }
        for attachment in imageAttachments { attachmentStore.remove(attachment) }
        imageAttachments.removeAll()
        imageAttachmentError = nil
    }

    private func finishAddingImages(_ staged: [AppImageAttachment]) {
        // Two adds can be in flight at once — the picker and a drop — and each
        // sized itself against the count it saw at admission, so the second to
        // land can push past the cap. Re-check against the real count here and
        // delete what does not fit, rather than leaving staged copies that
        // nothing references.
        defer { addingImagesCount = max(0, addingImagesCount - 1) }
        // The counter keeps `canRun` closed until every batch lands, so a run
        // should not be able to start underneath one. If it ever does, the run
        // has already snapshotted its images: appending here would attach them
        // to the *next* message with no way to take them off, which is worse
        // than saying so. `addImages` refuses a mid-run drop the same way.
        guard !isRunning else {
            for attachment in staged { attachmentStore.remove(attachment) }
            imageAttachmentError =
                "Wait for the current run to finish before attaching images."
            return
        }
        let capacity = maximumImageAttachments
        let available = max(0, capacity - imageAttachments.count)
        let accepted = staged.prefix(available)
        for attachment in staged.dropFirst(accepted.count) {
            attachmentStore.remove(attachment)
        }
        imageAttachments.append(contentsOf: accepted)
        if accepted.count < staged.count {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
        }
    }

    private func finishAddingImages(error: Error) {
        addingImagesCount = max(0, addingImagesCount - 1)
        imageAttachmentError = String(describing: error)
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
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: runtimeOptions,
                                             forceLogitsHead: forceLogitsHead)
        // The session is loaded with the same normalized options a run sends.
        // Loading with the raw settings instead meant a control that is off but
        // still carries a non-default value — RDADVISE off with its policy left
        // on `bounded` — produced a loaded session no run could match, and the
        // staleness check compares two normalized keys, so nothing ever offered
        // the reload that would have cleared it.
        let options = runtimeKey.options(prefillEnabled: runtimeOptions.prefillEnabled,
                                        prefillChunkTokens: runtimeOptions.prefillChunkTokens)
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        error = nil
        appliedLoadSequence = 0
        loadState = .loading(.validatingDirectory)
        let emitted = Mutex<UInt64>(0)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    // Stamped where the phase is emitted, in order; checked
                    // where it is applied, which is not.
                    let sequence = emitted.withLock { value -> UInt64 in
                        value += 1
                        return value
                    }
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation,
                                             sequence: sequence)
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

    public var hasPartialVisionPackDownload: Bool {
        guard let output = try? VisionPackLocation.companionURL(
            forTextModel: URL(fileURLWithPath: modelPathText, isDirectory: true)),
              let paths = try? RemoteInstallPaths(outputDirectory: output.path) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardVisionPackDownload: Bool {
        hasPartialVisionPackDownload && canBeginVisionCompanionOperation
    }

    public var canRemoveVisionPack: Bool {
        hasVisionPackDirectory && canBeginVisionCompanionOperation
    }

    public var hasVisionPackDirectory: Bool {
        guard let output = try? VisionPackLocation.companionURL(
            forTextModel: URL(fileURLWithPath: modelPathText, isDirectory: true)) else {
            return false
        }
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: output.path,
            isDirectory: &isDirectory) && isDirectory.boolValue
    }

    public func installVisionPack() {
        guard canInstallVisionPack else { return }
        visionInstallCancellationRequested = false
        visionInstallTask?.cancel()
        visionInstaller.cancel()
        let textModelDirectory = URL(
            fileURLWithPath: modelPathText,
            isDirectory: true).standardizedFileURL
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .checking
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                for try await event in visionInstaller.install(
                    textModelDirectory: textModelDirectory) {
                    guard let self else { return }
                    self.applyVisionInstallEvent(event, generation: generation)
                }
                self?.finishVisionInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishVisionInstallCancellation(generation: generation)
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelVisionInstall() {
        guard canCancelVisionInstall else { return }
        visionInstallCancellationRequested = true
        visionInstallState = .cancelling
        visionInstaller.cancel()
        // A cancel raised before the stream registers its own task would find no
        // active install; cancelling the consumer terminates the stream, which
        // routes through the same cooperative drain-to-checkpoint path.
        visionInstallTask?.cancel()
    }

    public func activateVisionPack() {
        guard canActivateVisionPack else { return }
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        resetVisionInstallETA()
        visionInstallState = .activating
        visionActivationProgress = 0
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                let output = try await visionInstaller.activatePreparedInstall(
                    textModelDirectory: directory,
                    onVerifyProgress: { [weak self] fraction in
                        Task { @MainActor in
                            guard let self,
                                  generation == self.visionInstallGeneration,
                                  case .activating = self.visionInstallState else { return }
                            // Each hop is its own task, and tasks are not
                            // ordered against each other, so a late one must
                            // not walk the bar backwards.
                            guard fraction >= (self.visionActivationProgress ?? 0)
                            else { return }
                            self.visionActivationProgress = fraction
                        }
                    })
                // Applied first: a progress hop still in flight is dropped
                // once the state is no longer `.activating`, so clearing before
                // this could be undone by a late update.
                self?.applyVisionInstallEvent(
                    .installed(output), generation: generation)
                self?.visionActivationProgress = nil
            } catch is CancellationError {
                // Cancellation can only land during verification, before
                // anything is renamed, so the prepared pack is untouched and
                // still activatable.
                self?.finishVisionActivationCancelled(generation: generation)
                self?.visionActivationProgress = nil
            } catch {
                self?.finishVisionInstallFailure(
                    error, generation: generation, phase: .activation)
                self?.visionActivationProgress = nil
            }
        }
    }

    private func finishVisionActivationCancelled(generation: UInt64) {
        guard generation == visionInstallGeneration else { return }
        resetVisionInstallETA()
        visionInstallTask = nil
        visionInstallCancellationRequested = false
        visionInstallState = .idle
        refreshVisionInstallReadiness()
    }

    public func discardVisionPackDownload() {
        guard canDiscardVisionPackDownload else { return }
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .discarding
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                try await visionInstaller.discardPartialInstall(
                    textModelDirectory: directory)
                guard let self, generation == self.visionInstallGeneration else { return }
                self.visionInstallTask = nil
                self.visionInstallState = .idle
                self.refreshVisionInstallReadiness()
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    /// Drives the confirmation the Model menu puts in front of `removeVisionPack`.
    ///
    /// The Inspector hides its own Remove button once the pack is installed, so
    /// the menu item is the only reachable way to delete 1.14 GB — and it called
    /// straight through, with the dialog sitting on an unreachable branch.
    public func requestVisionPackRemoval() {
        guard canRemoveVisionPack else { return }
        isConfirmingVisionPackRemoval = true
    }

    public func removeVisionPack() {
        isConfirmingVisionPackRemoval = false
        guard canRemoveVisionPack else { return }
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .discarding
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                try await visionInstaller.removeInstalled(
                    textModelDirectory: directory)
                guard let self, generation == self.visionInstallGeneration else { return }
                self.visionInstallTask = nil
                self.visionInstallState = .idle
                self.visionInstallationStatus = .missing
                self.refreshVisionInstallReadiness()
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
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
        refreshVisionInstallReadiness(at: directory)
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

    public func refreshVisionInstallReadiness() {
        refreshVisionInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true)
                .standardizedFileURL)
    }

    private func refreshVisionInstallReadiness(at textModelDirectory: URL) {
        visionInstallationStatus = AppVisionPackInstallationProbe.status(
            at: textModelDirectory)
        // Removing the companion leaves any attached image unsendable, and the
        // composer would keep offering it with nothing able to encode it.
        // Only once the dust has settled: the probe verifies the pack on disk,
        // and a companion operation renames that directory underneath it, so
        // refreshing mid-operation can briefly report no image support. Acting
        // on that would delete images the user had staged.
        if !isImageInputAvailable, !isVisionCompanionOperationInProgress,
           !imageAttachments.isEmpty {
            for attachment in imageAttachments { attachmentStore.remove(attachment) }
            imageAttachments.removeAll()
            // Say so. Clearing the error alongside the images removed them and
            // the only explanation for their absence in one step, so the
            // composer just quietly emptied itself.
            imageAttachmentError =
                "Image support is unavailable, so the attached images were removed."
        }
        guard isModelInstalled else {
            visionInstallReadiness = .failed("Install the text model first")
            return
        }
        guard !isVisionPackInstalled else { return }
        if visionInstaller.preparedInstallIsValid(
            textModelDirectory: textModelDirectory) {
            let output = try? VisionPackLocation.companionURL(
                forTextModel: textModelDirectory)
            // A pack that failed to activate must not be re-offered for
            // activation: `preparedInstallIsValid` does not hash the weights,
            // so a corrupt pack still looks ready and the user would loop
            // between Activate and the same failure.
            let reportedBroken: Bool
            switch visionInstallState {
            case .recoverable, .failed: reportedBroken = true
            default: reportedBroken = false
            }
            if let output, !isInstallingVisionPack, !reportedBroken {
                visionInstallState = .readyToActivate(output)
            }
        }
        visionInstallReadiness = .checking
        do {
            let requirement = try visionInstaller.checkInstallRequirement(
                textModelDirectory: textModelDirectory)
            visionInstallReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            visionInstallReadiness = .failed("\(error)")
        }
    }

    private func applyVisionInstallEvent(
        _ event: AppModelInstallEvent,
        generation: UInt64
    ) {
        guard generation == visionInstallGeneration else { return }
        if visionInstallCancellationRequested {
            switch event {
            case .readyToActivate, .installed:
                // Work that finished before the cancel landed is reported as it
                // actually ended, not as a pause.
                visionInstallCancellationRequested = false
            default:
                return
            }
        }
        switch event {
        case .checking:
            resetVisionInstallETA()
            visionInstallState = .checking
        case .downloadingMetadata:
            resetVisionInstallETA()
            visionInstallState = .downloadingMetadata
        case .planning:
            resetVisionInstallETA()
            visionInstallState = .planning
        case .reservingOutput:
            resetVisionInstallETA()
            visionInstallState = .reservingOutput
        case .copyingPayload(let reused, let downloaded, let total):
            visionInstallState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloaded,
                totalBytes: total)
            updateVisionInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloaded,
                totalBytes: total)
        case .hashingOutput(let file):
            resetVisionInstallETA()
            visionInstallState = .hashingOutput(file)
        case .finalizing:
            resetVisionInstallETA()
            visionInstallState = .finalizing
        case .readyToActivate(let directory):
            resetVisionInstallETA()
            visionInstallState = .readyToActivate(directory)
            visionInstallTask = nil
        case .installed:
            resetVisionInstallETA()
            let textModelDirectory = URL(
                fileURLWithPath: modelPathText,
                isDirectory: true).standardizedFileURL
            visionInstallationStatus = AppVisionPackInstallationProbe.status(
                at: textModelDirectory)
            guard isVisionPackInstalled else {
                finishVisionInstallFailure(
                    RepackError.configurationInvalid(
                        detail: "completed vision install failed verification"),
                    generation: generation)
                return
            }
            visionInstallState = .installed(modelDirectory: textModelDirectory)
            visionInstallTask = nil
        }
    }

    private func finishVisionInstallStream(generation: UInt64) {
        guard generation == visionInstallGeneration,
              visionInstallTask != nil else { return }
        if visionInstallCancellationRequested || visionInstallState == .cancelling {
            finishVisionInstallCancellation(generation: generation)
        } else if !isVisionPackInstalled {
            finishVisionInstallFailure(
                RepackError.configurationInvalid(
                    detail: "vision installer ended before completion"),
                generation: generation)
        }
    }

    private func finishVisionInstallCancellation(generation: UInt64) {
        guard generation == visionInstallGeneration else { return }
        resetVisionInstallETA()
        visionInstallCancellationRequested = false
        visionInstallTask = nil
        visionInstallState = .cancelled
        refreshVisionInstallReadiness()
    }

    /// Which phase failed. Only a download failure may leave a prepared pack
    /// that is worth activating; a verification failure must never send the
    /// user back to Activate, or the same corrupt pack is offered forever.
    enum VisionFailurePhase { case download, activation }

    func finishVisionInstallFailure(
        _ error: Error, generation: UInt64,
        phase: VisionFailurePhase = .download
    ) {
        guard generation == visionInstallGeneration else { return }
        // An error raised because the user cancelled is a pause with saved
        // progress, not an installation failure.
        guard !visionInstallCancellationRequested else {
            finishVisionInstallCancellation(generation: generation)
            return
        }
        resetVisionInstallETA()
        visionInstallTask = nil
        let hasSavedDownload = hasPartialVisionPackDownload
        let textModelDirectory = URL(
            fileURLWithPath: modelPathText,
            isDirectory: true).standardizedFileURL
        // A download that finished and verifies is activatable whatever went
        // wrong afterwards. Reporting it as "needs attention" hid the Activate
        // button behind a Resume that only repeats work already done. The one
        // failure that must not come back here is verification itself, or the
        // same corrupt pack is offered forever — but a lock held by another
        // process is contention, not corruption.
        let isContention: Bool
        if let repackError = error as? RepackError, case .installBusy = repackError {
            isContention = true
        } else {
            isContention = false
        }
        if phase == .download || isContention, hasSavedDownload,
           let output = try? VisionPackLocation.companionURL(
            forTextModel: textModelDirectory),
           visionInstaller.preparedInstallIsValid(
            textModelDirectory: textModelDirectory) {
            visionInstallState = .readyToActivate(output)
            refreshVisionInstallReadiness(at: textModelDirectory)
            return
        }
        visionInstallState = hasSavedDownload
            ? .recoverable("\(error)")
            : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            visionInstallReadiness = .insufficientSpace(AppModelInstallRequirement(
                probePath: path,
                requiredBytes: required,
                availableBytes: available))
        } else {
            refreshVisionInstallReadiness()
            if hasSavedDownload {
                visionInstallState = .recoverable("\(error)")
            }
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
        case .readyToActivate:
            finishInstallFailure(
                RepackError.configurationInvalid(
                    detail: "text installer returned a vision-only activation event"),
                generation: generation)
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
            refreshVisionInstallReadiness(at: directory)
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

    private func updateVisionInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let presentation = visionInstallETAEstimator.update(
            observation, timestamp: installETATimestamp)
        visionInstallETAPresentation = presentation
        visionInstallETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func resetVisionInstallETA() {
        visionInstallETAEstimator.reset()
        visionInstallETAPresentation = .hidden
        visionInstallETAText = nil
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
            prefillEnabled: settings.prefillEnabled,
            rdadvisePolicy: settings.rdadvisePolicy,
            // Pinned for the same reason as `init`: the app always releases the
            // image tower. Reading the persisted value here would let a
            // `keepReady` written by an older build resurrect ~1 GB of resident
            // tower on a machine with no control that shows or clears it.
            visionResidencyPolicy: .onDemand)
        maxContextTokens = settings.contextTokens
        temperature = settings.temperature
        topKEnabled = settings.topKEnabled
        topK = settings.topK
        topPEnabled = settings.topPEnabled
        topP = settings.topP
        newlineShortcut = settings.newlineShortcut
        showPromptExamples = settings.showPromptExamples
        sentPromptBehavior = settings.sentPromptBehavior
        loadModelOnLaunch = settings.loadModelOnLaunch
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
            showPromptExamples: showPromptExamples,
            sentPromptBehavior: sentPromptBehavior,
            visionResidencyPolicy: runtimeOptions.visionResidencyPolicy,
            rdadvisePolicy: runtimeOptions.rdadvisePolicy,
            loadModelOnLaunch: loadModelOnLaunch)
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

    /// `sequence` orders the phases a load emits. It is 0 for states this model
    /// raises itself, which bypass the ordering check.
    func applyLoadState(_ state: AppModelLoadState, generation: UInt64,
                        sequence: UInt64 = 0) {
        guard generation == loadGeneration else { return }
        if sequence > 0 {
            guard sequence > appliedLoadSequence else { return }
            appliedLoadSequence = sequence
        }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        // An outcome closes the load. Phases emitted before it but delivered
        // after it must not reopen one that has already finished: `.failed` is
        // raised here at sequence 0, so it never advanced the counter, and a
        // late `.loading` hop could put the UI back into a load with no task
        // left to cancel and no way to start another. `beginLoad` resets the
        // counter, so the seal lasts exactly one load.
        switch state {
        case .notLoaded, .ready, .failed:
            appliedLoadSequence = .max
        case .loading, .cancelling, .unloading:
            break
        }
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
            // The freshly loaded model's footprint, so the figure is right
            // before the first generation rather than after it.
            sampleLiveMemory()
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            error = loadError
        }
    }

    /// Releases the transcript's own references to the images it was showing.
    /// They are separate files from the composer's, so nothing else frees them.
    private func releaseTranscriptImages() {
        for attachment in outputImageAttachments { attachmentStore.remove(attachment) }
        outputImageAttachments = []
    }

    /// Deletes every file this session staged. Called when the app is quitting,
    /// which is the only moment they are all certainly unwanted.
    public func releaseAllAttachments() {
        releaseTranscriptImages()
        for attachment in imageAttachments { attachmentStore.remove(attachment) }
        imageAttachments.removeAll()
        attachmentStore.removeAll()
    }

    /// Memory comes from the process doing the work: the decode service when
    /// there is one, this process otherwise.
    private func sampleLiveMemory() {
        if let reporter = client as? any AppInferenceMemoryReporting {
            if let bytes = reporter.currentInferenceMemoryBytes {
                liveMemoryBytes = bytes
            }
            if let resident = reporter.currentInferenceResidentBytes {
                liveResidentBytes = resident
            }
            // Refreshed on every sample, so the tower figure tracks a run
            // instead of appearing only in its final diagnostics.
            if let tower = reporter.currentInferenceTowerBytes {
                visionTowerMappedBytes = tower
            }
        } else {
            liveMemoryBytes = memorySampler.sample()
            // The occupied figure, not the footprint again: this row is the one
            // that includes the mapped weights, and feeding it the footprint
            // made both numbers report the same thing.
            liveResidentBytes = memorySampler.occupiedSample()
        }
    }

    /// Resident bytes for display, on the same terms as
    /// `currentProcessMemoryBytes`.
    public var currentProcessResidentBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        if let liveResidentBytes { return liveResidentBytes }
        if let reporter = client as? any AppInferenceMemoryReporting {
            return reporter.currentInferenceResidentBytes
        }
        return memorySampler.occupiedSample()
    }

    public func clearOutput() {
        guard !isRunning else { return }
        outputPromptText = ""
        releaseTranscriptImages()
        outputText = ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
    }

    public func run() {
        guard canRun else { return }
        var request: AppGenerationRequest
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

        // The run reads the transcript's own hard links rather than the
        // composer's files, so clearing the composer below cannot delete an
        // image this request has not opened yet. One set of files, one owner.
        // A failed retain leaves no reference that is guaranteed to outlive
        // the composer, so the run is refused instead of started against files
        // that are about to be removed.
        var retained: [AppImageAttachment] = []
        do {
            for attachment in request.imageAttachments {
                retained.append(try attachmentStore.retain(attachment))
            }
        } catch {
            for attachment in retained { attachmentStore.remove(attachment) }
            imageAttachmentError = String(describing: error)
            self.error = .invalidRequest(
                "Could not prepare the attached images for this run: \(error)")
            return
        }
        request.imageAttachments = retained

        persistSettings()

        generationTranscriptMailbox?.reset()
        runIdentity &+= 1
        outputPromptText = request.prompt
        releaseTranscriptImages()
        outputImageAttachments = retained
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
        sampleLiveMemory()
        phase = .prefill
        runState = .running
        if sentPromptBehavior == .clear {
            promptText = ""
            // Images used to survive a cleared prompt, so the next Run silently
            // re-attached them to a completely different message. Dropping them
            // is safe because the request above was repointed at the retained
            // links: removing these files cannot pull the ground out from under
            // a run that has not opened its images yet.
            for attachment in imageAttachments { attachmentStore.remove(attachment) }
            imageAttachments.removeAll()
            imageAttachmentError = nil
        }

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
        // A run executes against the session that is actually loaded. Sending
        // the current settings instead meant that changing Context, Slots or
        // image residency and pressing Generate — without reloading first —
        // was refused outright with "generation runtime options do not match
        // the loaded session". The settings still apply on reload, which is
        // what the Memory section promises; they simply no longer break the
        // run in the meantime.
        let effective = loadedRuntimeKey ?? currentRuntimeKey
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            prompt: promptText,
            imageAttachments: imageAttachments,
            maxNewTokens: maxNewTokensOverride ?? effective.maxContextTokens,
            maxContextTokens: effective.maxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: effective.options(
                prefillEnabled: runtimeOptions.prefillEnabled,
                prefillChunkTokens: runtimeOptions.prefillChunkTokens))
        try request.validate(requireModelDirectory: true)
        return request
    }

    func apply(_ event: AppInferenceEvent) {
        switch event {
        case .memorySample:
            sampleLiveMemory()
        case .prefillProgress(let done, let total):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
            sampleLiveMemory()
        case .token(let token):
            phase = .decode
            liveTokenCount = token.index + 1
            liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
            sampleLiveMemory()
            if !token.textDelta.isEmpty {
                outputText += token.textDelta
            }
        case .finished(let diagnostics):
            visionTowerMappedBytes = diagnostics.visionTowerMappedBytes
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
