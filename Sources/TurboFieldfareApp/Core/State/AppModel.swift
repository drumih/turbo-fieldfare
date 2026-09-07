import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareRepackCore
import Observation

@MainActor
@Observable
public final class AppModel {
    public enum RunState: Equatable {
        case idle
        case running
    }

    /// How a turn ended.
    ///
    /// `committed` covers a reply that finished and a stop that landed at a
    /// token boundary: both are in the KV and on disk. `rewound` is a turn the
    /// runtime took back — a hard stop, a stream failure, a lost lineage — so
    /// it is in neither, and its message goes back to the composer. The cause
    /// is already in `error` by the time this is read.
    private enum TurnOutcome {
        case committed
        case rewound
    }

    public var modelPathText: String
    public var promptText: String = ""
    public private(set) var imageAttachments: [StagedImage] = []
    public private(set) var imageAttachmentError: String?
    private var visionAvailabilityAttachmentError: String?
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
    public internal(set) var outputPromptText: String = ""
    /// The newest turn's pictures, as the transcript draws them. A live turn's
    /// are staged; a reopened conversation's newest pair is drawn from its
    /// document, whose pictures the conversation store owns.
    public internal(set) var outputImageAttachments: [ChatImage] = []
    /// The open chat. The transcript renders it, and its turn order is what the
    /// decode service's gate checks every turn against.
    public internal(set) var conversation = AppConversation()
    /// What the window is showing, and in what state. One value: the six
    /// fields it replaces could disagree, and twelve defects were two of them
    /// disagreeing.
    var machine = ConversationScreenMachine()
    public var screen: ConversationScreen { machine.screen }
    /// The epoch the inference side has actually been told to open. Nil after a
    /// load or unload, both of which rebuild or release the KV; the next turn
    /// opens the conversation again before it sends anything.
    var serviceEpoch: UUID?
    public var outputText: String = ""
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    /// Settable only through `setMaxContextTokens`, because changing it changes
    /// which stored conversations can be continued — and what is on screen has
    /// to be redrawn from the answer that now applies, not the one taken when
    /// the row was clicked.
    public private(set) var maxContextTokens: Int = AppContextLengthOption.eightK.tokens
    public var temperature: Double = 0.2
    public var topKEnabled: Bool = true
    public var topK: Int = 64
    public var topPEnabled: Bool = true
    public var topP: Double = 0.95
    public private(set) var newlineShortcut: AppNewlineShortcut = .return
    public private(set) var showPromptExamples: Bool = true
    /// Whether the list of chats is showing. Persisted, so the window comes
    /// back the way it was left.
    public private(set) var isSidebarVisible: Bool = true
    /// Whether the Inspector is showing. Persisted alongside the sidebar, for
    /// the same reason: the window comes back the way it was left.
    public private(set) var isInspectorVisible: Bool = true
    /// Whether launching the app should load the model straight away. Off by
    /// default, because loading takes minutes and holds gigabytes.
    public private(set) var loadModelOnLaunch: Bool = false
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
    public internal(set) var installationStatus: AppModelInstallationStatus
    public private(set) var modelStorageMetrics: AppModelStorageMetrics?
    public private(set) var modelStorageMetricsError: String?
    public var visionInstallState: AppModelInstallState = .idle
    /// How far activation's hash of the companion weights has got, 0 to 1.
    /// Activation reads about 1.5 GB, which was a bare spinner with no way to
    /// tell a slow verify from a stuck one.
    public private(set) var visionActivationProgress: Double?
    public private(set) var visionInstallReadiness: AppModelInstallReadiness = .checking
    public private(set) var visionInstallationStatus: AppVisionPackInstallationStatus

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public internal(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    public internal(set) var livePrefillDone: Int = 0
    public internal(set) var livePrefillTotal: Int = 0
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

    let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private let visionInstaller: any AppVisionPackInstallerClient
    /// The message currently on its way through the send pipeline, from the
    /// click to the commit or the hand-back. `isTurnInFlight` reads it, so a
    /// second Generate is refused for the whole journey and not only while the
    /// runtime is generating.
    private var sendTask: Task<Void, Never>?
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
    private var hasShutDownForTermination = false
    /// What the generation stage left behind, read by the stage that decides
    /// whether the message is committed or goes back to the composer.
    private var turnOutcome: TurnOutcome = .committed
    /// The send waiting for its turn to end, with the run it belongs to.
    private var turnCompletion:
        (generation: Int, continuation: CheckedContinuation<Void, Never>)?
    private let memorySampler: AppMemorySampler
    let settingsPersistenceEnabled: Bool
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()
    private var visionInstallETAEstimator = DownloadETAEstimator()
    /// Internal rather than private: the history extension releases the staged
    /// copies of a conversation the KV is giving up, and lives in another file.
    let attachmentStore: AppImageAttachmentStore
    public let isVisionRuntimeSupported: Bool
    /// The stored conversations and which one is open.
    public let history = ConversationHistoryState()
    /// Nil when history is off for this instance — the tests that drive the
    /// model without a disk do exactly that, and so does a run with settings
    /// persistence disabled, which is the same "do not touch the user's files"
    /// switch.
    var conversationBinding: ConversationStoreBinding?
    var conversationStore: ConversationStore? { conversationBinding?.store }
    /// The directory the open chat is being written to. Created on the first
    /// send rather than at New Chat, so an empty chat never lands on disk.
    var storedConversationID: UUID?
    /// Read from the installed model's manifest, because the app process never
    /// loads the model. Nil until a model is installed.
    var conversationIdentity: ConversationIdentity? {
        get { conversationBinding?.identity }
        set { conversationBinding?.identity = newValue }
    }
    var conversationBindingGeneration: UInt64 = 0
    let conversationIdentityProvider: @Sendable (URL) throws -> ConversationIdentity
    let conversationStoreProvider: @Sendable (URL) -> ConversationStore
    var pendingRestoredConversationID: UUID?
    var pendingServiceRecoveryConversationID: UUID?
    /// Stored copies of the in-flight turn's images, written while the model is
    /// generating so the wait is not paid twice.
    var pendingTurnImageWrite: Task<TurnImageWriteOutcome, Never>?
    var persistenceTail: Task<Void, Never>?
    /// Reserves deletion across the persistence wait and the store actor hop.
    var conversationDeletionTask: Task<Void, Never>?
    var quarantinedPersistenceLineages: Set<PersistenceLineageKey> = []
    /// Why the store last refused to do something. Names, counts and paths
    /// only: a diagnostic carrying transcript text would put the user's
    /// conversation wherever this is collected.
    public internal(set) var historyDiagnostic: String?

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
                settingsPersistenceEnabled: Bool = false,
                conversationIdentityProvider: @escaping @Sendable (URL) throws -> ConversationIdentity = {
                    try ConversationIdentity.forModelDirectory($0)
                },
                conversationStoreProvider: @escaping @Sendable (URL) -> ConversationStore = {
                    ConversationStore(rootURL: $0)
                }) {
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
        self.isSidebarVisible = settings.sidebarVisible
        self.isInspectorVisible = settings.inspectorVisible
        self.loadModelOnLaunch = settings.loadModelOnLaunch
        self.pendingRestoredConversationID = settings.selectedConversationID
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.visionInstallationStatus = AppVisionPackInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.visionInstaller = visionInstaller
        self.memorySampler = memorySampler
        self.attachmentStore = attachmentStore
        self.isVisionRuntimeSupported = visionRuntimeSupported
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.conversationIdentityProvider = conversationIdentityProvider
        self.conversationStoreProvider = conversationStoreProvider
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        // History follows the same switch as settings: a model driven by tests
        // must not write to the user's Application Support directory.
        if settingsPersistenceEnabled {
            conversationBindingGeneration = 1
            let identity: ConversationIdentity?
            if installationStatus == .complete {
                do {
                    identity = try conversationIdentityProvider(directory)
                } catch {
                    identity = nil
                    historyDiagnostic = "the installed model identity could not be read: \(error)"
                }
            } else {
                identity = nil
            }
            conversationBinding = ConversationStoreBinding(
                modelDirectory: directory,
                identity: identity,
                generation: conversationBindingGeneration,
                storeProvider: conversationStoreProvider)
        } else {
            conversationBinding = nil
        }
        // Staged images of runs that were killed before they could clean up;
        // nothing else ever removes them.
        AppImageAttachmentStore.sweepAbandoned()
        refreshInstallReadiness()
        refreshVisionInstallReadiness()
        activateConversationStore()
    }

    public var isRunning: Bool { runState == .running }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    // The three lifecycle actions gate on the whole send, not on `isRunning`:
    // a deferred replay is a full prefill during which no turn is generating
    // yet, and an unload taken then dropped the KV under the replay, lost the
    // message it was carrying, and queued behind the prefill on the service
    // until the load timeout killed the connection.
    public var canLoadModel: Bool {
        isModelInstalled && !isTurnInFlight && !isVisionFilesystemMutationInProgress
            && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isTurnInFlight && !isVisionFilesystemMutationInProgress
            && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isTurnInFlight && !isVisionFilesystemMutationInProgress
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

    /// Any companion operation currently owns the installer transaction.
    public var isVisionCompanionOperationInProgress: Bool {
        visionInstallState.isInstalling
    }

    public var isVisionFilesystemMutationInProgress: Bool {
        switch visionInstallState {
        case .activating, .discarding: return true
        default: return false
        }
    }

    /// Pack activation, repair and removal mutate the runtime's files and need
    /// an unloaded model. Payload download does not use this gate.
    public var canBeginVisionCompanionOperation: Bool {
        !isRunning && !loadState.isLoading && !loadState.isReady
            && !isInstallingModel && !isVisionCompanionOperationInProgress
    }

    private var canBeginVisionDownload: Bool {
        !isRunning && !loadState.isLoading && !isInstallingModel
            && !isVisionCompanionOperationInProgress
    }

    public var canInstallVisionPack: Bool {
        guard isVisionRuntimeSupported else { return false }
        // A layout with nowhere to put a companion cannot be repaired by
        // downloading one, so do not offer to.
        guard visionInstallationStatus != .unsupportedLayout else { return false }
        guard isModelInstalled, !isVisionPackInstalled,
              case .ready = visionInstallReadiness else { return false }
        if case .readyToActivate = visionInstallState { return false }
        return canBeginVisionDownload
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
        // A deferred replay is not `isRunning` — no turn is generating yet —
        // but the message has been sent and the lineage it will be stamped
        // against does not exist yet. Without this a second Generate started a
        // second replay, and whichever finished last defined the epoch, so the
        // first one's request came back rejected as belonging to a conversation
        // that was no longer open.
        !isTurnInFlight && conversationDeletionTask == nil
            && !isAddingImages && isModelAvailable && !loadState.isLoading
            && !isVisionFilesystemMutationInProgress
            && !hasStaleLoadedRuntime
            // A conversation whose KV no longer matches it cannot take another
            // turn; only New chat clears that.
            && conversation.canSend
            // And the conversation on screen has to be one a message can go
            // to. The live one always is; a stored row only when the notice
            // says it can be continued. Asking only `conversation.canSend`
            // let a row the notice had refused be sent anyway.
            && screen.allowsSend
            && (!promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !imageAttachments.isEmpty)
            && (imageAttachments.isEmpty || isImageInputAvailable)
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    /// The message has been sent and nothing has come back yet.
    ///
    /// Wider than `isRunning`, which covers only a generation: a reopened
    /// conversation has to be replayed into the KV first, and that is a full
    /// prefill of everything it holds. For the length of it the window showed a
    /// composer that had been emptied and a transcript that had not changed, so
    /// the app looked like it had dropped the message.
    public var isTurnInFlight: Bool {
        isRunning || screen.isReplaying || sendTask != nil
    }

    /// Activity whose prompt/output has been published to the transcript.
    /// A queued send owns the composer before replacing the previous answer.
    public var isTranscriptTurnInFlight: Bool {
        isRunning || screen.isReplaying
    }

    public var hasOutputTranscript: Bool {
        // Same split as `transcriptHistory`: while a stored copy is on screen,
        // the live conversation belongs to another chat and is not what decides
        // whether this one has anything to show. A stored chat with no readable
        // turns has to fall through to the empty state rather than borrow the
        // held conversation's.
        switch screen {
        case .reading, .replaying:
            return !transcriptHistory.isEmpty
        case .live, .unreadable:
            return !conversation.outOfContextPairs.isEmpty
                || !conversation.isEmpty
                || !outputPromptText.isEmpty || !outputImageAttachments.isEmpty
                || !outputText.isEmpty
        }
    }

    public var shouldShowPromptExamples: Bool {
        showPromptExamples
            && promptText.isEmpty
            && !isRunning
            && !hasOutputTranscript
    }

    public var outputResponsePlainText: String {
        // The mailbox wins only when it has something to say. `??` alone fell
        // back to `outputText` when the mailbox was *absent*, never when it was
        // merely empty — so a reopened conversation, whose newest reply is put
        // straight into `outputText` with the mailbox freshly reset, drew an
        // "Answer" heading with nothing under it. During a run the two agree,
        // and a genuinely empty reply is empty either way.
        let streamed = generationTranscriptMailbox?.completeText ?? ""
        return streamed.isEmpty ? outputText : streamed
    }

    /// Completed turns the transcript draws *above* the live one.
    ///
    /// The live fields keep holding the last finished turn between runs — that
    /// is what leaves an answer on screen after it ends — so while nothing is
    /// decoding the newest pair is drawn live and must not also appear here.
    /// One property, used by both the transcript and Copy Conversation, so the
    /// two cannot disagree about which turn is which.
    /// The chat on screen is not the one the KV is holding.
    public var isShowingStoredCopy: Bool { screen.isShowingStoredCopy }

    /// The state of the conversation on screen. `.continuable` for a live chat
    /// and for one being replayed; the other two say why the composer is
    /// closed.
    public var openedConversationState: ConversationContinuability {
        switch screen {
        case .live, .replaying:
            return .continuable
        case .reading(_, _, let state, _):
            return state
        case .unreadable:
            // Nothing to put back and nothing to read: the same dead end as a
            // record with no tokens, and the notice says the same thing.
            return .cannotReplay(reason: .tokenCountUnknown)
        }
    }

    /// Whether the live fields belong under what is on screen.
    ///
    /// While a stored copy is merely being read they do not: they hold the chat
    /// the KV is keeping, which is a different conversation. But a send starts
    /// by replaying the chat being read, and for that stretch the live fields
    /// hold the message that started it — so suppressing them there drew an
    /// "Answer / Processing your prompt" with no question above it.
    public var showsLiveTurn: Bool { isTranscriptTurnInFlight || !isShowingStoredCopy }

    /// What the transcript is drawing, as an identity the incremental renderer
    /// can key on.
    ///
    /// The renderer appends and cannot take pairs back, so it has to be told
    /// when the thing on screen is a different conversation. The live
    /// conversation's epoch used to be enough because every open started a new
    /// one; now that browsing leaves the lineage alone, going out to a stored
    /// copy and back would be two changes the epoch cannot express, and the
    /// chat that was read stayed on screen under the row that was returned to.
    public var displayedTranscriptID: UUID {
        switch screen {
        case .live, .unreadable:
            return conversation.epoch
        case .reading(_, _, _, let renderID):
            return renderID
        case .replaying(_, _, _, let renderID):
            return renderID
        }
    }

    public var transcriptHistory: [(user: AppChatTurn, assistant: AppChatTurn)] {
        // A stored conversation draws in full: its pairs are not
        // `completedPairs`, so none of them is being held back for the live
        // fields to draw — and the live conversation belongs to another chat.
        switch screen {
        case .reading, .replaying:
            return screen.document?.pairs ?? []
        case .live, .unreadable:
            let pairs = conversation.completedPairs
            let live = conversation.hasTurnInFlight ? pairs
                : (pairs.isEmpty ? pairs : Array(pairs.dropLast()))
            return conversation.outOfContextPairs + live
        }
    }

    /// Where the transcript draws "earlier turns are no longer in context",
    /// counted in pairs from the top. Nil when everything on screen is still in
    /// the model's context.
    ///
    /// A fact about the live conversation, so it is drawn only under one: a
    /// stored copy read off disk is not in the model's context at all, and a
    /// rule under every turn of it says nothing a reader can act on.
    public var transcriptContextBreak: Int? {
        switch screen {
        case .reading, .replaying:
            return nil
        case .live, .unreadable:
            return conversation.outOfContextPairs.isEmpty
                ? nil : conversation.outOfContextPairs.count
        }
    }

    public var outputConversationPlainText: String {
        var history: [String] = []
        for pair in transcriptHistory {
            var prompt = pair.user.text
            if !pair.user.images.isEmpty {
                let names = pair.user.images.map(\.displayName).joined(separator: ", ")
                prompt = prompt.isEmpty ? "[images: \(names)]" : "[images: \(names)]\n\(prompt)"
            }
            history.append("You:\n\(prompt)")
            history.append("Answer:\n\(pair.assistant.text)")
        }
        let live = liveConversationPlainText
        if history.isEmpty { return live }
        if live.isEmpty { return history.joined(separator: "\n\n") }
        return history.joined(separator: "\n\n") + "\n\n" + live
    }

    private var liveConversationPlainText: String {
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
            isRunning: isTurnInFlight,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            lastStopReason: diagnostics?.stopReason,
            isVisionFilesystemMutationInProgress: isVisionFilesystemMutationInProgress))
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
        // Rebinding replaces the screen machine outright, so a replay in
        // flight would lose the message it is carrying.
        guard !isTurnInFlight else { return }
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
        endConversationForReleasedKV()
        diagnostics = nil
        error = nil
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(at: URL(fileURLWithPath: path))
        visionInstallationStatus = AppVisionPackInstallationProbe.status(
            at: URL(fileURLWithPath: path))
        refreshInstallReadiness()
        refreshVisionInstallReadiness()
        replaceConversationBinding(
            for: URL(fileURLWithPath: path, isDirectory: true),
            reason: "the model location changed")

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

    /// Changes the context, and redraws a stored conversation against it.
    ///
    /// Continuability is a function of the conversation's size and the context
    /// in force, and the window took its answer once, when the row was clicked.
    /// So raising the context from the notice left the chat continuable but
    /// still drawn read-only, under a boundary rule saying earlier turns were
    /// out of context when they no longer were. Re-opening rebuilds it from the
    /// state that now applies, in both directions.
    public func setMaxContextTokens(_ tokens: Int) {
        guard maxContextTokens != tokens else { return }
        maxContextTokens = tokens
        persistSettings()
        guard !isRunning, !screen.isReplaying,
              let id = screen.conversationID,
              let meta = history.entry(id) else { return }
        perform(machine.apply(.contextChanged(
            state: continuability(of: meta),
            heldID: storedConversationID,
            kvMatchesHeld: serviceEpoch == conversation.epoch,
            renderID: UUID())))
    }

    public func setSidebarVisible(_ visible: Bool) {
        guard isSidebarVisible != visible else { return }
        isSidebarVisible = visible
        persistSettings()
    }

    /// One action behind the strip button, the View menu item and its
    /// shortcut, so the three cannot disagree about what "shown" means.
    public func toggleSidebar() {
        setSidebarVisible(!isSidebarVisible)
    }

    /// Retries the writer lock a second window could not take at launch.
    ///
    /// `isReadOnly` was decided once, when the store was activated, so a window
    /// that opened while another held the lock said "this one can read them but
    /// not change them" for the rest of its life — including long after the
    /// other one had quit. Becoming the frontmost window is exactly when that
    /// is worth asking again: it is what the user does after closing the other
    /// copy.
    public func reacquireStoreIfPossible() {
        guard history.isReadOnlyStore, let store = conversationStore else { return }
        Task { [weak self] in
            // Nil is the ordinary answer — another window is open — and anything
            // else is a fault this one can act on. Discarded, a store that could
            // not be locked for any other reason read as "another instance holds
            // it" for the rest of the session.
            if let reason = await store.retryLock() {
                self?.recordHistoryDiagnostic(
                    "the conversation store could not be locked for writing: "
                        + reason)
            }
            await self?.refreshHistory()
        }
    }

    public func setInspectorVisible(_ visible: Bool) {
        guard isInspectorVisible != visible else { return }
        isInspectorVisible = visible
        persistSettings()
    }

    public func toggleInspector() {
        setInspectorVisible(!isInspectorVisible)
    }

    /// Whether New Chat can run right now.
    ///
    /// The same guard `newChat()` already has, made askable so the button and
    /// the menu item can be disabled rather than silently doing nothing, plus
    /// the installer: there is no chat to start before a model exists.
    public var canStartNewChat: Bool {
        !isTurnInFlight && !requiresModelInstallation
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
    nonisolated static let reservedPromptTokens = 1_024

    /// How many images this conversation can hold, derived from the context
    /// exactly as the server derives its budget. It used to be a fixed four,
    /// which meant the same set of images was accepted over the API and refused
    /// in the app.
    public var maximumImageAttachments: Int {
        // The context a run will actually use, which is the loaded session's
        // until it is reloaded. Capping on the pending setting instead let the
        // composer accept images the request then refused.
        //
        // The conversation counts against the same window, and the request
        // reserves it (`AppGenerationRequest.validate`). Reserving only a fixed
        // 1,024 here meant that past that point the composer kept offering
        // images Send would refuse — accepted on attach, rejected on the button.
        Self.imageAttachmentCapacity(
            maxContextTokens: effectiveMaxContextTokens,
            conversationTokens: conversation.kvTokens)
    }

    nonisolated static func imageAttachmentCapacity(
        maxContextTokens: Int,
        conversationTokens: Int?
    ) -> Int {
        guard let conversationTokens else { return 0 }
        // The reply's reserve too: `generate` refuses a prompt that leaves
        // less than it free, after every image has been encoded on the GPU.
        // Without it the composer offered image sets the turn then refused.
        return VisionImageTokenBudget.capacity(
            maxContext: maxContextTokens,
            reservedTextTokens: max(reservedPromptTokens, conversationTokens)
                + ConversationGenerationReserve.tokens)
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
        recheckVisionPackAtCurrentLocation()
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
            if !isImageInputAvailable {
                recordVisionAvailabilityError(
                    at: URL(fileURLWithPath: modelPathText, isDirectory: true))
            }
            discardSource()
            return
        }
        // The whole send, not just the generation: a picture attached during
        // a deferred replay met `restoreComposer`'s draft-wins rule when the
        // replay failed, which deleted the sent message's own pictures.
        guard !isTurnInFlight else {
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
        Task.detached(priority: .userInitiated) { [weak self, attachmentStore] in
            var staged: [StagedImage] = []
            defer {
                if let discardingSourceDirectory {
                    try? FileManager.default.removeItem(at: discardingSourceDirectory)
                }
            }
            do {
                for url in selected {
                    staged.append(try attachmentStore.stage(url))
                }
                await self?.finishAddingImages(staged)
            } catch {
                // The batch is all-or-nothing, so the copies made before the
                // failure are referenced by nothing and would never be deleted.
                for attachment in staged { attachmentStore.remove(attachment) }
                await self?.finishAddingImages(error: error)
            }
        }
    }

    /// Attaches image bytes that have no file behind them: an image copied out
    /// of another app arrives on the pasteboard as data, and a drag from an app
    /// that has not written the file yet arrives as a promise.
    public func addImageData(_ data: Data, displayName: String) {
        recheckVisionPackAtCurrentLocation()
        guard isImageInputAvailable else {
            recordVisionAvailabilityError(
                at: URL(fileURLWithPath: modelPathText, isDirectory: true))
            return
        }
        guard !isTurnInFlight else {
            imageAttachmentError =
                "Wait for the current run to finish before attaching images."
            return
        }
        let capacity = maximumImageAttachments
        guard imageAttachments.count < capacity else {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
            return
        }
        addingImagesCount += 1
        imageAttachmentError = nil
        Task.detached(priority: .userInitiated) { [weak self, attachmentStore] in
            do {
                let staged = try attachmentStore.stage(
                    data: data, displayName: displayName)
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
        guard !isTurnInFlight,
              let index = imageAttachments.firstIndex(where: { $0.id == id }) else { return }
        let attachment = imageAttachments.remove(at: index)
        attachmentStore.remove(attachment)
        imageAttachmentError = nil
    }

    /// Puts the composer in the state a completed pick leaves it in.
    ///
    /// `addImages` needs a verifiable companion pack to reach its staging path,
    /// which no unit test has, and the lifetime rules around these files are
    /// exactly what needs covering.
    func setComposerAttachmentsForTesting(_ attachments: [StagedImage]) {
        imageAttachments = attachments
    }

    public func clearImages() {
        guard !isTurnInFlight else { return }
        for attachment in imageAttachments { attachmentStore.remove(attachment) }
        imageAttachments.removeAll()
        imageAttachmentError = nil
    }

    private func finishAddingImages(_ staged: [StagedImage]) {
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
            endConversationForReleasedKV()
            self.clearUnloadTask(generation: generation)
        }
    }

    /// Ends the conversation because the KV behind it is gone.
    ///
    /// `applyLoadState`'s `.notLoaded` branch does this too, but nothing reaches
    /// it: every production transition to `.notLoaded` assigns `loadState`
    /// directly. Calling this from those sites is what actually runs it.
    private func endConversationForReleasedKV() {
        serviceEpoch = nil
        presentStoredConversationAfterReleasedKV()
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
            endConversationForReleasedKV()
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

    public func recheckVisionPackAtCurrentLocation() {
        refreshVisionInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true)
                .standardizedFileURL)
    }

    private func refreshInstallReadiness(at outputDirectory: URL) {
        installationStatus = AppModelInstallationProbe.status(
            at: outputDirectory,
            descriptor: installer.descriptor)
        do {
            modelStorageMetrics = FileManager.default.fileExists(atPath: outputDirectory.path)
                ? try AppModelStorageMetrics.measure(at: outputDirectory) : nil
            modelStorageMetricsError = nil
        } catch {
            modelStorageMetrics = nil
            modelStorageMetricsError = "installed storage could not be measured: \(error)"
        }
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
        // Removing the companion leaves any attached image unsendable. Keep the
        // draft intact and refuse the send until the reproducibility input is
        // present again.
        // Only once the dust has settled: the probe verifies the pack on disk,
        // and a companion operation renames that directory underneath it, so
        // refreshing mid-operation can briefly report no image support. Acting
        // on that would delete images the user had staged.
        if !isImageInputAvailable, !isVisionCompanionOperationInProgress,
           !imageAttachments.isEmpty {
            recordVisionAvailabilityError(at: textModelDirectory)
        } else if isImageInputAvailable,
                  imageAttachmentError == visionAvailabilityAttachmentError {
            imageAttachmentError = nil
            visionAvailabilityAttachmentError = nil
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

    private func recordVisionAvailabilityError(at textModelDirectory: URL) {
        let location: String
        do {
            location = try VisionPackLocation.companionURL(
                forTextModel: textModelDirectory).path
        } catch {
            location = "an unresolved companion path (\(error))"
        }
        let cause: String
        switch visionInstallationStatus {
        case .missing:
            cause = "the companion pack is missing"
        case .partial(let detail):
            cause = "the companion pack is incomplete: \(detail)"
        case .unsupportedLayout:
            cause = "this text-model layout cannot host a companion pack"
        case .complete:
            cause = isVisionRuntimeSupported
                ? "the companion pack is unavailable"
                : "image inference is unsupported on this device"
        }
        let message = "Image support is unavailable at \(location): \(cause). "
            + "Restore the companion pack before sending."
        visionAvailabilityAttachmentError = message
        imageAttachmentError = message
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
            endConversationForReleasedKV()
            refreshVisionInstallReadiness(at: directory)
            replaceConversationBinding(
                for: directory,
                reason: "the model installation completed")
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
        isSidebarVisible = settings.sidebarVisible
        isInspectorVisible = settings.inspectorVisible
        loadModelOnLaunch = settings.loadModelOnLaunch
        pendingRestoredConversationID = settings.selectedConversationID
    }

    func persistSettings() {
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
            sidebarVisible: isSidebarVisible,
            inspectorVisible: isInspectorVisible,
            visionResidencyPolicy: runtimeOptions.visionResidencyPolicy,
            rdadvisePolicy: runtimeOptions.rdadvisePolicy,
            loadModelOnLaunch: loadModelOnLaunch,
            selectedConversationID: history.selection)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        do {
            try MacAppSettingsFileStore.save(settings, forModelDirectory: modelDirectory)
        } catch {
            FileHandle.standardError.write(Data(
                "Saving Mac app settings failed: \(error)\n".utf8))
        }
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
            // Unloading released the runner and the KV, so whatever lineage was
            // open no longer has tokens behind it.
            serviceEpoch = nil
            archiveConversationContext()
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            // A load builds a new runner and an empty KV. The service ends the
            // lineage on its side for the same reason; leaving the app's epoch
            // in place would have the next turn claim to resume onto a cache
            // that had just been rebuilt.
            let isAlreadyWaitingForReplay = serviceEpoch == nil
                && storedConversationID != nil
                && screen.conversationID == storedConversationID
            serviceEpoch = nil
            // And the conversation itself is gone with that KV. Keeping the
            // turn list would leave the app numbering turns from where it left
            // off while the service, having just ended the lineage, expects
            // zero — so the gate would refuse the next turn and every turn
            // after it, for the rest of the session.
            if let id = pendingServiceRecoveryConversationID,
               screen.conversationID == id,
               storedConversationID == id {
                prepareServiceRecoveryForReplay()
                pendingServiceRecoveryConversationID = nil
            } else if let id = pendingServiceRecoveryConversationID,
                      screen.conversationID == id {
                // A replay lost the service with this row on screen and its
                // message handed back; the failed replay already ended the
                // held lineage. The row stays for the send that follows Retry
                // Load. Archived instead, its selection went and the retried
                // message opened a brand-new chat.
                pendingServiceRecoveryConversationID = nil
            } else if isAlreadyWaitingForReplay {
                // An explicit unload already put the durable row on screen.
                // Loading the new runner must not archive that row a second
                // time or clear the selection before its deferred replay.
            } else {
                presentStoredConversationAfterReleasedKV()
            }
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

    /// Deletes every file this session staged. Called when the app is quitting,
    /// which is the only moment they are all certainly unwanted.
    ///
    /// The whole staging directory at once rather than picture by picture: a
    /// release that walks the window's own references can only free what the
    /// window still remembers, and quitting frees the rest as well.
    public func releaseAllAttachments() {
        imageAttachments.removeAll()
        outputImageAttachments = []
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

    /// Ends the conversation and starts an empty one.
    ///
    /// No confirmation any more: the conversation being left is already on
    /// disk, turn by turn, and one click in the sidebar brings it back. An
    /// alert for something that is not lost is an alert for nothing.
    public func newChat() {
        // Refused during a replay as well as during a generation. The replay is
        // putting a conversation into the KV for a turn that has already been
        // sent; starting an empty chat under it left the window on the new chat
        // and the model on the old one, and whichever landed last defined what
        // the next turn was numbered against.
        guard !isTurnInFlight else { return }
        quarantinedPersistenceLineages.removeAll()
        pendingServiceRecoveryConversationID = nil
        // The empty chat is not written until its first send, so it takes no
        // row in the sidebar and the previous conversation stops being the one
        // this window is appending to.
        storedConversationID = nil
        // Every turn holds its own hard links, and dropping the turn list
        // without releasing them leaked one staged file per image per turn
        // until quit. The release is the machine's
        // `.releaseImagesOfHeldConversation` effect, so the transitions that
        // give the KV up and the transitions that free its pictures are one
        // list rather than two that drifted apart.
        perform(machine.apply(.newChat))
        // The composer's own attachments too. Released only from the transcript
        // and the conversation, a picture attached but never sent stayed in the
        // box and followed the user into the new chat — and its staged copy
        // stayed on disk until the app quit.
        clearImages()
        conversation.startNew()
        outputPromptText = ""
        outputText = ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
        // Only the intent is recorded here; the next turn opens the new lineage
        // on the inference side. Resetting eagerly as well raced that opening
        // and sent two resets for one new chat, and it buys nothing: the KV
        // allocation is fixed, so an unsent new chat holds no extra memory.
        serviceEpoch = nil
    }

    /// Starts an empty conversation because the KV behind the old one is gone.
    ///
    /// Distinct from `newChat()`: the user did not ask for this. The transcript
    /// stays — lifecycle actions are not supposed to discard it — but its turns
    /// move out of context, because the model can no longer see them. The
    /// alternative, letting `conversation` keep counting, desynchronises the app
    /// from the service's gate, which has just gone back to expecting turn zero;
    /// the gate would then refuse every turn for the rest of the session.
    private func archiveConversationContext() {
        var carried = conversation.completedPairs
        // The live fields hold the newest finished turn between runs, and it is
        // already in `completedPairs`; nothing extra to carry.
        if carried.isEmpty, !outputPromptText.isEmpty {
            carried = [(user: AppChatTurn(role: .user, text: outputPromptText,
                                          images: outputImageAttachments),
                        assistant: AppChatTurn(role: .assistant, text: outputText))]
        }
        // The read-only copy of a chat that was merely being looked at is not
        // part of the live conversation's history, and it cannot be mistaken
        // for one now: it lives in the screen and goes with the screen. When
        // the two were one array, a reload or an unload while reading a chat
        // that could not be continued drew that chat as the live conversation's
        // own out-of-context turns, under a context break with nothing below
        // it, and Re-read into a new chat built its prompt from a conversation
        // the user had only been reading.
        perform(machine.apply(.lineageEnded))
        let outOfContext = conversation.outOfContextPairs + carried
        conversation.startNew(carryingOutOfContext: outOfContext)
        // The stored conversation keeps every turn it had, and its row will say
        // whether it can still be continued. What must not happen is the next
        // turn appending to that file while the KV holds only the new lineage:
        // the transcript on disk would then claim a context the model does not
        // have. So this window stops writing to it and the next send opens a
        // new one.
        storedConversationID = nil
        // The archived pairs are what the transcript draws now. Leaving the
        // live fields holding the newest of them would draw that turn twice,
        // once as history and once as the turn still on screen.
        if !carried.isEmpty {
            outputPromptText = ""
            outputText = ""
            outputImageAttachments = []
            generationTranscriptMailbox?.reset()
        }
    }

    /// A persisted conversation remains the conversation on screen after its
    /// KV is released. It is now a readable stored copy, and the next send
    /// restores its exact token record before appending to the same directory.
    /// A non-persisted session has no such record and keeps the legacy visible
    /// context-break behavior instead.
    private func presentStoredConversationAfterReleasedKV() {
        // Browsing and KV ownership are independent. Releasing the held KV
        // must not switch the viewed row or redirect its next message.
        guard let id = screen.conversationID ?? storedConversationID,
              conversationStore != nil,
              history.entry(id) != nil else {
            archiveConversationContext()
            return
        }
        prepareServiceRecoveryForReplay()
        openConversation(id: id)
    }

    /// Opens the conversation on the inference side if it has not been opened
    /// yet. Called before every turn: a load or an unload ends the lineage
    /// there without the app being asked, and the next turn has to re-open it
    /// rather than resume onto a KV that was rebuilt empty.
    private func openConversationIfNeeded() async throws {
        guard serviceEpoch != conversation.epoch else { return }
        guard let lifecycle = client as? AppModelLifecycleClient else { return }
        try await lifecycle.resetConversation(epoch: conversation.epoch)
        serviceEpoch = conversation.epoch
    }

    // MARK: - The send path

    /// Hands a message over to the pipeline.
    ///
    /// Synchronous on purpose, and it empties the composer before it returns:
    /// `canRun` reads the composer, so a second Generate between the click and
    /// the pipeline's first suspension is refused by the same guard that
    /// refused the first one. Started inside the task instead, two clicks
    /// started two replays of the same chat under different epochs, and the
    /// turn the first had already stamped came back rejected as belonging to a
    /// conversation that was no longer open.
    public func send() {
        recheckVisionPackAtCurrentLocation()
        guard canRun else { return }
        let turn = takeComposer()
        sendTask = Task { [weak self] in
            guard let self else { return }
            await deliver(turn)
            sendTask = nil
        }
    }

    /// Takes the message out of the composer and gives it to the caller.
    ///
    /// The images move rather than being copied: the composer stops drawing
    /// them here and nothing deletes them, because the turn is now the only
    /// thing that refers to those files. A stage that fails puts both halves
    /// back through `restoreComposer`.
    private func takeComposer() -> PreparedTurn {
        let turn = PreparedTurn(prompt: promptText, images: imageAttachments)
        promptText = ""
        imageAttachments.removeAll()
        imageAttachmentError = nil
        return turn
    }

    /// The one path a message takes, in stages.
    ///
    /// Every stage either advances or ends in `restoreComposer`, and nothing
    /// here calls itself: the replay used to hand the message back to the
    /// composer and re-enter `run()`, so a replay that resolved without putting
    /// anything into the KV re-entered the same branch forever.
    private func deliver(_ turn: PreparedTurn) async {
        var turn = turn
        // The transcript draws this message as the current turn from the moment
        // a replay starts, so a stage that fails before the model saw it takes
        // the message off the screen with it. A turn the runtime rewound
        // afterwards keeps it: the window shows a stopped turn as the current
        // one until the next send replaces it.
        var isDrawnAsTheLiveTurn = false

        // 1. Replay. A reopened conversation is on screen but not in the
        //    model's context, and the ticket below has to be reserved against
        //    the restored lineage or the service's gate refuses a turn numbered
        //    against the wrong epoch.
        //
        //    Anything but the live conversation goes through here. Only
        //    `.reading` used to, so a send while the screen named an unreadable
        //    row skipped the replay and ran on the held chat: the turn was
        //    written into that chat's file while the sidebar, the notice and
        //    the saved selection all named the row that could not be read.
        if case .live = screen {} else {
            var replayID: UUID?
            for case .replay(let id, _) in machine.apply(.sendRequested(turn)) {
                replayID = id
            }
            guard let replayID else {
                // The machine refused: the row on screen cannot be continued,
                // which `canRun` already says. Handed back rather than run on
                // the held conversation under the wrong row.
                restoreComposer(turn, error: .invalidRequest(
                    "The conversation on screen cannot be continued."),
                    withdrawingFromTranscript: false)
                return
            }
            // The message moves into the transcript now, under the same prefill
            // placeholder an ordinary turn gets. A replay is a full prefill of
            // everything the conversation holds, and for the length of it the
            // window showed nothing at all — the composer was empty and the
            // transcript had not changed — which is what made a second Generate
            // the obvious thing to press.
            runIdentity &+= 1
            generationTranscriptMailbox?.reset()
            outputPromptText = turn.prompt
            outputImageAttachments = turn.images.map(ChatImage.staged)
            outputText = ""
            isDrawnAsTheLiveTurn = true
            if await !replayIntoKV(id: replayID, turn: turn) {
                // The machine's `.restoreComposer` effect has already handed
                // the message back with the reason it could not be replayed.
                return
            }
        }

        // 2. Reserve the position the service will check, before the request is
        //    built, so the position the transcript shows is the position sent.
        guard let ticket = conversation.beginTurn(text: turn.prompt) else {
            // `canRun` already required `conversation.canSend`, and the emptied
            // composer refuses a second send, so nothing should reach this.
            // Refused with the message given back rather than dropped.
            restoreComposer(turn, error: .invalidRequest(
                "This conversation is already sending a turn."),
                withdrawingFromTranscript: isDrawnAsTheLiveTurn)
            return
        }

        // 3. Build the request from the message, not from the composer: the
        //    composer was emptied at stage 1 and may already hold the next one.
        var request: AppGenerationRequest
        do {
            request = try makeRequest(turn, ticket: ticket)
        } catch {
            conversation.abandonTurn()
            restoreComposer(
                turn,
                error: (error as? AppInferenceError) ?? .unknown("\(error)"),
                withdrawingFromTranscript: isDrawnAsTheLiveTurn)
            return
        }

        // 4. Retain the images. The run reads the transcript's own hard links
        //    rather than the composer's files, so the hand-off below cannot
        //    delete an image this request has not opened yet. A failed retain
        //    leaves no reference guaranteed to outlive the composer, so the run
        //    is refused instead of started against files about to be removed.
        var retained: [StagedImage] = []
        do {
            for attachment in request.imageAttachments {
                retained.append(try attachmentStore.retain(attachment))
            }
        } catch {
            for attachment in retained { attachmentStore.remove(attachment) }
            conversation.abandonTurn()
            imageAttachmentError = String(describing: error)
            restoreComposer(turn, error: .invalidRequest(
                "Could not prepare the attached images for this run: \(error)"),
                withdrawingFromTranscript: isDrawnAsTheLiveTurn)
            return
        }
        request.imageAttachments = retained
        // The hand-off. The message is carried by its retained links from here,
        // and the copies it arrived with are referenced by nothing: handing
        // those back after a rewind would give the user thumbnails with no
        // files behind them.
        for attachment in turn.images { attachmentStore.remove(attachment) }
        turn = turn.carrying(retained)

        // 5. Start the turn.
        persistSettings()
        generationTranscriptMailbox?.reset()
        runIdentity &+= 1
        let generation = runIdentity
        outputPromptText = request.prompt
        // Not released: every turn of a conversation keeps its own images for
        // as long as the conversation shows them. They are hard links to files
        // that already exist, so holding them costs no additional bytes.
        conversation.attachImagesToPendingTurn(retained)
        outputImageAttachments = retained.map(ChatImage.staged)
        outputText = ""
        diagnostics = nil
        error = nil
        hasHandledTerminalEvent = false
        turnOutcome = .committed
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

        // 6. Generate. The directory is created here, on the first send, rather
        //    than at New Chat: a chat the user opens and never uses must leave
        //    nothing behind. The images start being written now, overlapped with
        //    the reply the user is already waiting for.
        if let storedID = await ensureStoredConversation(firstMessage: request.prompt) {
            beginStoringTurnImages(request.imageAttachments, in: storedID)
        }
        // Off the main actor: an event stream whose producer runs inline would
        // hold the window for the length of the run.
        let stream = Task.detached { [weak self, client, request, generation] in
            guard let self else { return }
            do {
                try await self.openConversationIfNeeded()
                for try await event in client.generate(request) {
                    await self.apply(event, generation: generation)
                }
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError, generation: generation)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"), generation: generation)
            }
        }
        // The turn is over at its terminal event, not when the stream happens
        // to close: a client that reports a failure and then goes quiet would
        // otherwise hold the composer, New Chat and every later send for as
        // long as it stayed quiet. The stream is watched as well, so a stream
        // that ends without a terminal event still releases this.
        await withCheckedContinuation { continuation in
            turnCompletion = (generation: generation, continuation: continuation)
            Task { [weak self] in
                await stream.value
                self?.endTurnWait(generation: generation)
            }
        }

        // 7. Commit or hand back. `finishSuccessfully` has already written the
        //    turn; a rewind reaches here with the cause already in `error`.
        guard case .rewound = turnOutcome else { return }
        restoreComposer(turn, error: nil, withdrawingFromTranscript: false)
        // Its images were written while the reply was generating, and the turn
        // that would have cited them is gone. Ended before the sweep rather
        // than beside it: the sweep deletes every file no record names, and a
        // write still running would have raced it into the same directory.
        await discardPendingTurnImages()
        await sweepImagesOfRewoundTurn()
    }

    /// Puts a stored conversation back into the KV and says whether the turn
    /// waiting on it may go.
    private func replayIntoKV(id: UUID, turn: PreparedTurn) async -> Bool {
        phase = .prefill
        livePrefillDone = 0
        // A denominator from the record, so the gauge counts up from the first
        // moment rather than from whenever the first progress event lands.
        livePrefillTotal = history.entry(id)?.kvTokens ?? 0
        let outcome = await replayOutcome(id: id)
        // Back out of the prefill phase the send put the window into, or the
        // gauge keeps claiming a replay that has already finished.
        phase = .idle
        livePrefillDone = 0
        livePrefillTotal = 0
        let effects = machine.apply(outcome)
        perform(effects)
        return effects.contains { if case .startTurn = $0 { return true } else { return false } }
    }

    /// Puts a turn that never reached the model, or one the runtime rewound,
    /// back in the composer.
    ///
    /// The one function that hands a message back; every failing stage reaches
    /// it exactly once. `error` is nil when the failure has already recorded
    /// itself — a rewound generation sets `error` as it ends, and a second copy
    /// of the same cause would replace what the window is already showing.
    ///
    /// `withdrawingFromTranscript` takes the message off the screen as well.
    /// True for every stage that failed before the model saw the turn, because
    /// what is drawn there is this very message with nothing underneath it;
    /// false for a turn the runtime rewound, which the window keeps drawing as
    /// the current one until the next send replaces it.
    func restoreComposer(_ turn: PreparedTurn,
                         error: AppInferenceError?,
                         withdrawingFromTranscript: Bool) {
        if withdrawingFromTranscript {
            outputPromptText = ""
            outputText = ""
            generationTranscriptMailbox?.reset()
        }
        // Dropped, never released: these are the same links the turn below is
        // carrying, and deleting them here handed the composer back a set of
        // pictures whose files had just been removed. Only when they are this
        // message's, though. A refusal before the model saw the turn — a
        // request that did not validate, a retain that failed — reaches here
        // with the live fields still drawing the previous turn, and clearing
        // them made that turn's pictures vanish from the transcript.
        var own: Set<UUID> = []
        for image in turn.images { own.insert(image.id) }
        var drawsThisMessage = withdrawingFromTranscript
        for image in outputImageAttachments {
            if let staged = image.staged, own.contains(staged.id) {
                drawsThisMessage = true
            }
        }
        if drawsThisMessage {
            outputImageAttachments = []
        }
        if promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            promptText = turn.prompt
        }
        if imageAttachments.isEmpty {
            imageAttachments = turn.images
        } else {
            // The composer was used again while the turn was in flight, and its
            // copies are the ones the next send will carry.
            for attachment in turn.images { attachmentStore.remove(attachment) }
        }
        if let error { self.error = error }
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        client.cancel()
    }

    /// The request the composer would build right now, for the callers that
    /// only need to know whether what is in it is valid.
    public func makeRequest(
        ticket: AppConversation.Ticket? = nil
    ) throws -> AppGenerationRequest {
        try makeRequest(
            PreparedTurn(prompt: promptText, images: imageAttachments),
            ticket: ticket)
    }

    /// The request one message makes.
    ///
    /// Built from the turn rather than from the composer, which the send path
    /// emptied before this is reached and which may already hold the next
    /// message.
    func makeRequest(
        _ turn: PreparedTurn,
        ticket: AppConversation.Ticket? = nil
    ) throws -> AppGenerationRequest {
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
            prompt: turn.prompt,
            imageAttachments: turn.images,
            maxNewTokens: maxNewTokensOverride ?? effective.maxContextTokens,
            maxContextTokens: effective.maxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: effective.options(
                prefillEnabled: runtimeOptions.prefillEnabled,
                prefillChunkTokens: runtimeOptions.prefillChunkTokens),
            // Carried whether or not a ticket exists: the image budget has to
            // fit around the conversation even while the composer is only being
            // validated.
            // The decode service overrides this from its own gate, but the
            // in-process client reads it directly — and without it that client
            // ran every turn through the single-prompt path while the app drew a
            // growing transcript, so the model saw only the newest message.
            continuesConversation: ticket != nil,
            // Unknown means the runtime committed a turn but did not report its
            // position. Reserve the whole window: text can still continue on
            // the service's own exact state, while every image fails closed.
            conversationTokens: conversation.kvTokens ?? effective.maxContextTokens,
            conversationEpoch: ticket?.epoch,
            turnIndex: ticket?.index)
        try request.validate(requireModelDirectory: true)
        return request
    }

    func apply(_ event: AppInferenceEvent, generation: Int? = nil) {
        guard generation == nil || generation == runIdentity else { return }
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
        let userText = outputPromptText
        let assistantText = outputText
        conversation.completeTurn(text: outputText, diagnostics: diagnostics)
        // Written here and nowhere else: this is the one path that commits a
        // turn to the KV, so it is the one path where the transcript on disk
        // and the model's context can be made to say the same thing. A turn
        // that threw was rewound by the runtime and reaches `finishWithError`,
        // which writes nothing.
        enqueueCompletedTurnPersistence(
            userText: userText, assistantText: assistantText,
            diagnostics: diagnostics)
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        error = .cancelled
        // A `.cancelled` event means the run threw `CancellationError`, and the
        // conversation rewound the turn: its tokens are not in the KV, and the
        // service did not count it either. Counting it here put the app one
        // ahead for the rest of the conversation, so the next turn carried an
        // index the gate refused — and that refusal reached the user as
        // "decode service runtime profile changed during generation".
        //
        // A stop that lands at a token boundary is a different event: the run
        // returns normally with `.cancelled` as its *stop reason*, arrives as
        // `.finished`, and is committed by `finishSuccessfully`.
        // Not counted. The live fields are not part of `conversation.turns`, so
        // what stays on screen is the stopped turn as the current one, exactly
        // as the single-prompt path left it — the next run replaces it, and it
        // never enters the history the transcript freezes.
        //
        // The turn itself is handed back rather than dropped: discarding it lost
        // the user's message and stranded its retained image links, which the
        // next run overwrote without releasing — one staged file per image,
        // until quit. Recorded rather than handed back here: the send pipeline
        // is holding the message, and it is the one place that gives it back.
        if conversation.abandonTurn() != nil { turnOutcome = .rewound }
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
        let abandoned: AppChatTurn?
        if case .conversationLineageLost = appError {
            // The transcript stays readable; nothing further can be sent until
            // New chat.
            abandoned = conversation.markLineageLost()
        } else {
            // The runtime rewound this turn, so it is in neither the KV nor the
            // transcript. Give the user their message back instead of making
            // them retype it.
            abandoned = conversation.abandonTurn()
        }
        if abandoned != nil { turnOutcome = .rewound }
        finishTerminalRun()
        if case .connectionLost = appError {
            recordLostServiceConnection(appError)
            if let id = storedConversationID, let meta = history.entry(id) {
                pendingServiceRecoveryConversationID = id
                perform(machine.apply(.rowClicked(
                    id: id, heldID: id, kvMatchesHeld: false,
                    state: continuability(of: meta), renderID: UUID())))
            }
        }
    }

    /// The loaded model is gone with the decode service that held it.
    ///
    /// One place for the state a lost connection leaves, reached by a
    /// generation that ends on it and by a replay that does: the runtime key,
    /// the memory readings and the epoch all belonged to a process that no
    /// longer exists, and `.failed` is what puts Retry Load on screen.
    func recordLostServiceConnection(_ appError: AppInferenceError) {
        loadedRuntimeKey = nil
        liveMemoryBytes = nil
        liveResidentBytes = nil
        loadState = .failed(appError)
        serviceEpoch = nil
    }

    private func finishStreamFailure(_ appError: AppInferenceError, generation: Int) {
        guard generation == runIdentity else { return }
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        phase = .idle
        runState = .idle
        isCancellationPending = false
        activeRunRuntimeKey = nil
        endTurnWait(generation: runIdentity)
    }

    /// Releases the send's last stage.
    ///
    /// Keyed on the run it belongs to: a stream from an abandoned run can close
    /// long after the run that replaced it started, and resuming the newer
    /// turn's wait would commit it before it had generated anything.
    private func endTurnWait(generation: Int) {
        guard let pending = turnCompletion, pending.generation == generation else {
            return
        }
        turnCompletion = nil
        pending.continuation.resume()
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

    public func shutdownForTermination() {
        guard !hasShutDownForTermination else { return }
        hasShutDownForTermination = true
        // Direct inspector bindings may have changed since the last saved action.
        persistSettings()
        client.cancel()
        (client as? AppModelLifecycleClient)?.shutdownForTermination()
        releaseAllAttachments()
    }
}
