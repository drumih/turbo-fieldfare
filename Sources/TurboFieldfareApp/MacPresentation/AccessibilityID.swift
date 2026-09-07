import SwiftUI

/// Every control the Mac app exposes to an acceptance pass, named once.
///
/// The pass drives the app through the accessibility API and addresses
/// controls by these identifiers, never by coordinates: both false readings of
/// the 2026-09-01 manual pass were clicks at stale coordinates. Raw values are
/// the strings the accessibility tree carries as `AXIdentifier`;
/// `Scripts/check_app_case_coverage.rb` fails when one has no case in the
/// `mac-app-acceptance` skill, so a new control cannot land without its case.
public enum AccessibilityID: String, CaseIterable, Sendable {
    case stripSidebar = "strip.sidebar"
    case stripNewChat = "strip.newChat"
    case stripInspector = "strip.inspector"

    case hudStatus = "hud.status"
    case hudPhase = "hud.phase"
    case hudRate = "hud.rate"
    case hudContext = "hud.context"
    case hudMemory = "hud.memory"
    case hudMemoryInfo = "hud.memory.info"

    case historySearch = "history.search"
    case historySearchClear = "history.search.clear"
    case historyRenameField = "history.rename.field"
    case historyRenameConfirm = "history.rename.confirm"
    case historyDeleteConfirm = "history.delete.confirm"

    case composerMessage = "composer.message"
    case composerGenerate = "composer.generate"
    case composerStop = "composer.stop"
    case composerAttach = "composer.attach"
    case composerClear = "composer.clear"
    case composerTips = "composer.tips"
    case examplesMore = "examples.more"

    case transcriptCopyResponse = "transcript.copyResponse"
    case transcriptLoad = "transcript.load"
    case transcriptReload = "transcript.reload"

    case noticeRaiseContext = "notice.raiseContext"
    case noticeNewChat = "notice.newChat"
    case noticeReread = "notice.reread"

    case bannerErrorDismiss = "banner.error.dismiss"
    case bannerModelAction = "banner.model.action"

    case installDownload = "install.download"
    case installCancel = "install.cancel"
    case installDiscard = "install.discard"
    case installCheckAgain = "install.checkAgain"

    case visionInstall = "vision.install"
    case visionCancel = "vision.cancel"
    case visionDiscard = "vision.discard"
    case visionActivate = "vision.activate"
    case visionRemove = "vision.remove"

    case inspectorContext = "inspector.context"
    case inspectorSlots = "inspector.slots"
    case inspectorTemperature = "inspector.temperature"
    case inspectorTopK = "inspector.topK"
    case inspectorTopKValue = "inspector.topK.value"
    case inspectorTopP = "inspector.topP"
    case inspectorTopPValue = "inspector.topP.value"
    case inspectorPrefill = "inspector.prefill"
    case inspectorRDAdvise = "inspector.rdadvise"
    case inspectorCopyPath = "inspector.copyPath"
    case inspectorUnload = "inspector.unload"

    /// Identifiers minted per item carry one of these prefixes, so a case can
    /// name the family and the driver can match one member exactly.
    public enum Prefix: String, CaseIterable, Sendable {
        case historyRow = "history.row."
        case historyRowMenu = "history.row.menu."
        case composerRemove = "composer.remove."
        case example = "examples."
    }

    public static func row(_ id: UUID) -> String { Prefix.historyRow.rawValue + id.uuidString }
    public static func rowMenu(_ id: UUID) -> String { Prefix.historyRowMenu.rawValue + id.uuidString }
    public static func remove(_ id: String) -> String { Prefix.composerRemove.rawValue + id }
    public static func example(_ id: String) -> String { Prefix.example.rawValue + id }
}

public extension View {
    func accessibilityIdentifier(_ id: AccessibilityID) -> some View {
        accessibilityIdentifier(id.rawValue)
    }
}
