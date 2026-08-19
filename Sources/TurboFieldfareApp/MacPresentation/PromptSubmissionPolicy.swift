import SwiftUI
import TurboFieldfareAppCore

public enum PromptSubmissionDecision: Equatable, Sendable {
    case submit
    case consume
    case deferToEditor
}

public enum PromptSubmissionPolicy {
    public static func decision(
        newlineShortcut: AppNewlineShortcut,
        modifiers: EventModifiers,
        canRun: Bool,
        hasMarkedText: Bool,
        isRepeat: Bool = false
    ) -> PromptSubmissionDecision {
        guard newlineShortcut == .shiftReturn else { return .deferToEditor }

        let nativeModifiers: EventModifiers = [.command, .shift, .option, .control]
        guard modifiers.intersection(nativeModifiers).isEmpty else {
            return .deferToEditor
        }
        if isRepeat { return .consume }
        guard !hasMarkedText else { return .deferToEditor }
        return canRun ? .submit : .consume
    }
}
