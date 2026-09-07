import Foundation
import TurboFieldfareValidationSupport
@testable import TurboFieldfareAppCore

/// One thing a user can do to their conversation history.
///
/// The alphabet a seeded walk draws from. Every serious defect this feature
/// produced needed three or four of these in a row — delete the chat the KV is
/// holding and send again, read one chat then another then the first — and
/// every test written for those defects afterwards is one hand-written
/// sequence. This is the same thing generated.
enum HistoryOperation: Sendable, CustomStringConvertible {
    case newChat
    /// `label` names the message: prompts are `m<label>`, so a failure trace
    /// carries positions and counts and never any content.
    case send(label: Int)
    /// The same message repeated until the turn is thousands of tokens long.
    ///
    /// The only way this alphabet reaches a conversation that no longer fits.
    /// Continuability is `kvTokens + 256 <= context`, the smallest context the
    /// window offers is 4,096, and an ordinary synthetic turn here costs seven
    /// tokens — so without this a walk would need five hundred turns in one
    /// chat before a row stopped being continuable, and `needsContext`, the
    /// state the read-only transcript and the "raise the context" notice exist
    /// for, would never be entered at all.
    case sendLong(label: Int)
    case sendWithImage(label: Int)
    /// Row indices are drawn blind and taken modulo the row count when applied,
    /// so a draw is reproducible from the seed alone.
    case open(row: Int)
    case delete(row: Int)
    case rename(row: Int, label: Int)
    case setContext(tokens: Int)
    case failNextRestore
    case relaunch
    /// Reload the model. A load builds a new runner and an empty KV, so the
    /// service ends the lineage and the app's conversation moves out of
    /// context — the one lifecycle action that keeps the window open while
    /// taking the model's memory of it away.
    case reload
    /// Unload the model. The same release as a reload without the rebuild:
    /// nothing can be sent until it comes back.
    case unload
    /// Send, then press Stop while the reply is still decoding.
    ///
    /// The in-process client's Stop is a hard abort: the runtime rewinds the
    /// turn, so it is in neither the KV nor the transcript, and the message
    /// comes back to the composer. Whether the stop reaches the run before it
    /// finishes is a race, so the outcome is read off disk rather than assumed.
    case stopMidTurn(label: Int)

    var description: String {
        switch self {
        case .newChat: return "newChat"
        case .send(let label): return "send(m\(label))"
        case .sendLong(let label): return "sendLong(m\(label))"
        case .sendWithImage(let label): return "sendWithImage(m\(label))"
        case .open(let row): return "open(#\(row))"
        case .delete(let row): return "delete(#\(row))"
        case .rename(let row, let label): return "rename(#\(row), t\(label))"
        case .setContext(let tokens): return "setContext(\(tokens))"
        case .failNextRestore: return "failNextRestore"
        case .relaunch: return "relaunch"
        case .reload: return "reload"
        case .unload: return "unload"
        case .stopMidTurn(let label): return "stopMidTurn(m\(label))"
        }
    }

    /// The contexts the walk switches between.
    ///
    /// Only values the window offers. `MacAppSettings.isValid` rejects anything
    /// else, and an invalid settings file is deleted and replaced with the
    /// defaults on the next launch — so a walk that set 2,048 (which is not one
    /// of the options, whatever the plan said) came back from a relaunch at
    /// 8,192, and the mismatch was the harness's, not the app's.
    static let contexts = [
        AppContextLengthOption.fourK.tokens,
        AppContextLengthOption.eightK.tokens,
        AppContextLengthOption.sixteenK.tokens,
    ]

    /// How many words a long message repeats to.
    ///
    /// One of them makes a conversation too big for a 4K context and small
    /// enough for an 8K one, so a single `setContext` moves a row in and out of
    /// `needsContext`.
    static let longMessageWords = 5_000

    /// Weights from the plan, with its 0.35 for sending split six to one
    /// between short and long messages: sending and browsing dominate
    /// because they are what a user does, and the destructive operations are
    /// frequent enough that a forty-step walk sees several of each.
    ///
    /// The three lifecycle operations take their 0.05 each out of the plain
    /// send, which is the only draw with enough weight to give it up and still
    /// dominate. A reload or an unload releases the KV, and every send after
    /// one has to open a new file rather than append to a conversation the
    /// model can no longer see.
    private static let weights:
        [(cumulative: Float, make: @Sendable (Int, Int, Int) -> HistoryOperation)] = [
        (0.15, { step, _, _ in .send(label: step) }),
        (0.20, { step, _, _ in .sendLong(label: step) }),
        (0.40, { _, row, _ in .open(row: row) }),
        (0.50, { _, _, _ in .newChat }),
        (0.60, { _, row, _ in .delete(row: row) }),
        (0.65, { step, row, _ in .rename(row: row, label: step) }),
        (0.70, { _, _, pick in .setContext(tokens: contexts[pick % contexts.count]) }),
        (0.75, { step, _, _ in .sendWithImage(label: step) }),
        (0.80, { _, _, _ in .failNextRestore }),
        (0.85, { _, _, _ in .relaunch }),
        (0.90, { _, _, _ in .reload }),
        (0.95, { _, _, _ in .unload }),
        (1.00, { step, _, _ in .stopMidTurn(label: step) }),
    ]

    static func next(step: Int, using generator: inout SplitMix64) -> HistoryOperation {
        let roll = generator.uniform(0, 1)
        let row = Int(generator.next() % 16)
        let pick = Int(generator.next() % 16)
        for entry in weights where roll < entry.cumulative {
            return entry.make(step, row, pick)
        }
        return .send(label: step)
    }
}
