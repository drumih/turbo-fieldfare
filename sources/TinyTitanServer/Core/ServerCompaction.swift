import TinyTitanCore
//
//  ServerCompaction.swift
//  TinyTitanServer
//
//  `/v1/responses/compact`: the payload that travels in a `compaction` item, the
//  instruction the summariser runs under, and the budget it is held to.
//

import Foundation
import TinyTitan
import TinyTitanLib

/// How a compaction was produced, so a replayed note can say what it is.
public enum CompactionMode: String, Codable, Sendable {
    /// One summariser pass under the handover instruction.
    case model
    /// A second pass, because the first note was over budget.
    case compressed
    /// No usable summary came back: the transcript was trimmed to the newest
    /// turns instead, so a compaction is never empty and never fails the caller.
    case extractive
}

/// What a `compaction` item's `encrypted_content` actually carries.
///
/// The spec's field is provider-opaque, not necessarily encrypted: it exists so a
/// provider can hand a client a payload it can round-trip without the client
/// reading it. This server is loopback-only and is summarising the caller's own
/// session, so the payload is a base64 JSON envelope the client is not expected to
/// inspect, and it is versioned so tomorrow's shape can be told apart from
/// today's. `docs/server-api.md` says this in as many words rather than letting
/// "encrypted" imply a key that does not exist.
public struct CompactionEnvelope: Codable, Equatable, Sendable {
    /// Envelope version, so a replayed payload from another build is refused by
    /// name instead of being misread.
    public static let version = 1

    public let v: Int
    public let model: String
    public let createdAt: Int
    public let mode: CompactionMode
    public let summary: String

    public init(model: String, createdAt: Int, mode: CompactionMode, summary: String) {
        v = Self.version
        self.model = model
        self.createdAt = createdAt
        self.mode = mode
        self.summary = summary
    }
}

public enum ServerCompaction {
    /// The default compacted size, as a fraction of the served context.
    ///
    /// A compaction is only worth making if it is dramatically smaller than what
    /// it replaces; a note that grows with the session has bought nothing. One
    /// eighth of the window leaves the rest for the work that follows, and the
    /// absolute floor keeps a tiny context from producing an unusable note.
    ///
    /// A non-positive `requested` falls to that default, which is a floor for the
    /// callers inside this repository. It is not the answer to a client: the wire
    /// refuses such a value by name, because a caller that set a budget should
    /// hear that its number was not the one used.
    public static func targetTokens(maxContext: Int, requested: Int?) -> Int {
        let share = max(256, maxContext / 8)
        let cap = min(4096, share)
        guard let requested, requested > 0 else { return cap }
        return min(requested, maxContext / 2)
    }

    /// The handover instruction.
    ///
    /// A generic "summarise this" loses exactly what an agent needs next: the
    /// constraint it was told to respect, the path it had already found, the
    /// option that was rejected and why. So this names what to keep, asks for
    /// exact wording where the wording *is* the fact, and forbids inventing.
    ///
    /// It is deliberately short and its headings are a *menu*, not a form: a
    /// long numbered instruction is read back out by a small model, which is how
    /// a compaction of a two-line session came back with the instruction's own
    /// items — including "Open questions" — as if they were content.
    public static func instruction(limit: Int) -> String {
        """
        You compact a working session into a handover note for a fresh context.

        Write only the note. Do not restate or copy these instructions, and leave
        out any heading that has nothing under it.

        Use short headings, and only the ones that apply:
        - Goal
        - Requirements (paths, names, numbers and flags quoted exactly)
        - Decisions (with the reason, and any option rejected)
        - Facts (files, commands that worked, errors verbatim, measurements)
        - Next step
        - Open questions

        Keep every requirement, constraint, path, identifier and number. Invent
        nothing that is not in the session. Note-form, no preamble. At most \
        \(limit) tokens.
        """
    }

    /// The plain retry, for a model that echoed the structured instruction.
    ///
    /// A short instruction with no headings to copy is what a small model needs;
    /// it is cheaper to ask again than to hand a caller a template as history.
    public static func plainInstruction(limit: Int) -> String {
        """
        Summarise the session below as a short handover note: what was decided \
        and why, the requirements and constraints, the facts worth keeping, and \
        the next step. Do not repeat this instruction. At most \(limit) tokens.
        """
    }

    /// Whether a note is degenerate rather than a summary.
    ///
    /// A small model under a long instruction can fall into a repetition loop:
    /// measured on the 2B, four transcript lines came back more than a dozen
    /// times and the "note" was longer than the session it replaced. A loop is
    /// not a compaction, and it is cheap to recognise — most of the note's lines
    /// being duplicates of each other — so it is treated as a failed pass instead
    /// of being replayed as history.
    public static func isDegenerate(_ note: String) -> Bool {
        let lines = note.split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { $0.count >= 16 }
        guard lines.count >= 6 else { return false }
        return Set(lines).count * 2 < lines.count
    }

    /// Remove lines a note copied from its own instruction.
    ///
    /// The instruction is known exactly, so an echo can be recognised and
    /// dropped deterministically rather than being replayed to the next turn as
    /// part of the session. A leading list marker is ignored when matching, since
    /// a model renumbers what it copies.
    public static func strippingInstructionEcho(_ note: String, instruction: String) -> String {
        func normalized(_ line: String) -> String {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let withoutMarker = trimmed.drop {
                $0.isNumber || $0 == "." || $0 == ")" || $0 == "("
                    || $0 == "-" || $0 == "*" || $0 == " "
            }
            return withoutMarker.lowercased().trimmingCharacters(in: .whitespaces)
        }
        let instructionLines = Set(
            instruction.split(separator: "\n")
                .map { normalized(String($0)) }
                .filter { !$0.isEmpty })

        var kept: [String] = []
        for line in note.split(separator: "\n", omittingEmptySubsequences: false) {
            let text = String(line)
            if !normalized(text).isEmpty, instructionLines.contains(normalized(text)) { continue }
            // Skip the blank runs a removal leaves behind.
            if text.trimmingCharacters(in: .whitespaces).isEmpty,
                kept.last?.trimmingCharacters(in: .whitespaces).isEmpty != false
            {
                continue
            }
            kept.append(text)
        }
        return kept.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The second pass, for a note that came back over budget.
    ///
    /// Compressing the note is better than truncating it: truncation drops the
    /// end of the session, which is the part the next turn is most likely to
    /// need. This pass keeps every fact and removes prose instead.
    public static func compressionInstruction(limit: Int) -> String {
        """
        The note below is a session compaction that is still too long: it has to \
        fit in \(limit) tokens. Rewrite it shorter. Keep every requirement, \
        constraint, decision, path, identifier and number; remove prose, \
        repetition and explanation. Emit only the shorter note, with no \
        preamble.
        """
    }

    /// The text a replayed note contributes to the next request's prompt.
    ///
    /// It says what it is before it says what happened, so the model treats the
    /// note as prior context rather than as a fresh instruction. The mapper puts
    /// it in the leading system block rather than as its own message: TinyTitan's
    /// chat template wants exactly one opening system message, and a compacted
    /// window stands in for the opening of the conversation anyway.
    public static func replayNote(_ envelope: CompactionEnvelope) -> String {
        """
        Compacted earlier session (TinyTitan compaction v\(envelope.v), \
        \(envelope.mode.rawValue)). Treat the note below as the conversation so \
        far, study it, and continue from it:

        \(envelope.summary)
        """
    }

    /// Base64 of the JSON envelope, which is what a `compaction` item carries.
    public static func encode(_ envelope: CompactionEnvelope) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(envelope).base64EncodedString()
    }

    /// Read a payload back, or refuse it by name.
    ///
    /// A payload that is not ours, is truncated, or comes from another envelope
    /// version is a request we cannot honour — and quietly dropping it would
    /// silently discard the caller's history, which is the one failure mode this
    /// endpoint must not have.
    public static func decode(_ payload: String) throws -> CompactionEnvelope {
        guard let data = Data(base64Encoded: payload),
            let envelope = try? JSONDecoder().decode(CompactionEnvelope.self, from: data)
        else {
            throw ServerRequestError.invalid(
                message: "compaction payload could not be read; it must be the "
                    + "encrypted_content this server returned",
                param: "input", code: "compaction_payload_invalid")
        }
        guard envelope.v == CompactionEnvelope.version else {
            throw ServerRequestError.invalid(
                message: "compaction payload version \(envelope.v) is not supported "
                    + "(this build reads v\(CompactionEnvelope.version))",
                param: "input", code: "compaction_payload_version")
        }
        return envelope
    }

    /// The last-resort note: the newest turns, trimmed to a character budget.
    ///
    /// Only reached when a summariser returns nothing usable. It keeps the head
    /// of the transcript (usually the instructions) and the tail (the newest
    /// work), which is the part a continuation needs, and says in the middle what
    /// happened to the rest rather than pretending nothing is missing.
    public static func extractiveSummary(transcript: String, characterBudget: Int) -> String {
        guard transcript.count > characterBudget else { return transcript }
        let marker = "\n\n…[earlier turns dropped to fit the compaction budget]…\n\n"
        let keep = max(0, characterBudget - marker.count)
        let head = keep / 3
        return String(transcript.prefix(head)) + marker
            + String(transcript.suffix(keep - head))
    }

    /// The conversation as role-labelled text, which is what the summariser is
    /// handed.
    ///
    /// A transcript rather than a second conversation: the summariser's own
    /// instruction is its system message, and handing it the messages directly
    /// would put a second system block in front of a template that wants one.
    /// Roles are spelled out so a note can say who decided what, and a tool call
    /// is rendered rather than dropped — its arguments are often the fact worth
    /// keeping.
    package static func transcript(_ messages: [OpenAIChatMessage]) -> String {
        messages.map { message in
            var text = ""
            if let content = message.content {
                switch content {
                case .text(let value): text = value
                case .parts(let parts): text = parts.compactMap(\.text).joined()
                }
            }
            if let calls = message.toolCalls, !calls.isEmpty {
                let rendered =
                    calls
                    .map { "\($0.function.name)\($0.function.arguments)" }
                    .joined(separator: ", ")
                text += text.isEmpty ? "(tool call: \(rendered))" : "\n(tool call: \(rendered))"
            }
            return "[\(message.role)] \(text)"
        }.joined(separator: "\n\n")
    }
}

extension OpenAIUsage {
    /// What one compaction cost, across its passes.
    ///
    /// The client is told the whole price of the note, not the last pass's
    /// share: a second pass exists to make the note fit, and hiding that it ran
    /// would make the number a lie of omission.
    func adding(_ other: OpenAIUsage) -> OpenAIUsage {
        OpenAIUsage(
            promptTokens: promptTokens + other.promptTokens,
            completionTokens: completionTokens + other.completionTokens,
            totalTokens: totalTokens + other.totalTokens,
            cachedTokens: promptTokensDetails.cachedTokens
                + other.promptTokensDetails.cachedTokens,
            reasoningTokens: completionTokensDetails.reasoningTokens
                + other.completionTokensDetails.reasoningTokens)
    }
}
