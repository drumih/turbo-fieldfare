import TinyTitanCore
//
//  HTTPServerHandler+Compact.swift
//  TinyTitanServer
//
//  `POST /v1/responses/compact`: a conversation in, a compacted window out.
//

import Foundation
import NIOCore
import NIOHTTP1
import Synchronization
import TinyTitanLib

extension ServerHTTPHandler {
    /// Compact a conversation into a note a fresh context can continue from.
    ///
    /// This is a *value* endpoint, not a turn: nothing is stored, no session
    /// begins, and the response is the window the client sends back as the base
    /// `input` of its next response. That is why the note travels inside the
    /// item and is decoded on the way back in by
    /// `ResponsesAPIMapper.chatMessages` — the round trip is the whole feature,
    /// and it needs no server-side state to work.
    func handleCompact(body: ByteBuffer, context: ChannelHandlerContext) {
        do {
            let decoded = try JSONDecoder().decode(
                CompactionRequest.self, from: Data(body.readableBytesView))
            // The spec requires `model`; it is decoded as optional so a missing
            // one is refused by name rather than as malformed JSON.
            guard let requestedModel = decoded.model, !requestedModel.isEmpty else {
                throw ServerRequestError.invalid(
                    message: "model is required",
                    param: "model", code: "invalid_value")
            }
            let target = try servedModel(named: requestedModel)
            let conversation = try ResponsesAPIMapper.chatMessages(
                items: decoded.inputItems, instructions: decoded.instructions)
            guard !conversation.isEmpty else {
                throw ServerRequestError.invalid(
                    message: "nothing to compact: input carries no messages",
                    param: "input", code: "invalid_value")
            }
            // A budget is the one number the caller chose, so a value that cannot
            // be read as one is refused rather than answered with the server's
            // own default: `targetTokens` keeps a positive floor for its own
            // callers, and that floor used to be the answer to `0` — a client
            // asking for no room at all got a 4096-token note and no signal.
            if let requested = decoded.maxCompactionTokens {
                guard requested > 0 else {
                    throw ServerRequestError.invalid(
                        message: "max_compaction_tokens must be greater than 0; omit it "
                            + "to let the server choose from the context window",
                        param: "max_compaction_tokens", code: "invalid_value")
                }
            }
            let budget = ServerCompaction.targetTokens(
                maxContext: target.maximumContext, requested: decoded.maxCompactionTokens)
            let resourceID =
                "resp_cmp_"
                + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            activeTask = childChannels.startTask {
                ServerLog.accepted(id: resourceID, streaming: false)
                do {
                    let result = try await self.compacted(
                        conversation: conversation, target: target, budget: budget)
                    ServerLog.compacted(
                        id: resourceID, mode: result.envelope.mode.rawValue,
                        noteTokens: result.noteTokens, budget: budget)
                    var output: [[String: Any]] = []
                    // The caller's instructions are preserved verbatim rather
                    // than summarised: the spec asks compaction to keep system
                    // prompts, and a paraphrase of a constraint is not the
                    // constraint.
                    if let instructions = decoded.instructions, !instructions.isEmpty {
                        output.append(
                            ResponsesAPIBuilder.messageItem(
                                id: ResponsesAPIBuilder.messageItemID(responseID: resourceID),
                                role: "developer", text: instructions, status: "completed"))
                    }
                    output.append(
                        ResponsesAPIBuilder.compactionItem(
                            id: "cmp_" + resourceID.dropFirst("resp_cmp_".count),
                            encryptedContent: try ServerCompaction.encode(result.envelope),
                            createdBy: "tinytitan"))
                    self.writeJSON(
                        contextBox.value, status: .ok,
                        object: ResponsesAPIBuilder.compactResource(
                            id: resourceID, created: created, output: output,
                            usage: result.usage))
                } catch {
                    self.handleAsyncFailure(
                        error, context: contextBox.value, id: resourceID,
                        phase: "compact", stream: false, outbox: nil,
                        streamState: StreamState(), surface: .responses)
                }
            }
        } catch let error as ServerRequestError {
            writeError(
                context, status: error == .unknownModel ? .notFound : .badRequest,
                error.envelope)
        } catch {
            writeError(
                context, status: .badRequest,
                RequestDecodingFailure.serverError(error, body: body).envelope)
        }
    }

    /// What one compaction produced.
    struct CompactionOutcome: Sendable {
        let envelope: CompactionEnvelope
        let usage: OpenAIUsage
        let noteTokens: Int
    }

    /// Summarise, measure, and degrade honestly.
    ///
    /// Pass one runs the handover instruction. The note is then measured with
    /// this server's own tokenizer — the one thing a client cannot do as cheaply —
    /// and a note over budget is **compressed by a second pass rather than
    /// truncated**: truncation drops the end of the session, which is exactly
    /// what a continuation needs. A note that turns out to be its own instruction
    /// read back is retried with a plainer one, and a pass that still produces
    /// nothing usable falls back to the newest text trimmed to the budget, so the
    /// caller always gets a window: a compact endpoint that fails the turn is
    /// worse than one that compacts badly and says which way.
    func compacted(
        conversation: [OpenAIChatMessage], target: ServedModel,
        budget: Int
    ) async throws -> CompactionOutcome {
        let transcript = ServerCompaction.transcript(conversation)
        let created = Int(Date().timeIntervalSince1970)
        let structured = ServerCompaction.instruction(limit: budget)
        let first = try await summarise(
            instruction: structured, body: transcript,
            target: target, budget: budget)
        var usage = first.usage
        var note = ServerCompaction.strippingInstructionEcho(
            first.content,
            instruction: structured)
        var mode = CompactionMode.model

        // A pass that echoed its instruction, or looped, has not summarised
        // anything. Ask again without a menu to copy; at temperature 0 a second
        // identical attempt would only repeat itself, and a repetition loop is
        // the failure a small model actually produces.
        func unusable(_ text: String) -> Bool {
            text.isEmpty || ServerCompaction.isDegenerate(text)
        }
        if unusable(note) {
            let plain = ServerCompaction.plainInstruction(limit: budget)
            let retry = try await summarise(
                instruction: plain, body: transcript,
                target: target, budget: budget)
            usage = usage.adding(retry.usage)
            note = ServerCompaction.strippingInstructionEcho(retry.content, instruction: plain)
        }

        if unusable(note) {
            mode = .extractive
            note = ServerCompaction.extractiveSummary(
                transcript: transcript,
                characterBudget: budget * 3)
        } else if try await tokenCount(of: note, target: target) > budget {
            let instruction = ServerCompaction.compressionInstruction(limit: budget)
            let second = try await summarise(
                instruction: instruction, body: note,
                target: target, budget: budget)
            usage = usage.adding(second.usage)
            let compressed = ServerCompaction.strippingInstructionEcho(
                second.content,
                instruction: instruction)
            if !compressed.isEmpty {
                note = compressed
                mode = .compressed
            }
            // One compression pass, then trim: a model that missed the budget
            // twice will miss it a third time, and the caller is waiting.
            if try await tokenCount(of: note, target: target) > budget {
                note = ServerCompaction.extractiveSummary(
                    transcript: note,
                    characterBudget: budget * 3)
                mode = .extractive
            }
        }
        let envelope = CompactionEnvelope(
            model: target.id, createdAt: created,
            mode: mode, summary: note)
        return CompactionOutcome(
            envelope: envelope, usage: usage,
            noteTokens: try await tokenCount(of: note, target: target))
    }

    /// One summariser pass through the normal queue, so a compaction waits its
    /// turn and is shed under load like any other generation.
    private func summarise(
        instruction: String, body: String, target: ServedModel,
        budget: Int
    ) async throws -> ServerCompletion {
        let request = OpenAIChatRequest(
            model: target.id,
            messages: [
                OpenAIChatMessage(
                    role: "system", content: .text(instruction),
                    toolCalls: nil, toolCallID: nil, name: nil),
                OpenAIChatMessage(
                    role: "user", content: .text(body),
                    toolCalls: nil, toolCallID: nil, name: nil),
            ],
            stream: false,
            // Greedy: a compaction should be the same note for the same session,
            // and the caller may re-run one after a failure.
            temperature: 0,
            maxCompletionTokens: budget,
            // The summariser must not think. A model that reasons inside its own
            // output cap spends the cap on thoughts and returns an empty note —
            // the same reason `plugins/dsh-tinytitan` forces this for the
            // harness's auxiliary calls.
            reasoningEffort: "off")
        let validated = try validate(request, for: target)
        return try await coordinator.run {
            try Task.checkCancellation()
            return try await self.backend.generate(validated) { _ in }
        }
    }

    /// How many tokens a note occupies once rendered, or 0 when the backend
    /// cannot count (the budget check is a quality guard, not correctness).
    private func tokenCount(of text: String, target: ServedModel) async throws -> Int {
        guard let counting = backend as? any PromptTokenCounting else { return 0 }
        let request = OpenAIChatRequest(
            model: target.id,
            messages: [
                OpenAIChatMessage(
                    role: "user", content: .text(text),
                    toolCalls: nil, toolCallID: nil, name: nil)
            ],
            stream: false,
            maxCompletionTokens: 1)
        return try await counting.countPromptTokens(try validate(request, for: target))
    }
}
