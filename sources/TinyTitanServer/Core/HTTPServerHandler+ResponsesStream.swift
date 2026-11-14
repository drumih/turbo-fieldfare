//
//  HTTPServerHandler+ResponsesStream.swift
//  TinyTitanServer
//
//  The Responses API streaming half: the stream opener, the per-event
//  enqueueing and the terminal flush.
//
//  Split out of `HTTPServerHandler+Responses.swift` (2026-09-28) under the
//  500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitan
import TinyTitanLib
import TinyTitanCore

extension ServerHTTPHandler {

    func beginResponsesStream(
        _ context: ChannelHandlerContext,
        id: String,
        created: Int,
        echo: ResponsesAPIEcho,
        itemState: ResponsesStreamState
    ) -> EventLoopFuture<Void> {
        let response = ResponsesAPIBuilder.responseObject(
            id: id, created: created, model: responseModelID, status: "in_progress",
            output: [], usage: nil, echo: echo)
        var frames = Data()
        for name in ["response.created", "response.in_progress"] {
            let event = ResponsesAPIBuilder.event(
                name, sequence: itemState.nextSequence(),
                ["response": response])
            if let frame = Self.eventFrame(name: name, object: event) {
                frames.append(frame)
            }
        }
        return writeStreamHead(context, initialFrames: frames, extraHeaders: [])
    }

    /// Enqueue one Responses-API event, numbering it in stream order.
    func responsesEvent(
        _ type: String,
        _ fields: [String: Any],
        itemState: ResponsesStreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        let object = ResponsesAPIBuilder.event(type, sequence: itemState.nextSequence(), fields)
        guard let frame = Self.eventFrame(name: type, object: object) else {
            failStream(
                outbox: outbox, context: context,
                message: "stream response could not be encoded",
                code: "internal_error", surface: .responses)
            return
        }
        guard outbox.enqueue(frame) else {
            failStream(
                outbox: outbox, context: context,
                message: "stream backpressure limit exceeded; client is too slow",
                code: "stream_overflow", surface: .responses)
            return
        }
    }

    func enqueueResponsesEvent(
        _ event: ServerInferenceEvent,
        id: String,
        echo: ResponsesAPIEcho,
        itemState: ResponsesStreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        switch event {
        case .promptProcessed:
            // The Responses stream already carried its input-token count in
            // `response.created`; there is no event for this one.
            break
        case .promptProgress:
            // Reading progress is a chat-completions extension; this surface
            // has no event for it and the flag is not parsed here.
            break
        case .content(let text):
            enqueueResponsesContentDelta(
                id: id, text: text, itemState: itemState,
                outbox: outbox, context: context)
        case .reasoning(let text):
            enqueueResponsesReasoningDelta(
                id: id, text: text, itemState: itemState,
                outbox: outbox, context: context)
        case .toolCall(let call):
            enqueueResponsesToolCall(
                id: id, call: call, itemState: itemState,
                namespace: echo.namespaces[call.name],
                outbox: outbox, context: context)
        }
    }

    /// Thoughts stream as a reasoning item's summary text, ahead of the
    /// message, the way the API's own reasoning models order them.
    func enqueueResponsesReasoningDelta(
        id: String,
        text: String,
        itemState: ResponsesStreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        let (index, ordinal, first) = itemState.appendReasoning(text)
        let itemID = ResponsesAPIBuilder.reasoningItemID(responseID: id, index: ordinal)
        if first {
            responsesEvent(
                "response.output_item.added",
                [
                    "output_index": index,
                    "item": ["id": itemID, "type": "reasoning", "summary": []],
                ],
                itemState: itemState, outbox: outbox, context: context)
            responsesEvent(
                "response.reasoning_summary_part.added",
                [
                    "item_id": itemID, "output_index": index, "summary_index": 0,
                    "part": ResponsesAPIBuilder.summaryTextPart(""),
                ],
                itemState: itemState, outbox: outbox, context: context)
        }
        responsesEvent(
            "response.reasoning_summary_text.delta",
            [
                "item_id": itemID, "output_index": index, "summary_index": 0,
                "delta": text,
            ],
            itemState: itemState, outbox: outbox, context: context)
    }

    /// Finish the open reasoning item, if any: the model has moved on to its
    /// answer or a call, and a client renders the thought as complete.
    func closeResponsesReasoning(
        id: String,
        itemState: ResponsesStreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        guard let (index, ordinal, text) = itemState.closeReasoning() else { return }
        let itemID = ResponsesAPIBuilder.reasoningItemID(responseID: id, index: ordinal)
        responsesEvent(
            "response.reasoning_summary_text.done",
            [
                "item_id": itemID, "output_index": index, "summary_index": 0,
                "text": text,
            ],
            itemState: itemState, outbox: outbox, context: context)
        responsesEvent(
            "response.reasoning_summary_part.done",
            [
                "item_id": itemID, "output_index": index, "summary_index": 0,
                "part": ResponsesAPIBuilder.summaryTextPart(text),
            ],
            itemState: itemState, outbox: outbox, context: context)
        responsesEvent(
            "response.output_item.done",
            [
                "output_index": index,
                "item": ResponsesAPIBuilder.reasoningItem(id: itemID, text: text),
            ],
            itemState: itemState, outbox: outbox, context: context)
    }

    func enqueueResponsesContentDelta(
        id: String,
        text: String,
        itemState: ResponsesStreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        closeResponsesReasoning(id: id, itemState: itemState, outbox: outbox, context: context)
        let itemID = ResponsesAPIBuilder.messageItemID(responseID: id)
        let (index, first) = itemState.announceMessage()
        if first {
            responsesEvent(
                "response.output_item.added",
                [
                    "output_index": index,
                    "item": [
                        "id": itemID, "type": "message", "role": "assistant",
                        "status": "in_progress", "content": [],
                    ],
                ],
                itemState: itemState, outbox: outbox, context: context)
            responsesEvent(
                "response.content_part.added",
                [
                    "item_id": itemID, "output_index": index, "content_index": 0,
                    "part": ResponsesAPIBuilder.outputTextPart(""),
                ],
                itemState: itemState, outbox: outbox, context: context)
        }
        responsesEvent(
            "response.output_text.delta",
            [
                "item_id": itemID, "output_index": index, "content_index": 0,
                "delta": text, "logprobs": [],
            ],
            itemState: itemState, outbox: outbox, context: context)
    }

    /// A tool call arrives complete from the decoder, so its whole item life
    /// cycle is streamed at once: added, argument deltas, done, item done.
    func enqueueResponsesToolCall(
        id: String,
        call: ParsedToolCall,
        itemState: ResponsesStreamState,
        namespace: String?,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        closeResponsesReasoning(id: id, itemState: itemState, outbox: outbox, context: context)
        let (index, ordinal) = itemState.allocateCall()
        let itemID = ResponsesAPIBuilder.functionCallItemID(responseID: id, index: ordinal)
        responsesEvent(
            "response.output_item.added",
            [
                "output_index": index,
                "item": ResponsesAPIBuilder.functionCallItem(
                    id: itemID, name: call.name, arguments: "", callID: call.id,
                    status: "in_progress", namespace: namespace),
            ],
            itemState: itemState, outbox: outbox, context: context)
        for fragment in utf8Fragments(call.argumentsJSON, maximumBytes: 1024) {
            responsesEvent(
                "response.function_call_arguments.delta",
                ["item_id": itemID, "output_index": index, "delta": fragment],
                itemState: itemState, outbox: outbox, context: context)
        }
        responsesEvent(
            "response.function_call_arguments.done",
            [
                "item_id": itemID, "output_index": index, "name": call.name,
                "arguments": call.argumentsJSON,
            ],
            itemState: itemState, outbox: outbox, context: context)
        responsesEvent(
            "response.output_item.done",
            [
                "output_index": index,
                "item": ResponsesAPIBuilder.functionCallItem(
                    id: itemID, name: call.name, arguments: call.argumentsJSON,
                    callID: call.id, status: "completed", namespace: namespace),
            ],
            itemState: itemState, outbox: outbox, context: context)
    }

    /// Close the message item, then end the stream with the terminal object:
    /// `response.completed`, or `response.incomplete` when the output cap
    /// cut the generation. Returns the object so it can be stored.
    func finishResponsesStream(
        _ context: ChannelHandlerContext,
        id: String,
        created: Int,
        echo: ResponsesAPIEcho,
        completion: ServerCompletion,
        itemState: ResponsesStreamState,
        outbox: SSEOutbox
    ) -> [String: Any] {
        let itemID = ResponsesAPIBuilder.messageItemID(responseID: id)
        // A turn cut off mid-thought still finishes its reasoning item.
        closeResponsesReasoning(id: id, itemState: itemState, outbox: outbox, context: context)
        // A turn with no text and no calls still has one (empty) message
        // item, as the API's own output does.
        if itemState.messageIndex == nil, completion.toolCalls.isEmpty {
            enqueueResponsesContentDelta(
                id: id, text: "", itemState: itemState,
                outbox: outbox, context: context)
        }
        if let index = itemState.messageIndex {
            responsesEvent(
                "response.output_text.done",
                [
                    "item_id": itemID, "output_index": index, "content_index": 0,
                    "text": completion.content, "logprobs": [],
                ],
                itemState: itemState, outbox: outbox, context: context)
            responsesEvent(
                "response.content_part.done",
                [
                    "item_id": itemID, "output_index": index, "content_index": 0,
                    "part": ResponsesAPIBuilder.outputTextPart(completion.content),
                ],
                itemState: itemState, outbox: outbox, context: context)
            responsesEvent(
                "response.output_item.done",
                [
                    "output_index": index,
                    "item": ResponsesAPIBuilder.messageItem(
                        id: itemID, role: "assistant",
                        text: completion.content, status: "completed"),
                ],
                itemState: itemState, outbox: outbox, context: context)
        }
        let final = finalResponsesObject(
            id: id, created: created, echo: echo,
            completion: completion, itemState: itemState)
        let name =
            (final["status"] as? String) == "incomplete"
            ? "response.incomplete" : "response.completed"
        responsesEvent(
            name, ["response": final],
            itemState: itemState, outbox: outbox, context: context)
        // The Responses API has no [DONE] terminator; the final event is it.
        outbox.enqueueTerminal([], closeWhenDrained: false)
        return final
    }
}
