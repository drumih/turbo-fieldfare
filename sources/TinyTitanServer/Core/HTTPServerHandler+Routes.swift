//
//  HTTPServerHandler+Routes.swift
//  TinyTitanServer
//
//  Request dispatch and the model-management endpoints, which every API surface shares.
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitanCore
import TinyTitanLib

extension ServerHTTPHandler {
    func route(
        head: HTTPRequestHead,
        body: ByteBuffer,
        context: ChannelHandlerContext
    ) {
        // S27: HTTP/1.1 requires a Host header; reject requests without one.
        if head.version == .http1_1, head.headers.first(name: "host") == nil {
            writeError(
                context, status: .badRequest,
                OpenAIErrorEnvelope(
                    message: "missing Host header",
                    code: "missing_host"))
            return
        }
        let path =
            head.uri.split(
                separator: "?", maxSplits: 1,
                omittingEmptySubsequences: false
            ).first.map(String.init) ?? head.uri
        // A client that sends anthropic-version is speaking the Messages API;
        // the shared paths (/v1/models, 404s) answer in its shape.
        let anthropic =
            head.headers.first(name: "anthropic-version") != nil
            || path.hasPrefix("/v1/messages")
        let segments = path.split(separator: "/").map(String.init)
        let jsonBody =
            head.headers.first(name: "content-type")?
            .lowercased().hasPrefix("application/json") == true
        // AUD-147's gate for a route that reads *no* body. `jsonBody` cannot be
        // copied here: `curl -X POST` sends no content-type, and that is how this
        // route is documented and tested -- the absent-header path is the CLI's
        // contract, which is the AUD-132 lesson. What still has to be refused is a
        // browser form POST, and a form always *does* name a content-type, so
        // "absent" is the CLI and "present but not JSON" is the form.
        let notAForm =
            head.headers.first(name: "content-type").map {
                $0.lowercased().hasPrefix("application/json")
            } ?? true
        // S28: only the two read routes answer HEAD; see `refuseUnsupportedHEAD`.
        if refuseUnsupportedHEAD(head, path: path, context: context) { return }
        switch (head.method, path) {
        case (.GET, "/health"):
            writeJSON(context, status: .ok, object: ["status": "ok"])
        case (.GET, "/v1/models"):
            if let router {
                writeModelList(router.servedModels, anthropic: anthropic, context: context)
                return
            }
            // Advertise the base model plus the "<model>-fast" alias, which
            // serves the same weights with the CLI-strip heuristic enabled.
            if anthropic {
                writeJSON(
                    context, status: .ok,
                    object: AnthropicBuilder.modelList(ids: [modelID, modelID + "-fast"]),
                    surface: .anthropic)
                return
            }
            let response = OpenAIModelList(
                object: "list",
                data: [
                    .init(
                        id: modelID,
                        object: "model",
                        created: nil,
                        ownedBy: "tinytitan"),
                    .init(
                        id: modelID + "-fast",
                        object: "model",
                        created: nil,
                        ownedBy: "tinytitan"),
                ])
            writeCodable(context, status: .ok, response)
        case (.HEAD, "/health"), (.HEAD, "/v1/models"):
            // S28: HEAD is answered with headers only.
            writeHeadOnly(context, status: .ok)
        case (.POST, "/v1/chat/completions"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .chat)
                return
            }
            handleCompletion(
                body: body,
                context: context,
                // From the *local* head: `self.head` is cleared when `.end`
                // arrives, before this runs. Reading it here returned nil
                // every time, which is a fix that compiles, ships and does
                // nothing -- caught only because the isolation scenario
                // refuses to run until it sees the header take effect.
                workspace: WorkspaceHeader.value(in: head))
        case (.POST, "/v1/responses"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .responses)
                return
            }
            handleResponses(
                body: body,
                context: context,
                workspace: WorkspaceHeader.value(in: head))
        case (.POST, "/v1/responses/compact"):
            // The spec's compaction endpoint: a conversation in, a compacted
            // window out. No model state and no stored response, so it is a
            // plain JSON route rather than anything the response store sees.
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .responses)
                return
            }
            handleCompact(body: body, context: context)
        case (.POST, "/v1/messages"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .anthropic)
                return
            }
            handleMessages(
                body: body, context: context,
                workspace: WorkspaceHeader.value(in: head))
        case (.POST, "/v1/messages/count_tokens"):
            guard jsonBody else {
                writeUnsupportedMediaType(context, surface: .anthropic)
                return
            }
            handleCountTokens(body: body, context: context)
        case (.POST, "/v1/models/unload"):
            guard notAForm else {
                writeUnsupportedMediaType(context, surface: .chat)
                return
            }
            handleUnload(context: context)
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"), (_, "/v1/responses"),
            (_, "/v1/responses/compact"),
            (_, "/v1/models/unload"), (_, "/v1/messages"), (_, "/v1/messages/count_tokens"):
            writeRequestError(
                context,
                .invalid(
                    message: "method not allowed", param: nil,
                    code: "method_not_allowed"),
                status: .methodNotAllowed,
                surface: anthropic ? .anthropic : .chat)
        default:
            routeBySegment(
                segments: segments, method: head.method,
                anthropic: anthropic, context: context)
        }
    }

    /// Paths that are not in the table above: a model id, a stored response, or
    /// nothing. Kept out of the switch so the switch stays a table.
    func routeBySegment(
        segments: [String], method: HTTPMethod, anthropic: Bool,
        context: ChannelHandlerContext
    ) {
        if segments.count == 3, segments[0] == "v1", segments[1] == "models" {
            handleModel(id: segments[2], method: method, anthropic: anthropic, context: context)
        } else if segments.count >= 3, segments[0] == "v1", segments[1] == "responses" {
            handleStoredResponse(
                segments: Array(segments.dropFirst(2)),
                method: method, context: context)
        } else {
            writeRequestError(
                context, .notFound(message: "route not found", param: nil),
                status: .notFound,
                surface: anthropic ? .anthropic : .chat)
        }
    }

    /// S28: a HEAD response must never carry a body, and only the two read routes
    /// support HEAD. Everything else — a POST route, a per-id route, or nothing
    /// at all — would otherwise answer with the body its ordinary path writes,
    /// which on a keep-alive connection is read as the *next* response's head by
    /// a simple client. Every other method error on a known route is already a
    /// 405 (`method_not_allowed`), so answering HEAD the same way keeps this one
    /// rule rather than a second route table; the cost is that a HEAD for an
    /// unknown path says 405 where a GET says 404, and both are head-only.
    ///
    /// - Returns: true when the request has been answered and routing should stop.
    func refuseUnsupportedHEAD(
        _ head: HTTPRequestHead, path: String,
        context: ChannelHandlerContext
    ) -> Bool {
        guard head.method == .HEAD, path != "/health", path != "/v1/models" else { return false }
        writeHeadOnly(context, status: .methodNotAllowed)
        return true
    }

    func writeUnsupportedMediaType(_ context: ChannelHandlerContext, surface: APISurface) {
        writeRequestError(
            context,
            .invalid(
                message: "content-type must be application/json",
                param: nil, code: "unsupported_media_type"),
            status: .unsupportedMediaType, surface: surface)
    }

    /// The model a request names. Resolved before validation, so omitted
    /// sampling and the max_tokens bound come from that model rather than
    /// whichever one is resident. Without a router this is the one model,
    /// answered from the backend as before, and the validator still refuses
    /// any other name.
    func servedModel(named name: String) throws -> ServedModel {
        guard let router else {
            return ServedModel(
                id: modelID, displayName: modelID,
                maximumContext: backend.maximumContext,
                sampling: backend.samplingDefaults,
                reasoningProfile: reasoningProfile)
        }
        guard let model = router.servedModel(named: name) else {
            throw ServerRequestError.unknownModel
        }
        return model
    }

    /// Validates against `target` and binds the request to it, so the router
    /// loads the model that was validated and the response names it.
    func validate(
        _ request: OpenAIChatRequest,
        for target: ServedModel
    ) throws -> ValidatedChatRequest {
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: target.id, maxContext: target.maximumContext,
            reasoningProfile: target.reasoningProfile,
            sampling: target.sampling
        )
        .withModel(target.id)
        // Best-effort reasoning: say what was applied when a client asked for
        // a level this model cannot render. Never an error — see
        // `ReasoningFallback`.
        if !validated.reasoningNotes.isEmpty {
            ServerLog.reasoningFallback(id: target.id, notes: validated.reasoningNotes)
        }
        responseModelID = target.id
        return validated
    }

    /// Every catalog model in the shape the client speaks. The "-fast"
    /// aliases are accepted but not listed: doubling a catalog into twice as
    /// many menu entries helps nobody choose between models.
    func writeModelList(
        _ models: [ServedModel], anthropic: Bool,
        context: ChannelHandlerContext
    ) {
        if anthropic {
            writeJSON(
                context, status: .ok,
                object: AnthropicBuilder.modelList(
                    models: models.map { (id: $0.id, displayName: $0.displayName) }),
                surface: .anthropic)
            return
        }
        writeCodable(
            context, status: .ok,
            OpenAIModelList(
                object: "list",
                data: models.map {
                    .init(id: $0.id, object: "model", created: nil, ownedBy: "tinytitan")
                }))
    }

    /// `GET /v1/models/{id}` in either shape.
    func handleModel(
        id: String, method: HTTPMethod, anthropic: Bool,
        context: ChannelHandlerContext
    ) {
        let surface: APISurface = anthropic ? .anthropic : .chat
        guard method == .GET else {
            writeRequestError(
                context,
                .invalid(
                    message: "method not allowed", param: nil,
                    code: "method_not_allowed"),
                status: .methodNotAllowed, surface: surface)
            return
        }
        let displayName: String
        if let router {
            guard let model = router.servedModel(named: id) else {
                writeRequestError(context, .unknownModel, status: .notFound, surface: surface)
                return
            }
            displayName = model.displayName
        } else {
            guard id == modelID || id == modelID + "-fast" else {
                writeRequestError(context, .unknownModel, status: .notFound, surface: surface)
                return
            }
            displayName = id
        }
        if anthropic {
            writeJSON(
                context, status: .ok,
                object: AnthropicBuilder.modelObject(id: id, displayName: displayName),
                surface: .anthropic)
        } else {
            writeCodable(
                context, status: .ok,
                OpenAIModelList.Model(id: id, object: "model", created: nil, ownedBy: "tinytitan"))
        }
    }

    /// `GET|DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`,
    /// `GET /v1/responses/{id}/input_items`: the stored side of the
    /// Responses API. Nothing here runs the model.
    func handleStoredResponse(
        segments: [String], method: HTTPMethod,
        context: ChannelHandlerContext
    ) {
        let id = segments[0]
        let notFound = ServerRequestError.notFound(
            message: "Response with id '\(id)' not found.", param: "id")
        switch (method, segments.count, segments.count > 1 ? segments[1] : "") {
        case (.GET, 1, _):
            guard let entry = responseStore.get(id) else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses)
                return
            }
            writeData(context, status: .ok, data: entry.responseJSON)
        case (.DELETE, 1, _):
            guard responseStore.delete(id) else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses)
                return
            }
            writeJSON(
                context, status: .ok, object: ["id": id, "object": "response", "deleted": true])
        case (.POST, 2, "cancel"):
            guard responseStore.get(id) != nil else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses)
                return
            }
            // Every response this server produces is foreground and already
            // finished by the time it has an id to cancel.
            writeRequestError(
                context,
                .invalid(
                    message: "Only responses created with background=true can be cancelled.",
                    param: "id", code: "invalid_request"), status: .badRequest, surface: .responses)
        case (.GET, 2, "input_items"):
            guard let entry = responseStore.get(id) else {
                writeRequestError(context, notFound, status: .notFound, surface: .responses)
                return
            }
            do {
                writeData(
                    context, status: .ok,
                    data: try ResponsesAPIBuilder.inputItemsList(entry.inputItems))
            } catch {
                writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            }
        case (_, 1, _), (_, 2, "cancel"), (_, 2, "input_items"):
            writeRequestError(
                context,
                .invalid(
                    message: "method not allowed", param: nil,
                    code: "method_not_allowed"),
                status: .methodNotAllowed, surface: .responses)
        default:
            writeRequestError(
                context, .notFound(message: "route not found", param: nil),
                status: .notFound, surface: .responses)
        }
    }

    /// Control endpoint: release the model's memory on demand. With residency
    /// managed (--lazy-load / --idle-unload-seconds) this waits for in-flight
    /// requests to drain, then unloads; with a plain session it is a no-op.
    func handleUnload(context: ChannelHandlerContext) {
        let contextBox = SendableContext(context)
        activeTask = childChannels.startTask {
            // Only a residency-managing backend has anything to release; a
            // plain session reports false without the inference protocol
            // needing to know residency exists.
            let released: Bool
            if let managing = self.backend as? any ResidencyManaging {
                released = await managing.unload()
            } else {
                released = false
            }
            self.writeJSON(
                contextBox.value, status: .ok,
                object: ["status": "ok", "unloaded": released])
        }
    }

}
