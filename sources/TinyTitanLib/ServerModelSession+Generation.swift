// The request orchestrator: cache resolution, decode and publish.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import TinyTitan

extension ServerModelSession {
    /// lint:allow-long the request orchestrator: prompt preparation, cache
    /// resolution, decode, publish, and the completion. Each of those is its
    /// own method; what remains is the sequence plus a nested failure builder
    /// that closes over eight locals -- hoisting it would mean an
    /// eight-parameter signature for a twenty-line body.
    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        // One slot per in-flight generation. The coordinator bounds concurrency
        // to this session's width, so a slot is normally free immediately; the
        // wait is a safety net if the two ever disagree.
        let slot = try await acquireSlot()
        defer { releaseSlot(slot) }
        // The routing history this request produced follows the install to the
        // next start (Project Maya's expert profile). Best effort, after the
        // response is complete; `TINYTITAN_EXPERT_PROFILE=0` disables it.
        defer { model.saveExpertUsageProfile() }
        // §5bk: an opt-in second command queue that hands the device a
        // micro-dispatch through the gaps between the engine's own launches.
        // Inert unless the operator names TINYTITAN_GPU_KEEPWARM_US, and a value
        // outside the bound refuses the request rather than running a control
        // that reads as the arm.
        let keepWarm = try GPUKeepWarmRunner.startIfEnabled(context: context)
        defer { keepWarm?.stop() }
        // Stage-split measurement (TINYTITAN_RUNNER_STATS): snapshot the runner's
        // lifetime counters so the footer can report this request's delta.
        let runnerSnapshot = RunnerCounterSnapshot(
            cb1: runner.totalCb1Nanos,
            io: runner.totalIoNanos,
            cb2: runner.totalCb2Nanos,
            head: runner.totalHeadNanos,
            headFused: runner.totalHeadFusedNanos,
            rdadvise: runner.totalRDAdviseNanos,
            rdadviseCalls: runner.totalRDAdviseCalls,
            rdadviseBytes: runner.totalRDAdviseBytes,
            wait: runner.totalWaitNanos,
            body: runner.totalBodyNanos,
            prefetchIssued: runner.totalPrefetchIssued,
            prefetchAdopted: runner.totalPrefetchAdopted,
            preamble: runner.totalPreambleNanos,
            preambleRelease: runner.totalPreambleReleaseNanos,
            preamblePin: runner.totalPreamblePinNanos,
            preambleReserve: runner.totalPreambleReserveNanos,
            embed: runner.totalEmbedNanos,
            gather: runner.totalGatherNanos,
            loopSample: runner.totalLoopSampleNanos,
            loopProgress: runner.totalLoopProgressNanos,
            loopOther: runner.totalLoopOtherNanos,
            missIo: runner.totalMissIoNanos,
            exposedIo: runner.totalExposedIoNanos,
            hitFixupLayers: runner.totalHitFixupLayers,
            routerReadback: runner.totalRouterReadbackNanos,
            cachePlan: runner.totalCachePlanNanos,
            ioQueue: runner.totalIOQueueNanos,
            ioCompletionToFixup: runner.totalIOCompletionToFixupSubmitNanos,
            ioHostWaits: runner.totalExpertIOHostWaits,
            ioHostWaitsAvoided: runner.totalExpertIOHostWaitsAvoided,
            gpuClassifiedHits: runner.totalGPUClassifiedHits,
            gpuClassifiedMisses: runner.totalGPUClassifiedMisses,
            gpuAllHitLayers: runner.totalGPUResidencyAllHitLayers,
            expertStreaming: runner.expertStreamingStatistics())
        runner.resetKernelGPUTimings()
        var completed = false
        defer {
            if !completed {
                if promptCacheMode == .singlePrefix {
                    promptCache.invalidate()
                }
                activePromptCacheEntryID = nil
                // One sequence failed; only its slot's KV/GDN is suspect. A
                // whole-runner reset would wipe the other slots' live state.
                if slots > 1 {
                    runner.reset(slot: slot)
                } else {
                    runner.reset()
                }
                mtpDecoder?.reset()
            }
        }
        // B6: an engine-internal generation is never watched. Everything
        // else gets the configured set, which is inert unless the operator
        // turned watchdogs on.
        let watchdogs =
            request.isEngineInternal
            ? WatchdogSupervisor.inert
            : WatchdogSupervisor(configuration: WatchdogConfiguration.shared)
        let watchdogTicker = watchdogs.startTicker()
        defer { watchdogTicker?.cancel() }
        // B2: a tool loop shows up in the incoming message history, not in
        // the output stream, so it is judged before anything is generated.
        if !request.isEngineInternal {
            watchdogs.record(
                pingPong: PingPongWatchdog.inspect(
                    request.messages, configuration: watchdogs.configuration))
        }
        // There is no safe intervention from here -- withholding the tools
        // leaves a tool-templated prompt with a decoder that allows none,
        // which fails the request outright. `WatchdogKind.canAct` carries the
        // reasoning; ping-pong observes, and the client, which owns the loop,
        // decides.
        // A request that names a different thinking mode or effort than the
        // session loaded at is a mid-session switch: resolve the tokenizer for
        // it here, so the render, the special tokens and the decoder all
        // follow the switch. `nil` (the common case) reuses the session's.
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        // A raw completion carries its own token ids: it is never re-rendered
        // through the template, never matched against the prompt cache, and
        // never interpreted by the think/tool decoder. A chat request takes the
        // path it always did. Both share everything below -- prefill, the
        // decode loop, diagnostics and the completion -- so there is one
        // orchestrator, not two.
        let isRaw = request.renderedPromptIDs != nil
        let promptIDs: [Int32]
        let cacheRequest: ValidatedChatRequest
        let needsToolTemplate: Bool
        if let renderedPromptIDs = request.renderedPromptIDs {
            promptIDs = renderedPromptIDs
            cacheRequest = request
            needsToolTemplate = false
        } else {
            let prepared = try preparePrompt(request, renderTokenizer: renderTokenizer)
            promptIDs = prepared.promptIDs
            cacheRequest = prepared.cacheRequest
            needsToolTemplate = prepared.needsToolTemplate
        }

        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if isRaw {
            guard promptIDs.count < maxContext else {
                throw ServerRequestError.invalid(
                    message: "prompt exceeds the configured context",
                    param: "messages",
                    code: "context_length_exceeded")
            }
            effectivePromptIDs = promptIDs
            completionStart = .reset
        } else {
            let resolved = try await resolveCacheStart(
                cacheRequest: cacheRequest,
                promptIDs: promptIDs,
                requestedReasoning: request.reasoning)
            effectivePromptIDs = resolved.effectivePromptIDs
            completionStart = resolved.start
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        // The chat path clears the loop's own stop matcher because the
        // assistant output applies the same strings after the think/tool
        // decoder. A raw completion has no decoder, so the loop's matcher is
        // the one that has to run -- which is exactly the pre-facade CLI's
        // behaviour.
        if !isRaw { config.stopStrings = [] }
        // Structured output is a per-request grammar: a fresh constraint per
        // request (its state is the document parsed so far), over a table that
        // is built once per model.
        if let node = request.jsonSchema {
            config.constraint = JSONConstraint(
                table: structuredOutputTable(), node: node,
                vocab: model.config.vocabSize)
        }

        // The full render, not the cache-trimmed suffix, decides whether the
        // generation prompt left a thought open; both end in the same
        // generation prompt, but only the render is always whole. The decoder
        // runs for every generation, so a thought the *model* opens while the
        // switch is off is still split out of the answer rather than streamed
        // as it.
        //
        // A raw completion gets no decoder: the caller's text was not rendered
        // with a thought block, and the CLI's `--prompt` mode has always shown
        // whatever the model wrote, `<think>` markers included.
        let decoder: StructuredAssistantDecoder? =
            isRaw
            ? nil
            : StructuredAssistantDecoder.forGeneration(
                tokenizer: renderTokenizer,
                promptIDs: promptIDs,
                allowedTools: needsToolTemplate ? Set(request.tools.map(\.name)) : nil)
        // The stall clock starts at the first visible token, so a long
        // thought before the answer cannot trip it. Reasoning is watched for
        // loops alone, in a window of its own.
        //
        // A raw completion's output applies no stop matcher of its own: the
        // loop's owns the strings, and a second identical matcher over already
        // filtered text would only hold the same text a second time.
        let state = GenerationDecodeState(
            decoder: decoder,
            output: AssistantOutput(
                stops: isRaw ? [] : request.generationConfig.stopStrings,
                onEvent: onEvent,
                observeVisible: { watchdogs.observe($0) },
                observeReasoning: { watchdogs.observeReasoning($0) }))

        // MTP drafts several tokens ahead of the sampler and never consults a
        // grammar, so a constrained request takes the ordinary decode path
        // (`runRawCompletion` refuses the MTP producer outright).
        let activeProducer: any LogitProducer =
            if config.isPureGreedy,
                config.constraint == nil,
                let mtpDecoder,
                promptIDs.count + config.maxNewTokens
                    <= mtpDecoder.draftMaxContext
            {
                mtpDecoder
            } else {
                batchedProducer ?? runner
            }
        let activeStart: RawCompletionStart =
            activeProducer is StreamingMTPDecoder
            ? .reset : completionStart
        let activePromptIDs =
            activeProducer is StreamingMTPDecoder
            ? promptIDs : effectivePromptIDs
        // `@Sendable`: `runRawCompletion` is @concurrent, so a progress closure
        // that is still actor-isolated cannot be sent into it (Swift 6.4).
        // Everything these touch lives in the Sendable box above.
        let publish: @Sendable ([StructuredAssistantEvent], Bool) -> Void = { events, isToken in
            state.output.publish(events, isToken: isToken)
            if state.output.isStopped { state.shouldStop = true }
        }
        // The prompt's accounting, published once: at the end of prefill, or
        // immediately when there is nothing left to prefill — a fully cached
        // prompt produces no progress callback at all, and a caller still needs
        // to be told the prompt was read. `tokens` is the prompt's length and
        // `cachedTokens` the part of it the cache already held.
        let cachedPromptTokenCount =
            if case .resume(let cached) = activeStart { cached } else { 0 }
        let publishPromptProcessed: @Sendable (Int) -> Void = { total in
            guard !state.publishedPromptProgress else { return }
            state.publishedPromptProgress = true
            onEvent(
                .promptProcessed(tokens: total, cachedTokens: cachedPromptTokenCount))
        }
        if activePromptIDs.isEmpty {
            publishPromptProcessed(promptIDs.count)
        }
        let interiorCapture = interiorCheckpointCapture(
            promptIDs: activePromptIDs,
            messages: cacheRequest.messages,
            imStartID: renderTokenizer.imStartID,
            enabled: !isRaw && activeProducer is RealForwardRunner)
        // `renderTokenizer` is the one this request's reasoning resolves to, and
        // it is already what rendered the prompt and what the assistant decoder
        // was built with. `tokenizer` is the session's -- the level the model was
        // *loaded* at -- so a mid-session reasoning switch had the decoder on one
        // tokenizer and the detokenizer plus the stop-id check on another. Their
        // stop ids and special tokens coincide across a loaded folder today,
        // which is why this looked harmless; it is not guaranteed, and the
        // generation loop is the wrong place to rely on it.
        let result = try await runRawCompletion(
            producer: activeProducer,
            tokenizer: renderTokenizer,
            promptIds: activePromptIDs,
            config: config,
            context: context,
            scratch: scratches[slot],
            prefillConfig: prefillConfig,
            start: activeStart,
            slot: slot,
            checkpointPositions: interiorCapture.positions,
            onPrefillCheckpoint: interiorCapture.onCheckpoint,
            wideVerify: wideVerifyOptions,
            // A watchdog stop is polled here, between tokens, alongside the
            // stop-string matcher's own flag.
            shouldStop: { @Sendable in state.shouldStop || watchdogs.wantsStop },
            onProgress: { @Sendable progress in
                guard state.decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill(let done, let total):
                        // The runtime reports prefill chunk by chunk; the prompt
                        // is read when the last chunk lands. A request that
                        // asked for reading progress gets each chunk before
                        // that, which is the only signal a long prompt has.
                        if done >= total {
                            publishPromptProcessed(total)
                        } else if request.returnProgress {
                            onEvent(.promptProgress(done: done, total: total))
                        }
                    case .token(_, let tokenID, let delta):
                        if let decoder = state.decoder {
                            publish(try decoder.consume(tokenID: tokenID, delta: delta), true)
                        } else {
                            publish(delta.isEmpty ? [] : [.content(delta)], true)
                        }
                    case .tail(let text):
                        if let decoder = state.decoder {
                            publish(try decoder.consumeTail(text), false)
                        } else {
                            publish(text.isEmpty ? [] : [.content(text)], false)
                        }
                    }
                } catch {
                    state.decodingError = error
                    state.shouldStop = true
                }
            })
        emitGenerationDiagnostics(
            activeProducer: activeProducer,
            result: result,
            snapshot: runnerSnapshot)
        func structuredFailure(
            kind: StructuredOutputFailureKind,
            cause: StructuredOutputFailureCause
        ) -> StructuredOutputFailure {
            StructuredOutputFailure(
                kind: kind,
                cause: cause,
                diagnostics: StructuredOutputFailureDiagnostics(
                    renderedPromptIDs: promptIDs,
                    effectivePromptIDs: effectivePromptIDs,
                    result: result,
                    maxCompletionTokens: config.maxNewTokens,
                    decodedCalls: state.output.calls.count,
                    visibleBytes: state.output.content.utf8.count,
                    stopStringMatched: state.output.isStopped,
                    toolStartID: tokenizer.toolCallStartID,
                    toolEndID: tokenizer.toolCallEndID,
                    toolResponseID: tokenizer.toolResponseID,
                    toolResponseEndID: tokenizer.toolResponseEndID))
        }
        if let decodingError = state.decodingError {
            throw structuredFailure(
                kind: .decoderConsume,
                cause: .classify(decodingError))
        }
        if let decoder {
            do {
                try decoder.finish()
            } catch {
                throw structuredFailure(
                    kind: .decoderFinish,
                    cause: .classify(error))
            }
        }
        if needsToolTemplate, result.reason == .toolCalls, state.output.calls.isEmpty {
            throw structuredFailure(kind: .orphanToolResponse, cause: .none)
        }
        state.output.finish()
        var content = state.output.content
        let calls = state.output.calls
        var reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        // The *last user message*, not the whole prompt: a long system
        // prompt in front of "hi" is still a short question, and an agent
        // harness puts a long system prompt in front of everything.
        let asked = request.messages.last { $0.role == .user }?.content?.utf8.count ?? 0
        watchdogs.finish(
            visibleBytes: content.utf8.count,
            requestBytes: asked,
            finishReason: reason)
        // B4: neither protocol has an honest reason for "the server stopped
        // this", and inventing one breaks clients. The mapping and the note
        // live in `WatchdogSet.resolve`, which is testable without a model.
        let outcome = watchdogs.resolve(content: content, finishReason: reason)
        // The cache entry must carry what was GENERATED. The note is written
        // by the server after the fact and has no tokens behind it in the KV
        // range, so publishing it would leave an entry whose text and KV
        // disagree, and a later continuation would splice the difference in.
        let generated = content
        if let note = outcome.note {
            content = outcome.content
            reason = outcome.finishReason
            onEvent(.content(note))
        }
        // A raw completion is never published to the prompt cache: there is no
        // message list to re-render a continuation tail from, so an entry would
        // be unusable by the only path that could match it.
        if !isRaw {
            publishCacheEntry(
                cacheRequest: cacheRequest,
                content: generated,
                calls: calls,
                result: result,
                stopStringFiltered: state.output.isStopped)
        }
        completed = true
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            // S26: completion_tokens reports the number of GENERATED tokens,
            // matching OpenAI's "completion_tokens = tokens in the generated
            // completion". A stop-string-hidden suffix is therefore counted as
            // generated even though it is filtered from the visible content.
            usage: OpenAIUsage(
                promptTokens: result.prefillTokens,
                completionTokens: result.newTokens,
                totalTokens: result.prefillTokens + result.newTokens,
                cachedTokens: result.cachedPromptTokens,
                reasoningTokens: state.output.reasoningTokens),
            watchdogTrips: watchdogs.trips,
            stopSequence: state.output.matchedStop,
            reasoning: state.output.reasoning,
            // The render's mode, not the loaded session's: a request that
            // switched thinking off per request is the one whose thought is
            // unrequested.
            unrequestedReasoning: renderTokenizer.thinkingMode.isEnabled
                ? 0 : state.output.reasoning.count,
            prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds,
            engineStopReason: result.reason)
    }

}
