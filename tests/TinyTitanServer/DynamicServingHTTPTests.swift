import Foundation
import NIOCore
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
import TinyTitanCore

private func send(
    _ port: Int, _ method: String, _ path: String, json: String? = nil,
    headers: [String: String] = [:]
) async throws -> (Data, HTTPURLResponse) {
    var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)\(path)")))
    request.httpMethod = method
    if let json {
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(json.utf8)
    }
    for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
    let (data, response) = try await URLSession.shared.data(for: request)
    return (data, try #require(response as? HTTPURLResponse))
}

private func object(_ data: Data) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

private func chat(_ model: String, extra: String = "") -> String {
    #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]\#(extra)}"#
}

private func withServer<T>(
    backend: any ServerInferenceBackend,
    router: (any ModelRouting)? = nil,
    queueLimit: Int = 4,
    maxConcurrentSequences: Int = 1,
    _ body: (Int) async throws -> T
) async throws -> T {
    let server = TinyTitanHTTPServer(
        modelID: "test-model", queueLimit: queueLimit,
        maxConcurrentSequences: maxConcurrentSequences,
        backend: backend, router: router)
    let channel = try await server.start(port: 0)
    let port = try #require(channel.localAddress?.port)
    do {
        let result = try await body(port)
        try await server.shutdown()
        return result
    } catch {
        try await server.shutdown()
        throw error
    }
}

private func withRouter<T>(
    log: RoutingEventLog, delay: Duration? = nil,
    _ body: (Int, ModelRouter) async throws -> T
) async throws -> T {
    let router = try RoutingFixture.router(log: log, delay: delay)
    try await router.preload()
    return try await withServer(backend: router, router: router) { port in
        try await body(port, router)
    }
}

/// Holds every generation until `releaseAll`, recording the peak number running
/// at once. Used to observe the coordinator's admission width through HTTP.
private actor ConcurrencyProbe: ServerInferenceBackend {
    let maximumContext = 262_144
    private var active = 0
    private var peak = 0
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        active += 1
        peak = max(peak, active)
        if !open {
            await withCheckedContinuation { waiters.append($0) }
        }
        active -= 1
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    func peakConcurrency() -> Int { peak }

    func releaseAll() {
        open = true
        let held = waiters
        waiters.removeAll()
        for waiter in held { waiter.resume() }
    }
}

@Suite("Dynamic serving over HTTP", .serialized)
struct DynamicServingHTTPTests {
    private let anthropic = ["anthropic-version": "2023-06-01"]

    @Test func modelsListsEveryCatalogModelInTheOpenAIShape() async throws {
        try await withRouter(log: RoutingEventLog()) { port, _ in
            let (data, response) = try await send(port, "GET", "/v1/models")
            #expect(response.statusCode == 200)
            let list = try object(data)
            #expect(list["object"] as? String == "list")
            let models = try #require(list["data"] as? [[String: Any]])
            // The dense install is listed once with its alternative engine as
            // a real choice; a single-engine install is listed bare.
            #expect(
                models.compactMap { $0["id"] as? String }
                    == [
                        "alpha_4-Bit", "flash_8-Bit", "small-2b",
                        RoutingFixture.dense.id, "\(RoutingFixture.dense.id)@cpu",
                    ])
            #expect(models.allSatisfy { $0["object"] as? String == "model" })
            #expect(models.allSatisfy { $0["owned_by"] as? String == "tinytitan" })
        }
    }

    @Test func modelsAnswersInTheAnthropicShapeWhenAskedTo() async throws {
        try await withRouter(log: RoutingEventLog()) { port, _ in
            let (data, _) = try await send(port, "GET", "/v1/models", headers: anthropic)
            let list = try object(data)
            #expect(list["has_more"] as? Bool == false)
            #expect(list["first_id"] as? String == "alpha_4-Bit")
            #expect(list["last_id"] as? String == "\(RoutingFixture.dense.id)@cpu")
            let models = try #require(list["data"] as? [[String: Any]])
            #expect(
                models.compactMap { $0["display_name"] as? String }
                    == [
                        "Alpha 35B", "Flash 125B", "Small 2B",
                        "Dense 2B", "Dense 2B (CPU)",
                    ])
            #expect(models.allSatisfy { $0["type"] as? String == "model" })
            #expect(models.allSatisfy { $0["created_at"] is String })
            #expect(Set(models.flatMap(\.keys)) == ["type", "id", "display_name", "created_at"])

            let (one, status) = try await send(
                port, "GET", "/v1/models/small-2b", headers: anthropic)
            #expect(status.statusCode == 200)
            #expect(try object(one)["display_name"] as? String == "Small 2B")
            let (_, missing) = try await send(port, "GET", "/v1/models/nothing")
            #expect(missing.statusCode == 404)
        }
    }

    @Test func requestsRunOnTheModelTheyNameAndSayWhichItWas() async throws {
        let log = RoutingEventLog()
        try await withRouter(log: log) { port, router in
            let (data, response) = try await send(
                port, "POST", "/v1/chat/completions",
                json: chat("small-2b"))
            #expect(response.statusCode == 200)
            let completion = try object(data)
            #expect(completion["model"] as? String == "small-2b")
            #expect(await router.residentModelID == "small-2b")

            // The "-fast" alias works for every catalog model and still
            // routes to (and reports) the base model.
            let (fast, fastResponse) = try await send(
                port, "POST", "/v1/chat/completions",
                json: chat("flash_8-Bit-fast"))
            #expect(fastResponse.statusCode == 200)
            #expect(try object(fast)["model"] as? String == "flash_8-Bit")
            let last = try #require(log.requests.last)
            #expect(last.stripCLIPrompt)
            #expect(last.model == "flash_8-Bit")

            let (message, messageResponse) = try await send(
                port, "POST", "/v1/messages",
                json:
                    #"{"model":"alpha_4-Bit","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
                headers: anthropic)
            #expect(messageResponse.statusCode == 200)
            #expect(try object(message)["model"] as? String == "alpha_4-Bit")
            #expect(
                log.loads == [
                    "load alpha_4-Bit", "load small-2b", "load flash_8-Bit", "load alpha_4-Bit",
                ])
        }
    }

    @Test func anUnknownModelIsRefusedOnEverySurface() async throws {
        let log = RoutingEventLog()
        try await withRouter(log: log) { port, _ in
            let (chatData, chatResponse) = try await send(
                port, "POST", "/v1/chat/completions",
                json: chat("nothing"))
            #expect(chatResponse.statusCode == 404)
            let error = try #require(try object(chatData)["error"] as? [String: Any])
            #expect(error["code"] as? String == "model_not_found")

            let (_, responses) = try await send(
                port, "POST", "/v1/responses",
                json: #"{"model":"nothing","input":"hi"}"#)
            #expect(responses.statusCode == 404)
            let (_, messages) = try await send(
                port, "POST", "/v1/messages",
                json:
                    #"{"model":"nothing","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
                headers: anthropic)
            #expect(messages.statusCode == 404)
            #expect(log.loads == ["load alpha_4-Bit"])
        }
    }

    /// The request's omitted values and its token bound come from the model it
    /// names, decided before that model is loaded -- not from the resident one.
    @Test func omittedSamplingAndTheTokenBoundComeFromTheNamedModel() async throws {
        let log = RoutingEventLog()
        try await withRouter(log: log) { port, router in
            // Alpha is resident; the CPU model's context is 32768.
            let (_, tooLong) = try await send(
                port, "POST", "/v1/chat/completions",
                json: chat("small-2b", extra: #","max_tokens":40000"#))
            #expect(tooLong.statusCode == 400)
            #expect(await router.residentModelID == "alpha_4-Bit")
            let (_, fits) = try await send(
                port, "POST", "/v1/chat/completions",
                json: chat("alpha_4-Bit", extra: #","max_tokens":40000"#))
            #expect(fits.statusCode == 200)

            _ = try await send(port, "POST", "/v1/chat/completions", json: chat("flash_8-Bit"))
            let flash = try #require(log.requests.last)
            // Qwen3.8 has two published rows. This fixture serves it with
            // thinking off, so the instruct row applies -- not the thinking row
            // the model was catalogued with.
            #expect(flash.generationConfig.temperature == 0.7)
            #expect(flash.generationConfig.topP == 0.80)
            #expect(flash.generationConfig.presencePenalty == 1.5)
            #expect(flash.maximumCompletionTokens == RoutingFixture.configuredContext)

            _ = try await send(port, "POST", "/v1/chat/completions", json: chat("small-2b"))
            let small = try #require(log.requests.last)
            #expect(small.generationConfig.temperature == 0.6)
            #expect(small.maximumCompletionTokens == CPUModelBackend.contextCeiling)
        }
    }

    /// Four clients at once against the default queue limit: every one is
    /// admitted and answered, one generation at a time. The width is the
    /// *parsed* default rather than a literal, so a launched server and this
    /// test cannot disagree about what "default" means.
    @Test func fourConcurrentRequestsAllComplete() async throws {
        let log = RoutingEventLog()
        let backend = RoutedStubModel(
            id: "test-model", log: log, gate: nil,
            delay: .milliseconds(150))
        let defaults = try ServerArguments.parse(["--model", "/m"], environment: [:])
        try await withServer(
            backend: backend, queueLimit: defaults.queueLimit,
            maxConcurrentSequences: defaults.maxConcurrentSequences
        ) { port in
            let statuses = try await withThrowingTaskGroup(of: Int.self) { group in
                for _ in 0..<4 {
                    group.addTask {
                        try await send(
                            port, "POST", "/v1/chat/completions",
                            json: chat("test-model")
                        ).1.statusCode
                    }
                }
                return try await group.reduce(into: []) { $0.append($1) }
            }
            #expect(statuses == [200, 200, 200, 200])
        }
        #expect(log.requests.count == 4)
        #expect(log.maxConcurrentGenerations == 1)
    }

    /// With a width of four, four generations run at once through one server;
    /// the fifth queues behind the `queueLimit` of one and the sixth is shed
    /// with 429. The batched admission rule, observed at the HTTP boundary.
    ///
    /// Kept to five concurrent requests because URLSession opens at most six
    /// connections per host: a wider fan-out would never reach the server.
    @Test func fourGenerationsRunAtOnceAndTheSixthIsShed() async throws {
        let probe = ConcurrencyProbe()
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1,
            maxConcurrentSequences: 4, backend: probe)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let statuses = try await withThrowingTaskGroup(of: Int.self) { group in
                for _ in 0..<5 {
                    group.addTask {
                        try await send(
                            port, "POST", "/v1/chat/completions",
                            json: chat("test-model")
                        ).1.statusCode
                    }
                }
                // Wait until all five are admitted: four running (held by the
                // probe) and one queued behind them.
                let deadline = ContinuousClock.now + .seconds(10)
                while ContinuousClock.now < deadline {
                    let peak = await probe.peakConcurrency()
                    let queued = await server.queuedRequestCount
                    if peak >= 4 && queued >= 1 { break }
                    try? await Task.sleep(for: .milliseconds(5))
                }
                // Sent while all five are still admitted and held, so it cannot
                // be admitted whatever the scheduling order.
                let sixth = try await send(
                    port, "POST", "/v1/chat/completions",
                    json: chat("test-model")
                ).1.statusCode
                #expect(sixth == 429)
                await probe.releaseAll()
                var out: [Int] = []
                for try await status in group { out.append(status) }
                return out
            }
            #expect(statuses == Array(repeating: 200, count: 5))
            #expect(
                await probe.peakConcurrency() == 4,
                "exactly the configured width ran at once")
            try await server.shutdown()
        } catch {
            try await server.shutdown()
            throw error
        }
    }

    @Test func fourConcurrentRequestsAcrossModelsAllComplete() async throws {
        let log = RoutingEventLog()
        let names = ["alpha_4-Bit", "small-2b", "alpha_4-Bit", "flash_8-Bit"]
        try await withRouter(log: log, delay: .milliseconds(50)) { port, _ in
            let answered = try await withThrowingTaskGroup(of: (String, Int, String?).self) {
                group in
                for name in names {
                    group.addTask {
                        let (data, response) = try await send(
                            port, "POST", "/v1/chat/completions",
                            json: chat(name))
                        return (name, response.statusCode, try object(data)["model"] as? String)
                    }
                }
                return try await group.reduce(into: []) { $0.append($1) }
            }
            #expect(answered.count == 4)
            for (name, status, model) in answered {
                #expect(status == 200)
                #expect(model == name)
            }
        }
        #expect(log.maxConcurrentGenerations == 1)
    }

    /// Without a router the server is the one it always was: one model plus
    /// its "-fast" alias, the backend's own defaults, every other name refused.
    @Test func aSingleModelServerIsUnchanged() async throws {
        let log = RoutingEventLog()
        let backend = RoutedStubModel(id: "test-model", log: log, gate: nil, delay: nil)
        try await withServer(backend: backend) { port in
            let (data, _) = try await send(port, "GET", "/v1/models")
            let ids = try #require(try object(data)["data"] as? [[String: Any]])
                .compactMap { $0["id"] as? String }
            #expect(ids == ["test-model", "test-model-fast"])

            let (_, other) = try await send(
                port, "POST", "/v1/chat/completions", json: chat("small-2b"))
            #expect(other.statusCode == 404)
            let (reply, ok) = try await send(
                port, "POST", "/v1/chat/completions", json: chat("test-model"))
            #expect(ok.statusCode == 200)
            #expect(try object(reply)["model"] as? String == "test-model")
        }
        let request = try #require(log.requests.first)
        #expect(request.generationConfig.temperature == GenerationDefaults.house.temperature)
        #expect(request.maximumCompletionTokens == backend.maximumContext)
    }
}

@Suite("Dynamic serving arguments")
struct DynamicServingArgumentTests {
    private func parse(_ input: [String]) throws -> ServerArguments {
        try ServerArguments.parse(input, environment: [:])
    }

    @Test func withoutAModelsDirectoryNothingChanges() throws {
        let arguments = try parse(["--model", "/m"])
        #expect(arguments.modelsDirectory == nil)
        #expect(!arguments.catalogOnly)
        #expect(arguments.reasoningLevel == nil)
        #expect(arguments.requestedReasoningLevel == .off)
    }

    @Test func theCatalogNeedsADirectoryButNoModel() throws {
        let arguments = try parse(["--catalog", "--models-dir", "/models"])
        #expect(arguments.catalogOnly)
        #expect(arguments.modelsDirectory == "/models")
        #expect(throws: ServerArgumentError.self) { try parse(["--catalog"]) }
        #expect(throws: ServerArgumentError.invalid("--model is required")) {
            try parse(["--models-dir", "/models"])
        }
    }

    @Test func reasoningReplacesTheOlderFlags() throws {
        #expect(
            try parse(["--model", "/m", "--reasoning", "high"]).requestedReasoningLevel == .high)
        #expect(try parse(["--model", "/m", "--thinking", "on"]).requestedReasoningLevel == .on)
        #expect(
            try parse(["--model", "/m", "--thinking", "on", "--reasoning-effort", "xhigh"])
                .requestedReasoningLevel == .xhigh)
        #expect(throws: ServerArgumentError.self) {
            try parse(["--model", "/m", "--reasoning", "high", "--thinking", "on"])
        }
        #expect(throws: ServerArgumentError.self) {
            try parse(["--model", "/m", "--reasoning", "loud"])
        }
    }

    @Test func singleModelFlagsAreRefusedWithACatalog() {
        for flags in [
            ["--model-id", "x"], ["--cpu"], ["--mtp-model", "/d"], ["--idle-unload-seconds", "60"],
        ] {
            #expect(throws: ServerArgumentError.self) {
                try parse(["--models-dir", "/models", "--model", "a"] + flags)
            }
        }
    }

    /// One active plus `queueLimit` queued: the default must admit four
    /// concurrent clients without shedding any.
    @Test func theDefaultQueueAdmitsFourClients() throws {
        #expect(try parse(["--model", "/m"]).queueLimit + 1 >= 4)
    }

    /// The batched width is opt-in and a power of two, so an agentic workload can
    /// ask for many while a typo is still refused. What a machine really builds
    /// is clamped by the memory budget at load, not by this argument.
    @Test func concurrentSequenceCountIsAPowerOfTwoUpToTheEngineCeiling() throws {
        #expect(try parse(["--model", "/m"]).maxConcurrentSequences == 1)
        for good in ["1", "2", "4", "8", "16", "256"] {
            #expect(
                try parse(["--model", "/m", "--max-concurrent-sequences", good])
                    .maxConcurrentSequences == Int(good))
        }
        for bad in ["0", "3", "5", "6", "7", "-1", "512", "abc"] {
            #expect(throws: ServerArgumentError.self) {
                try parse(["--model", "/m", "--max-concurrent-sequences", bad])
            }
        }
        // The argument's ceiling is the engine's slot cap, so the two cannot
        // drift apart: a width the server accepts is one the engine can build.
        #expect(KVCacheManager.maximumSlots == 256)
    }

    /// More than one slot turns the single-sequence prompt cache off rather
    /// than risk restoring a prefix into the wrong slot; MTP does the same.
    @Test func batchingDisablesTheSessionWidePromptCache() {
        #expect(
            ServerModelSession.effectivePromptCacheMode(
                requested: .multiPrefix, mtpEnabled: false, slots: 1) == .multiPrefix)
        #expect(
            ServerModelSession.effectivePromptCacheMode(
                requested: .multiPrefix, mtpEnabled: false, slots: 4) == .off)
        #expect(
            ServerModelSession.effectivePromptCacheMode(
                requested: .singlePrefix, mtpEnabled: false, slots: 2) == .off)
        #expect(
            ServerModelSession.effectivePromptCacheMode(
                requested: .multiPrefix, mtpEnabled: true, slots: 1) == .off)
    }

    /// The routing banner states the initial model's cache mode, and that
    /// depends on the engine as well as the width: a CPU entry has no cache to
    /// turn on, so it reports off whatever the server was asked for, and a
    /// raised width turns a GPU entry's session-wide cache off.
    @Test func theInitialModelsPromptCacheFollowsItsEngineAndTheWidth() {
        for slots in [1, 2, 4] {
            #expect(
                ServerModelSession.initialPromptCacheMode(
                    backend: .cpu, requested: .multiPrefix,
                    maxConcurrentSequences: slots) == .off)
        }
        #expect(
            ServerModelSession.initialPromptCacheMode(
                backend: .gpu, requested: .multiPrefix, maxConcurrentSequences: 1) == .multiPrefix)
        #expect(
            ServerModelSession.initialPromptCacheMode(
                backend: .gpu, requested: .multiPrefix, maxConcurrentSequences: 4) == .off)
        #expect(
            ServerModelSession.initialPromptCacheMode(
                backend: .gpu, requested: .singlePrefix, maxConcurrentSequences: 2) == .off)
    }

    /// Every plan is built through `ModelSessionPlan.from`, which carries the
    /// configured width. The catalog loader once built its own plan and dropped
    /// `slots`, so `--models-dir` sessions ran one sequence while the
    /// coordinator admitted four; this pins the width through the shared
    /// factory so that cannot recur silently.
    @Test func thePlanFactoryCarriesTheConfiguredWidth() throws {
        let eight = try parse(["--model", "/m", "--max-concurrent-sequences", "8"])
        let plan = ModelSessionPlan.from(
            arguments: eight,
            modelDirectory: URL(fileURLWithPath: "/m"),
            thinking: .off, reasoningEffort: nil, mtpModelDirectory: nil)
        #expect(plan.slots == 8)

        let mtp = try parse(["--model", "/m", "--mtp-model", "/d"])
        let mtpPlan = ModelSessionPlan.from(
            arguments: mtp,
            modelDirectory: URL(fileURLWithPath: "/m"),
            thinking: .off, reasoningEffort: nil,
            mtpModelDirectory: URL(fileURLWithPath: "/d"))
        #expect(mtpPlan.slots == 1, "MTP is single-sequence")
    }
}
