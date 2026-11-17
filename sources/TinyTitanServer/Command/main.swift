import TinyTitanCore
import Darwin
import Foundation
// `MemberImportVisibility`: `ServerArguments` exposes `ModelFamily` (an TinyTitan
// type) and its `rawValue`, so this file names that module directly rather than
// relying on TinyTitanServerCore re-exporting it.
import TinyTitan
import TinyTitanLib
import TinyTitanLegacyEngine
import TinyTitanMemory
import TinyTitanServerCore

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}
if let line = arguments.calibrationLogLine {
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

// A launcher parses stdout, so the catalog is the only thing written there;
// skipped directories go to stderr.
if arguments.catalogOnly, let directory = arguments.modelsDirectory {
    let catalog = ModelCatalog.scan(directory: URL(fileURLWithPath: directory))
    catalog.reportSkipped()
    do {
        FileHandle.standardOutput.write(try catalog.jsonData() + Data("\n".utf8))
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

do {
    // The listener binds after the model is loaded, so an occupied port would
    // otherwise cost a full multi-gigabyte load before it was reported. The
    // bind below stays authoritative; this only moves the common refusal ahead
    // of the load.
    if !ServerPortCheck.isAvailable(port: arguments.port) {
        FileHandle.standardError.write(
            Data("error: --port \(arguments.port) is already in use on 127.0.0.1\n".utf8))
        exit(2)
    }
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    // Without --reasoning these are the --thinking / --reasoning-effort
    // values verbatim and nothing is read from disk; the router fits the
    // level per model itself, so it needs nothing from here.
    let reasoning =
        arguments.modelsDirectory == nil
        ? try arguments.singleModelReasoning(directory: modelURL)
        : (thinking: arguments.thinkingMode, effort: arguments.reasoningEffort)
    // Built lazily: the CPU engine serves an affine snapshot, which has no
    // manifest and would be rejected by a plan that expects an install.
    // Constructing it eagerly printed that rejection on every CPU launch.
    // Batched serving: one runner with this many KV/GDN slots, and a
    // coordinator that admits that many at once. `sessionSlots` caps the width
    // to one under MTP, and both the plan and the coordinator read it so they
    // cannot disagree.
    let concurrency = arguments.sessionSlots
    let makePlan = {
        ModelSessionPlan.from(
            arguments: arguments,
            modelDirectory: modelURL,
            thinking: reasoning.thinking,
            reasoningEffort: reasoning.effort,
            mtpModelDirectory: arguments.mtpModel.map {
                URL(fileURLWithPath: $0).standardizedFileURL
            })
    }

    let backend: any ServerInferenceBackend
    let facts: ModelSessionFacts
    var managed: ManagedModelBackend?
    var router: ModelRouter?
    /// The initial model's engine, set on the catalog path so the routing banner
    /// can name it beside `prompt_cache`: the two only make sense together,
    /// because a CPU entry has no cache and reports `.off` rather than the mode
    /// the server was asked for.
    var initialEngine: String?
    // The reasoning profile comes from an install's manifest, which a CPU
    // snapshot does not have. These models carry no reasoning-effort control
    // either, so the profile is the family's plain default.
    var reasoningProfile: ServerReasoningProfile?

    if let modelsDirectory = arguments.modelsDirectory {
        var catalog = ModelCatalog.scan(directory: URL(fileURLWithPath: modelsDirectory))
        // --model may name a directory outside the models directory; it is
        // served beside the catalog rather than refused.
        var initial = catalog.entry(idOrPath: arguments.model)
        if initial == nil, FileManager.default.fileExists(atPath: modelURL.path) {
            initial = catalog.add(probing: modelURL)
        }
        catalog.reportSkipped()
        guard let initial else { throw ModelRouterError.notInCatalog(arguments.model) }
        let routing = try ModelRouter(
            catalog: catalog,
            initialModelID: initial.id,
            reasoning: arguments.requestedReasoningLevel,
            maximumContext: arguments.maxContext,
            loader: ModelRouter.standardLoader(arguments: arguments))
        if !arguments.lazyLoad {
            try await routing.preload()
        }
        let gpu = catalog.entries.filter { $0.backend == .gpu }.count
        print(
            "catalog: \(catalog.entries.count) models (\(gpu) gpu, "
                + "\(catalog.entries.count - gpu) cpu) in \(catalog.directory.path); "
                + (arguments.lazyLoad
                    ? "\(initial.id) loads on the first request" : "loaded \(initial.id)"))
        router = routing
        backend = routing
        initialEngine = initial.backend.rawValue
        facts = ModelSessionFacts(
            modelID: initial.id,
            prefillChunkTokens: 0,
            // The mode the initial model will really run,
            // not the one requested: the rule lives in
            // `initialPromptCacheMode` so the banner, the
            // residency line and a test all read the same
            // answer, and the CPU arm is testable without
            // a catalog on disk.
            // The probe backend keeps no prompt cache, so an entry the
            // catalogue lists for it reports `.off` rather than the mode the
            // server was asked for.
            promptCacheMode: initial.kind == .gpu(.k2Horizon)
                ? .off
                : ServerModelSession.initialPromptCacheMode(
                    backend: initial.backend,
                    requested: arguments.promptCacheMode,
                    maxConcurrentSequences: concurrency))
        reasoningProfile = routing.servedModel(named: initial.id)?.reasoningProfile
    } else if arguments.cpu {
        // A different engine entirely: no Metal context, no expert
        // streaming, no prompt cache. Everything above it -- both API
        // surfaces, the memory subsystem, the watchdogs -- is unchanged,
        // which is the point of putting it behind the same protocol.
        let directory = URL(fileURLWithPath: arguments.model).standardizedFileURL
        let started = ContinuousClock.now
        let cpuBackend = try await CPUModelBackend(
            snapshotDirectory: directory,
            maximumContext: arguments.maxContext,
            resident: arguments.cpuResident,
            thinkingMode: reasoning.thinking)
        backend = cpuBackend
        let elapsed = started.duration(to: .now)
        let seconds =
            Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        let identifier =
            arguments.modelIDOverride
            ?? directory.lastPathComponent
        facts = ModelSessionFacts(
            modelID: identifier,
            prefillChunkTokens: 0,
            promptCacheMode: .off,
            expertCacheSlots: 0)
        print(
            String(
                format: "CPU engine: %@ %@in %.1fs, %d threads",
                identifier,
                cpuBackend.residentBytes > 0
                    ? "\(cpuBackend.residentBytes / 1_000_000) MB resident, " : "",
                seconds, cpuBackend.threads))
        reasoningProfile = ServerReasoningProfile(
            family: ModelFamily.qwen36.rawValue,
            thinkingMode: reasoning.thinking,
            reasoningEffort: nil,
            reasoningControl: ModelFamily.qwen36.engineReasoningControl)
    } else if (try? ManifestReader.peekFamily(directoryURL: modelURL)) == .k2Horizon {
        // K2 has a graph, but not this runner's: the probe is the engine. It
        // is a backend of its own, like the CPU path, because
        // `RealForwardRunner` sizes a Qwen-shaped pipeline set and refuses the
        // family. No prompt cache, no batching, plain FP16 KV caches.
        let slots = Int(
            ProcessInfo.processInfo.environment["TINYTITAN_K2_SLOTS"] ?? ""
        )
        .flatMap { $0 > 0 ? $0 : nil }
        let identity = try? ManifestReader.peekIdentity(directoryURL: modelURL)
        let resolvedSlots = slots ?? ((identity?.weightBits ?? 4) == 4 ? 64 : 16)
        let contextLimit = min(
            arguments.maxContext, K2ProbeBackend.contextCeiling)
        let identifier =
            arguments.modelIDOverride
            ?? modelURL.lastPathComponent
        let k2Facts = ModelSessionFacts(
            modelID: identifier,
            prefillChunkTokens: 0,
            promptCacheMode: .off,
            expertCacheSlots: resolvedSlots)
        facts = k2Facts
        if arguments.managesResidency {
            // --lazy-load and --idle-unload-seconds work here too: the wrapper
            // builds a probe on demand and drops it when idle, which is what
            // returns the install's ~22 GB. The Metal context is reused across
            // reloads.
            let residency = ManagedModelBackend(
                facts: k2Facts,
                maximumContext: contextLimit,
                samplingDefaults: SamplingProfile(
                    temperature: GenerationDefaults.forFamily(.k2Horizon).temperature,
                    topK: GenerationDefaults.forFamily(.k2Horizon).topK,
                    topP: GenerationDefaults.forFamily(.k2Horizon).topP,
                    minP: GenerationDefaults.forFamily(.k2Horizon).minP,
                    presencePenalty: GenerationDefaults.forFamily(.k2Horizon).presencePenalty),
                idleTimeout: arguments.idleUnloadSeconds > 0
                    ? .seconds(arguments.idleUnloadSeconds) : nil,
                loader: { context in
                    try await K2ProbeBackend(
                        modelDirectory: modelURL,
                        maximumContext: contextLimit,
                        thinking: reasoning.thinking,
                        reasoningEffort: reasoning.effort,
                        slots: slots,
                        reusingContext: context)
                })
            managed = residency
            backend = residency
            print(
                "K2 probe engine: \(identifier) loads on the first request"
                    + (arguments.idleUnloadSeconds > 0
                        ? ", unloads after \(arguments.idleUnloadSeconds)s idle"
                        : ""))
        } else {
            let started = ContinuousClock.now
            let k2 = try await K2ProbeBackend(
                modelDirectory: modelURL,
                maximumContext: contextLimit,
                thinking: reasoning.thinking,
                reasoningEffort: reasoning.effort,
                slots: slots)
            backend = k2
            let elapsed = started.duration(to: .now)
            let seconds =
                Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            print(
                String(
                    format: "K2 probe engine: %@ in %.1fs, context %d, %d expert slots",
                    identifier, seconds, k2.maximumContext, k2.expertCacheSlots))
        }
        reasoningProfile = ServerReasoningProfile(
            family: ModelFamily.k2Horizon.rawValue,
            thinkingMode: reasoning.thinking,
            reasoningEffort: reasoning.effort,
            reasoningControl: ModelFamily.k2Horizon.engineReasoningControl)
    } else if arguments.managesResidency {
        let plan = makePlan()
        // Reads manifest.json only; a bad --model still fails here at launch
        // rather than on the first request.
        facts = try plan.previewFacts(
            modelIDOverride: arguments.modelIDOverride, factory: EngineFactories.sixXGPU())
        let residency = ManagedModelBackend(
            plan: plan,
            facts: facts,
            idleTimeout: arguments.idleUnloadSeconds > 0
                ? .seconds(arguments.idleUnloadSeconds) : nil)
        managed = residency
        backend = residency
    } else {
        let plan = makePlan()
        let session = try await plan.makeSession(factory: EngineFactories.sixXGPU())
        backend = session
        facts = ModelSessionFacts(
            modelID: arguments.modelIDOverride ?? session.defaultModelID,
            prefillChunkTokens: session.prefillChunkTokens,
            promptCacheMode: session.promptCacheMode,
            expertCacheSlots: session.expertCacheSlots)
    }

    // The coordinator owns the "a client generation is in flight" signal, and
    // the resident side-engine reads it to choose its width, so it is built
    // here and handed to both.
    let coordinator = ServerCoordinator(
        queueLimit: arguments.queueLimit,
        width: concurrency)

    // Persistent memory wraps whatever backend was built: one decorator on
    // the way in, and nothing at all when it is disabled. The side-engine is
    // built inside, from `TINYTITAN_SIDE_ENGINE` or the default install.
    let servingBackend = ServerMemoryFactory.wrap(
        backend,
        modelsDirectory: arguments.modelsDirectory,
        isClientGenerating: { coordinator.generating.isBusy })

    let server = TinyTitanHTTPServer(
        modelID: facts.modelID,
        queueLimit: arguments.queueLimit,
        maxConcurrentSequences: concurrency,
        backend: servingBackend,
        reasoningProfile: try reasoningProfile
            ?? makePlan().reasoningProfile(factory: EngineFactories.sixXGPU()),
        router: router,
        coordinator: coordinator)
    _ = try await server.start(port: arguments.port)
    let diskCache =
        facts.promptCacheMode == .off
        ? "off" : arguments.promptCacheDiskDirectory ?? "off"
    let cacheMemoryMiB =
        facts.promptCacheMode == .off
        ? 0 : arguments.promptCacheMemoryMiB
    let mtp = arguments.mtpModel == nil ? "off" : "on:\(arguments.mtpMemoryMiB)MiB"
    let residencyBanner =
        arguments.managesResidency
        ? " lazy_load=on idle_unload=\(arguments.idleUnloadSeconds > 0 ? "\(arguments.idleUnloadSeconds)s" : "off")"
        : ""
    if let router {
        // Prefill chunk and expert slots belong to whichever install is
        // loaded, so the routing banner states the server's own settings. The
        // exception is `prompt_cache`, which belongs to the model: it reports
        // the initial model's engine and real mode, and the residency line
        // reports both again on every load and switch.
        let engine = initialEngine.map { " engine=\($0)" } ?? ""
        print(
            "TinyTitanServer \(ServerVersion.current) ready at http://127.0.0.1:\(arguments.port) models=\(router.servedModels.count) initial=\(facts.modelID)\(engine) context=\(arguments.maxContext) concurrency=\(concurrency) prompt_cache=\(facts.promptCacheMode.rawValue) reasoning=\(arguments.requestedReasoningLevel.rawValue) dynamic=on"
        )
    } else {
        print(
            "TinyTitanServer \(ServerVersion.current) ready at http://127.0.0.1:\(arguments.port) model=\(facts.modelID) context=\(arguments.maxContext) concurrency=\(concurrency) prefill_chunk=\(facts.prefillChunkTokens)\(facts.expertCacheSlots > 0 ? " expert_slots=\(facts.expertCacheSlots)" : "") prompt_cache=\(facts.promptCacheMode.rawValue) prompt_cache_memory_mib=\(cacheMemoryMiB) prompt_cache_disk=\(diskCache) thinking=\(reasoning.thinking.rawValue) mtp=\(mtp)\(residencyBanner)"
        )
    }
    WatchdogConfiguration.shared.announce()
    if arguments.unloadDiscardsWarmCache {
        FileHandle.standardError.write(
            Data(
                ("warning: --idle-unload-seconds drops the in-memory prompt cache with "
                    + "the model; add --prompt-cache-disk <dir> so entries survive an "
                    + "unload, or the first request after each unload pays a full "
                    + "cold prefill\n").utf8))
    }

    _ = await signals.wait()
    try await server.shutdown()
    // After the server, so nothing is still writing: this flushes memory that
    // has not reached a session boundary and releases the workspace lock.
    if let memory = servingBackend as? MemoryBackend {
        await memory.shutDown()
    }
    // After the server, so the reaper cannot outlive it.
    await managed?.shutdown()
    await router?.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
