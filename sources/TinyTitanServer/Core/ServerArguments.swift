import Foundation
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib
import TinyTitanVariants

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let mtpModel: String?
    public let mtpMemoryMiB: Int
    public let port: Int
    /// Explicit --model-id value; nil derives the API ID from the installed
    /// manifest (for example qwen3.6-35b-a3b or ornith-1.5-35b-a3b).
    public let modelIDOverride: String?
    public let maxContext: Int
    /// Whether `--max-context` was explicitly passed on the command line.
    public var maxContextExplicit = false
    public let queueLimit: Int
    /// Generations that may run at once through the batched engine. One is the
    /// historical single-generation server and the default; the excess still
    /// queues. Above one, each sequence holds its own KV cache and scratch, and
    /// the prompt cache is switched off (it is session-wide, so a prefix could
    /// otherwise be restored into the wrong sequence).
    public let maxConcurrentSequences: Int

    /// The width a session's runner and scratch may actually be built with.
    /// MTP is single-sequence, so it pins the width to one; every plan factory
    /// reads this rather than re-deriving it, so the single-model path and the
    /// catalog loader cannot disagree.
    public var sessionSlots: Int {
        mtpModel == nil ? maxConcurrentSequences : 1
    }
    package let promptCacheMode: ServerPromptCacheMode
    public let promptCacheMaximumEntries: Int
    public let promptCacheMemoryMiB: Int
    public let promptCacheDiskDirectory: String?
    public let promptCacheDiskMiB: Int
    public let prefillChunkTokens: Int?
    /// Path to a measured calibration profile, if one was requested.
    public let calibrationProfile: String?
    /// One line naming what the profile contributed, for the startup log.
    public let calibrationLogLine: String?
    public let kvCachePrecision: KVCachePrecision
    /// Whether `--kv-bits` was given. Inkling 2 refuses an explicit 8 or 4 and
    /// serves the default at 16 bits (TT-123), so it has to tell them apart.
    public var kvCachePrecisionExplicit = false
    /// Host-KV window: full-attention rows stay in a host arena and only this
    /// many rows per layer/slot live in Metal. Nil is the contiguous cache.
    public let hostKVWindowTokens: Int?
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let thinkingMode: ModelThinkingMode
    /// Reasoning-effort override for effort-aware model families; nil keeps
    /// the template default. Family support is validated at startup against
    /// the installed manifest.
    public let reasoningEffort: ModelReasoningEffort?
    /// Serve on the CPU instead of the GPU, from an affine snapshot rather
    /// than an install.
    ///
    /// For models small enough that a GPU is not the point: a 2B reads about
    /// 1.9 GB per token, which the CPU does at twenty tokens a second on
    /// four performance cores. It leaves the GPU entirely free, which is why
    /// the same engine can run a side model beside a 35B.
    public let cpu: Bool
    /// Keep the whole snapshot faulted into memory rather than leaving
    /// residency to the page cache. On by default with `--cpu`, because a
    /// model that gets evicted between requests pays to fault itself back in.
    public let cpuResident: Bool
    public let expertCacheSlots: Int?
    /// Bytes the routed-expert cache may use. Slots are derived from it and the
    /// model's own expert stride, so this is the knob and the slot count is the
    /// outcome. `--expert-cache-slots` still wins if both are given.
    public let expertCacheBudgetBytes: Int?
    /// Defer the model load to the first inference request.
    public let lazyLoad: Bool
    /// Release the weights after this many idle seconds; 0 disables unloading.
    public let idleUnloadSeconds: Int
    /// Serve every model under this directory, one resident at a time, with
    /// `model` naming the one loaded first. Nil keeps the single-model server
    /// exactly as it was, which the benchmark harness depends on.
    public let modelsDirectory: String?
    /// Print the catalog of `modelsDirectory` as JSON and exit.
    public let catalogOnly: Bool
    /// The explicit `--reasoning` level; nil when the older flags were used.
    public let reasoningLevel: TinyTitanCore.ReasoningLevel?

    /// lint:allow-long a flag table: one `case` per option plus its
    /// validation. Splitting it into per-group parsers would hide the
    /// exhaustive switch that makes an unhandled flag a compile-visible gap.
    public static func parse(
        _ input: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) throws -> ServerArguments {
        var model: String?
        var mtpModel: String?
        var mtpMemoryMiB = RuntimeLimits.defaultMTPMemoryMiB
        var port = 8080
        var modelIDOverride: String?
        var maxContext = 262_144
        var maxContextWasSet = false
        var queueLimit = 4
        // One generation at a time unless the operator asks for more: the
        // batched path holds one KV cache and scratch buffer per sequence, so
        // the memory bill and the per-answer slowdown are opt-in, not a
        // surprise on a default launch.
        var maxConcurrentSequences = 1
        var promptCacheMode: ServerPromptCacheMode = .multiPrefix
        var promptCacheMaximumEntries = 4
        // Derived, not constant: `--prompt-cache-memory-mib` below still
        // overrides it when the caller names one.
        var promptCacheMemoryMiB = Self.defaultPromptCacheMemoryMiB(
            physicalMemory: physicalMemory)
        var promptCacheDiskDirectory: String?
        var promptCacheDiskMiB = 8_192
        var prefillChunkTokens: Int?
        var calibrationProfile: String?
        var calibrationLogLine: String?
        var kvCachePrecision: KVCachePrecision = .int8
        var kvCachePrecisionExplicit = false
        var hostKVWindowTokens: Int?
        var ropeScalingMode: RuntimeRoPEScalingMode = .none
        var thinkingMode = ModelThinkingMode.resolved(environment: environment)
        var reasoningEffort = ModelReasoningEffort.resolved(environment: environment)
        var expertCacheSlots: Int?
        var cpu = false
        var cpuResident = true
        var expertCacheBudgetBytes: Int?
        var lazyLoad = false
        var idleUnloadSeconds = 0
        var modelsDirectory: String?
        var catalogOnly = false
        var reasoningLevel: TinyTitanCore.ReasoningLevel?
        // Tracked apart from the values, which the environment can also set:
        // only flags typed next to --reasoning conflict with it.
        var thinkingWasSet = false
        var effortWasSet = false
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            if flag == "--catalog" {
                catalogOnly = true
                index += 1
                continue
            }
            // Valueless flags are consumed before the "requires a value" guard
            // below; otherwise `--lazy-load --port 9999` would swallow --port
            // as this flag's value and then reject it as unknown.
            if flag == "--lazy-load" {
                lazyLoad = true
                index += 1
                continue
            }
            if flag == "--cpu" {
                cpu = true
                index += 1
                continue
            }
            if flag == "--no-cpu-resident" {
                cpuResident = false
                index += 1
                continue
            }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--mtp-model":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--mtp-model must not be empty")
                }
                mtpModel = value
            case "--mtp-memory-mib":
                guard let parsed = Int(value),
                    RuntimeLimits.allowedMTPMemoryMiB.contains(parsed)
                else {
                    throw ServerArgumentError.invalid(
                        "--mtp-memory-mib must be between 256 and 512")
                }
                mtpMemoryMiB = parsed
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                    (1...RuntimeLimits.maximumContextTokens).contains(parsed)
                else {
                    // Not "is not supported": that sentence belongs to the rung
                    // gate below, and a typo read as a policy refusal.
                    throw ServerArgumentError.invalid(
                        "--max-context must be a token count between 1 and "
                            + "\(RuntimeLimits.maximumContextTokens)")
                }
                maxContext = parsed
                maxContextWasSet = true
            case "--rope-scaling":
                guard let parsed = RuntimeRoPEScalingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--rope-scaling must be none or yarn")
                }
                ropeScalingMode = parsed
            case "--queue-limit":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid("--queue-limit must be between 1 and 64")
                }
                queueLimit = parsed
            case "--max-concurrent-sequences":
                // A power of two, up to the engine's slot ceiling. The old bound
                // of 4 was a policy limit, not an engineering one: the batched
                // stores are sized per slot and `BatchedMemoryBudget` clamps the
                // width to what the machine can actually hold at load, logging
                // what it built. So the argument may ask for more than a given
                // Mac will run — that is the operator's call, and the clamp is
                // what keeps it honest.
                guard let parsed = Int(value), parsed >= 1,
                    parsed <= RuntimeLimits.maximumConcurrentSequences,
                    parsed & (parsed - 1) == 0
                else {
                    throw ServerArgumentError.invalid(
                        "--max-concurrent-sequences must be a power of two between 1 and "
                            + "\(RuntimeLimits.maximumConcurrentSequences)")
                }
                maxConcurrentSequences = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off, single-prefix, or multi-prefix")
                }
                promptCacheMode = parsed
            case "--prompt-cache-entries":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-entries must be between 1 and 64")
                }
                promptCacheMaximumEntries = parsed
            case "--prompt-cache-memory-mib":
                guard let parsed = Int(value), (0...4_096).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-memory-mib must be between 0 and 4096")
                }
                promptCacheMemoryMiB = parsed
            case "--prompt-cache-disk":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-disk must not be empty")
                }
                promptCacheDiskDirectory = value
            case "--prompt-cache-disk-mib":
                guard let parsed = Int(value), (0...65_536).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-disk-mib must be between 0 and 65536")
                }
                promptCacheDiskMiB = parsed
            case "--calibration-profile":
                calibrationProfile = value
            case "--prefill-chunk":
                guard let parsed = Int(value),
                    RuntimeLimits.allowedPrefillChunkTokens.contains(parsed)
                else {
                    throw ServerArgumentError.invalid(
                        "--prefill-chunk must be one of "
                            + Self.prefillChunkTokensHelp)
                }
                prefillChunkTokens = parsed
            case "--kv-bits":
                guard let bits = Int(value),
                    let parsed = KVCachePrecision(rawValue: bits)
                else {
                    throw ServerArgumentError.invalid("--kv-bits must be 4, 8, or 16")
                }
                kvCachePrecision = parsed
                kvCachePrecisionExplicit = true
            case "--host-kv-window":
                guard let parsed = Int(value),
                    (1...RuntimeLimits.maximumContextTokens).contains(parsed)
                else {
                    throw ServerArgumentError.invalid(
                        "--host-kv-window must be a token count between 1 and "
                            + "\(RuntimeLimits.maximumContextTokens)")
                }
                hostKVWindowTokens = parsed
            case "--thinking":
                guard let parsed = ModelThinkingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--thinking must be off or on")
                }
                thinkingMode = parsed
                thinkingWasSet = true
            case "--reasoning-effort":
                guard let parsed = ModelReasoningEffort(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--reasoning-effort must be low, medium, high, or xhigh")
                }
                reasoningEffort = parsed
                effortWasSet = true
            case "--reasoning":
                guard let parsed = TinyTitanCore.ReasoningLevel(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--reasoning must be one of "
                            + TinyTitanCore.ReasoningLevel.allCases.map(\.rawValue).joined(
                                separator: ", "))
                }
                reasoningLevel = parsed
            case "--models-dir":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--models-dir must not be empty")
                }
                modelsDirectory = value
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                    RuntimeLimits.allowedExpertCacheSlots.contains(parsed)
                else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots must be one of \(RuntimeLimits.allowedExpertCacheSlots)"
                    )
                }
                expertCacheSlots = parsed
            case "--ram-budget":
                guard let parsed = RuntimeLimits.parseBudgetBytes(value) else {
                    throw ServerArgumentError.invalid(
                        "--ram-budget must be a positive size such as 4G, 512M or a byte count")
                }
                // Below 4 GiB the process cannot honour the number: the resident
                // weights plus the 8-slot minimum cache are already ~4.7 GiB on a
                // Qwen3.8 4-bit install. Refuse rather than accept a target that
                // silently overshoots by 2x.
                guard parsed >= RuntimeLimits.minimumProcessTargetBytes else {
                    throw ServerArgumentError.invalid(
                        "--ram-budget must be at least "
                            + "\(RuntimeLimits.minimumProcessTargetBytes >> 30)G: a "
                            + "streaming install holds its weights plus a minimum expert cache "
                            + "beside them, which is about 4.7G for Qwen3.8 4-bit")
                }
                expertCacheBudgetBytes = parsed
            case "--idle-unload-seconds":
                guard let parsed = Int(value), (0...86_400).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--idle-unload-seconds must be between 0 and 86400")
                }
                idleUnloadSeconds = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        if catalogOnly, modelsDirectory == nil {
            throw ServerArgumentError.invalid("--catalog requires --models-dir")
        }
        // Listing the catalog loads nothing, so it needs no initial model.
        guard let model = model ?? (catalogOnly ? "" : nil) else {
            throw ServerArgumentError.invalid("--model is required")
        }
        if reasoningLevel != nil, thinkingWasSet || effortWasSet {
            throw ServerArgumentError.invalid(
                "--reasoning replaces --thinking and --reasoning-effort; pass one or the other")
        }
        if modelsDirectory != nil, !catalogOnly {
            // Each of these configures one model; with a catalog there is no
            // single model for it to mean.
            let conflicts: [(present: Bool, why: String)] = [
                (modelIDOverride != nil, "--model-id: every catalog model keeps its own id"),
                (mtpModel != nil, "--mtp-model: a draft head belongs to one model"),
                (cpu, "--cpu: the catalog knows which engine each model uses"),
                (
                    idleUnloadSeconds > 0,
                    "--idle-unload-seconds: POST /v1/models/unload releases the resident model"
                ),
            ]
            if let conflict = conflicts.first(where: \.present) {
                throw ServerArgumentError.invalid(
                    "--models-dir cannot be combined with \(conflict.why)")
            }
        }
        if ropeScalingMode == .yarn {
            if !maxContextWasSet {
                maxContext = RuntimeLimits.defaultYaRNContextTokens
            }
            guard RuntimeLimits.supportedYaRNContextTokens.contains(maxContext) else {
                throw ServerArgumentError.invalid(
                    "YaRN --max-context must be one of \(Self.supportedYaRNContextRungsHelp)")
            }
        } else {
            guard RuntimeLimits.supportedContextTokens.contains(maxContext) else {
                throw ServerArgumentError.invalid(
                    "--max-context must be one of \(Self.supportedContextRungsHelp)")
            }
        }
        if ropeScalingMode == .yarn, mtpModel != nil {
            throw ServerArgumentError.invalid(
                "--mtp-model cannot be combined with --rope-scaling yarn")
        }
        if let effort = reasoningEffort, !thinkingMode.isEnabled {
            throw ServerArgumentError.invalid(
                "--reasoning-effort \(effort.rawValue) requires --thinking on; "
                    + "the chat template ignores effort while thinking is off")
        }
        if let path = calibrationProfile {
            do {
                let profile = try CalibrationProfile.load(
                    fileAt: URL(fileURLWithPath: path))
                // The profile is evidence about one model; peeking the
                // install's identity rejects a profile measured for a
                // different checkpoint before its knobs are applied. A
                // catalog server has no single model to bind to, so the
                // check applies only when `--model` names one.
                let identity: EngineModelMetadata? =
                    try model.isEmpty
                    ? nil
                    : {
                        let modelURL = URL(fileURLWithPath: model)
                        return try EngineFactories.gpuFactory(directoryURL: modelURL)
                            .metadata(modelURL)
                    }()
                let application = try profile.application(
                    for: CalibrationProfile.currentMachine(),
                    contextTokens: maxContext,
                    modelID: identity?.modelID,
                    family: identity?.family,
                    weightBits: identity?.weightBits,
                    explicitPrefillChunkTokens: prefillChunkTokens,
                    explicitExpertCacheSlots: expertCacheSlots)
                if prefillChunkTokens == nil {
                    prefillChunkTokens = application.prefillChunkTokens
                }
                if expertCacheSlots == nil {
                    expertCacheSlots = application.expertCacheSlots
                }
                calibrationLogLine = application.logLine
            } catch {
                throw ServerArgumentError.invalid(
                    "--calibration-profile: \(error)")
            }
        }
        return ServerArguments(
            model: model,
            mtpModel: mtpModel,
            mtpMemoryMiB: mtpMemoryMiB,
            port: port,
            modelIDOverride: modelIDOverride,
            maxContext: maxContext,
            maxContextExplicit: maxContextWasSet,
            queueLimit: queueLimit,
            maxConcurrentSequences: maxConcurrentSequences,
            promptCacheMode: promptCacheMode,
            promptCacheMaximumEntries: promptCacheMaximumEntries,
            promptCacheMemoryMiB: promptCacheMemoryMiB,
            promptCacheDiskDirectory: promptCacheDiskDirectory,
            promptCacheDiskMiB: promptCacheDiskMiB,
            prefillChunkTokens: prefillChunkTokens,
            calibrationProfile: calibrationProfile,
            calibrationLogLine: calibrationLogLine,
            kvCachePrecision: kvCachePrecision,
            kvCachePrecisionExplicit: kvCachePrecisionExplicit,
            hostKVWindowTokens: hostKVWindowTokens,
            ropeScalingMode: ropeScalingMode,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort,
            cpu: cpu,
            cpuResident: cpuResident,
            expertCacheSlots: expertCacheSlots,
            expertCacheBudgetBytes: expertCacheBudgetBytes,
            lazyLoad: lazyLoad,
            idleUnloadSeconds: idleUnloadSeconds,
            modelsDirectory: modelsDirectory,
            catalogOnly: catalogOnly,
            reasoningLevel: reasoningLevel)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
