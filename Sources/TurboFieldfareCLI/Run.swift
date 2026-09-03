import Foundation
import Metal
import TurboFieldfare

/// One row of the `--messages-file` JSON.
private struct MessageJSON: Decodable {
    let role: String
    let content: MessageContentJSON
}

private enum MessageContentJSON: Decodable {
    case text(String)
    case parts([MessagePartJSON])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([MessagePartJSON].self))
        }
    }
}

private struct MessagePartJSON: Decodable {
    let type: String
    let text: String?
    let path: String?
}

enum PromptInput {
    case raw(String)
    case messages([GFTokenizer.Message])
    case multimodal(messages: [MultimodalMessage], images: [UUID: URL])

    /// Whether this prompt carries images. `--messages-file` reaches the image
    /// path with `args.images` empty, so the runtime checks that depend on
    /// images ask the parsed input, not the flag list.
    var hasImages: Bool {
        if case .multimodal = self { return true }
        return false
    }
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let input = try parseInput(args: args)
        var selectedDevice: MTLDevice?
        if input.hasImages {
            guard let device = MetalContext.makeSystemDefaultDevice() else {
                return errored(stderr, "no Metal device", 1)
            }
            try VisionRuntime.requireSupportedDevice(device)
            selectedDevice = device
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        var promptIds: [Int32]
        var multimodalMessages: [MultimodalMessage]?
        var imageURLs: [UUID: URL] = [:]
        switch input {
        case .raw(let text):
            multimodalMessages = nil
            promptIds = tokenizer.encode(text, addBOS: true)
        case .messages(let messages):
            multimodalMessages = nil
            promptIds = tokenizer.encode(
                try tokenizer.applyChatTemplate(messages), addBOS: false)
        case .multimodal(let messages, let images):
            multimodalMessages = messages
            imageURLs = images
            // Filled in after the tower runs; the renderer needs the projected
            // features to lay out the placeholder spans.
            promptIds = []
        }
        if multimodalMessages == nil {
            guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        }
        guard multimodalMessages != nil || promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        func makeConfig(maxNewTokens: Int) -> GenerationConfig {
            GenerationConfig(
                maxNewTokens: maxNewTokens,
                temperature: args.temperature,
                topK: args.topK,
                topP: args.topP,
                repetitionPenalty: args.repetitionPenalty,
                seed: args.seed,
                stopStrings: args.stops,
                extraStopTokens: [])
        }
        // Hoisted above the `auto` estimate, which needs a device to read image
        // geometry. Planning never touches the GPU, but building the plan does
        // need the device the run will use.
        guard let device = selectedDevice ?? MetalContext.makeSystemDefaultDevice() else {
            return errored(stderr, "no Metal device", 1)
        }

        var effectiveArgs = args
        if args.resolvesPrefillChunkAuto {
            let promptTokens = try estimatedPromptTokens(
                input: input, tokenizer: tokenizer, device: device)
            effectiveArgs.prefillChunkTokens =
                PrefillRuntimeConfig.autoChunkTokens(promptTokens: promptTokens)
            if !args.quiet {
                let line = "[prefill chunk auto: "
                    + "\(effectiveArgs.prefillChunkTokens) tokens for a "
                    + "\(promptTokens)-token prompt]\n"
                stderr.write(Data(line.utf8))
            }
        }
        let args = effectiveArgs
        let runtime = try args.resolvedRuntimeConfiguration(
            forceLogitsHead: !makeConfig(maxNewTokens: args.maxNew).isPureGreedy,
            imagePrompt: input.hasImages)

        if !args.quiet,
           let notice = prefillCoercionNotice(
            hasImages: input.hasImages, config: runtime.prefillConfig) {
            stderr.write(Data((notice + "\n").utf8))
        }

        // An image prompt that cannot fit is decided by geometry, so decide it
        // before spending a model load and a GPU encode on it. The plan reads
        // metadata only, and the same plan is reused at encode time so each
        // image is opened and parsed once.
        var imagePlans: [UUID: VisionImagePlan] = [:]
        if let multimodalMessages {
            // Dictionary order depends on a per-process hash seed, so planning
            // straight from `imageURLs` named a different image on each run when
            // two were bad. The prompt's own part order is the stable one.
            let ordered = orderedImageIDs(messages: multimodalMessages, images: imageURLs)
            // The device from the guard above, not a fresh `MetalContext`:
            // building one compiles every shader module, and planning never
            // touches the GPU.
            let preprocessor = Gemma4ImagePreprocessor(device: device, config: VisionConfig())
            var projected = 0
            for id in ordered {
                guard let url = imageURLs[id] else { continue }
                let plan = try preprocessor.plan(fileURL: url)
                imagePlans[id] = plan
                projected += plan.geometry.softTokenCount
                    + VisionImageTokenBudget.markerTokensPerImage
            }
            // The text counts too. The tokenizer is already loaded, so this is
            // free and it closes the common case: a prompt whose text overflows
            // used to be refused only after the pack opened and every image had
            // been encoded. A lower bound - the template's framing tokens are
            // not counted - so it can only refuse what could never fit.
            for message in multimodalMessages {
                for part in message.content {
                    if case .text(let text) = part {
                        projected += tokenizer.encode(text, addBOS: false).count
                    }
                }
            }
            guard projected < args.maxContext else {
                return errored(
                    stderr,
                    "context overflow: prompt needs at least \(projected) tokens, "
                        + "which reaches maxContext \(args.maxContext)",
                    2)
            }
        }

        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        let multimodalInput: MultimodalPrefillInput?
        if let multimodalMessages {
            let vision = try VisionRuntime.open(
                textModelURL: modelURL,
                context: context,
                visionPackURL: args.visionPack.map { URL(fileURLWithPath: $0) })
            var features: [UUID: VisionFeatures] = [:]
            features.reserveCapacity(imageURLs.count)
            let visionStarted = ContinuousClock.now
            for id in orderedImageIDs(messages: multimodalMessages, images: imageURLs) {
                guard let url = imageURLs[id] else { continue }
                try Task.checkCancellation()
                if let plan = imagePlans[id] {
                    features[id] = try vision.encodeImage(
                        plan: plan,
                        languageModel: model,
                        residencyPolicy: args.visionResidency,
                        checkCancellation: { try Task.checkCancellation() })
                } else {
                    features[id] = try vision.encodeImage(
                        at: url,
                        languageModel: model,
                        residencyPolicy: args.visionResidency,
                        checkCancellation: { try Task.checkCancellation() })
                }
            }
            if !args.quiet {
                let duration = visionStarted.duration(to: .now)
                let seconds = Double(duration.components.seconds)
                    + Double(duration.components.attoseconds) / 1e18
                stderr.write(Data(
                    "[vision images=\(imageURLs.count) encode=\(String(format: "%.3f", seconds))s]\n".utf8))
            }
            let rendered = try MultimodalPromptRenderer.render(
                messages: multimodalMessages,
                featuresByID: features,
                tokenizer: tokenizer)
            promptIds = rendered.effectiveTokenIDs
            multimodalInput = rendered
            guard promptIds.count < args.maxContext else {
                return errored(
                    stderr,
                    "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                    2)
            }
        } else {
            multimodalInput = nil
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = makeConfig(maxNewTokens: effectiveMaxNew)

        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            multimodalInput: multimodalInput,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch let error as ArgsError {
        return errored(stderr, "\(error)", 2)
    } catch let error as VisionImageError {
        return errored(stderr, "\(error)", 3)
    } catch let error as VisionPackError {
        return errored(stderr, "\(error)", 4)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}


/// The order the messages reference their images, so a repeated command plans
/// and encodes them the same way every run.
func orderedImageIDs(messages: [MultimodalMessage],
                     images: [UUID: URL]) -> [UUID] {
    var ordered: [UUID] = []
    var seen: Set<UUID> = []
    for message in messages {
        for part in message.content {
            if case .image(let id) = part, images[id] != nil, seen.insert(id).inserted {
                ordered.append(id)
            }
        }
    }
    for id in images.keys.sorted(by: { $0.uuidString < $1.uuidString })
    where !seen.contains(id) {
        ordered.append(id)
    }
    return ordered
}

/// The line to print when an image prompt overrides the requested prefill mode.
///
/// The override itself happens deep in `RawCompletion`, silently, while
/// `RUNTIME_CONTROLS.md` says `--prefill off` disables that path. A flag that is
/// ignored without a word is worse than one that is refused, so the CLI says it
/// before the model loads.

/// A prompt-length estimate good enough to choose a chunk size.
///
/// Deliberately cheap: it tokenises text and reads image geometry, and never
/// loads the model. Being a little low only costs one extra chunk.
func estimatedPromptTokens(
    input: PromptInput,
    tokenizer: GFTokenizer,
    device: MTLDevice
) throws -> Int {
    switch input {
    case .raw(let text):
        return tokenizer.encode(text, addBOS: true).count
    case .messages(let messages):
        return tokenizer.encode(try tokenizer.applyChatTemplate(messages),
                                addBOS: false).count
    case .multimodal(let messages, let images):
        var total = 0
        for message in messages {
            for part in message.content {
                if case .text(let text) = part {
                    total += tokenizer.encode(text, addBOS: false).count
                }
            }
        }
        let preprocessor = Gemma4ImagePreprocessor(
            device: device, config: VisionConfig())
        for url in images.values {
            // Not `try?`: an image that cannot be planned would count as zero
            // soft tokens, so `auto` would size the chunk for the text alone and
            // put the multimodal prompt back on the many-chunk path this flag
            // exists to avoid. An unplannable image fails the run anyway, and
            // failing here says why, before the load.
            total += try preprocessor.plan(fileURL: url).geometry.softTokenCount
            total += VisionImageTokenBudget.markerTokensPerImage
        }
        return total
    }
}

func prefillCoercionNotice(
    hasImages: Bool, config: PrefillRuntimeConfig
) -> String? {
    guard hasImages, config.coercedForImagePrompt() != nil else { return nil }
    return "[prefill coerced to chunked: image prompts require it]"
}

func parseInput(args: Args) throws -> PromptInput {
    if let prompt = args.prompt {
        return .raw(prompt)
    }
    if let chatPrompt = args.chatPrompt {
        guard !args.images.isEmpty else {
            return .messages([GFTokenizer.Message(role: .user, content: chatPrompt)])
        }
        var images: [UUID: URL] = [:]
        var content: [MultimodalContentPart] = []
        for path in args.images {
            let id = UUID()
            images[id] = URL(fileURLWithPath: path)
            content.append(.image(id: id))
        }
        if !chatPrompt.isEmpty { content.append(.text(chatPrompt)) }
        return .multimodal(
            messages: [MultimodalMessage(role: .user, content: content)],
            images: images)
    }
    guard let path = args.messagesFile else {
        throw ArgsError.modeMissing
    }
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
    let base = url.deletingLastPathComponent()
    var images: [UUID: URL] = [:]
    let messages: [MultimodalMessage] = try rows.map { row in
        guard let role = GFTokenizer.Role(rawValue: row.role) else {
            throw ArgsError.invalidValue(flag: "--messages-file", value: "unknown role: \(row.role)")
        }
        let parts: [MultimodalContentPart]
        switch row.content {
        case .text(let text):
            parts = [.text(text)]
        case .parts(let rows):
            parts = try rows.map { part in
                switch part.type {
                case "text":
                    guard let text = part.text else {
                        throw ArgsError.invalidValue(
                            flag: "--messages-file", value: "text part is missing text")
                    }
                    return .text(text)
                case "image_file":
                    guard role == .user, let path = part.path else {
                        throw ArgsError.invalidValue(
                            flag: "--messages-file", value: "image_file requires a user role and path")
                    }
                    let id = UUID()
                    // Test the string, not a URL: `URL(fileURLWithPath:)`
                    // resolves against the process directory, so its `path` is
                    // always absolute and the relative branch never ran. A
                    // relative path belongs to the messages file's directory.
                    images[id] = path.hasPrefix("/")
                        ? URL(fileURLWithPath: path)
                        : base.appendingPathComponent(path)
                    return .image(id: id)
                default:
                    throw ArgsError.invalidValue(
                        flag: "--messages-file", value: "unknown content type: \(part.type)")
                }
            }
        }
        return MultimodalMessage(role: role, content: parts)
    }
    if images.isEmpty {
        return .messages(messages.map { message in
            GFTokenizer.Message(
                role: message.role,
                content: message.content.compactMap {
                    if case .text(let text) = $0 { text } else { nil }
                }.joined())
        })
    }
    return .multimodal(messages: messages, images: images)
}

/// An image prompt's token count follows from geometry, so an oversized one can
/// be refused in milliseconds instead of after a model load and a GPU encode.
func validatePromptSize(
    tokens: Int,
    args: Args,
    stderr: FileHandle
) -> RunResult? {
    guard tokens + args.maxNew <= args.maxContext else {
        return errored(
            stderr,
            "context overflow: prompt \(tokens) + maxNew \(args.maxNew) exceeds maxContext \(args.maxContext)",
            2)
    }
    return nil
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
