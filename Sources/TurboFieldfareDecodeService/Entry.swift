import Darwin
import TurboFieldfare
import Foundation
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

enum DecodeServiceError: Error, CustomStringConvertible {
    case attachmentOutsideStore(path: String)

    var description: String {
        switch self {
        case .attachmentOutsideStore(let path):
            "image attachment is not a staged attachment: \(path)"
        }
    }
}

@main enum TurboFieldfareDecodeServiceMain {
    static func main() async {
        let socketPath = argument(after: "--socket")
        let launchLabel = argument(after: "--launch-label")
        let handles: (input: FileHandle, output: FileHandle)
        do {
            handles = if let socketPath {
                try DecodeUnixSocket.listenAndAccept(path: socketPath)
            } else {
                (.standardInput, .standardOutput)
            }
        } catch {
            FileHandle.standardError.write(Data("Decode service transport failed: \(error)\n".utf8))
            Foundation.exit(1)
        }
        defer {
            if let socketPath { unlink(socketPath) }
            if let launchLabel { retireLaunchJob(launchLabel) }
        }

        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
        let client = RealInferenceClient()
        let commands = DecodeCommandQueue()
        let input = Thread {
            do {
                while true {
                    let command = try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handles.input)
                    if case .cancel = command {
                        // Cooperative: end the turn at the next token boundary
                        // and keep what it produced, so the conversation can
                        // continue from it. Cancelling the task instead throws
                        // out of the decode loop and the turn is rewound.
                        client.stop()
                    }
                    commands.append(command)
                    if case .shutdown = command { break }
                }
            } catch {
                commands.close()
            }
        }
        input.name = "TurboFieldfare.DecodeService.Input"
        input.qualityOfService = .userInitiated
        input.start()

        var modelDirectory: URL?
        var loadedOptions: DecodeRuntimeOptions?
        var conversation = DecodeConversationGate()
        while let command = await nextCommand(commands) {
            switch command {
            case .load(let request):
                let directory = URL(fileURLWithPath: request.modelPath)
                do {
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    try await client.ensureLoaded(
                        modelDirectory: directory,
                        maxContextTokens: request.maxContextTokens,
                        options: options,
                        forceLogitsHead: request.forceLogitsHead) { _ in }
                    modelDirectory = directory
                    loadedOptions = request.runtimeOptions
                    // A load builds a new runner and a new KV, so whatever
                    // lineage was open no longer has tokens behind it.
                    conversation.endLineage()
                    let memory = AppMemorySampler().sample()
                    try write(DecodeServiceEvent(
                        kind: .ready, generationID: request.requestID,
                        currentMemoryBytes: memory, peakMemoryBytes: memory),
                        to: handles.output)
                } catch {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.requestID,
                        error: "\(error)"), to: handles.output)
                }
            case .resetConversation(let request):
                conversation.reset(to: request.epoch)
                await client.resetConversation()
                // Not `try?`. The gate has already reset; if the app never
                // hears so it waits out the whole timeout for a reply that
                // cannot come, and then cannot tell that from a slow service.
                // Closing the stream makes it an EOF the client rebuilds from.
                do { try write(DecodeServiceEvent(
                    kind: .conversationReset, generationID: request.requestID,
                    conversationTokenCount: 0,
                    conversationEpoch: request.epoch), to: handles.output)
                } catch {
                    let message = "Decode service closing after a lost "
                        + "conversation reset: \(error)\n"
                    FileHandle.standardError.write(Data(message.utf8))
                    await client.unload()
                    try? handles.output.close()
                    return
                }
            case .generate(let request):
                guard let modelDirectory else {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "model is not loaded"), to: handles.output)
                    continue
                }
                // Prefill is chosen per request, not at load, so it must not be
                // part of this comparison: toggling it and pressing Generate was
                // refused as a mismatched session.
                var comparable = request.runtimeOptions
                comparable.prefillEnabled = loadedOptions?.prefillEnabled
                    ?? comparable.prefillEnabled
                comparable.prefillChunkTokens = loadedOptions?.prefillChunkTokens
                    ?? comparable.prefillChunkTokens
                guard comparable == loadedOptions else {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "generation runtime options do not match the loaded session"),
                        to: handles.output)
                    continue
                }
                // Fail closed before the model is touched. The decision
                // lives in `DecodeConversationGate`, where its boundary cases
                // are tested without a socket or a model.
                let admission: DecodeConversationGate.Admission
                switch conversation.admit(request) {
                case .success(let value):
                    admission = value
                case .failure(let rejection):
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: rejection.message,
                        conversationEpoch: conversation.openEpoch), to: handles.output)
                    continue
                }
                let isConversationTurn: Bool
                if case .turn = admission { isConversationTurn = true }
                else { isConversationTurn = false }
                let outbox = DecodeServiceOutbox(
                    generationID: request.generationID,
                    towerBytes: { client.currentVisionTowerBytes },
                    conversationTokens: {
                        isConversationTurn ? client.currentConversationTokens : nil
                    })
                let writerFinished = DispatchSemaphore(value: 0)
                let writer = Thread {
                    defer { writerFinished.signal() }
                    do { try outbox.runWriter(to: handles.output) }
                    catch {
                        FileHandle.standardError.write(Data("IPC writer failed: \(error)\n".utf8))
                    }
                }
                writer.name = "TurboFieldfare.DecodeService.Writer"
                writer.qualityOfService = .userInitiated
                writer.start()

                do {
                    // The trust boundary: these paths arrive over a socket, and
                    // this process opens and hashes whatever they name. Without
                    // this, a peer could use the service to report on any file
                    // the user can read.
                    if let outside = (request.imageAttachments ?? []).first(where: {
                        !AppImageAttachmentStore.contains(URL(fileURLWithPath: $0.path))
                    }) {
                        throw DecodeServiceError.attachmentOutsideStore(
                            path: outside.path)
                    }
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    // The conversation already in the KV is what the image
                    // budget has to fit around; reserving zero admits an image
                    // that only fits an empty context.
                    let carried = await client.conversationTokenCount
                    let continues = isConversationTurn
                    let generation = AppGenerationRequest(
                        modelDirectory: modelDirectory, prompt: request.prompt,
                        imageAttachments: (request.imageAttachments ?? []).map {
                            AppImageAttachment(
                                id: $0.id,
                                fileURL: URL(fileURLWithPath: $0.path),
                                displayName: $0.displayName,
                                encodedBytes: $0.encodedBytes,
                                sha256: $0.sha256)
                        },
                        maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        topK: request.topK,
                        topP: request.topP,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: options,
                        continuesConversation: continues,
                        conversationTokens: continues ? carried : 0)
                    for try await event in client.generate(generation) { outbox.publish(event) }
                    // Reached only when the stream completed. A turn that threw
                    // was rewound by the conversation (or broke its lineage), so
                    // its tokens are not in the KV and it must not advance the
                    // order the next turn has to match — the app does not count
                    // it either, and a one-sided count rejects every later turn.
                    // A turn stopped by the user does reach here: it ends at a
                    // token boundary with its partial reply committed.
                    conversation.commit(admission)
                    outbox.finish()
                } catch {
                    outbox.finish(error: error)
                }
                await withCheckedContinuation { continuation in
                    DispatchQueue.global(qos: .userInitiated).async {
                        writerFinished.wait()
                        continuation.resume()
                    }
                }
            case .cancel:
                break
            case .unload(let requestID):
                await client.unload()
                modelDirectory = nil
                loadedOptions = nil
                conversation.endLineage()
                try? write(DecodeServiceEvent(
                    kind: .unloaded, generationID: requestID), to: handles.output)
            case .shutdown:
                await client.unload()
                return
            }
        }
    }

    private static func nextCommand(_ commands: DecodeCommandQueue)
        async -> DecodeServiceCommand? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: commands.next())
            }
        }
    }

    private static func write(_ event: DecodeServiceEvent,
                              to handle: FileHandle) throws {
        try handle.write(contentsOf: DecodeFrameCodec.encode(event))
    }

    private static func appRuntimeOptions(_ options: DecodeRuntimeOptions) throws
        -> AppRuntimeOptions {
        guard let cachePolicy = AppExpertCachePolicy(
            rawValue: options.expertCachePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown expert cache policy \(options.expertCachePolicy)")
        }
        guard let rdadvisePolicy = AppRDAdvicePolicy(
            rawValue: options.rdadvisePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown RDADVISE policy \(options.rdadvisePolicy)")
        }
        guard let modelVerification = AppModelVerification(
            rawValue: options.modelVerification) else {
            throw AppInferenceError.invalidRequest(
                "unknown model verification \(options.modelVerification)")
        }
        // An unknown policy is a request for behaviour that does not exist;
        // absent means the shipped default.
        guard let visionResidencyPolicy = VisionResidencyPolicy(
            rawValue: options.visionResidencyPolicy ?? VisionResidencyPolicy.onDemand.rawValue)
        else {
            throw AppInferenceError.invalidRequest(
                "unknown vision residency policy \(options.visionResidencyPolicy ?? "")")
        }
        let resolved = AppRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: cachePolicy,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: rdadvisePolicy,
            modelVerification: modelVerification,
            visionResidencyPolicy: visionResidencyPolicy)
        try resolved.validate()
        return resolved
    }

    private static func argument(after name: String) -> String? {
        let arguments = CommandLine.arguments
        guard let index = arguments.firstIndex(of: name),
              arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }

    private static func retireLaunchJob(_ label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
    }
}
