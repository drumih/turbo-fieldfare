import Darwin
import Foundation
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

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

        let client = RealInferenceClient()
        let commands = DecodeCommandQueue()
        let input = Thread {
            do {
                while true {
                    let command = try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handles.input)
                    if case .cancel = command { client.cancel() }
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
            case .generate(let request):
                guard let modelDirectory else {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "model is not loaded"), to: handles.output)
                    continue
                }
                guard request.runtimeOptions == loadedOptions else {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "generation runtime options do not match the loaded session"),
                        to: handles.output)
                    continue
                }

                let outbox = DecodeServiceOutbox(generationID: request.generationID)
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
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    let generation = AppGenerationRequest(
                        modelDirectory: modelDirectory, prompt: request.prompt,
                        maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: options)
                    for try await event in client.generate(generation) {
                        outbox.publish(event)
                    }
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
        let resolved = AppRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: cachePolicy,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: rdadvisePolicy,
            modelVerification: modelVerification)
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
