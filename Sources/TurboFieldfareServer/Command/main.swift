import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareServerCore

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

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    // .gturbo v2 bundles run the Kimi K3 stack; v1 runs Gemma.
    let bundle = try GTurboBundleProbe.probe(bundleURL: modelURL)
    let backend: any ServerInferenceBackend
    let requestValidation: ServerRequestValidation
    let modelID: String
    let integrityPolicy: ModelIntegrityPolicy = arguments.modelVerification == "trusted-install"
        ? .sizeCheckTrustedReceipt
        : .fullSha256
    if bundle.isK3 {
        let prefetchPolicy: K3ExpertPrefetchPolicy
        switch arguments.expertPredict {
        case "selective": prefetchPolicy = .selective
        case "on", "full": prefetchPolicy = .predict
        default: prefetchPolicy = .off
        }
        let ioWorkers: K3ExpertIOWorkers = arguments.expertIOWorkers == "auto"
            ? .adaptive
            : .fixed(Int(arguments.expertIOWorkers)!)
        backend = try K3ServerModelSession.load(
            bundleURL: modelURL,
            maxContext: arguments.maxContext,
            prefillMode: .chunked(chunkTokens: arguments.prefillChunk),
            promptCacheMode: arguments.promptCacheMode,
            prefetchPolicy: prefetchPolicy,
            residentExpertCacheBytes: UInt64(arguments.expertCacheGiB) << 30,
            expertShardRoots: arguments.expertShardRoots.map {
                URL(fileURLWithPath: $0).standardizedFileURL
            },
            ioWorkers: ioWorkers,
            ioSplits: arguments.expertIOSplits,
            ioCachePolicy: K3ExpertIOCachePolicy(
                rawValue: arguments.expertIOCache)!,
            integrityPolicy: integrityPolicy)
        requestValidation = .k3
        modelID = arguments.modelIDExplicit ? arguments.modelID : bundle.modelID
    } else {
        guard arguments.maxContext <= 65_536 else {
            throw ServerArgumentError.invalid(
                "--max-context above 65536 is supported only for K3 bundles")
        }
        backend = try await ServerModelSession.load(
            modelDirectory: modelURL,
            maxContext: arguments.maxContext,
            promptCacheMode: arguments.promptCacheMode,
            integrityPolicy: integrityPolicy)
        requestValidation = .gemma
        modelID = arguments.modelID
    }
    let server = TurboFieldfareHTTPServer(
        modelID: modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        requestValidation: requestValidation)
    _ = try await server.start(port: arguments.port)
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue) k3_prefill_chunk=\(arguments.prefillChunk) expert_predict=\(arguments.expertPredict) expert_cache_gib=\(arguments.expertCacheGiB) expert_shards=\(arguments.expertShardRoots.count) expert_io=\(arguments.expertIOWorkers)x\(arguments.expertIOSplits)/\(arguments.expertIOCache) verification=\(arguments.modelVerification)")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
