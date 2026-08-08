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
        backend = try K3ServerModelSession.load(
            bundleURL: modelURL,
            maxContext: arguments.maxContext,
            promptCacheMode: arguments.promptCacheMode,
            integrityPolicy: integrityPolicy)
        requestValidation = .k3
        modelID = arguments.modelIDExplicit ? arguments.modelID : bundle.modelID
    } else {
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
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue) verification=\(arguments.modelVerification)")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
