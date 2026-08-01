import Darwin
import Foundation
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
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        promptCacheMode: arguments.promptCacheMode)
    let modelID = arguments.modelIDOverride ?? backend.defaultModelID
    let server = TurboFieldfareHTTPServer(
        modelID: modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        chatDialect: backend.chatDialect)
    _ = try await server.start(port: arguments.port)
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue)")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
