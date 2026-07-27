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
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        promptCacheMode: arguments.promptCacheMode)
    let server = TurboFieldfareHTTPServer(
        modelID: arguments.modelID,
        queueLimit: arguments.queueLimit,
        backend: backend)
    let channel = try await server.start(port: arguments.port)
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(arguments.modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue)")

    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    let signals = [SIGINT, SIGTERM].map { value in
        let source = DispatchSource.makeSignalSource(signal: value, queue: .global())
        source.setEventHandler {
            channel.close(promise: nil)
        }
        source.resume()
        return source
    }
    try await channel.closeFuture.get()
    for source in signals { source.cancel() }
    try await server.shutdown()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
