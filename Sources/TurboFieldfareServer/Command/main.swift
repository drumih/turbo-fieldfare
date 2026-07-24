import Foundation
import TurboFieldfareServerCore

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let args: ServerArgs
    do {
        args = try ServerArgs.parse(Array(values.dropFirst()))
    } catch ServerArgsError.helpRequested {
        print(ServerArgs.usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(ServerArgs.usage)")
        return 2
    }

    do {
        try await runServer(args: args)
        return 0
    } catch {
        printError("error: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
