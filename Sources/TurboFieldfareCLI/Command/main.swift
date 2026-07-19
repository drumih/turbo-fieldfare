import Darwin
import Foundation
import TurboFieldfareCLICore

let rawArgv = Array(CommandLine.arguments.dropFirst())
let parsedArgs: Args
do {
    parsedArgs = try Args.parse(rawArgv)
} catch ArgsError.helpRequested {
    print(Args.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
    FileHandle.standardError.write(Data(Args.usage.utf8))
    FileHandle.standardError.write(Data("\n".utf8))
    exit(2)
}

// Keep the cancellable task off the top-level executor so the blocking signal
// bridge cannot prevent it from starting.
final class RunBox: @unchecked Sendable {
    var code: Int32 = 0
    var task: Task<Void, Never>?
}

func drive(_ args: Args) -> Int32 {
    let box = RunBox()
    let sem = DispatchSemaphore(value: 0)
    box.task = Task {
        let result = await run(args: args)
        box.code = result.exitCode
        sem.signal()
    }

    signal(SIGINT, SIG_IGN)
    let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    sigintSource.setEventHandler { box.task?.cancel() }
    sigintSource.resume()

    sem.wait()
    return box.code
}

exit(drive(parsedArgs))
