import Darwin
import Foundation
import TinyTitanCLICore

let rawArgv = Array(CommandLine.arguments.dropFirst())
if rawArgv.first == "calibrate" {
    do {
        let options = try CalibrationOptions.parse(Array(rawArgv.dropFirst()))
        exit(try await CalibrationCommand.run(options))
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
        FileHandle.standardError.write(Data(CalibrationOptions.usage.utf8))
        FileHandle.standardError.write(Data("\n".utf8))
        exit(2)
    }
}
if rawArgv.first == "doctor" {
    do {
        let options = try DoctorOptions.parse(Array(rawArgv.dropFirst()))
        exit(try DoctorCommand.run(options))
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
        FileHandle.standardError.write(Data(DoctorOptions.usage.utf8))
        FileHandle.standardError.write(Data("\n".utf8))
        exit(2)
    }
}
if rawArgv.first == "catalog" {
    do {
        let options = try CatalogOptions.parse(Array(rawArgv.dropFirst()))
        exit(CatalogCommand.run(options))
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
        FileHandle.standardError.write(Data(CatalogOptions.usage.utf8))
        FileHandle.standardError.write(Data("\n".utf8))
        exit(2)
    }
}
if rawArgv.first == "plugin-check" {
    do {
        let options = try PluginCheckOptions.parse(Array(rawArgv.dropFirst()))
        exit(PluginCheckCommand.run(options))
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
        FileHandle.standardError.write(Data(PluginCheckOptions.usage.utf8))
        FileHandle.standardError.write(Data("\n".utf8))
        exit(2)
    }
}
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
/// unchecked-invariant: a hand-off box between the top-level task and the
/// blocking signal wait, published exactly once before the semaphore is
/// signalled and read only after -- the semaphore is the ordering.
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
