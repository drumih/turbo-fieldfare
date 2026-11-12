import Foundation
import Metal
import TinyTitan
import TinyTitanLib

/// `TinyTitanBench verifyprobe <install> [width=8] [dumpDir] [promptFile]`: the
/// wide-verify identity bisect (docs/plan-wide-verify.md, task 3).
///
/// At one boundary deep in an edit prompt, decodes `width` tokens scalar with
/// activation dumps on, restores the state captured before them, and verifies
/// the same tokens as one `verifyGreedyRows` pass with dumps on. Each pass row
/// i lands in `pos{P+i}/` beside scalar decode's dumps for that position:
/// `L{L}_rows_after` (pass) against `L{L+1}_entry` / `stack_out` (scalar) and
/// `verify_logits` against `logits`. `benchmark/verify_identity_bisect.py` reads
/// them and names the first layer where each row leaves scalar decode.
///
/// Loaded with the logits head so scalar decode writes logits to compare.
extension TinyTitanBench {
    static func runVerifyProbe(
        install: String, width: Int, dumpDirectory: String, promptFile: String?
    ) throws {
        /// unchecked-invariant: written only inside the Task below and read
        /// only after `semaphore.wait()` returns; no concurrent access.
        final class Box: @unchecked Sendable { var error: Error? }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                try await runVerifyProbeAsync(
                    install: install, width: width, dumpDirectory: dumpDirectory,
                    promptFile: promptFile)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.error { throw error }
    }

    private static func runVerifyProbeAsync(
        install: String, width: Int, dumpDirectory: String, promptFile: String?
    ) async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw PassCostError.noDevice }
        try FileManager.default.createDirectory(
            atPath: dumpDirectory, withIntermediateDirectories: true)
        // The runner reads the dump directory when it is built.
        setenv("TINYTITAN_ACT_DUMP", dumpDirectory, 1)
        let session = try await ServerModelSession.load(
            modelDirectory: URL(fileURLWithPath: install),
            maxContext: 8192,
            forceLogitsHead: true,
            device: device)
        let runner = session.benchmarkRunner
        // Nothing is dumped until the window is moved onto the probe.
        runner.activationDumpPositionCount = 0
        let prompt = session.benchmarkTokenizer.encode(
            try passCostPrompt(promptFile: promptFile), addBOS: true)
        let installURL = URL(fileURLWithPath: install)
        let vocab = try ArchConfig.resolved(
            forFamily: ManifestReader.peekFamily(directoryURL: installURL),
            directoryURL: installURL
        ).vocabSize
        guard
            let logits = device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride, options: .storageModeShared)
        else { throw PassCostError.noDevice }
        guard width >= 2, width <= runner.wideVerifyMaximumRows else {
            throw PrefillError.chunkedUnsupported(
                "verifyprobe width must be 2...\(runner.wideVerifyMaximumRows)")
        }

        runner.reset()
        let seed = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
        _ = seed
        var token = argmaxHalf(logits, count: vocab)
        var position = prompt.count
        for _ in 0..<32 {
            try await runner.produce(token: token, position: position, into: logits)
            token = argmaxHalf(logits, count: vocab)
            position += 1
        }

        // Scalar decode of the probe window, dumped.
        let snapshot = try runner.captureInferenceState()
        let auxiliary = try runner.captureAuxiliaryState()
        runner.activationDumpFirstPosition = position
        runner.activationDumpPositionCount = width
        var inputs: [Int32] = []
        var scalarNext: [Int32] = []
        var current = token
        for step in 0..<width {
            inputs.append(current)
            try await runner.produce(token: current, position: position + step, into: logits)
            current = argmaxHalf(logits, count: vocab)
            scalarNext.append(current)
        }

        // The same inputs as one verify pass, dumped beside them.
        try runner.restoreInferenceState(snapshot, auxiliary: auxiliary)
        try runner.prepareForContinuation(expectedPosition: position)
        let result = try await runner.verifyGreedyRows(inputs, startPosition: position)
        runner.activationDumpPositionCount = 0

        print("probe boundary position \(position), width \(width), dumps in \(dumpDirectory)")
        print("row  input    scalar   verify  same")
        for row in 0..<width {
            print(
                String(
                    format: "%3d  %6d  %7d  %7d  %@", row, inputs[row], scalarNext[row],
                    result.predictions[row],
                    scalarNext[row] == result.predictions[row] ? "yes" : "NO"))
        }
        print(
            "compare: python3 benchmark/verify_identity_bisect.py \(dumpDirectory) "
                + "\(position) \(width)")
    }

    /// The wide-verify commands (docs/plan-wide-verify.md): `passcost` and
    /// `verifyprobe`. Returns false for any other name.
    static func runWideVerifyCommand(_ name: String) throws -> Bool {
        switch name {
        case "passcost":
            // Gate 2 of docs/plan-wide-verify.md: install, rounds per width,
            // pass path (chunked | dedicated | both), optional prompt file. Loads a model, so it runs before the
            // kernel benches' Metal context like the other model commands.
            let args = CommandLine.arguments
            guard args.count > 2 else {
                throw PassCostError.usage
            }
            let path: PassCostPath
            if args.count > 4 {
                guard let parsed = PassCostPath(rawValue: args[4]) else {
                    throw PassCostError.usage
                }
                path = parsed
            } else {
                path = .chunked
            }
            try runPassCost(
                install: args[2],
                roundsPerWidth: args.count > 3 ? Int(args[3]) ?? 6 : 6,
                path: path,
                promptFile: args.count > 5 ? args[5] : nil)
        case "verifyprobe":
            // The wide-verify identity bisect: install, width, dump directory,
            // optional prompt file. See VerifyProbeCommands.
            let args = CommandLine.arguments
            guard args.count > 2 else {
                print(
                    "usage: TinyTitanBench verifyprobe <install> [width=8] [dumpDir] [promptFile]")
                return true
            }
            try runVerifyProbe(
                install: args[2],
                width: args.count > 3 ? Int(args[3]) ?? 8 : 8,
                dumpDirectory: args.count > 4 ? args[4] : ".build/verify-probe",
                promptFile: args.count > 5 ? args[5] : nil)
        default:
            return false
        }
        return true
    }
}
