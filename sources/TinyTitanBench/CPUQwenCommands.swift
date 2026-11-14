//
//  CPUQwenCommands.swift
//  TinyTitanBench
//
//  The Qwen3.5 side-engine commands: the continuation check, perplexity, text
//  generation, batch generation and the tokenizer load they share.
//
//  Split out of `CPUCommands.swift` (2026-09-28) under the 500-line-per-file
//  rule (Task 8 of the cleanup runbook) as pure code motion; `TinyTitanBench.main`
//  still dispatches through `runCPUCommand` in CPUCommands.swift.

import Foundation
import Metal
import TinyTitan
import TinyTitanCore

extension TinyTitanBench {

    /// Qwen3.5-2B on the CPU, checked against the continuations that define
    /// correctness for the numpy reference.
    ///
    /// The token ids come out of the model itself -- its `vocab.json` in a
    /// snapshot, its tokenizer in a `.ssdai` install -- so this cannot drift
    /// from what the reference does.
    static func runCPUQwen35(snapshot path: String, dump: URL? = nil) throws {
        let directory = URL(fileURLWithPath: path)
        let started = ContinuousClock.now
        let snapshot = try Self.loadDenseSnapshot(path)
        // Width is the side-engine's scheduling knob, so it is settable
        // here: the measurement that produced the policy is a sweep of it.
        let requested = ProcessInfo.processInfo.environment["TINYTITAN_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        func seconds(_ from: ContinuousClock.Instant) -> Double {
            let elapsed = from.duration(to: .now)
            return Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
        }
        print(
            "\(path): \(snapshot.configuration.layers) layers, "
                + "hidden \(snapshot.configuration.hiddenSize), "
                + "rotary \(snapshot.configuration.rotaryDim)/"
                + "\(snapshot.configuration.headDim), "
                + "loaded in \(String(format: "%.2fs", seconds(started)))")
        print("threads: \(model.threads)")

        let vocabulary = try Self.denseVocabulary(path, directory: directory)

        let checks: [([String], String)] = [
            (["Once", "\u{120}upon", "\u{120}a"], "\u{120}time"),
            (
                ["The", "\u{120}capital", "\u{120}of", "\u{120}France", "\u{120}is"],
                "\u{120}Paris"
            ),
            (
                [
                    "The", "\u{120}quick", "\u{120}brown", "\u{120}fox", "\u{120}jumps",
                    "\u{120}over", "\u{120}the", "\u{120}lazy",
                ], "\u{120}dog"
            ),
        ]
        var failures = 0
        for (index, (words, expected)) in checks.enumerated() {
            model.reset()
            var logits: [Float] = []
            let run = ContinuousClock.now
            for word in words {
                guard let id = vocabulary.idFor(word) else {
                    print("  no token for \(word)")
                    failures += 1
                    break
                }
                logits = try model.step(token: id)
            }
            guard !logits.isEmpty else { continue }
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            let want = vocabulary.idFor(expected) ?? -1
            let ok = best == want
            failures += ok ? 0 : 1
            let prompt = words.map { $0.replacingOccurrences(of: "\u{120}", with: " ") }
                .joined()
            let rate = Double(words.count) / seconds(run)
            print(
                String(
                    format: "  %@ %-46@ -> %@ (%.2f), wanted %@  [%.1f tok/s]",
                    ok ? "ok " : "FAIL", prompt as NSString,
                    vocabulary.textFor(best), logits[best],
                    vocabulary.labelFor(expected), rate))
            if let dump {
                try? FileManager.default.createDirectory(
                    at: dump, withIntermediateDirectories: true)
                let file = dump.appendingPathComponent("check\(index).f32")
                let payload = logits.withUnsafeBufferPointer { Data(buffer: $0) }
                try? payload.write(to: file)
            }
        }
        print(
            failures == 0
                ? "all continuations correct"
                : "\(failures) of \(checks.count) wrong")
        // Exit non-zero on a mismatch. Without this the process returned 0
        // after printing "N of 3 wrong", so a scripted run -- and this command
        // exists to be scripted -- read a dead forward pass as a pass. The
        // whole point of `cpu35` is to be the check that says the Swift forward
        // pass matches the oracle.
        exit(failures == 0 ? 0 : 1)
    }

    /// Held-out text through the CPU forward pass, as mean negative
    /// log-likelihood and perplexity.
    ///
    /// The twenty-prompt A/B (`benchmark/quant_quality_ab.py`) is a floor: a
    /// short answerable prompt is a coarse instrument and a small perplexity
    /// difference sits below it. This is the sharper one. `step` returns the
    /// logits over the whole vocabulary for every position, so scoring a fixed
    /// text needs no sampling and no generation — one pass per install over
    /// the *same* tokens, and the difference between two quantizations of one
    /// model becomes a paired number rather than a pass/fail.
    ///
    ///     TinyTitanBench cpu35ppl <snapshot> <text-file> [maxTokens] [nll-out]
    ///
    /// `nll-out`, when given, is one negative log-likelihood per scored token,
    /// so a caller can compare two installs position by position instead of
    /// comparing two means.
    static func runCPUQwen35Perplexity(
        snapshot path: String,
        text: URL,
        maximumTokens: Int,
        nllOutput: URL?
    ) throws {
        let directory = URL(fileURLWithPath: path)
        let snapshot = try Self.loadDenseSnapshot(path)
        let requested = ProcessInfo.processInfo.environment["TINYTITAN_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        guard let tokenizer = try loadTokenizer(directory) else {
            FileHandle.standardError.write(Data("no tokenizer in \(path)\n".utf8))
            exit(2)
        }
        // lint:allow-unbounded-read the corpus is the operator's own `--text` file,
        // named on the command line of the run they started; nothing loads it on
        // their behalf, and scoring it is the benchmark's whole point.
        let body = try String(contentsOf: text, encoding: .utf8)
        var ids = tokenizer.encode(body, addBOS: false).map(Int.init)
        if ids.count > maximumTokens { ids = Array(ids.prefix(maximumTokens)) }
        guard ids.count >= 2 else {
            FileHandle.standardError.write(Data("text is too short to score\n".utf8))
            exit(2)
        }

        model.reset()
        var nlls: [Double] = []
        nlls.reserveCapacity(ids.count - 1)
        let started = ContinuousClock.now
        var previous = ids[0]
        for index in 1..<ids.count {
            let logits = try model.step(token: previous)
            nlls.append(Self.negativeLogLikelihood(logits, target: ids[index]))
            previous = ids[index]
        }
        let elapsed = started.duration(to: .now)
        let seconds =
            Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18

        let mean = nlls.reduce(0, +) / Double(nlls.count)
        // A token hash, so two runs can be shown to have scored the same text
        // rather than merely the same file name.
        var tokenHash: UInt64 = 0xcbf2_9ce4_8422_2325
        for id in ids {
            tokenHash =
                (tokenHash ^ UInt64(UInt32(truncatingIfNeeded: id)))
                &* 0x0000_0100_0000_01b3
        }
        print(
            String(
                format: "%@: %d tokens scored, threads %d, token hash %016llx",
                path as NSString, nlls.count, model.threads, tokenHash))
        print(
            String(
                format: "mean nll %.6f  perplexity %.6f  seconds %.1f  (%.1f tok/s)",
                mean, exp(mean), seconds, Double(nlls.count) / seconds))
        if let nllOutput {
            let lines = nlls.map { String(format: "%.6f", $0) }.joined(separator: "\n")
            try (lines + "\n").write(to: nllOutput, atomically: true, encoding: .utf8)
        }
    }

    /// `-log softmax(logits)[target]`, computed with the max subtracted so a
    /// logit above ~88 does not overflow `exp`. Returns infinity for a target
    /// outside the vocabulary, which cannot happen for a token the same
    /// tokenizer produced but is not worth a crash if it ever does.
    static func negativeLogLikelihood(_ logits: [Float], target: Int) -> Double {
        guard logits.indices.contains(target) else { return .infinity }
        var peak = -Float.infinity
        for value in logits where value > peak { peak = value }
        guard peak.isFinite else { return .infinity }
        var total = 0.0
        for value in logits { total += Double(expf(value - peak)) }
        return Double(peak) + log(total) - Double(logits[target])
    }

    /// The side-engine answering in text, which is what everything above was
    /// for. The tokenizer is the engine's own, loaded straight out of the
    /// snapshot the converter wrote.
    static func runCPUQwen35Generation(
        snapshot path: String,
        prompt: String,
        limit: Int
    ) throws {
        let directory = URL(fileURLWithPath: path)
        let snapshot = try Self.loadDenseSnapshot(path)
        let requested = ProcessInfo.processInfo.environment["TINYTITAN_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        guard let tokenizer = try loadTokenizer(directory) else {
            // Non-zero: a run that never checked anything is not a pass.
            FileHandle.standardError.write(Data("no tokenizer in \(path)\n".utf8))
            exit(2)
        }
        let ids = tokenizer.encode(prompt, addBOS: false).map(Int.init)
        print(
            "prompt: \(prompt.debugDescription) -> \(ids.count) tokens, "
                + "threads \(model.threads)")
        let started = ContinuousClock.now
        let produced = try model.generate(
            prompt: ids, maximumTokens: limit,
            stopping: [Int(tokenizer.eosID)])
        let elapsed = started.duration(to: .now)
        let seconds =
            Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        print("output: " + tokenizer.decode(produced.map(Int32.init)).debugDescription)
        print(
            String(
                format: "%d prompt + %d generated in %.1fs (%.1f tok/s)",
                ids.count, produced.count, seconds,
                Double(ids.count + produced.count) / seconds))
    }

    /// Run a file of prompts through the side-engine.
    ///
    /// The model loads once and the session resets between prompts, which is
    /// the shape every experiment wants and the shape a resident service
    /// will have: two gigabytes mapped once, then many short jobs.
    static func runCPUQwen35Batch(
        snapshot path: String,
        input: URL,
        output: URL
    ) throws {
        let directory = URL(fileURLWithPath: path)
        let snapshot = try Self.loadDenseSnapshot(path)
        let requested = ProcessInfo.processInfo.environment["TINYTITAN_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        let tokenizer = try loadTokenizer(directory)
        guard let tokenizer else {
            // Non-zero: a run that never checked anything is not a pass.
            FileHandle.standardError.write(Data("no tokenizer in \(path)\n".utf8))
            exit(2)
        }

        // lint:allow-unbounded-read as in `runCPUQwen35Perplexity`: one line per
        // prompt in the operator's own `--input` file, read to be scored.
        let lines = try String(contentsOf: input, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: true)
        var results: [String] = []
        let started = ContinuousClock.now
        var tokens = 0
        for (index, line) in lines.enumerated() {
            guard let data = line.data(using: .utf8),
                let job = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let prompt = job["prompt"] as? String
            else { continue }
            let limit = (job["max"] as? Int) ?? 64
            model.reset()
            // `chat` renders the model's own template, which an
            // instruction-tuned model needs to answer rather than continue.
            // Raw continuation stays the default: the parity checks depend
            // on it.
            let rendered: String
            if (job["chat"] as? Bool) == true {
                var messages: [GFTokenizer.Message] = []
                if let system = job["system"] as? String {
                    messages.append(GFTokenizer.Message(role: .system, content: system))
                }
                messages.append(GFTokenizer.Message(role: .user, content: prompt))
                rendered = (try? tokenizer.applyChatTemplate(messages)) ?? prompt
            } else {
                rendered = prompt
            }
            let ids = tokenizer.encode(rendered, addBOS: false).map(Int.init)
            let produced = try model.generate(
                prompt: ids, maximumTokens: limit,
                stopping: [Int(tokenizer.eosID)])
            tokens += ids.count + produced.count
            var record = job
            record["completion"] = tokenizer.decode(produced.map(Int32.init))
            record["prompt_tokens"] = ids.count
            record["completion_tokens"] = produced.count
            let encoded = try JSONSerialization.data(withJSONObject: record)
            results.append(encoded.lossyUTF8String)
            if (index + 1) % 10 == 0 {
                FileHandle.standardError.write(Data("  \(index + 1)/\(lines.count)\n".utf8))
            }
        }
        try results.joined(separator: "\n").appending("\n").write(
            to: output, atomically: true, encoding: .utf8)
        let elapsed = started.duration(to: .now)
        let seconds =
            Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        print(
            String(
                format: "%d prompts, %d tokens in %.1fs (%.1f tok/s) -> %@",
                results.count, tokens, seconds, Double(tokens) / seconds,
                output.path as NSString))
    }

    /// GFTokenizer loads asynchronously and these commands are one-shot
    /// tools, so they wait rather than restructuring `main` around it.
    ///
    /// The folder resolution is the shared one, so a shipped `.ssdai` install
    /// (tokenizer in a `tokenizer/` sidecar) and a flat HF snapshot
    /// (`tokenizer.json` at the top level) are both reached without a second
    /// copy of that rule living here.
    static func loadTokenizer(_ directory: URL) throws -> GFTokenizer? {
        // unchecked-invariant: written only inside the Task below and read
        // only after `semaphore.wait()` returns, which the signal orders after
        // the last write. There is no concurrent access.
        final class Box: @unchecked Sendable { var value: GFTokenizer? }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        guard let folder = GFTokenizer.resolvedTokenizerFolder(forModelDirectory: directory) else {
            return nil
        }
        Task {
            do {
                box.value = try await GFTokenizer.load(from: folder)
            } catch {
                FileHandle.standardError.write(Data("tokenizer load failed: \(error)\n".utf8))
            }
            semaphore.signal()
        }
        semaphore.wait()
        return box.value
    }
}
