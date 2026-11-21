import Foundation
import TinyTitanCore
import TinyTitanInkling2

/// The text-only Inkling 2 Mini path for the one-shot CLI. The model is a raw
/// HuggingFace directory, not a `.ssdai` install, so it bypasses the engine
/// factory and drives the Inkling generator directly.
enum Inkling2Run {
    static func isInkling2(_ directory: URL) -> Bool {
        let configURL = directory.appendingPathComponent("config.json")
        guard
            let data = try? BoundedMetadataRead.read(
                fileAt: configURL, maxBytes: 1 << 20),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = root["model_type"] as? String
        else { return false }
        return type.hasPrefix("inkling2")
    }

    static func run(
        args: Args, stdout: FileHandle, stderr: FileHandle
    ) -> RunResult {
        do {
            guard let userPrompt = args.prompt else {
                throw Inkling2RunError.missingPrompt
            }
            let directory = URL(fileURLWithPath: args.model)
            let loadStart = Date()
            let config = try Inkling2Config.load(directory: directory)
            let store = try Inkling2ExpertStore(directory: directory)
            let tokenizer = try Inkling2Tokenizer(directory: directory)
            let model = try Inkling2TextModel(
                directory: directory, config: config, store: store)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            var generator = Inkling2Generator(model: model, tokenizer: tokenizer)
            let prompt =
                "<|message_user|><|content_text|>\(userPrompt)<|end_message|>"
                + "<|message_model|><|content_text|>"
            let result = try generator.generate(
                prompt: prompt, maxTokens: args.maxNew)
            stdout.write(Data((result.text + "\n").utf8))
            if Inkling2Profiler.enabled {
                stderr.write(Data((Inkling2Profiler.report() + "\n").utf8))
            }
            if !args.quiet {
                stderr.write(
                    Data(
                        String(
                            format:
                                "inkling: load %.1fs, prefill %.1fs, decode %.1fs, %d tokens (%.3f tok/s)\n",
                            loadSeconds, generator.lastPrefillSeconds,
                            generator.lastDecodeSeconds, result.tokens.count,
                            generator.lastDecodeSeconds > 0
                                ? Double(result.tokens.count) / generator.lastDecodeSeconds : 0
                        ).utf8))
            }
            return RunResult(exitCode: 0)
        } catch {
            stderr.write(Data("error: \(error)\n".utf8))
            return RunResult(exitCode: 1)
        }
    }
}

enum Inkling2RunError: Error, CustomStringConvertible {
    case missingPrompt

    var description: String {
        switch self {
        case .missingPrompt: "--prompt is required for an Inkling 2 model"
        }
    }
}
