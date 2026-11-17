//
//  main.swift
//  TinyTitanLib terminal demo
//
//  The smallest real program that uses the library as a *linked binary* rather
//  than as a SwiftPM dependency. The same source builds two ways:
//
//      ./build-static.sh    -> tinytitan-demo-static   (libTinyTitanLib.a)
//      ./build-dynamic.sh   -> tinytitan-demo-dynamic  (libTinyTitanLib.dylib)
//
//  The only difference between the two apps is the link step, which is the
//  point of having both: everything below is the library's public API.
//
//  Run it against an installed model:
//
//      ./tinytitan-demo-static --model models/qwen3.5_9B_4Bit
//      ./tinytitan-demo-dynamic --model models/qwen3.5_9B_4Bit
//
//  The build scripts copy the resource bundles (and, for the dynamic app, the
//  dylib) next to the executable, because the runtime resolves its Metal
//  shaders relative to the running binary.
//

import Darwin
import Foundation
import Metal
import TinyTitanLib
import TinyTitanLegacyEngine

// Which library form this binary was linked against. The build scripts define
// exactly one of these, so the demo can say what it is instead of guessing.
#if STATIC_LINK
    let linkForm = "static  — linked from libTinyTitanLib.a"
#elseif DYNAMIC_LINK
    let linkForm = "dynamic — linked from libTinyTitanLib.dylib"
#else
    let linkForm = "unknown — build with build-static.sh or build-dynamic.sh"
#endif

let defaultPrompt = "difference swift vs c++ in detail"
let defaultModel = "models/qwen3.5_9B_4Bit"

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("demo: \(message)\n".utf8))
    exit(1)
}

/// Framing goes to stderr and the answer goes to stdout, the same split the CLI
/// uses: `demo > answer.txt` should leave you a file of generated text, and the
/// library's own diagnostics travel on stderr beside this.
func status(_ line: String) {
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

func usage() -> Never {
    print(
        """
        TinyTitanLib demo — \(linkForm)

        usage: \(CommandLine.arguments.first ?? "demo") [options]

          --model <dir>        a .ssdai install (default: \(defaultModel))
          --prompt <text>      the prompt (default: "\(defaultPrompt)")
          --max-tokens <n>     how many tokens to generate (default: 256)
          --temperature <f>    override the model's own temperature (0 = greedy)
          --repetition-penalty <f>
                               the engine's default is 1.0; about 1.15 keeps
                               this model from repeating itself on long answers
          -h, --help           this text
        """)
    exit(0)
}

// Unbuffered stdout: a streaming demo should show tokens as they arrive, not in
// blocks when a buffer happens to fill.
setvbuf(stdout, nil, _IONBF, 0)

var modelDirectory = defaultModel
var prompt = defaultPrompt
/// Long enough to answer a "compare these in detail" question, short enough to
/// stay clear of the point where this model starts repeating itself (see the
/// repetition penalty below).
var maxTokens = 256
/// `nil` means "whatever this model's own sampling row says", which is what the
/// CLI uses and what the engine exposes through `samplingDefaults`.
var temperature: Double?
var repetitionPenalty = 1.0

var arguments = CommandLine.arguments.dropFirst().makeIterator()
while let argument = arguments.next() {
    switch argument {
    case "--model":
        guard let value = arguments.next() else { fail("--model needs a directory") }
        modelDirectory = value
    case "--prompt":
        guard let value = arguments.next() else { fail("--prompt needs text") }
        prompt = value
    case "--max-tokens":
        guard let value = arguments.next(), let parsed = Int(value), parsed > 0 else {
            fail("--max-tokens needs a positive whole number")
        }
        maxTokens = parsed
    case "--temperature":
        guard let value = arguments.next(), let parsed = Double(value) else {
            fail("--temperature needs a number")
        }
        temperature = parsed
    case "--repetition-penalty":
        guard let value = arguments.next(), let parsed = Double(value), parsed > 0 else {
            fail("--repetition-penalty needs a positive number")
        }
        repetitionPenalty = parsed
    case "-h", "--help":
        usage()
    default:
        fail("unknown argument \(argument)")
    }
}

status("TinyTitanLib demo")
status("  library: \(linkForm)")
status("  model:   \(modelDirectory)")
status("  prompt:  \(prompt)")
status("")

guard FileManager.default.fileExists(atPath: modelDirectory) else {
    fail(
        "no install at \(modelDirectory). Pass one with --model, or install one with "
            + "tools/install_models.sh in the source repository.")
}
guard let device = MTLCreateSystemDefaultDevice() else {
    fail("no Metal device on this machine")
}

do {
    // One engine per install: it owns the resident weights, the expert cache and
    // the Metal pipelines, and it is expensive to create.
    let engine = try await Engine(
        directory: URL(fileURLWithPath: modelDirectory),
        device: device,
        factory: EngineFactories.sixXGPU())

    let descriptor = engine.descriptor
    status(
        "  loaded:  \(descriptor.id) [\(descriptor.family)], context "
            + "\(descriptor.contextWindow), \(descriptor.weightBytes / 1_000_000) MB on disk")
    status("")

    // A session is one conversation. Cheap: the model is already resident.
    //
    // The sampling values come from the model itself unless the caller
    // overrides them: every family has its own row (Qwen 3.8 outside thinking
    // mode wants a presence penalty, for one), and the engine exposes it so an
    // embedder does not have to guess. Hardcoding a temperature here is how the
    // demo's first version produced "multi-paradigm, multi-paradigm, compiled,
    // compiled" — the model was fine, the defaults were not.
    let defaults = engine.samplingDefaults
    let activeTemperature = temperature ?? defaults.temperature
    status(
        "  sampling: temperature \(activeTemperature) · top-k \(defaults.topK) · "
            + "top-p \(defaults.topP) · presence penalty \(defaults.presencePenalty) · "
            + "repetition penalty \(repetitionPenalty)")

    let session = await engine.session()
    let clock = Date()
    let summary = try await session.respond(
        to: [ChatMessage(role: .user, content: prompt)],
        options: GenerationOptions(
            maxTokens: maxTokens,
            temperature: activeTemperature,
            topP: defaults.topP,
            topK: defaults.topK,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: defaults.presencePenalty)
    ) { event in
        // The callback is where the tokens arrive. They go to stdout, so
        // redirecting this program gives a file of just the answer.
        if case .token(let text) = event {
            print(text, terminator: "")
        }
    }
    print("")

    let wall = Date().timeIntervalSince(clock)
    let rate =
        summary.decodeSeconds > 0
        ? Double(summary.completionTokens) / summary.decodeSeconds : 0
    status("")
    status(
        "  \(summary.completionTokens) tokens · prompt \(summary.promptTokens) tokens · "
            + "prefill \(String(format: "%.2f", summary.prefillSeconds))s · "
            + "decode \(String(format: "%.2f", summary.decodeSeconds))s · "
            + "\(String(format: "%.2f", rate)) tok/s · "
            + "stop \(summary.decodeStopReason.rawValue) · "
            + "wall \(String(format: "%.2f", wall))s")

    // Release the resident weights. The process is about to exit anyway; an app
    // that keeps running would call this to hand the memory back.
    await engine.unload()
} catch {
    fail("\(error)")
}
