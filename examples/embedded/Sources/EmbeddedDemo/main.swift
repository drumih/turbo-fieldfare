//
//  main.swift
//  EmbeddedDemo
//
//  The library, used the way another program uses it: depend on `TinyTitanLib`,
//  open an install with an `Engine`, take a `Session` and stream tokens. There
//  is no subprocess and no HTTP anywhere in this file.
//
//  Usage:
//    EmbeddedDemo --model <ssdai install> [--prompt <text>] [--max-tokens <n>]
//
//  Weights do not ship with this repository and the library never downloads
//  them, so a run needs an install from the installer or `tools/install_models.sh`.
//  With no `--model` the demo only proves that the library resolves and links,
//  which is what CI can check — the model store is gitignored.
//

import Foundation
import Metal
import TinyTitanLib
import TinyTitanLegacyEngine

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("EmbeddedDemo: \(message)\n".utf8))
    exit(1)
}

var modelDirectory: String?
var prompt = "The capital of France is"
var maxTokens = 16

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
        guard let value = arguments.next(), let parsed = Int(value) else {
            fail("--max-tokens needs a whole number")
        }
        maxTokens = parsed
    case "--help", "-h":
        print("usage: EmbeddedDemo --model <ssdai install> [--prompt <text>] [--max-tokens <n>]")
        exit(0)
    default:
        fail("unknown argument \(argument)")
    }
}

guard let modelDirectory else {
    print("EmbeddedDemo: TinyTitanLib linked (\(String(describing: Engine.self)))")
    print("no --model given: resolution and link verified, nothing loaded")
    exit(0)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fail("no Metal device on this machine, so an install cannot be opened")
}

do {
    let engine = try await Engine(
        directory: URL(fileURLWithPath: modelDirectory),
        device: device,
        factory: EngineFactories.sixXGPU())
    let descriptor = engine.descriptor
    print(
        "loaded \(descriptor.id) [\(descriptor.family)], context \(descriptor.contextWindow), "
            + "\(descriptor.weightBytes / 1_000_000) MB on disk")

    let session = await engine.session()
    let summary = try await session.respond(
        to: [ChatMessage(role: .user, content: prompt)],
        options: GenerationOptions(maxTokens: maxTokens, temperature: 0)
    ) { event in
        if case .token(let text) = event {
            print(text, terminator: "")
        }
    }
    print("")
    print(
        "done: \(summary.completionTokens) completion tokens, "
            + "\(summary.promptTokens) prompt tokens, stop \(summary.stopReason)")
} catch {
    fail("\(error)")
}
