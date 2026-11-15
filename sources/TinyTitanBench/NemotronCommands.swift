import Foundation
import Metal
import TinyTitan
import TinyTitanCore

/// `TinyTitanBench nemotron <install> "<prompt>" [n]`: the Nemotron 3.5 Bolt
/// text graph, its tokenizer and timings in one command.
///
/// The 74B-A9B checkpoint is not a Qwen shape, so it does not go through
/// `RealForwardRunner`; it loads its `.ssdai` install directly and runs the
/// Mamba-2 / LatentMoE / attention blocks in `NemotronModel`. Prefill is
/// token-by-token, so the first number is the real TTFT and the second is
/// steady-state decode.
extension TinyTitanBench {
    /// The `nemotron*` command shapes, parsed in one place so the bench's main
    /// switch stays under the function-length gate.
    static func runNemotronCommand(_ arguments: [String]) throws {
        let command = arguments.count > 1 ? arguments[1] : "nemotron"
        let install =
            arguments.count > 2
            ? arguments[2] : "models/nemotron-3.5-bolt_74B_A9B_4Bit"
        switch command {
        case "nemotroninfo":
            try runNemotronInfo(install: install)
        case "nemotronlogits":
            let tokens = arguments.count > 3 ? arguments[3] : "5"
            try runNemotronLogits(
                install: install, tokens: tokens,
                gpu: arguments.contains("--gpu"),
                graph: arguments.contains("--graph"))
        default:
            let prompt =
                arguments.count > 3
                ? arguments[3] : "The capital of France is"
            let limit = arguments.count > 4 ? Int(arguments[4]) ?? 16 : 16
            try runNemotron(
                install: install, prompt: prompt, limit: limit,
                gpu: ["nemotrongpu", "nemotrongraph"].contains(command),
                graph: command == "nemotrongraph")
        }
    }

    static func runNemotron(
        install path: String, prompt: String, limit: Int, gpu: Bool = false,
        graph: Bool = false
    ) throws {
        /// unchecked-invariant: written only inside the Task and read after
        /// `semaphore.wait()` orders it; no concurrent access.
        final class Box: @unchecked Sendable {
            var error: Error?
        }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                try await runNemotronAsync(
                    install: URL(fileURLWithPath: path), prompt: prompt,
                    limit: limit, gpu: gpu, graph: graph)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.error { throw error }
    }

    private static func runNemotronAsync(
        install: URL, prompt: String, limit: Int, gpu: Bool, graph: Bool
    ) async throws {
        let loaded = try NemotronInstall(directory: install)
        let thinking =
            ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_THINKING"] == "1"
        let tokenizer = try await NemotronTokenizer(
            directory: install, thinking: thinking)
        let chat = ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_CHAT"] == "1"
        let ids = try tokenizer.encode(prompt: prompt, chat: chat)
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("the prompt tokenized to nothing\n".utf8))
            exit(2)
        }
        if ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_PRINT_IDS"] == "1" {
            print("ids: " + ids.map(String.init).joined(separator: ","))
        }
        let sequence = ids.count + limit + 8
        let model: NemotronModel?
        let graphModel: NemotronMetalGraph?
        if graph || gpu {
            let context = try MetalContext.runtimeDefault()
            print("device: \(context.device.name)\(graph ? " (graph)" : "")")
            if graph {
                graphModel = try NemotronMetalGraph(
                    install: loaded, context: context, maxSequence: sequence)
                model = nil
            } else {
                graphModel = nil
                model = try NemotronModel(
                    install: loaded, maxSequence: sequence,
                    matvec: try NemotronMetal(install: loaded, context: context))
            }
        } else {
            graphModel = nil
            model = try NemotronModel(install: loaded, maxSequence: sequence)
        }
        func advance(_ token: Int) throws -> [Float] {
            if let graphModel { return try graphModel.step(token: token) }
            guard let model else {
                throw NemotronInstallError.unsupported("no Nemotron engine was built")
            }
            return try model.step(token: token)
        }

        let prefillStart = Date()
        var logits: [Float] = []
        for token in ids { logits = try advance(Int(token)) }
        let prefill = Date().timeIntervalSince(prefillStart)

        let ignoreEOS =
            ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_IGNORE_EOS"] == "1"
        var produced: [Int32] = []
        let decodeStart = Date()
        for _ in 0..<limit {
            guard let best = greedy(logits) else { break }
            if !ignoreEOS, loaded.geometry.eosTokens.contains(best) { break }
            produced.append(Int32(best))
            logits = try advance(best)
        }
        let decode = Date().timeIntervalSince(decodeStart)

        print(
            "prompt \(ids.count) tokens: \(String(format: "%.2f", prefill))s "
                + "(\(String(format: "%.2f", Double(ids.count) / prefill)) tok/s)")
        if produced.isEmpty {
            print("decode 0 tokens")
        } else {
            print(
                "decode \(produced.count) tokens: \(String(format: "%.2f", decode))s "
                    + "(\(String(format: "%.2f", Double(produced.count) / decode)) tok/s)")
        }
        print("tokens: " + produced.map { String(Int($0)) }.joined(separator: ","))
        print("output: \(tokenizer.decode(produced))")
    }

    /// `TinyTitanBench nemotronlogits <install> <token,token,...> [--gpu]`:
    /// the logits after a token prefix, with no tokenizer in the way. This is
    /// the instrument the NumPy reference is compared against.
    static func runNemotronLogits(
        install path: String, tokens: String, gpu: Bool, graph: Bool
    ) throws {
        let loaded = try NemotronInstall(directory: URL(fileURLWithPath: path))
        let ids = tokens.split(separator: ",").compactMap { Int($0) }
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("no token ids\n".utf8))
            exit(2)
        }
        let backend: NemotronMatvec?
        var graphModel: NemotronMetalGraph?
        if graph || gpu {
            let context = try MetalContext.runtimeDefault()
            if graph {
                graphModel = try NemotronMetalGraph(
                    install: loaded, context: context, maxSequence: ids.count + 8)
                backend = nil
            } else {
                backend = try NemotronMetal(install: loaded, context: context)
            }
        } else {
            backend = nil
        }
        let model = try NemotronModel(
            install: loaded, maxSequence: ids.count + 8, matvec: backend)
        var logits: [Float] = []
        for token in ids {
            if let graphModel {
                logits = try graphModel.step(token: token)
            } else {
                logits = try model.step(token: token)
            }
        }
        print(logits.map { String(format: "%.6f", $0) }.joined(separator: ","))
        var best = 0
        for index in logits.indices where logits[index] > logits[best] { best = index }
        print("argmax \(best)")
        let top = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)
        print(
            "top: "
                + top.map { "\($0)=\(String(format: "%.3f", logits[$0]))" }
                .joined(separator: " "))
    }

    /// `TinyTitanBench nemotroninfo <install>`: what the install declares and
    /// what the reader found, before a token is run.
    static func runNemotronInfo(install path: String) throws {
        let loaded = try NemotronInstall(directory: URL(fileURLWithPath: path))
        let g = loaded.geometry
        print("family: nemotron_h")
        print(
            "layers \(g.numLayers): "
                + "\(g.layerKinds.filter { $0 == .mamba }.count) mamba, "
                + "\(g.layerKinds.filter { $0 == .moe }.count) moe, "
                + "\(g.layerKinds.filter { $0 == .attention }.count) attention")
        let carried = g.layerKinds.indices
            .filter { g.layerKinds[$0] == .moe }
            .map { g.expertCounts[$0] }
        print(
            "hidden \(g.hiddenSize), vocab \(g.vocabSize), latent \(g.moeLatentSize), "
                + "experts \(carried.min() ?? 0)...\(carried.max() ?? 0) "
                + "top-\(g.topKExperts)")
        print(
            "mamba \(g.mambaHeads)x\(g.mambaHeadDim), state \(g.ssmStateSize), "
                + "groups \(g.mambaGroups), conv \(g.convKernel)")
        print("attention \(g.attentionHeads)/\(g.attentionKVHeads)x\(g.attentionHeadDim)")
        print("resident tensors: \(loaded.resident.count)")
        print("expert layers: \(loaded.experts.count)")
        print("eos: \(g.eosTokens.map(String.init).joined(separator: ","))")
    }

    private static func greedy(_ logits: [Float]) -> Int? {
        guard !logits.isEmpty else { return nil }
        var best = 0
        for index in logits.indices where logits[index] > logits[best] { best = index }
        return best
    }
}
