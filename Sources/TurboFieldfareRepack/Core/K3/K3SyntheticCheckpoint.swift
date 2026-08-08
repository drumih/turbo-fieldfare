import Foundation

/// Test support: fabricates a tiny Kimi K3 checkpoint in the official
/// safetensors layout (96-shard naming conventions, `text_config` config
/// schema, MXFP4-packed per-expert tensors, head-padded F32 `A_log`, no
/// conv1d biases — exactly the source facts the K3 repack profile is written
/// against) inside a local directory. The installer never calls this; the K3
/// repack tests and the engine end-to-end test share it from the core target
/// because both need the identical fixture.
///
/// Dims mirror `K3ForwardRunnerTests.tinyConfig()`: hidden 256, 5 layers
/// (0 KDA+dense, 1 KDA, 2 KDA, 3 MLA, 4 KDA), KDA 4 heads x 32 conv 4,
/// decay low-rank 64, MLA qLora 64 / kvLora 64 / rope 8 / nope 16 / v 16 /
/// heads 4, AttnRes block 2, 16 experts top-4, latent 128, expert
/// intermediate 64, 1 shared expert (intermediate 64), dense intermediate
/// 256, vocab 512, bos 500 / eos 501.
enum K3SyntheticCheckpoint {

    struct Snapshot {
        let directory: String
        let shardPaths: [String]
        let configPath: String
        let indexPath: String
        let tiktokenPath: String
        let tokenizerConfigPath: String
    }

    struct Dims: Sendable {
        let hidden = 256
        let vocab = 512
        let numLayers = 5
        let denseLayers1 = [1]
        let kdaLayers1 = [1, 2, 3, 5]
        let fullAttnLayers1 = [4]
        let kdaHeads = 4
        let kdaHeadDim = 32
        let kdaConvWidth = 4
        let kdaDecayLowRank = 64
        let mlaHeads = 4
        let mlaQLora = 64
        let mlaKVLora = 64
        let mlaRope = 8
        let mlaNope = 16
        let mlaV = 16
        let attnResBlock = 2
        let experts = 16
        let topK = 4
        let latent = 128
        let expertInter = 64
        let sharedExperts = 1
        let sharedInter = 64
        let denseInter = 256
        let bos = 500
        let eos = 501

        var kdaP: Int { kdaHeads * kdaHeadDim }
        var kdaLayers0: [Int] { kdaLayers1.map { $0 - 1 } }
        var mlaLayers0: [Int] { fullAttnLayers1.map { $0 - 1 } }
        var moeLayers0: [Int] { (0..<numLayers).filter { !denseLayers1.contains($0 + 1) } }
    }

    static func build(at dir: String,
                      seed: UInt64 = 0xD3C0_AFE1_5EED_0001) throws -> Snapshot {
        let dims = Dims()
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)
        var rng = K3SyntheticPRNG(seed: seed)

        var trunk: [(String, String, [Int], [UInt8])] = []
        var experts: [(String, String, [Int], [UInt8])] = []

        func bf16(_ name: String, _ shape: [Int], _ values: [Float]) {
            precondition(shape.reduce(1, *) == values.count, "\(name) shape/values drift")
            var bytes: [UInt8] = []
            bytes.reserveCapacity(values.count * 2)
            for v in values {
                let bits = K3TrunkQuantizer.bf16Bits(v)
                bytes.append(UInt8(truncatingIfNeeded: bits))
                bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
            }
            trunk.append((name, "BF16", shape, bytes))
        }
        func f32(_ name: String, _ shape: [Int], _ values: [Float]) {
            precondition(shape.reduce(1, *) == values.count, "\(name) shape/values drift")
            var bytes: [UInt8] = []
            bytes.reserveCapacity(values.count * 4)
            for var v in values {
                withUnsafeBytes(of: &v) { bytes.append(contentsOf: $0) }
            }
            trunk.append((name, "F32", shape, bytes))
        }
        func uniform(_ count: Int, _ lo: Float, _ hi: Float) -> [Float] {
            (0..<count).map { _ in rng.uniform(lo, hi) }
        }
        /// Magnitude policy mirrors `K3ForwardRunnerTests.masterValues` so the
        /// repacked bundle produces well-behaved logits for the engine test.
        func masterValues(name: String, count: Int) -> [Float] {
            if name.hasSuffix("embed_tokens.weight") { return uniform(count, -0.5, 0.5) }
            if name.hasSuffix("lm_head.weight") { return uniform(count, -0.2, 0.2) }
            if name.hasSuffix("_res_proj.weight") { return uniform(count, -0.15, 0.15) }
            if name.hasSuffix("_res_norm.weight")
                || name.hasSuffix("routed_expert_norm.weight") {
                return uniform(count, 0.7, 1.3)
            }
            if name.hasSuffix("layernorm.weight") || name.hasSuffix("norm.weight") {
                return uniform(count, 0.7, 1.3)
            }
            if name.hasSuffix("block_sparse_moe.gate.weight") {
                return uniform(count, -0.05, 0.05)
            }
            return uniform(count, -0.06, 0.06)
        }
        func matrix(_ name: String, _ rows: Int, _ columns: Int) {
            bf16(name, [rows, columns], masterValues(name: name, count: rows * columns))
        }

        let H = dims.hidden
        let P = dims.kdaP
        matrix("language_model.model.embed_tokens.weight", dims.vocab, H)
        matrix("language_model.lm_head.weight", dims.vocab, H)
        bf16("language_model.model.norm.weight", [H], uniform(H, 0.7, 1.3))
        bf16("language_model.model.output_attn_res_proj.weight", [1, H], uniform(H, -0.15, 0.15))
        bf16("language_model.model.output_attn_res_norm.weight", [H], uniform(H, 0.7, 1.3))

        for layer in 0..<dims.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            bf16("\(prefix).input_layernorm.weight", [H], uniform(H, 0.7, 1.3))
            bf16("\(prefix).post_attention_layernorm.weight", [H], uniform(H, 0.7, 1.3))
            bf16("\(prefix).self_attention_res_proj.weight", [1, H], uniform(H, -0.15, 0.15))
            bf16("\(prefix).self_attention_res_norm.weight", [H], uniform(H, 0.7, 1.3))
            bf16("\(prefix).mlp_res_proj.weight", [1, H], uniform(H, -0.15, 0.15))
            bf16("\(prefix).mlp_res_norm.weight", [H], uniform(H, 0.7, 1.3))

            if dims.kdaLayers0.contains(layer) {
                let attn = "\(prefix).self_attn"
                matrix("\(attn).q_proj.weight", P, H)
                matrix("\(attn).k_proj.weight", P, H)
                matrix("\(attn).v_proj.weight", P, H)
                matrix("\(attn).g_proj.weight", P, H)
                matrix("\(attn).o_proj.weight", H, P)
                matrix("\(attn).f_a_proj.weight", dims.kdaDecayLowRank, H)
                matrix("\(attn).f_b_proj.weight", P, dims.kdaDecayLowRank)
                matrix("\(attn).b_proj.weight", dims.kdaHeads, H)
                for qkv in ["q", "k", "v"] {
                    // Official layout: F32 [channels, 1, kernel], no bias.
                    f32("\(attn).\(qkv)_conv1d.weight", [P, 1, dims.kdaConvWidth],
                        uniform(P * dims.kdaConvWidth, -0.25, 0.25))
                }
                // Official layout: A_log is head-padded to head_dim with a
                // zero tail; only the first `kdaHeads` values are trained.
                f32("\(attn).A_log", [dims.kdaHeadDim],
                    uniform(dims.kdaHeads, 0.0, 2.8)
                        + [Float](repeating: 0, count: dims.kdaHeadDim - dims.kdaHeads))
                f32("\(attn).dt_bias", [P], uniform(P, -0.3, 0.3))
                f32("\(attn).o_norm.weight", [dims.kdaHeadDim],
                    uniform(dims.kdaHeadDim, 0.5, 1.5))
            } else {
                let attn = "\(prefix).self_attn"
                let qElements = dims.mlaHeads * (dims.mlaNope + dims.mlaRope)
                let cacheRow = dims.mlaKVLora + dims.mlaRope
                let kvbRows = dims.mlaHeads * (dims.mlaNope + dims.mlaV)
                let outElements = dims.mlaHeads * dims.mlaV
                matrix("\(attn).q_a_proj.weight", dims.mlaQLora, H)
                bf16("\(attn).q_a_layernorm.weight", [dims.mlaQLora],
                     uniform(dims.mlaQLora, 0.7, 1.3))
                matrix("\(attn).q_b_proj.weight", qElements, dims.mlaQLora)
                matrix("\(attn).kv_a_proj_with_mqa.weight", cacheRow, H)
                bf16("\(attn).kv_a_layernorm.weight", [dims.mlaKVLora],
                     uniform(dims.mlaKVLora, 0.7, 1.3))
                matrix("\(attn).kv_b_proj.weight", kvbRows, dims.mlaKVLora)
                matrix("\(attn).g_proj.weight", outElements, H)
                matrix("\(attn).o_proj.weight", H, outElements)
            }

            if dims.moeLayers0.contains(layer) {
                let moe = "\(prefix).block_sparse_moe"
                matrix("\(moe).gate.weight", dims.experts, H)
                f32("\(moe).gate.e_score_correction_bias", [dims.experts],
                    uniform(dims.experts, -0.25, 0.25))
                matrix("\(moe).routed_expert_down_proj.weight", dims.latent, H)
                matrix("\(moe).routed_expert_up_proj.weight", H, dims.latent)
                bf16("\(moe).routed_expert_norm.weight", [dims.latent],
                     uniform(dims.latent, 0.7, 1.3))
                matrix("\(moe).shared_experts.gate_proj.weight", dims.sharedInter, H)
                matrix("\(moe).shared_experts.up_proj.weight", dims.sharedInter, H)
                matrix("\(moe).shared_experts.down_proj.weight", H, dims.sharedInter)

                for expert in 0..<dims.experts {
                    let base = "\(moe).experts.\(expert)"
                    for stem in ["w1", "w2", "w3"] {
                        let rows = stem == "w2" ? dims.latent : dims.expertInter
                        let columns = stem == "w2" ? dims.expertInter : dims.latent
                        var packed = [UInt8](repeating: 0, count: rows * columns / 2)
                        for i in packed.indices { packed[i] = UInt8(rng.next() & 0xFF) }
                        var scales = [UInt8](repeating: 0, count: rows * columns / 32)
                        for i in scales.indices {
                            // 2^-9 ... 2^-4 group scales; 255 (zero-group
                            // guard) avoided, mirroring K3ForwardRunnerTests.
                            scales[i] = UInt8(118 + rng.next() % 6)
                        }
                        experts.append(("\(base).\(stem).weight_packed", "U8",
                                        [rows, columns / 2], packed))
                        experts.append(("\(base).\(stem).weight_scale", "U8",
                                        [rows, columns / 32], scales))
                    }
                }
            } else {
                let mlp = "\(prefix).mlp"
                matrix("\(mlp).gate_proj.weight", dims.denseInter, H)
                matrix("\(mlp).up_proj.weight", dims.denseInter, H)
                matrix("\(mlp).down_proj.weight", H, dims.denseInter)
            }
        }

        // Excluded-from-the-text-bundle tensors, present to prove the
        // planner drops them.
        bf16("vision_tower.encoder.layers.0.self_attn.q_proj.weight", [H, H],
             uniform(H * H, -0.06, 0.06))
        bf16("mm_projector.weight", [H, H], uniform(H * H, -0.06, 0.06))

        let trunkShard = "model-00001-of-00002.safetensors"
        let expertShard = "model-00002-of-00002.safetensors"
        let trunkPath = (dir as NSString).appendingPathComponent(trunkShard)
        let expertPath = (dir as NSString).appendingPathComponent(expertShard)
        try writeShard(path: trunkPath, tensors: trunk)
        try writeShard(path: expertPath, tensors: experts)

        var weightMap: [String: String] = [:]
        for (name, _, _, _) in trunk { weightMap[name] = trunkShard }
        for (name, _, _, _) in experts { weightMap[name] = expertShard }
        let totalSize = (trunk + experts).reduce(UInt64(0)) { $0 + UInt64($1.3.count) }
        let indexObj: [String: Any] = [
            "metadata": ["total_size": totalSize],
            "weight_map": weightMap,
        ]
        let indexData = try JSONSerialization.data(withJSONObject: indexObj,
                                                   options: [.sortedKeys])
        let indexPath = (dir as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))

        let config = configJSON(dims: dims)
        let configData = try JSONSerialization.data(withJSONObject: config,
                                                    options: [.sortedKeys])
        let configPath = (dir as NSString).appendingPathComponent("config.json")
        try configData.write(to: URL(fileURLWithPath: configPath))

        let tiktokenPath = (dir as NSString).appendingPathComponent("tiktoken.model")
        try Data("a3N5bnRoZXRpYyB0aWt0b2tlbiBmaXh0dXJlIDAK".utf8)
            .write(to: URL(fileURLWithPath: tiktokenPath))
        let tokenizerConfigPath = (dir as NSString)
            .appendingPathComponent("tokenizer_config.json")
        try Data(#"{"tokenizer_class":"KimiTokenizer","_synthetic":true}"#.utf8)
            .write(to: URL(fileURLWithPath: tokenizerConfigPath))

        return Snapshot(directory: dir,
                        shardPaths: [trunkPath, expertPath],
                        configPath: configPath,
                        indexPath: indexPath,
                        tiktokenPath: tiktokenPath,
                        tokenizerConfigPath: tokenizerConfigPath)
    }

    /// The official K3 config schema (`text_config` wrapper,
    /// `linear_attn_config` 1-based layer lists) with tiny values.
    private static func configJSON(dims: Dims) -> [String: Any] {
        let linearAttn: [String: Any] = [
            "full_attn_layers": dims.fullAttnLayers1,
            "kda_layers": dims.kdaLayers1,
            "num_heads": dims.kdaHeads,
            "head_dim": dims.kdaHeadDim,
            "short_conv_kernel_size": dims.kdaConvWidth,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": true,
        ]
        let textConfig: [String: Any] = [
            "model_type": "kimi_linear",
            "hidden_size": dims.hidden,
            "vocab_size": dims.vocab,
            "num_hidden_layers": dims.numLayers,
            "intermediate_size": dims.denseInter,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "hidden_act": "situ",
            "bos_token_id": dims.bos,
            "eos_token_id": dims.eos,
            "first_k_dense_replace": dims.denseLayers1.count,
            "num_attention_heads": dims.mlaHeads,
            "q_lora_rank": dims.mlaQLora,
            "kv_lora_rank": dims.mlaKVLora,
            "qk_nope_head_dim": dims.mlaNope,
            "qk_rope_head_dim": dims.mlaRope,
            "v_head_dim": dims.mlaV,
            "mla_use_output_gate": true,
            "attn_res_block_size": dims.attnResBlock,
            "num_experts": dims.experts,
            "num_experts_per_token": dims.topK,
            "routed_expert_hidden_size": dims.latent,
            "moe_intermediate_size": dims.expertInter,
            "num_shared_experts": dims.sharedExperts,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "moe_renormalize": true,
            "linear_attn_config": linearAttn,
        ]
        return [
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "bos_token_id": dims.bos,
            "eos_token_id": dims.eos,
            "tie_word_embeddings": false,
            "text_config": textConfig,
        ]
    }

    private static func writeShard(path: String,
                                   tensors: [(String, String, [Int], [UInt8])]) throws {
        var off: UInt64 = 0
        var headerDict: [String: Any] = [:]
        for (name, dtype, shape, bytes) in tensors {
            let begin = off
            let end = begin + UInt64(bytes.count)
            headerDict[name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, end],
            ]
            off = end
        }
        let headerData = try JSONSerialization.data(withJSONObject: headerDict,
                                                    options: [.sortedKeys])
        var padded = headerData
        while padded.count % 8 != 0 { padded.append(0x20) }  // space pad

        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        precondition(fd >= 0, "open failed for \(path)")
        defer { close(fd) }
        var headerLenLE = UInt64(padded.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { raw in
            _ = write(fd, raw.baseAddress, 8)
        }
        padded.withUnsafeBytes { raw in
            _ = write(fd, raw.baseAddress, padded.count)
        }
        for (_, _, _, bytes) in tensors {
            bytes.withUnsafeBufferPointer { ptr in
                _ = write(fd, ptr.baseAddress, ptr.count)
            }
        }
    }
}

/// Deterministic PRNG for fixture bytes (same SplitMix64 as the house
/// synthetic snapshots; duplicated here because the core target must not see
/// the test-only one).
struct K3SyntheticPRNG {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ lo: Float, _ hi: Float) -> Float {
        let unit = Float(next() >> 40) / Float(1 << 24)
        return lo + (hi - lo) * unit
    }
}
