import Foundation
import TinyTitanCore

// Loading a snapshot: the two readers that turn a directory into an
// `AffineSnapshot` -- a plain affine safetensors snapshot, which the converter
// writes, and a `.ssdai` install, which is what every other model here is --
// plus the one architecture fact the manifest does not record.
//
// Split out of `AffineSnapshot.swift` (2026-10-07) under the 500-line-per-file
// rule as pure code motion. The types, the width rules and the accessors stay
// with the struct.
extension AffineSnapshot {

    public init(
        directory: URL,
        maxBytes: UInt64 = ManifestReader.defaultMaxBytes
    ) throws {
        self.directory = directory
        // A snapshot directory may have been copied off another machine, which is
        // the same attacker model the `.ssdai` loader fences with
        // `ManifestReader.defaultMaxBytes`, so this document shares that ceiling
        // instead of carrying a second one that can drift below it. The real
        // `config.json` measures 12,935 bytes installed, about 5,000x under it.
        let configData = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("config.json"),
            maxBytes: maxBytes)
        let config = try JSONSerialization.jsonObject(with: configData)
        guard let config = config as? [String: Any] else {
            throw SafeTensorsFile.Failure.malformed("config.json is not an object")
        }
        func integer(_ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = config[key] as? Int { return value }
            if let fallback { return fallback }
            throw SafeTensorsFile.Failure.malformed("config.json lacks \(key)")
        }
        func double(_ key: String, _ fallback: Double) -> Double {
            (config[key] as? NSNumber)?.doubleValue ?? fallback
        }
        configuration = Configuration(
            hiddenSize: try integer("hidden_size"),
            layers: try integer("num_hidden_layers"),
            heads: try integer("num_attention_heads"),
            keyValueHeads: try integer("num_key_value_heads"),
            headDim: try integer("head_dim"),
            fullAttentionInterval: try integer("full_attention_interval", 4),
            linearKeyHeads: try integer("linear_num_key_heads"),
            linearValueHeads: try integer("linear_num_value_heads"),
            linearKeyHeadDim: try integer("linear_key_head_dim"),
            linearValueHeadDim: try integer("linear_value_head_dim"),
            convKernel: try integer("linear_conv_kernel_dim", 4),
            intermediateSize: try integer("intermediate_size"),
            vocabulary: try integer("vocab_size"),
            normEpsilon: Float(double("rms_norm_eps", 1e-6)),
            ropeTheta: Float(double("rope_theta", 10_000_000)),
            partialRotaryFactor: double("partial_rotary_factor", 0.25),
            tiedEmbedding: (config["tie_word_embeddings"] as? Bool) ?? true,
            maxPositions: try integer("max_position_embeddings", 32_768))

        modelType = config["model_type"] as? String
        guard let quantization = config["quantization"] as? [String: Any],
            let bits = quantization["bits"] as? Int,
            let group = quantization["group_size"] as? Int
        else {
            throw SafeTensorsFile.Failure.malformed("config.json lacks a quantization block")
        }
        baseBits = bits
        groupSize = group
        var overrides: [String: Int] = [:]
        for (stem, value) in quantization {
            if let entry = value as? [String: Any], let width = entry["bits"] as? Int {
                overrides[stem] = width
            }
        }
        widths = overrides
        try Self.validateWidths(baseBits: bits, widths: overrides)

        let index = try JSONSerialization.jsonObject(
            with: try BoundedMetadataRead.read(
                fileAt: directory.appendingPathComponent("model.safetensors.index.json"),
                maxBytes: maxBytes))
        guard let index = index as? [String: Any],
            let map = index["weight_map"] as? [String: String]
        else {
            throw SafeTensorsFile.Failure.malformed("index has no weight_map")
        }
        // The shard names are index content from a directory that can arrive copied
        // off another machine, so they open through `SSDAIModelDirectory`: a
        // non-canonical name is refused and every component is opened `O_NOFOLLOW`
        // relative to the directory's own descriptor, so no name reaches a file
        // outside the snapshot. `LocalSnapshotLoader` fences the same contract here.
        let root = try SSDAIModelDirectory(rootURL: directory)
        var opened: [String: SafeTensorsFile] = [:]
        for file in Set(map.values) {
            opened[file] = try SafeTensorsFile(
                url: directory.appendingPathComponent(file),
                descriptor: try root.openFile(file))
        }
        storage = .safetensors(shards: opened, placement: map)
    }

    /// Load a `.ssdai` install as CPU weights.
    ///
    /// The install's `manifest.json` carries the same architecture facts the
    /// converter used to write `config.json`, and its resident index carries
    /// the tensor offsets, so this needs no second architecture description --
    /// it is the snapshot's reader pointed at a different file layout.
    ///
    /// Two facts live only in the snapshot config and are re-derived here:
    /// `full_attention_interval`, from the spacing of the manifest's
    /// full-attention mask, and the norm epsilon, which the manifest does not
    /// record and which is 1e-6 for every Qwen 3.5-family model. The first is
    /// checked for regularity rather than assumed, because a wrong interval
    /// silently changes which layers use DeltaNet and which use attention.
    public init(ssdai directory: URL) throws {
        self.directory = directory
        let manifest = try ManifestReader.read(directoryURL: directory)
        let arch = manifest.arch
        // The family is an identity fact, not an arch field; reading it keeps a
        // GPU install from being offered to the CPU engine by mistake.
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        guard identity.family == .qwen35Dense else {
            throw SafeTensorsFile.Failure.malformed(
                "not a dense install: manifest declares \(identity.family.rawValue)")
        }
        modelType = CPUModelFamily.qwen35Dense.rawValue

        // The DeltaNet geometry is optional in the manifest because older
        // installs predate it. A dense install must carry it -- these values
        // decide the linear-attention arithmetic -- so a missing one is a
        // refusal, not a default.
        func required(_ value: Int?, _ field: String) throws -> Int {
            guard let value else {
                throw SafeTensorsFile.Failure.malformed(
                    "manifest.arch lacks \(field), which the CPU engine needs")
            }
            return value
        }
        configuration = Configuration(
            hiddenSize: arch.hiddenSize,
            layers: arch.numLayers,
            heads: arch.numHeads,
            keyValueHeads: arch.numKVHeads,
            headDim: arch.headDim,
            fullAttentionInterval: try Self.fullAttentionInterval(of: arch),
            linearKeyHeads: try required(arch.linearNumKHeads, "linearNumKHeads"),
            linearValueHeads: try required(arch.linearNumVHeads, "linearNumVHeads"),
            linearKeyHeadDim: try required(arch.linearKeyHeadDim, "linearKeyHeadDim"),
            linearValueHeadDim: try required(arch.linearValueHeadDim, "linearValueHeadDim"),
            convKernel: try required(arch.linearConvKernelSize, "linearConvKernelSize"),
            intermediateSize: arch.ffnIntermediate,
            vocabulary: arch.vocabSize,
            normEpsilon: 1e-6,
            ropeTheta: Float(arch.ropeTheta),
            partialRotaryFactor: arch.partialRotaryFactor,
            tiedEmbedding: arch.tieWordEmbeddings,
            maxPositions: 262_144)

        guard let quant = manifest.quant else {
            throw SafeTensorsFile.Failure.malformed("manifest.quant is missing")
        }
        baseBits = quant.attention.weightBits
        groupSize = quant.attention.groupSize
        let embeddingBits = quant.embedding.weightBits
        let weightsURL = directory.appendingPathComponent("model_weights.bin")
        let index = try ResidentIndexReader.load(fileURL: weightsURL)
        // Mapped once, held for the snapshot's life; see `ResidentWeights`.
        let weights = ResidentWeights(
            // lint:allow-unbounded-read `.alwaysMapped` is a mapping, not a copy:
            // the pages fault in as the kernels touch them, so this is the one
            // whole-file read here that does not allocate the file's size up front
            // — which is the defect the gate hunts. It is also the design: a blob
            // read this way is what lets an 8-bit install run, and its shape comes
            // from the verified index loaded above, not from this read.
            data: try Data(contentsOf: weightsURL, options: .alwaysMapped))

        // Per-tensor width overrides, keyed by stem, exactly as the snapshot
        // reads them from its `quantization` block.
        //
        // The manifest names *slots* plus a `quantization` block keyed by the
        // same stem the snapshot uses, and those keys are what a 4-bit build
        // leans on: its embedding and every attention K/V are 8-bit where the
        // body is 4. Ignoring the block made `k_proj`/`v_proj` dequantize as
        // 4-bit -- the bytes are the same, but they get unpacked wrongly, and
        // that is enough to turn the answer into nonsense while every shape
        // still checks out. The snapshot has always honoured this; a reader
        // that does not is worse than one that refuses.
        var widths: [String: Int] = manifest.quantOverrides
        for name in index.entries.keys {
            let stem =
                name.hasSuffix(".weight")
                ? String(name.dropLast(".weight".count)) : name
            // The embedding is the head too when the output is tied, and the
            // manifest's override may not name it; the slot carries its width.
            if stem.hasSuffix("embed_tokens") || stem.hasSuffix("lm_head") {
                widths[stem] = widths[stem] ?? embeddingBits
            }
        }
        self.widths = widths
        // Not checked here: on this route the widths are the manifest's slots and
        // overrides, and `SSDAIManifestQuantV1.init(from:)` already refused
        // anything outside `supportedWeightBits` while decoding.
        storage = .ssdai(index: index, weights: weights)
    }

    /// The `full_attention_interval` the manifest's mask encodes.
    ///
    /// `1` marks full attention and `2` gated DeltaNet, and every Qwen 3.5
    /// dense model puts full attention on the last layer of each group of
    /// `interval`. Deriving it is exact for a regular mask and refuses an
    /// irregular one rather than guessing, because the interval decides which
    /// layers take the arithmetic path.
    private static func fullAttentionInterval(
        of arch: ManifestArch
    ) throws -> Int {
        let mask = arch.fullAttentionLayerMask
        let full = mask.enumerated().filter { $0.element == 1 }.map(\.offset)
        guard full.count >= 2 else {
            throw SafeTensorsFile.Failure.malformed(
                "manifest's attention mask has fewer than two full-attention "
                    + "layers, so no interval can be derived: full at \(full)")
        }
        // The gap between *consecutive* full-attention layers. Every Qwen 3.5
        // model puts full attention on the last layer of each group, so a
        // regular mask has one gap throughout.
        let gaps = Set(zip(full, full.dropFirst()).map { $1 - $0 })
        guard gaps.count == 1, let interval = gaps.first, interval > 1 else {
            throw SafeTensorsFile.Failure.malformed(
                "manifest's attention mask is not a regular interval: full at \(full)")
        }
        return interval
    }
}
