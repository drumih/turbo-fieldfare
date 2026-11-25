import Foundation

/// A pinned upstream checkpoint the installer knows how to repack. Each value
/// fixes the repo, revision and index fingerprint so installs are exactly
/// reproducible.
public struct SupportedModelSource: Sendable, Equatable {
    /// CLI selector value (`--model <name>`).
    public let name: String
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    /// Value recorded as `manifest.modelID` when the source fingerprint matches.
    public let modelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64
    /// Set for an Inkling 2 source, which the installer plans from its own
    /// tensors instead of affine index metadata.
    public var inkling2: RemoteInkling2Source?

    public func installOptions(
        outputDirectory: URL,
        overwrite: Bool,
        token: String?,
        resume: Bool = false
    )
        -> RemoteStreamingRepackOptions
    {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume,
            installDraftHead: installsDraftHead,
            inkling2: inkling2)
    }

    /// Whether this source claims a repository's `mtp.*` namespace rather
    /// than the model itself. Only Qwen3.8-Flash-Next ships the two together;
    /// every other draft has a repository of its own, where the config
    /// already says what it is.
    public var installsDraftHead: Bool {
        repoID == Self.qwen38flash.repoID && name.hasSuffix("-mtp")
    }

    /// Download estimate covers the `language_model.*` tensors plus tokenizer
    /// and metadata sidecars. Installed bytes add the resident index and
    /// per-expert 16 KB page rounding plus layout/manifest sidecars.
    public static let qwen36 = SupportedModelSource(
        name: "qwen36",
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        modelID: "qwen3.6-35b-a3b-4bit",
        approximateDownloadBytes: 19_529_025_048,
        installedBytes: 19_546_491_213,
        reserveBytes: 1_073_741_824)

    public static let qwen36_8bit = SupportedModelSource(
        name: "qwen36-8bit",
        displayName: "Qwen3.6 35B-A3B 8-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-8bit",
        revision: "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
        sourceIndexSHA256:
            "3db12edeebeb65cab9a6eeb63cd74be4e0c74139a75f672701290b98230501cf",
        modelID: "qwen3.6-35b-a3b-8bit",
        approximateDownloadBytes: 37_741_392_345,
        installedBytes: 37_800_000_000,
        reserveBytes: 1_073_741_824)

    /// Text-only Ornith 1.5 uses the same qwen3_5_moe tensor contract as the
    /// Qwen 3.6 target. The official MLX conversion intentionally contains
    /// only `language_model.*` tensors; vision and MTP are separate follow-up
    /// features rather than silently entering the text runtime.
    public static let ornith15 = SupportedModelSource(
        name: "ornith15",
        displayName: "Ornith 1.5 35B-A3B 4-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        revision: "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
        sourceIndexSHA256:
            "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36",
        modelID: "ornith-1.5-35b-a3b-4bit",
        approximateDownloadBytes: 19_528_995_943,
        installedBytes: 19_530_000_000,
        reserveBytes: 1_073_741_824)

    public static let ornith15_8bit = SupportedModelSource(
        name: "ornith15-8bit",
        displayName: "Ornith 1.5 35B-A3B 8-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
        revision: "02440c39bdf7365c494a7f55f2a8b104ba87562f",
        sourceIndexSHA256:
            "83c641a791aa957df7d280eef1b0c8faf7a2ec9b19dd3355fb13abae8ae0ed15",
        modelID: "ornith-1.5-35b-a3b-8bit",
        approximateDownloadBytes: 36_848_194_663,
        installedBytes: 36_850_000_000,
        reserveBytes: 1_073_741_824)

    /// One-layer native Qwen3.6 MTP draft. It contains no embedding or LM
    /// head; the runtime reuses those tensors from whichever 4/6/8-bit target
    /// is loaded. The routed experts remain SSD-streamed in `.ssdai` form.
    public static let qwen36MTP = SupportedModelSource(
        name: "qwen36-mtp",
        displayName: "Qwen3.6 35B-A3B native MTP draft 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
        revision: "0295b81421bf4d0fccca9a7c0fcfb1418dda3516",
        sourceIndexSHA256:
            "00e220ddb21ceeb6290a3a1161f97339c553f3d27fc4319900a96edb5cfae74c",
        modelID: "qwen3.6-35b-a3b-mtp-4bit",
        approximateDownloadBytes: 475_130_833,
        installedBytes: 475_300_000,
        reserveBytes: 536_870_912)

    /// Qwen3.8-Flash-Next, the Qwen4 preview architecture. Community-quantized
    /// rather than mlx-community: no official or mlx-community MLX release
    /// exists, and this artifact was verified against the official bf16
    /// weights before being pinned -- every 4-bit tensor lands within 1.02x of
    /// an ideal affine g64 round-trip, i.e. as accurate as the format allows
    /// (`benchmark/tinytitan_quant_fidelity.py`). Group size is 64, matching the
    /// format TinyTitan already uses.
    ///
    /// `approximateDownloadBytes` covers the backbone plus the 102.4 GB fp16
    /// n-gram table; the MTP draft installs separately from its own shard.
    public static let qwen38flash = SupportedModelSource(
        name: "qwen38flash",
        displayName: "Qwen3.8-Flash-Next 125B-A6B 4-bit",
        repoID: "RockTalk/Qwen3.8-Flash-Next-MLX-4bit",
        revision: "478474da92599ad0cf9f8bd447e658b29cb8480a",
        sourceIndexSHA256:
            "d7fe03ad2d1365e24ae2e305c829f600b80fac845d128383421a7be4adbdda1b",
        modelID: "qwen3.8-flash-next-4bit",
        approximateDownloadBytes: 173_825_909_304,
        installedBytes: 174_228_562_488,
        reserveBytes: 4 * 1_073_741_824)

    /// The Qwen3.8-Flash-Next MTP draft head.
    ///
    /// Same repository and revision as the target, because the draft ships
    /// inside it -- but in its own `mtp-weights.safetensors` shard, so the
    /// install fetches 1.37 GB rather than re-reading the backbone. It has no
    /// embedding or head of its own; both are shared from the target it
    /// drafts for.
    public static let qwen38flashMTP = SupportedModelSource(
        name: "qwen38flash-mtp",
        displayName: "Qwen3.8-Flash-Next 125B-A6B native MTP draft 4-bit",
        repoID: "RockTalk/Qwen3.8-Flash-Next-MLX-4bit",
        revision: "478474da92599ad0cf9f8bd447e658b29cb8480a",
        sourceIndexSHA256:
            "d7fe03ad2d1365e24ae2e305c829f600b80fac845d128383421a7be4adbdda1b",
        modelID: "qwen3.8-flash-next-mtp-4bit",
        approximateDownloadBytes: 1_471_026_000,
        installedBytes: 1_480_000_000,
        reserveBytes: 2 * 1_073_741_824)

    /// Inkling 2 Mini from Unsloth's Q4_K_XL GGUF (TT-121): routed experts
    /// NVFP4 (Q8_0 in the last four layers), the rest Q8_0/F32/BF16, all kept
    /// as the GGUF stores them. The fingerprint is the SHA-256 of the GGUF's
    /// header (metadata, tokenizer and tensor table), which pins every tensor's
    /// encoding and offset. Converted, not yet served: the variant that serves
    /// it is TT-121 phase 2.
    public static let inkling2GGUFQ4 = SupportedModelSource(
        name: "inkling-q4-k-xl-unsloth",
        displayName: "Inkling 2 Mini UD-Q4_K_XL (Unsloth)",
        repoID: "unsloth/Inkling-2-Mini-GGUF",
        revision: "45fd08764f2b0bf16b03a5483a80e5c7aaec8772",
        sourceIndexSHA256:
            "e35ee6baa44933adb00cd5196be291a80c7e6b35aa75c2b01549b056057387e0",
        modelID: "inkling-2-mini-gguf-q4",
        approximateDownloadBytes: 17_629_759_456,
        installedBytes: 17_664_255_634,
        reserveBytes: 1_073_741_824,
        inkling2: .gguf(fileName: "Inkling-2-Mini-UD-Q4_K_XL.gguf"))

    /// Inkling 2 Mini from the original ModelOpt checkpoint (TT-121): routed
    /// experts NVFP4 (MXFP8 in the last four layers), the rest BF16/F32, the
    /// MTP head beside it, the vision tower left out. The fingerprint is the
    /// SHA-256 of `config.json` and both safetensors headers. Converted, not
    /// yet served: the variant that serves it is TT-121 phase 3.
    public static let inkling2NVFP4 = SupportedModelSource(
        name: "inkling2mini",
        displayName: "Inkling 2 Mini",
        repoID: "thinkingmachines/Inkling-2-Mini",
        revision: "ed40e7da625f594fc9c4990f5f66be24873c22d7",
        sourceIndexSHA256:
            "0b94cadc1237f8d183c2fabf35dd610d4aa7364862be2afda0371eaacef45750",
        modelID: "inkling-2-mini-nvfp4",
        approximateDownloadBytes: 19_555_000_000,
        installedBytes: 19_592_286_535,
        reserveBytes: 1_073_741_824,
        inkling2: .checkpoint)

    /// Default source when no `--model` selector is given.
    public static let `default` = ornith15_8bit

    public static let all: [SupportedModelSource] = [
        qwen36, qwen36_8bit, ornith15, ornith15_8bit, qwen36MTP, qwen38flash,
        qwen38flashMTP, inkling2GGUFQ4, inkling2NVFP4,
    ]

    public static func named(_ name: String) -> SupportedModelSource? {
        if let direct = all.first(where: { $0.name == name }) { return direct }
        switch name.lowercased() {
        case "inkling2-gguf-q4", "inkling_q4_k_xl_unsloth", "inkling2-ud-q4-k-xl-unsloth":
            return inkling2GGUFQ4
        case "inkling2-nvfp4":
            return inkling2NVFP4
        default:
            return nil
        }
    }
}
