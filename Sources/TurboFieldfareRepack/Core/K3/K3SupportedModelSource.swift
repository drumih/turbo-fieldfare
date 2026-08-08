import Foundation

/// The pinned official Kimi K3 checkpoint the K3 repack profile installs.
public enum K3SupportedModelSource {
    public static let displayName = "Kimi K3 2.78T MoE (int4 trunk, MXFP4 experts)"
    public static let repoID = "moonshotai/Kimi-K3"
    // Pinned 2026-08-07: `main` @ 9f62e4e9fffbd0a83ddd60e1c209d828994b3569
    // (lastModified 2026-07-27T16:29:18Z), from
    // GET https://huggingface.co/api/models/moonshotai/Kimi-K3.
    public static let revision = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    // SHA-256 of `model.safetensors.index.json` at the pinned revision,
    // cross-checked against the LFS blob OID returned by
    // POST /api/models/moonshotai/Kimi-K3/paths-info/<revision> (the two
    // agree, so the pinned bytes are content-addressed upstream).
    public static let sourceIndexSHA256 =
        "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
    public static let sourceIndexBytes: UInt64 = 59_764_096
    /// `metadata.total_size` of the pinned index. Slight overestimate of the
    /// real transfer: it includes the excluded vision_tower/mm_projector
    /// tensors, some of which still arrive inside coalesced range gaps.
    public static let approximateDownloadBytes: UInt64 = 1_560_860_324_864
    /// ~1.4465 TB packed experts (92 x 896 x 17,547,264) + ~35.2 GB int4
    /// resident trunk + layout/manifest/tokenizer sidecars.
    public static let installedBytes: UInt64 = 1_481_700_000_000
    /// Transient BF16 staging for the trunk quantization phase (~113.5 GB at
    /// int4), accounted in the driver's disk check and deleted before the
    /// bundle is promoted.
    public static let approximateStagingBytes: UInt64 = 113_500_000_000
    public static let reserveBytes: UInt64 = 1_073_741_824

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false,
                                      trunkQuant: K3TrunkQuant = .int4)
        -> K3RemoteStreamingRepackOptions {
        K3RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            trunkQuant: trunkQuant,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}
