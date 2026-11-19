import Foundation
import TinyTitanCore

// The ANE prefill sidecar's on-disk contract: where its directory lives for a
// configured chunk width, what its `ane_prefill.json` records, and the bounded
// read of that document.
//
// Split out of `ANEPrefillAttention.swift` (2026-10-06) under the
// 500-line-per-file rule as pure code motion; nothing widened, because nothing
// here was `private`.
extension ANEPrefillAttention {
    /// The geometry the sidecar's Core ML graph was built for. The exporter
    /// records it and `init` refuses a sidecar that does not match the model:
    /// a mismatch computes a *different* attention, and plausibly.
    struct SidecarGeometry: Decodable {
        let family: String
        let hiddenSize: Int
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let chunkTokens: Int
        let fullAttentionLayers: [Int]?
    }

    struct SidecarMetadata: Decodable {
        let version: Int
        let family: String
        let chunkTokens: Int
        let histories: [Int]
        let layers: [Int]
        /// Present from the generalized exporter on; absent in a sidecar built
        /// while the graph was hard-coded to the 35B-A3B geometry.
        let geometry: SidecarGeometry?
        /// SHA-256 of the `model_weights.bin` the sidecar was exported from,
        /// copied out of that model's install receipt at export time.
        let weightsSha256: String?
        /// True only when the exporter watched the Neural Engine compile every
        /// variant and saw no compiler error. Core ML reports such a failure on
        /// the native stderr and still exits 0, so a sidecar from an exporter
        /// without this flag may silently run the whole prefill on the CPU.
        let aneCompileVerified: Bool?
        /// True when the exporter built this sidecar for a family whose
        /// full-attention layers pick keys with a sparse indexer, so the
        /// runtime has to fold that selection into the mask. Absent on a
        /// sidecar for a dense family; the runtime refuses to load a
        /// sparse-indexed model's sidecar that does not record it.
        let selectionFolded: Bool?
        /// "split" when the graph returns the attention core ("attn") and the
        /// pre-sigmoid gate ("gate") and the runtime applies gate + wo on the
        /// GPU. Absent means the historical fused graph ("out" after wo).
        let tail: String?
    }

    static let expectedVersion = 1

    /// The sidecar directory for a configured prefill chunk.
    ///
    /// A model may carry one sidecar per chunk width — `ane_prefill-1024`
    /// beside the historical `ane_prefill` (4,096) — because the width that
    /// wins depends on the prompt: 4,096 for long ones, a smaller chunk to
    /// reach the band below it at all. The configured chunk picks the
    /// directory; `init` then insists the sidecar found there was built for
    /// exactly that chunk, so a nearer width is refused rather than run.
    static func sidecarDirectory(
        modelDirectory: URL,
        configChunkTokens: Int
    ) -> URL {
        let specific = modelDirectory.appendingPathComponent(
            "ane_prefill-\(configChunkTokens)", isDirectory: true)
        let meta = specific.appendingPathComponent("ane_prefill.json")
        if FileManager.default.fileExists(atPath: meta.path) {
            return specific
        }
        return modelDirectory.appendingPathComponent(
            "ane_prefill",
            isDirectory: true)
    }

    /// The sidecar's own reader, bounded like every other metadata reader here
    /// (`ManifestReader.load`, `VerifiedInstallReceiptReader.load`,
    /// `PackedExpertsLayoutReader.load`). A sidecar directory is copied off
    /// another machine, so its `ane_prefill.json` may be arbitrarily large
    /// before anyone looks: the cap is applied to the descriptor's size before
    /// the buffer is allocated, never to the bytes after they have been read
    /// (K17), and a link in place of the document is refused rather than
    /// followed.
    static func loadSidecarMetadata(
        at url: URL,
        maxBytes: UInt64 = ManifestReader.defaultMaxBytes
    ) throws -> SidecarMetadata {
        let metaData: Data
        do {
            metaData = try BoundedMetadataRead.read(fileAt: url, maxBytes: maxBytes)
        } catch ModelError.metadataOverBound(let document, let bytes, let cap) {
            // `PrefillError.chunkedUnsupported`, because a sidecar over bound is
            // a re-export question and every caller of this already handles that
            // case as one.
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar metadata \(url.deletingLastPathComponent().path)/\(document) "
                    + "is \(bytes) bytes, over the \(cap)-byte metadata bound; re-export it")
        }
        return try JSONDecoder().decode(SidecarMetadata.self, from: metaData)
    }
}
