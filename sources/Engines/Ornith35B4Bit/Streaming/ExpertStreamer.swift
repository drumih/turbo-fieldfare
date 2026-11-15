import Foundation
import TinyTitanFormat

enum StreamerError: Error, CustomStringConvertible {
    case openFailed(path: String, errno: Int32)
    case sizeMismatch(expected: UInt64, actual: UInt64)
    case offsetOutOfRange(UInt64)
    case bufferWrapFailed
    case preadFailed(errno: Int32)
    case allocFailed(errno: Int32)
    case slotOutOfRange(Int)

    public var description: String {
        switch self {
        case .openFailed(let path, let error):
            return "open(\(path)) failed: errno \(error)"
        case .sizeMismatch(let expected, let actual):
            return "file size mismatch: expected \(expected), got \(actual)"
        case .offsetOutOfRange(let offset):
            return "offset \(offset) is outside the streamed range"
        case .bufferWrapFailed:
            return "failed to wrap expert cache memory in an MTLBuffer"
        case .preadFailed(let error):
            return "pread failed: errno \(error)"
        case .allocFailed(let error):
            return "posix_memalign failed: errno \(error)"
        case .slotOutOfRange(let slot):
            return "expert cache slot \(slot) is out of range"
        }
    }
}

/// Byte layout of one routed-expert layer file.
public struct StreamLayout: Sendable {
    public let path: String
    public let streamOffset: UInt64
    public let streamSize: UInt64
    public let expertsPerLayer: Int
    public let expertStride: UInt64
    public let expertOffsets: [UInt64]?
    /// Per-expert payload bytes from the layer layout. `expertStride` is the
    /// slot capacity; a tiered layer's cold experts are shorter than it, so a
    /// read must ask for `size`. `nil` means every expert fills the stride.
    package let expertSizes: [UInt64]?
    /// Per-expert tensor maps for a tiered layer: present only for a cold
    /// expert (weights stored at 2-bit), empty for a hot one. The streamer
    /// runs `TwoBitExpand.expandExpertInPlace` after reading such an expert, so
    /// the slot ends up in the 4-bit layout the kernels expect.
    package let expertTensors: [[ExpertBlobTensor]]?
    /// Bytes of the expert blob that decode's phase 1 (gate+up) consumes,
    /// when the manifest records where the `down` tensor starts. `nil` means
    /// whole-region reads; a split needs a boundary strictly inside the
    /// stride, which the streamer refuses anything else for.
    public let phase1Bytes: UInt64?

    /// Whether every expert fills the slot stride. A tiered layer stores cold
    /// experts shorter than the slot, so the bounded reader reads each expert's
    /// own length and skips the phase-1 split.
    package var hasUniformExpertSizes: Bool {
        expertSizes?.allSatisfy { $0 == expertStride } ?? true
    }

    /// Whether the bounded reader can publish the phase-1 stage for this layer:
    /// the split needs a phase boundary *and* uniform expert sizes. A tiered
    /// layer's read branch never calls `firstStage`, so declaring a stage there
    /// makes the early-miss fixup await the whole load (AUD2-045).
    package var splitsPhase1: Bool {
        phase1Bytes != nil && hasUniformExpertSizes
    }

    /// Per-expert compressed frames, when this layer's blobs are compressed.
    ///
    /// `nil` means the layer is stored raw and every expert is `expertStride`
    /// bytes at `expertOffset(...)`. Present means those offsets address the
    /// *index* instead: each expert is one or two LZFSE frames, and a split
    /// layer has one frame per half so the reader keeps its phase-1 overlap.
    ///
    /// `package` rather than `public`: the frame type lives in
    /// `TinyTitanFormat`, whose types are package-scoped, and nothing outside
    /// this package builds a `StreamLayout`.
    package let compressedFrames: [[ExpertBlobIndex.Blob]]?
    /// The decoded size of one block in a v3 index, or 0 for a legacy (v2)
    /// index whose entries are whole-phase frames.
    package let compressedBlockBytes: UInt64
    /// The rANS sidecar, when the layer is entropy-coded for the GPU.
    package let rans: ExpertRANSIndex.Layout?
    /// Phase 0/1 per block index, recomputed from the layer's tensors.
    package let ransPhases: [UInt8]

    package init(
        path: String,
        streamOffset: UInt64,
        streamSize: UInt64,
        expertsPerLayer: Int,
        expertStride: UInt64,
        expertOffsets: [UInt64]? = nil,
        expertSizes: [UInt64]? = nil,
        expertTensors: [[ExpertBlobTensor]]? = nil,
        phase1Bytes: UInt64? = nil,
        compressedFrames: [[ExpertBlobIndex.Blob]]? = nil,
        compressedBlockBytes: UInt64 = 0,
        rans: ExpertRANSIndex.Layout? = nil,
        ransPhases: [UInt8] = []
    ) {
        self.path = path
        self.streamOffset = streamOffset
        self.streamSize = streamSize
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.expertOffsets = expertOffsets
        self.expertSizes = expertSizes
        self.expertTensors = expertTensors
        self.phase1Bytes = phase1Bytes
        self.compressedFrames = compressedFrames
        self.compressedBlockBytes = compressedBlockBytes
        self.rans = rans
        self.ransPhases = ransPhases
    }

    /// The byte offset of one expert within this layer's file.
    ///
    /// Unchecked on purpose, and safe only because the type's contract is
    /// validated where the layout is built: `perLayer` must be representable
    /// (`expertStride * expertsPerLayer`, checked in `Model` before the
    /// `StreamLayout` is formed) and `layer`/`expert` must be within the model's
    /// and the layout's bounds. Checking here would put a reporting multiply on
    /// every expert read, which is the whole decode path.
    @inline(always)
    public func expertOffset(layer: Int, expert: Int) -> UInt64 {
        if layer == 0, let expertOffsets, expert >= 0, expert < expertOffsets.count {
            return expertOffsets[expert]
        }
        let perLayer = UInt64(expertsPerLayer) * expertStride
        return UInt64(layer) * perLayer + UInt64(expert) * expertStride
    }
}
