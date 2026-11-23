import Foundation

// The repack plan value types: the page layout constant and the resident,
// layer, passthrough and plan structures the planner produces.
//
// Split out of `RepackPlanner.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
enum Layout {
    static let pageBytes: UInt64 = 16_384
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]  // per-entry offsets into the table
    let indexSize: UInt64  // header + entries + table + padding
    let residentSize: UInt64  // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String  // "gate" | "up" | "down"
    let component: String  // "weights" | "scales" | "biases"
    let dtype: UInt8  // 0=U32, 1=BF16
    let logicalShape: [UInt64]  // per-expert logical shape
    let offsetInExpertBlob: UInt64  // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?  // 4 for routed expert weight; nil for scales/biases
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(
        layerIndex: Int,
        path: String,
        expertsPerLayer: Int,
        expertStride: UInt64,
        subTensors: [PerExpertTensorSlice]
    ) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

/// A source file copied into the install verbatim, outside the tensor
/// payload. Qwen3.8-Flash-Next's n-gram table is 102 GB of fp16 rows that the
/// runtime gathers per token straight off storage: it is already
/// row-addressable, so restructuring it would only cost a rewrite of the
/// largest file in the install.
struct PassthroughFile: Sendable, Equatable {
    let sourceName: String
    let destinationName: String
    let size: UInt64
    /// A missing optional file leaves the model runnable in a reduced mode
    /// rather than failing the install.
    let required: Bool
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String  // "affine"
    let baseGroupSize: Int  // 64
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
    let passthroughFiles: [PassthroughFile]
    /// Per-slot quantization scheme names, for a family whose tensors keep a
    /// source encoding (`GGUF_NVFP4`, `GGUF_Q8_0`, ...); a slot absent here is
    /// written with `baseMode`, which is every affine family.
    var slotSchemes: [String: String] = [:]

    /// The stride `manifest.json` records: the largest layer's. Equal to every
    /// layer's for a uniform install; a family whose layers hold experts of
    /// different encodings (Inkling's NVFP4 and Q8_0 layers) writes compact
    /// layers inside it, which the format allows.
    var manifestExpertStride: UInt64 {
        layers.filter { $0.expertsPerLayer > 0 }.map(\.expertStride).max() ?? 0
    }
}
