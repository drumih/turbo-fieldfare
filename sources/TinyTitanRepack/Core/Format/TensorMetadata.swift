import Foundation

/// One physical tensor in a safetensors shard. Coordinates are absolute file
/// offsets so the writer can map and copy a single tensor without re-parsing
/// the shard header.
struct SourceTensor: Sendable, Hashable {
    enum Dtype: UInt8, Sendable, Hashable {
        case u32 = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
        /// Bytes copied verbatim in a source encoding the plan entry names
        /// (a GGUF block, NVFP4, MXFP8): the source tensor is a byte range, and
        /// the destination's dtype code says what the bytes are.
        case opaque = 255

        var elementBytes: Int {
            switch self {
            case .u32: 4
            case .bf16: 2
            case .fp16: 2
            case .fp32: 4
            case .opaque: 1
            }
        }
    }

    let name: String
    let shardPath: String
    let dtype: Dtype
    let shape: [UInt64]
    let absoluteOffset: UInt64
    let sizeBytes: UInt64
}

/// Bit-width override resolved from `config.json -> quantization`.
struct QuantSpec: Sendable, Hashable {
    let bits: Int
}
