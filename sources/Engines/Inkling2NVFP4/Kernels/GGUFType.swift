// Provenance: trimmed from sources/TinyTitan/Infrastructure/ModelIO/GGUFType.swift
// (TT-121 phase 2) under the variant-isolation rule: only the ggml types the
// Inkling 2 Q4_K_XL install keeps, with their block geometry and names.

/// A ggml tensor type the variant runs.
enum GGUFType: UInt32, CaseIterable, Sendable {
    case f32 = 0
    case f16 = 1
    case q8_0 = 8
    case bf16 = 30
    case nvfp4 = 40
    /// Not a ggml type: the checkpoint's MXFP8 experts repacked into 34-byte
    /// blocks of 32 values (E8M0 scale, a zero byte, 32 E4M3 values), TT-125.
    case mxfp8 = 0x4D58_3800

    /// Values per block.
    var blockSize: UInt64 {
        switch self {
        case .f32, .f16, .bf16: return 1
        case .q8_0, .mxfp8: return 32
        case .nvfp4: return 64
        }
    }

    /// Bytes per block.
    var blockBytes: UInt64 {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q8_0, .mxfp8: return 34
        case .nvfp4: return 36
        }
    }

    /// GGML's spelling, for diagnostics.
    var name: String {
        switch self {
        case .f32: return "F32"
        case .f16: return "F16"
        case .q8_0: return "Q8_0"
        case .bf16: return "BF16"
        case .nvfp4: return "NVFP4"
        case .mxfp8: return "MXFP8"
        }
    }
}

/// A tensor encoding the variant has no kernel for.
enum GGUFDequantError: Error, Equatable, CustomStringConvertible {
    case unsupportedType(String)

    var description: String {
        switch self {
        case .unsupportedType(let name):
            return "the Inkling 2 Q4 variant has no kernel for \(name)"
        }
    }
}
