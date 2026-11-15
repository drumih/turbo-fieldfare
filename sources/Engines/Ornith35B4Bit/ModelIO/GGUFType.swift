import Foundation

package enum GGUFType: UInt32, CaseIterable, Sendable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2K = 10
    case q3K = 11
    case q4K = 12
    case q5K = 13
    case q6K = 14
    case q8K = 15
    case iq2XXS = 16
    case iq2XS = 17
    case iq3XXS = 18
    case iq1S = 19
    case iq4NL = 20
    case iq3S = 21
    case iq2S = 22
    case iq4XS = 23
    case i8 = 24
    case i16 = 25
    case i32 = 26
    case i64 = 27
    case f64 = 28
    case iq1M = 29
    case bf16 = 30
    case tq1_0 = 34
    case tq2_0 = 35
    case mxfp4 = 39
    case nvfp4 = 40
    case q1_0 = 41
    case q2_0 = 42
    case f8E4M3 = 43

    /// Elements per block. 1 for the unpacked types.
    package var blockSize: UInt64 {
        switch self {
        case .f32, .f16, .i8, .i16, .i32, .i64, .f64, .bf16, .nvfp4, .f8E4M3:
            return 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4NL, .mxfp4:
            return 32
        case .q2_0:
            return 64
        case .q1_0:
            return 128
        case .q2K, .q3K, .q4K, .q5K, .q6K, .q8K,
            .iq2XXS, .iq2XS, .iq3XXS, .iq1S, .iq3S, .iq2S, .iq4XS, .iq1M,
            .tq1_0, .tq2_0:
            return 256
        }
    }

    /// Bytes per block, the packed size including scales.
    package var blockBytes: UInt64 {
        switch self {
        case .f32, .i32: return 4
        case .f16, .i16, .bf16: return 2
        case .i8, .f8E4M3: return 1
        case .i64, .f64: return 8
        case .q4_0, .iq4NL: return 18
        case .q4_1: return 20
        case .q5_0: return 22
        case .q5_1: return 24
        case .q8_0: return 34
        case .q8_1: return 36
        case .q2K: return 84
        case .q3K, .iq3S: return 110
        case .q4K: return 144
        case .q5K: return 176
        case .q6K: return 210
        case .q8K: return 292
        case .iq2XXS: return 66
        case .iq2XS: return 74
        case .iq3XXS: return 98
        case .iq1S: return 50
        case .iq2S: return 82
        case .iq4XS: return 136
        case .iq1M: return 56
        case .tq1_0: return 54
        case .tq2_0: return 66
        case .mxfp4: return 17
        case .nvfp4: return 36
        case .q1_0, .q2_0: return 18
        }
    }

    /// GGML's spelling, for diagnostics and the inventory report.
    package var name: String {
        switch self {
        case .f32: return "F32"
        case .f16: return "F16"
        case .q4_0: return "Q4_0"
        case .q4_1: return "Q4_1"
        case .q5_0: return "Q5_0"
        case .q5_1: return "Q5_1"
        case .q8_0: return "Q8_0"
        case .q8_1: return "Q8_1"
        case .q2K: return "Q2_K"
        case .q3K: return "Q3_K"
        case .q4K: return "Q4_K"
        case .q5K: return "Q5_K"
        case .q6K: return "Q6_K"
        case .q8K: return "Q8_K"
        case .iq2XXS: return "IQ2_XXS"
        case .iq2XS: return "IQ2_XS"
        case .iq3XXS: return "IQ3_XXS"
        case .iq1S: return "IQ1_S"
        case .iq4NL: return "IQ4_NL"
        case .iq3S: return "IQ3_S"
        case .iq2S: return "IQ2_S"
        case .iq4XS: return "IQ4_XS"
        case .i8: return "I8"
        case .i16: return "I16"
        case .i32: return "I32"
        case .i64: return "I64"
        case .f64: return "F64"
        case .iq1M: return "IQ1_M"
        case .bf16: return "BF16"
        case .tq1_0: return "TQ1_0"
        case .tq2_0: return "TQ2_0"
        case .mxfp4: return "MXFP4"
        case .nvfp4: return "NVFP4"
        case .q1_0: return "Q1_0"
        case .q2_0: return "Q2_0"
        case .f8E4M3: return "F8_E4M3"
        }
    }
}

/// A metadata value, in the GGUF wire types.
