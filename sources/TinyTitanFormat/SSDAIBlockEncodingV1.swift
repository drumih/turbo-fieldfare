import Foundation

/// Block-quantized tensor encodings an install stores byte for byte from its
/// source, rather than re-encoding them to the affine u32 packing (TT-121).
///
/// The format knows their byte layouts and nothing else: what a block means
/// numerically is the reading variant's business, and a variant that did not
/// compile a kernel for an encoding refuses it. Each case pins exactly how many
/// bytes a `rows x columns` matrix occupies, so the index and the expert layout
/// can refuse a payload of the wrong length instead of reading past it — the
/// same guarantee the affine scales span gives `dtype 0`.
///
/// | Encoding | Source | Bytes |
/// | --- | --- | --- |
/// | `GGUF_Q8_0` | ggml `block_q8_0` | 34 per 32 values: f16 scale, 32 int8 |
/// | `GGUF_NVFP4` | ggml `block_nvfp4` | 36 per 64 values: 4 UE4M3 sub-block scales, 32 bytes E2M1 |
/// | `NVFP4` | ModelOpt NVFP4 | values: 2 E2M1 a byte; scales: 1 E4M3 byte per 16 values, separate |
/// | `MXFP8` | OCP MX FP8 | values: 1 E4M3 byte each; scales: 1 UE8M0 byte per 32 values, separate |
///
/// The GGUF encodings carry their scales inside each block, so an entry has no
/// separate scale range. `NVFP4` and `MXFP8` keep the source's separate scale
/// tensor; in the resident index it is the entry's scale range, and in an
/// expert blob it is a sibling sub-tensor. Per-tensor or per-expert global
/// scales (an NVFP4 `amax`, a GGUF per-expert `scale`) are ordinary `F32`
/// tensors of their own.
package enum SSDAIBlockEncodingV1: String, CaseIterable, Sendable {
    case ggufQ8_0 = "GGUF_Q8_0"
    case ggufNVFP4 = "GGUF_NVFP4"
    case nvfp4 = "NVFP4"
    case mxfp8 = "MXFP8"

    /// The resident-index `dtype` an entry in this encoding carries.
    package var residentDType: SSDAIFormatV1.DType {
        switch self {
        case .ggufQ8_0: .ggufQ8_0
        case .ggufNVFP4: .ggufNVFP4
        case .nvfp4: .nvfp4
        case .mxfp8: .mxfp8
        }
    }

    package init?(residentDType: SSDAIFormatV1.DType) {
        guard let match = Self.allCases.first(where: { $0.residentDType == residentDType })
        else { return nil }
        self = match
    }

    /// Values one block (GGUF) or one scale (NVFP4, MXFP8) covers; a row's
    /// length must be a multiple of it.
    package var valuesPerBlock: Int {
        switch self {
        case .ggufQ8_0: 32
        case .ggufNVFP4: 64
        case .nvfp4: 16
        case .mxfp8: 32
        }
    }

    /// Bytes of the values payload of a `rows x columns` matrix.
    package func payloadBytes(rows: UInt64, columns: UInt64, field: String) throws -> UInt64 {
        try checkColumns(columns, field: field)
        let values = try ssdaiCheckedMultiply(rows, columns, field: field)
        switch self {
        case .ggufQ8_0: return try ssdaiCheckedMultiply(values / 32, 34, field: field)
        case .ggufNVFP4: return try ssdaiCheckedMultiply(values / 64, 36, field: field)
        case .nvfp4: return values / 2
        case .mxfp8: return values
        }
    }

    /// Bytes of the separate scale tensor; zero for the GGUF encodings, whose
    /// scales are inside each block.
    package func scaleBytes(rows: UInt64, columns: UInt64, field: String) throws -> UInt64 {
        try checkColumns(columns, field: field)
        let values = try ssdaiCheckedMultiply(rows, columns, field: field)
        switch self {
        case .ggufQ8_0, .ggufNVFP4: return 0
        case .nvfp4: return values / 16
        case .mxfp8: return values / 32
        }
    }

    private func checkColumns(_ columns: UInt64, field: String) throws {
        guard columns > 0, columns % UInt64(valuesPerBlock) == 0 else {
            throw TinyTitanFormatError.invalid(
                field: field,
                reason:
                    "\(rawValue) needs a row length that is a multiple of \(valuesPerBlock), got \(columns)"
            )
        }
    }

    /// `rows x columns` of a shape: the last non-zero dimension is the row
    /// length and every dimension before it multiplies into the row count, so a
    /// 3-D `[experts, rows, columns]` tensor reads as `experts * rows` rows.
    package static func matrix(of shape: [UInt32], field: String) throws -> (
        rows: UInt64, columns: UInt64
    ) {
        let dimensions = shape.prefix { $0 > 0 }.map(UInt64.init)
        guard let columns = dimensions.last else {
            throw TinyTitanFormatError.invalid(field: field, reason: "empty shape")
        }
        var rows: UInt64 = 1
        for dimension in dimensions.dropLast() {
            rows = try ssdaiCheckedMultiply(rows, dimension, field: field)
        }
        return (rows, columns)
    }
}

/// Element types an expert blob's auxiliary sub-tensors may use beside the
/// affine `U32`/`BF16` and the block encodings: global scales (`F32`) and the
/// separate NVFP4/MXFP8 scale tensors (`F8_E4M3`, `UE8M0`).
package enum SSDAIAuxiliaryDTypeV1: String, CaseIterable, Sendable {
    case f32 = "F32"
    case f8e4m3 = "F8_E4M3"
    case ue8m0 = "UE8M0"

    package var bytesPerElement: UInt64 {
        switch self {
        case .f32: 4
        case .f8e4m3, .ue8m0: 1
        }
    }
}

extension SSDAIPackedExpertsLayoutV1 {
    /// The sub-tensor element types an expert blob may hold: the affine
    /// `U32`/`BF16`, a block encoding, or an auxiliary scale type.
    static func isKnownSubTensorDType(_ dtype: String) -> Bool {
        dtype == "U32" || dtype == "BF16"
            || SSDAIBlockEncodingV1(rawValue: dtype) != nil
            || SSDAIAuxiliaryDTypeV1(rawValue: dtype) != nil
    }

    /// Byte-exact sizes for the encodings that define them: a block-encoded
    /// sub-tensor holds its values only (its separate scales are a sibling),
    /// and an auxiliary one holds `elements x width`. The affine types keep
    /// their existing range-only checks.
    static func validateSubTensorSize(_ tensor: SSDAISubTensorV1, field: String) throws {
        let expected: UInt64
        if let encoding = SSDAIBlockEncodingV1(rawValue: tensor.dtype) {
            let (rows, columns) = try SSDAIBlockEncodingV1.matrix(of: tensor.shape, field: field)
            expected = try encoding.payloadBytes(rows: rows, columns: columns, field: field)
        } else if let auxiliary = SSDAIAuxiliaryDTypeV1(rawValue: tensor.dtype) {
            var elements: UInt64 = 1
            for dimension in tensor.shape {
                elements = try ssdaiCheckedMultiply(elements, UInt64(dimension), field: field)
            }
            expected = try ssdaiCheckedMultiply(elements, auxiliary.bytesPerElement, field: field)
        } else {
            return
        }
        guard tensor.size == expected else {
            throw TinyTitanFormatError.invalid(
                field: field,
                reason:
                    "\(tensor.dtype) \(tensor.shape) needs \(expected) bytes, declares \(tensor.size)"
            )
        }
    }
}
