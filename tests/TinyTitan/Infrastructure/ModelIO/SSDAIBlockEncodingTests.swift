import Foundation
import Testing

@testable import TinyTitanFormat

/// The `.ssdai` block encodings and variant geometry (TT-121 phase 0).
///
/// The sizes are Inkling 2 Mini's own: one expert's fused gate/up projection is
/// 2,304 rows of 2,304 values, stored as GGUF NVFP4 (the Q4_K_XL file) or as
/// ModelOpt NVFP4 / OCP MXFP8 (the original checkpoint). The GGUF byte count is
/// the per-expert slice measured in the file (2,985,984), and the checkpoint's
/// scale shapes are its header's (`[..., 144]` for NVFP4, `[..., 72]` for MXFP8).
@Suite struct SSDAIBlockEncodingTests {
    @Test func byteCountsMatchTheInklingSources() throws {
        let rows: UInt64 = 2_304
        let columns: UInt64 = 2_304
        let f = "t"
        #expect(
            try SSDAIBlockEncodingV1.ggufNVFP4.payloadBytes(rows: rows, columns: columns, field: f)
                == 2_985_984)
        #expect(
            try SSDAIBlockEncodingV1.ggufNVFP4.scaleBytes(rows: rows, columns: columns, field: f)
                == 0)
        #expect(
            try SSDAIBlockEncodingV1.ggufQ8_0.payloadBytes(rows: rows, columns: columns, field: f)
                == 5_640_192)
        #expect(
            try SSDAIBlockEncodingV1.nvfp4.payloadBytes(rows: rows, columns: columns, field: f)
                == 2_654_208)
        #expect(
            try SSDAIBlockEncodingV1.nvfp4.scaleBytes(rows: rows, columns: columns, field: f)
                == rows * 144)
        #expect(
            try SSDAIBlockEncodingV1.mxfp8.payloadBytes(rows: rows, columns: columns, field: f)
                == 5_308_416)
        #expect(
            try SSDAIBlockEncodingV1.mxfp8.scaleBytes(rows: rows, columns: columns, field: f)
                == rows * 72)
    }

    @Test func aRowThatDoesNotFillABlockIsRefused() {
        #expect(throws: TinyTitanFormatError.self) {
            _ = try SSDAIBlockEncodingV1.ggufNVFP4.payloadBytes(rows: 1, columns: 96, field: "t")
        }
        #expect(throws: TinyTitanFormatError.self) {
            _ = try SSDAIBlockEncodingV1.mxfp8.scaleBytes(rows: 1, columns: 48, field: "t")
        }
    }

    @Test func aThreeDimensionalShapeReadsAsStackedRows() throws {
        let matrix = try SSDAIBlockEncodingV1.matrix(of: [128, 2_304, 2_304, 0], field: "t")
        #expect(matrix.rows == 128 * 2_304)
        #expect(matrix.columns == 2_304)
    }

    @Test func everyEncodingHasItsOwnResidentDType() {
        let codes = SSDAIBlockEncodingV1.allCases.map(\.residentDType.rawValue)
        #expect(Set(codes).count == SSDAIBlockEncodingV1.allCases.count)
        #expect(codes.allSatisfy { $0 >= 4 })
        for encoding in SSDAIBlockEncodingV1.allCases {
            #expect(SSDAIBlockEncodingV1(residentDType: encoding.residentDType) == encoding)
        }
        #expect(SSDAIBlockEncodingV1(residentDType: .bf16) == nil)
    }
}

extension SSDAIBlockEncodingTests {
    /// A one-entry resident index, 16 KiB, with the payload right after it.
    static func index(
        dtype: UInt8, shape: [UInt32], size: UInt64, scaleSize: UInt64 = 0
    ) -> (bytes: [UInt8], header: SSDAIResidentIndexHeaderV1) {
        let indexSize = Int(SSDAIFormatV1.alignmentBytes)
        var bytes = [UInt8](repeating: 0, count: indexSize)
        func put(_ value: UInt64, _ offset: Int, _ width: Int) {
            for byte in 0..<width {
                bytes[offset + byte] = UInt8((value >> (8 * UInt64(byte))) & 0xff)
            }
        }
        let name = Array("blk.0.attn_q.weight".utf8)
        let entry = 24
        let nameAt = entry + 72
        put(UInt64(indexSize), 0, 8)
        put(size + scaleSize, 8, 8)
        put(1, 16, 8)
        put(UInt64(nameAt), entry, 4)
        put(UInt64(name.count), entry + 4, 2)
        bytes[entry + 6] = dtype
        put(UInt64(indexSize), entry + 8, 8)
        put(size, entry + 16, 8)
        for (axis, dimension) in shape.enumerated() {
            put(UInt64(dimension), entry + 24 + 4 * axis, 4)
        }
        if scaleSize > 0 {
            put(UInt64(indexSize) + size, entry + 40, 8)
            put(scaleSize, entry + 48, 8)
        }
        bytes.replaceSubrange(nameAt..<nameAt + name.count, with: name)
        let header = SSDAIResidentIndexHeaderV1(
            indexSize: UInt64(indexSize), residentSize: size + scaleSize, entryCount: 1)
        return (bytes, header)
    }

    static func decode(_ built: (bytes: [UInt8], header: SSDAIResidentIndexHeaderV1)) throws
        -> [SSDAIResidentIndexEntryV1]
    {
        try built.bytes.withUnsafeBytes {
            try SSDAIResidentIndexCodec.decodeRegion($0, header: built.header)
        }
    }

    @Test func aCorrectlySizedBlockEntryDecodes() throws {
        let gguf = Self.index(dtype: 5, shape: [2_304, 2_304, 0, 0], size: 2_985_984)
        #expect(try Self.decode(gguf).first?.dtype == 5)
        let nvfp4 = Self.index(
            dtype: 6, shape: [2_304, 2_304, 0, 0], size: 2_654_208, scaleSize: 2_304 * 144)
        #expect(try Self.decode(nvfp4).first?.dtype == 6)
    }

    @Test func aMisSizedBlockEntryIsRefused() {
        // One block short: a kernel would read 36 bytes past the tensor.
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(Self.index(dtype: 5, shape: [2_304, 2_304, 0, 0], size: 2_985_948))
        }
        // NVFP4 without its separate scales.
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(Self.index(dtype: 6, shape: [2_304, 2_304, 0, 0], size: 2_654_208))
        }
        // An unknown code is still refused by name.
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(Self.index(dtype: 9, shape: [16, 0, 0, 0], size: 64))
        }
    }
}

extension SSDAIBlockEncodingTests {
    static func layout(gateUpSize: UInt64) -> SSDAIPackedExpertsLayoutV1 {
        let stride: UInt64 = 183 * 16_384
        let expert = SSDAIExpertV1(
            expert: nil, physicalRank: nil, offset: 0, size: gateUpSize + 4,
            tensors: [
                "gate_up": SSDAISubTensorV1(
                    offset: 0, size: gateUpSize, dtype: "GGUF_NVFP4", shape: [2_304, 2_304], bits: 4
                ),
                "gate_up_scale": SSDAISubTensorV1(
                    offset: gateUpSize, size: 4, dtype: "F32", shape: [1], bits: nil),
            ])
        return SSDAIPackedExpertsLayoutV1(
            expertStride: stride, numLayers: 1, expertsPerLayer: 1,
            layers: [SSDAILayerV1(layer: 0, file: "layer_00.bin", experts: [expert])])
    }

    @Test func anExpertBlobMayHoldBlockEncodedTensors() throws {
        try SSDAIV1StructuralValidator.validate(Self.layout(gateUpSize: 2_985_984))
    }

    @Test func aMisSizedExpertTensorIsRefused() {
        #expect(throws: TinyTitanFormatError.self) {
            try SSDAIV1StructuralValidator.validate(Self.layout(gateUpSize: 2_985_948))
        }
    }

    @Test func variantGeometryRoundTripsAndIsValidated() throws {
        let geometry = SSDAIVariantGeometryV1(
            scalars: ["loopPasses": 3, "relativeClip": 30.5],
            lists: ["localLayers": [0, 1, 2, 3, 4, 6]])
        let decoded = try JSONDecoder().decode(
            SSDAIVariantGeometryV1.self, from: JSONEncoder().encode(geometry))
        #expect(decoded == geometry)
        try decoded.validate(field: "arch.variantGeometry")
        #expect(throws: TinyTitanFormatError.self) {
            try SSDAIVariantGeometryV1(scalars: ["softcap": .infinity]).validate(field: "g")
        }
        #expect(throws: TinyTitanFormatError.self) {
            try SSDAIVariantGeometryV1(scalars: ["x": 1], lists: ["x": [1]]).validate(field: "g")
        }
    }
}
