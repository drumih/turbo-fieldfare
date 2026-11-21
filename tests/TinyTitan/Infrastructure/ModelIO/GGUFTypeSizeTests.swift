import Testing

@testable import TinyTitan

/// Every GGUF type's (elements per block, bytes per block) pinned to ggml's
/// `GGML_QUANT_SIZES` (llama.cpp `gguf-py/gguf/constants.py`). A wrong row
/// mis-sizes every tensor of that type: NVFP4 was declared one element per
/// block, so the reader computed tensors 64x too large and refused both
/// Inkling-2-Mini GGUFs as running past the end of the file.
@Suite struct GGUFTypeSizeTests {
    static let ggml: [(GGUFType, UInt64, UInt64)] = [
        (.f32, 1, 4),
        (.f16, 1, 2),
        (.q4_0, 32, 18),
        (.q4_1, 32, 20),
        (.q5_0, 32, 22),
        (.q5_1, 32, 24),
        (.q8_0, 32, 34),
        (.q8_1, 32, 36),
        (.q2K, 256, 84),
        (.q3K, 256, 110),
        (.q4K, 256, 144),
        (.q5K, 256, 176),
        (.q6K, 256, 210),
        (.q8K, 256, 292),
        (.iq2XXS, 256, 66),
        (.iq2XS, 256, 74),
        (.iq3XXS, 256, 98),
        (.iq1S, 256, 50),
        (.iq4NL, 32, 18),
        (.iq3S, 256, 110),
        (.iq2S, 256, 82),
        (.iq4XS, 256, 136),
        (.i8, 1, 1),
        (.i16, 1, 2),
        (.i32, 1, 4),
        (.i64, 1, 8),
        (.f64, 1, 8),
        (.iq1M, 256, 56),
        (.bf16, 1, 2),
        (.tq1_0, 256, 54),
        (.tq2_0, 256, 66),
        (.mxfp4, 32, 17),
        (.nvfp4, 64, 36),
        (.q1_0, 128, 18),
        (.q2_0, 64, 18),
        (.f8E4M3, 1, 1),
    ]

    @Test func everyTypeMatchesGGML() {
        #expect(Self.ggml.count == GGUFType.allCases.count)
        for (type, elements, bytes) in Self.ggml {
            #expect(type.blockSize == elements, "\(type.name) block size")
            #expect(type.blockBytes == bytes, "\(type.name) block bytes")
        }
    }

    @Test func nvfp4TensorBytes() {
        // blk.0.ffn_gate_up_exps.weight of UD-Q4_K_XL: 2304 x 2304 x 128.
        let elements: UInt64 = 2_304 * 2_304 * 128
        #expect(elements / GGUFType.nvfp4.blockSize * GGUFType.nvfp4.blockBytes == 382_205_952)
    }
}
