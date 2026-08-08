import Foundation

package enum GTurboFormatV2 {
    package static let magic = "GTURBO"
    package static let versionMajor = 2
    package static let versionMinor = 0
    package static let alignmentBytes: UInt64 = 16_384

    package static let knownFlags: Set<String> = [
        "streamingPresent", "latentMoE", "kdaLayers", "attnRes",
    ]

    // Quant scheme identifiers recorded in the v2 manifest quant slots.
    // affine4-g64 / affine8-g64 are the MLX-style group-64 affine trunk
    // formats shared with the v1 runtime; mxfp4E2M1G32E8M0 is the K3
    // routed-expert format (E2M1 values, one E8M0 scale byte per group of 32)
    // and appears only inside packed_experts blobs, never in the resident file.
    package static let quantSchemeAffine4G64 = "affine4-g64"
    package static let quantSchemeAffine8G64 = "affine8-g64"
    package static let quantSchemeBF16 = "bf16"
    package static let quantSchemeFP32 = "fp32"
    package static let quantSchemeMxfp4E2M1G32E8M0 = "mxfp4E2M1G32E8M0"

    // MXFP4 expert blob schema: six subtensors per expert (v1 blobs had nine).
    // w1/w3 map latent -> intermediate, w2 maps intermediate -> latent.
    package static let mxfp4GroupSize = 32
    package static let mxfp4PackedDType = "U8"
    package static let mxfp4ScaleDType = "U8"
    package static let mxfp4PackedBits = 4
    package static let mxfp4ScaleBits = 8
    package static let mxfp4ExpertMatrixStems = ["w1", "w2", "w3"]
    package static let mxfp4ExpertTensorNames: [String] = [
        "w1_packed", "w1_scales", "w2_packed", "w2_scales", "w3_packed", "w3_scales",
    ]
}
