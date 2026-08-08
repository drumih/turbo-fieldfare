import Foundation

package struct GTurboSubTensorV2: Codable, Equatable, Sendable {
    package let offset: UInt64
    package let size: UInt64
    package let dtype: String
    package let shape: [UInt32]
    package let bits: Int?

    package init(offset: UInt64, size: UInt64, dtype: String,
                 shape: [UInt32], bits: Int?) {
        self.offset = offset
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.bits = bits
    }
}

package struct GTurboExpertV2: Codable, Equatable, Sendable {
    package let expert: Int?
    package let physicalRank: Int?
    package let offset: UInt64
    package let size: UInt64
    package let tensors: [String: GTurboSubTensorV2]

    package init(expert: Int?, physicalRank: Int?, offset: UInt64, size: UInt64,
                 tensors: [String: GTurboSubTensorV2]) {
        self.expert = expert
        self.physicalRank = physicalRank
        self.offset = offset
        self.size = size
        self.tensors = tensors
    }
}

package struct GTurboLayerV2: Codable, Equatable, Sendable {
    package let layer: Int
    package let file: String
    package let experts: [GTurboExpertV2]

    package init(layer: Int, file: String, experts: [GTurboExpertV2]) {
        self.layer = layer
        self.file = file
        self.experts = experts
    }
}

// `numLayers` is the full model depth (matching the manifest). `layers`
// covers exactly the MoE layers: dense layers (manifest arch.denseLayers)
// have no packed-expert file and no entry here.
package struct GTurboPackedExpertsLayoutV2: Codable, Equatable, Sendable {
    package let expertStride: UInt64
    package let numLayers: Int
    package let expertsPerLayer: Int
    package let layers: [GTurboLayerV2]

    package init(expertStride: UInt64, numLayers: Int, expertsPerLayer: Int,
                 layers: [GTurboLayerV2]) {
        self.expertStride = expertStride
        self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer
        self.layers = layers
    }
}

package enum GTurboPackedExpertsLayoutCodecV2 {
    package static func decode(_ data: Data) throws -> GTurboPackedExpertsLayoutV2 {
        let layout: GTurboPackedExpertsLayoutV2
        do { layout = try JSONDecoder().decode(GTurboPackedExpertsLayoutV2.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "packed_experts/layout.json", reason: "\(error)") }
        try GTurboV2StructuralValidator.validate(layout)
        return layout
    }

    package static func encode(_ layout: GTurboPackedExpertsLayoutV2) throws -> Data {
        try GTurboV2StructuralValidator.validate(layout)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(layout))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw GTurboFormatError.invalid(
                field: "packed_experts/layout.json", reason: "\(error)")
        }
    }
}
