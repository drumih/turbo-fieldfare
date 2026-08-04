import Foundation

package struct GTurboSubTensorV1: Codable, Equatable, Sendable {
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

package struct GTurboExpertV1: Codable, Equatable, Sendable {
    package let expert: Int?
    package let physicalRank: Int?
    package let offset: UInt64
    package let size: UInt64
    package let tensors: [String: GTurboSubTensorV1]

    package init(expert: Int?, physicalRank: Int?, offset: UInt64, size: UInt64,
                 tensors: [String: GTurboSubTensorV1]) {
        self.expert = expert
        self.physicalRank = physicalRank
        self.offset = offset
        self.size = size
        self.tensors = tensors
    }
}

package struct GTurboLayerV1: Codable, Equatable, Sendable {
    package let layer: Int
    package let file: String
    package let experts: [GTurboExpertV1]

    package init(layer: Int, file: String, experts: [GTurboExpertV1]) {
        self.layer = layer
        self.file = file
        self.experts = experts
    }
}

package struct GTurboPackedExpertsLayoutV1: Codable, Equatable, Sendable {
    package let expertStride: UInt64
    package let numLayers: Int
    package let expertsPerLayer: Int
    package let layers: [GTurboLayerV1]

    package init(expertStride: UInt64, numLayers: Int, expertsPerLayer: Int,
                 layers: [GTurboLayerV1]) {
        self.expertStride = expertStride
        self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer
        self.layers = layers
    }
}

package enum GTurboPackedExpertsLayoutCodec {
    package static func decode(_ data: Data) throws -> GTurboPackedExpertsLayoutV1 {
        let layout: GTurboPackedExpertsLayoutV1
        do { layout = try JSONDecoder().decode(GTurboPackedExpertsLayoutV1.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "packed_experts/layout.json", reason: "\(error)") }
        try GTurboV1StructuralValidator.validate(layout)
        return layout
    }

    package static func encode(_ layout: GTurboPackedExpertsLayoutV1) throws -> Data {
        try GTurboV1StructuralValidator.validate(layout)
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

package enum GTurboV1StructuralValidator {
    package static func validate(_ layout: GTurboPackedExpertsLayoutV1) throws {
        guard layout.numLayers > 0, layout.expertsPerLayer > 0,
              layout.expertStride > 0,
              layout.expertStride % GTurboFormatV1.alignmentBytes == 0,
              layout.layers.count == layout.numLayers else {
            throw GTurboFormatError.invalid(field: "layout", reason: "invalid dimensions or stride")
        }
        var layerIDs = Set<Int>()
        var layerFiles = Set<String>()
        for layer in layout.layers {
            guard layer.layer >= 0, layer.layer < layout.numLayers,
                  layerIDs.insert(layer.layer).inserted else {
                throw GTurboFormatError.invalid(field: "layout.layers", reason: "duplicate or invalid layer")
            }
            try GTurboPathValidator.validateBasename(layer.file,
                                                      field: "layout.layers[\(layer.layer)].file")
            let fileKey = GTurboPathValidator.appleFilesystemKey(layer.file)
            guard fileKey != "layout.json" else {
                throw GTurboFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].file",
                    reason: "reserved packed-expert filename")
            }
            guard layerFiles.insert(fileKey).inserted else {
                throw GTurboFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].file",
                    reason: "duplicate layer filename")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw GTurboFormatError.invalid(field: "layout.layers[\(layer.layer)].experts",
                                                reason: "wrong expert count")
            }
            var logicalIDs = Set<Int>()
            var physicalRanks = Set<Int>()
            var offsets = Set<UInt64>()
            let hasExplicitLogicalIDs = layer.experts.map(\.expert)
            guard hasExplicitLogicalIDs.allSatisfy({ $0 == nil })
                    || hasExplicitLogicalIDs.allSatisfy({ $0 != nil }) else {
                throw GTurboFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].experts",
                    reason: "expert ids must be either all explicit or all positional")
            }
            for (position, expert) in layer.experts.enumerated() {
                let logical = expert.expert ?? position
                let physical = expert.physicalRank ?? logical
                guard logical >= 0, logical < layout.expertsPerLayer,
                      physical >= 0, physical < layout.expertsPerLayer,
                      logicalIDs.insert(logical).inserted,
                      physicalRanks.insert(physical).inserted,
                      offsets.insert(expert.offset).inserted else {
                    throw GTurboFormatError.invalid(field: "layout.layers[\(layer.layer)].experts",
                                                    reason: "duplicate or invalid expert mapping")
                }
                let expectedOffset = try gturboCheckedMultiply(UInt64(physical), layout.expertStride,
                                                               field: "expert.offset")
                guard expert.offset == expectedOffset, expert.size == layout.expertStride else {
                    throw GTurboFormatError.invalid(field: "expert[\(logical)]",
                                                    reason: "offset or size does not match physical rank")
                }
                var tensorRanges: [(start: UInt64, end: UInt64, name: String)] = []
                for (name, tensor) in expert.tensors {
                    guard tensor.dtype == "U32" || tensor.dtype == "BF16",
                          !name.isEmpty, tensor.size > 0,
                          !tensor.shape.isEmpty,
                          tensor.shape.allSatisfy({ $0 > 0 }),
                          tensor.bits.map({ $0 > 0 && $0 <= 32 }) ?? true else {
                        throw GTurboFormatError.invalid(field: "expert[\(logical)].tensors.\(name)",
                                                        reason: "invalid dtype or shape")
                    }
                    let end = try gturboCheckedAdd(tensor.offset, tensor.size,
                                                   field: "tensor.\(name).range")
                    guard end <= expert.size else {
                        throw GTurboFormatError.invalid(field: "expert[\(logical)].tensors.\(name)",
                                                        reason: "range exceeds expert blob")
                    }
                    tensorRanges.append((tensor.offset, end, name))
                }
                let sortedRanges = tensorRanges.sorted {
                    $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start
                }
                for pair in zip(sortedRanges, sortedRanges.dropFirst())
                    where pair.0.end > pair.1.start {
                    throw GTurboFormatError.invalid(
                        field: "expert[\(logical)].tensors",
                        reason: "overlapping ranges \(pair.0.name) and \(pair.1.name)")
                }
            }
        }
    }

    package static func crossValidate(manifest: GTurboManifestV1,
                                      layout: GTurboPackedExpertsLayoutV1) throws {
        try crossValidate(
            manifestNumLayers: manifest.numLayers,
            manifestExpertsPerLayer: manifest.expertsPerLayer,
            manifestExpertStride: manifest.expertStride,
            manifestFileSizes: manifest.files.mapValues(\.size),
            layout: layout)
    }

    package static func crossValidate(
        manifestNumLayers: Int,
        manifestExpertsPerLayer: Int,
        manifestExpertStride: UInt64,
        manifestFileSizes: [String: UInt64],
        layout: GTurboPackedExpertsLayoutV1
    ) throws {
        guard manifestNumLayers == layout.numLayers,
              manifestExpertsPerLayer == layout.expertsPerLayer,
              manifestExpertStride == layout.expertStride else {
            throw GTurboFormatError.invalid(field: "manifest/layout",
                                            reason: "dimension mismatch")
        }
        let expectedLayerSize = try gturboCheckedMultiply(UInt64(layout.expertsPerLayer),
                                                          layout.expertStride,
                                                          field: "layout.layerSize")
        for layer in layout.layers {
            let path = "packed_experts/\(layer.file)"
            guard manifestFileSizes[path] == expectedLayerSize else {
                throw GTurboFormatError.invalid(field: "manifest.files.\(path)",
                                                reason: "missing or wrong layer size")
            }
        }
    }
}
