import Foundation

package enum GTurboV2StructuralValidator {
    package static func validate(_ layout: GTurboPackedExpertsLayoutV2) throws {
        guard layout.numLayers > 0, layout.expertsPerLayer > 0,
              layout.expertStride > 0,
              layout.expertStride % GTurboFormatV2.alignmentBytes == 0,
              !layout.layers.isEmpty, layout.layers.count <= layout.numLayers else {
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
                guard Set(expert.tensors.keys) == Set(GTurboFormatV2.mxfp4ExpertTensorNames) else {
                    throw GTurboFormatError.invalid(field: "expert[\(logical)].tensors",
                                                    reason: "expected the six MXFP4 subtensors")
                }
                for stem in GTurboFormatV2.mxfp4ExpertMatrixStems {
                    let packed = expert.tensors["\(stem)_packed"]!
                    let scales = expert.tensors["\(stem)_scales"]!
                    try validateMXFP4Tensor(packed, bits: GTurboFormatV2.mxfp4PackedBits,
                                            field: "expert[\(logical)].tensors.\(stem)_packed")
                    try validateMXFP4Tensor(scales, bits: GTurboFormatV2.mxfp4ScaleBits,
                                            field: "expert[\(logical)].tensors.\(stem)_scales")
                    // One E8M0 scale byte per group of 32 packed values.
                    guard scales.shape[0] == packed.shape[0],
                          UInt64(scales.shape[1]) * UInt64(GTurboFormatV2.mxfp4GroupSize)
                            == UInt64(packed.shape[1]) else {
                        throw GTurboFormatError.invalid(
                            field: "expert[\(logical)].tensors.\(stem)_scales",
                            reason: "scale shape does not match packed shape")
                    }
                }
                var tensorRanges: [(start: UInt64, end: UInt64, name: String)] = []
                for (name, tensor) in expert.tensors {
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

    private static func validateMXFP4Tensor(_ tensor: GTurboSubTensorV2,
                                            bits: Int, field: String) throws {
        guard tensor.dtype == GTurboFormatV2.mxfp4PackedDType,
              tensor.bits == bits, tensor.size > 0,
              tensor.shape.count == 2,
              tensor.shape.allSatisfy({ $0 > 0 }) else {
            throw GTurboFormatError.invalid(field: field, reason: "invalid dtype or shape")
        }
        let elements = UInt64(tensor.shape[0]) * UInt64(tensor.shape[1])
        let bitCount = try gturboCheckedMultiply(elements, UInt64(bits), field: field)
        guard bitCount % 8 == 0, tensor.size == bitCount / 8 else {
            throw GTurboFormatError.invalid(field: field, reason: "size does not match shape and bits")
        }
    }

    package static func crossValidate(manifest: GTurboManifestV2,
                                      layout: GTurboPackedExpertsLayoutV2) throws {
        try GTurboManifestCodecV2.validate(manifest)
        try crossValidate(
            manifestNumLayers: manifest.numLayers,
            manifestExpertsPerLayer: manifest.expertsPerLayer,
            manifestExpertStride: manifest.expertStride,
            manifestDenseLayers: manifest.arch.denseLayers,
            manifestFileSizes: manifest.files.mapValues(\.size),
            layout: layout)
    }

    package static func crossValidate(
        manifestNumLayers: Int,
        manifestExpertsPerLayer: Int,
        manifestExpertStride: UInt64,
        manifestDenseLayers: [Int],
        manifestFileSizes: [String: UInt64],
        layout: GTurboPackedExpertsLayoutV2
    ) throws {
        guard manifestNumLayers == layout.numLayers,
              manifestExpertsPerLayer == layout.expertsPerLayer,
              manifestExpertStride == layout.expertStride else {
            throw GTurboFormatError.invalid(field: "manifest/layout",
                                            reason: "dimension mismatch")
        }
        guard manifestDenseLayers.allSatisfy({ $0 >= 1 && $0 <= manifestNumLayers }) else {
            throw GTurboFormatError.invalid(field: "manifest/layout",
                                            reason: "dense layer id out of range")
        }
        // Layout entries cover exactly the MoE layers: full depth minus the
        // dense layers (recorded 1-based in the manifest, 0-based here).
        let denseZeroBased = Set(manifestDenseLayers.map { $0 - 1 })
        let expectedLayers = Set(0..<manifestNumLayers).subtracting(denseZeroBased)
        guard Set(layout.layers.map(\.layer)) == expectedLayers else {
            throw GTurboFormatError.invalid(field: "manifest/layout",
                                            reason: "layout layers do not match the MoE layer set")
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
