import Foundation

package struct SSDAISubTensorV1: Codable, Equatable, Sendable {
    package let offset: UInt64
    package let size: UInt64
    package let dtype: String
    package let shape: [UInt32]
    package let bits: Int?

    package init(
        offset: UInt64, size: UInt64, dtype: String,
        shape: [UInt32], bits: Int?
    ) {
        self.offset = offset
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.bits = bits
    }
}

package struct SSDAIExpertV1: Codable, Equatable, Sendable {
    package let expert: Int?
    package let physicalRank: Int?
    package let offset: UInt64
    package let size: UInt64
    package let tensors: [String: SSDAISubTensorV1]

    package init(
        expert: Int?, physicalRank: Int?, offset: UInt64, size: UInt64,
        tensors: [String: SSDAISubTensorV1]
    ) {
        self.expert = expert
        self.physicalRank = physicalRank
        self.offset = offset
        self.size = size
        self.tensors = tensors
    }
}

package struct SSDAILayerV1: Codable, Equatable, Sendable {
    package let layer: Int
    package let file: String
    package let experts: [SSDAIExpertV1]
    /// Per-layer expert count when this layer differs from the install's
    /// `expertsPerLayer`; nil means it uses the install-wide value.
    ///
    /// A mixed dense/sparse model needs it: K2-Horizon's first three layers
    /// are dense and the other 45 carry 100 experts each, so one value cannot
    /// describe both. Optional so every install written before mixed layers
    /// existed decodes unchanged.
    package let expertsPerLayer: Int?

    package init(
        layer: Int, file: String, experts: [SSDAIExpertV1],
        expertsPerLayer: Int? = nil
    ) {
        self.layer = layer
        self.file = file
        self.experts = experts
        self.expertsPerLayer = expertsPerLayer
    }
}

/// How an install stores its routed-expert blobs.
///
/// Absent from `layout.json` means raw: expert `n` is `n * expertStride` bytes
/// and the layer file is exactly `expertsPerLayer * expertStride` long. Present
/// means each expert is one or two LZFSE frames addressed by a
/// `layer_NN.idx` sidecar, and the layer file is the compressed length instead.
package enum SSDAIExpertCompression: String, Codable, Sendable {
    /// Whole-phase blocks as independent LZFSE frames, decoded on the CPU.
    case lzfse
    /// Tensor-role blocks entropy-coded with static rANS, decoded on the GPU.
    /// The payload is not byte-addressable by raw offset; the `.ridx` sidecar
    /// carries the model and the block map.
    case rans
}

package struct SSDAIPackedExpertsLayoutV1: Codable, Equatable, Sendable {
    package let expertStride: UInt64
    package let numLayers: Int
    package let expertsPerLayer: Int
    /// `nil` for a raw install; `lzfse` for a compressed one.
    ///
    /// Optional so every install written before compression existed decodes
    /// unchanged. A reader that does not know the value refuses the install at
    /// decode rather than reading frames as weights.
    package let compression: SSDAIExpertCompression?
    package let layers: [SSDAILayerV1]

    package init(
        expertStride: UInt64, numLayers: Int, expertsPerLayer: Int,
        compression: SSDAIExpertCompression? = nil,
        layers: [SSDAILayerV1]
    ) {
        self.expertStride = expertStride
        self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer
        self.compression = compression
        self.layers = layers
    }
}

package enum SSDAIPackedExpertsLayoutCodec {
    package static func decode(_ data: Data) throws -> SSDAIPackedExpertsLayoutV1 {
        let layout: SSDAIPackedExpertsLayoutV1
        do { layout = try JSONDecoder().decode(SSDAIPackedExpertsLayoutV1.self, from: data) } catch
        {
            throw TinyTitanFormatError.invalid(
                field: "packed_experts/layout.json", reason: "\(error)")
        }
        try SSDAIV1StructuralValidator.validate(layout)
        return layout
    }
}

package enum SSDAIV1StructuralValidator {
    package static func validate(_ layout: SSDAIPackedExpertsLayoutV1) throws {
        // A dense model packs no experts, and the repacker writes that honestly:
        // `expertsPerLayer: 0`, `expertStride: 0`, an empty `layers` list, while
        // `numLayers` still counts the transformer layers. Everything below is
        // about packed payload, so the degenerate document is valid for exactly
        // the case that has none -- and only that case: a layer tuple without
        // experts, or experts without a stride, stays invalid.
        if layout.expertsPerLayer == 0 {
            // The repacker writes one entry per transformer layer anyway, each
            // with an empty expert list. Nothing is packed, nothing is
            // streamed, and the file names those entries carry are never read
            // -- so the document is valid exactly while every entry declares no
            // experts, and an entry that names one without a stride is still
            // refused below.
            guard layout.expertStride == 0, layout.numLayers > 0,
                layout.layers.allSatisfy({ $0.experts.isEmpty }),
                layout.layers.isEmpty || layout.layers.count == layout.numLayers
            else {
                throw TinyTitanFormatError.invalid(
                    field: "layout",
                    reason: "invalid dimensions or stride")
            }
            // There are no blobs to compress, so a compression marker on a
            // dense install is a contradiction: either the marker or the empty
            // expert list is wrong, and both are the install's own claim.
            guard layout.compression == nil else {
                throw TinyTitanFormatError.invalid(
                    field: "layout.compression",
                    reason: "a dense install has no expert blobs to compress")
            }
            return
        }
        guard layout.numLayers > 0, layout.expertsPerLayer > 0,
            layout.expertStride > 0,
            layout.expertStride % SSDAIFormatV1.alignmentBytes == 0,
            layout.layers.count == layout.numLayers
        else {
            throw TinyTitanFormatError.invalid(
                field: "layout", reason: "invalid dimensions or stride")
        }
        var layerIDs = Set<Int>()
        var layerFiles = Set<String>()
        for layer in layout.layers {
            guard layer.layer >= 0, layer.layer < layout.numLayers,
                layerIDs.insert(layer.layer).inserted
            else {
                throw TinyTitanFormatError.invalid(
                    field: "layout.layers", reason: "duplicate or invalid layer")
            }
            try SSDAIPathValidator.validateBasename(
                layer.file,
                field: "layout.layers[\(layer.layer)].file")
            let fileKey = SSDAIPathValidator.appleFilesystemKey(layer.file)
            guard fileKey != "layout.json" else {
                throw TinyTitanFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].file",
                    reason: "reserved packed-expert filename")
            }
            guard layerFiles.insert(fileKey).inserted else {
                throw TinyTitanFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].file",
                    reason: "duplicate layer filename")
            }
            let layerExperts = layer.expertsPerLayer ?? layout.expertsPerLayer
            guard layerExperts >= 0,
                layerExperts <= layout.expertsPerLayer,
                layer.experts.count == layerExperts
            else {
                throw TinyTitanFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].experts",
                    reason: "wrong expert count")
            }
            var logicalIDs = Set<Int>()
            var physicalRanks = Set<Int>()
            var offsets = Set<UInt64>()
            var expertRanges: [(start: UInt64, end: UInt64)] = []
            let hasExplicitLogicalIDs = layer.experts.map(\.expert)
            guard
                hasExplicitLogicalIDs.allSatisfy({ $0 == nil })
                    || hasExplicitLogicalIDs.allSatisfy({ $0 != nil })
            else {
                throw TinyTitanFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].experts",
                    reason: "expert ids must be either all explicit or all positional")
            }
            for (position, expert) in layer.experts.enumerated() {
                let logical = expert.expert ?? position
                let physical = expert.physicalRank ?? logical
                guard logical >= 0, logical < layerExperts,
                    physical >= 0, physical < layerExperts,
                    logicalIDs.insert(logical).inserted,
                    physicalRanks.insert(physical).inserted,
                    offsets.insert(expert.offset).inserted
                else {
                    throw TinyTitanFormatError.invalid(
                        field: "layout.layers[\(layer.layer)].experts",
                        reason: "duplicate or invalid expert mapping")
                }
                // A tiered install packs a layer into per-expert extents that
                // need not fill `expertStride`: hot experts are written at the
                // 4-bit width and cold ones at 2-bit, so an extent is at most
                // the stride and the offsets are compact rather than
                // `rank * stride`. What must still hold is a positive extent no
                // larger than one slot, and non-overlapping ranges; the
                // file-size check in `crossValidate` bounds them in the file.
                guard expert.size > 0, expert.size <= layout.expertStride else {
                    throw TinyTitanFormatError.invalid(
                        field: "expert[\(logical)]",
                        reason:
                            "expert size \(expert.size) is not within 1...\(layout.expertStride)")
                }
                let (rangeEnd, overflowed) = expert.offset.addingReportingOverflow(expert.size)
                guard !overflowed else {
                    throw TinyTitanFormatError.invalid(
                        field: "expert[\(logical)]",
                        reason: "expert range overflows")
                }
                let layerCapacity = try ssdaiCheckedMultiply(
                    UInt64(layout.expertsPerLayer), layout.expertStride,
                    field: "layout.layerCapacity")
                guard rangeEnd <= layerCapacity else {
                    throw TinyTitanFormatError.invalid(
                        field: "expert[\(logical)]",
                        reason: "expert range ends past the layer's capacity")
                }
                for existing in expertRanges
                where expert.offset < existing.end && existing.start < rangeEnd {
                    throw TinyTitanFormatError.invalid(
                        field: "expert[\(logical)]",
                        reason: "expert range overlaps another expert")
                }
                expertRanges.append((expert.offset, rangeEnd))
                var tensorRanges: [(start: UInt64, end: UInt64, name: String)] = []
                for (name, tensor) in expert.tensors {
                    guard SSDAIPackedExpertsLayoutV1.isKnownSubTensorDType(tensor.dtype),
                        !name.isEmpty, tensor.size > 0,
                        !tensor.shape.isEmpty,
                        tensor.shape.allSatisfy({ $0 > 0 }),
                        tensor.bits.map({ $0 > 0 && $0 <= 32 }) ?? true
                    else {
                        throw TinyTitanFormatError.invalid(
                            field: "expert[\(logical)].tensors.\(name)",
                            reason: "invalid dtype or shape")
                    }
                    try SSDAIPackedExpertsLayoutV1.validateSubTensorSize(
                        tensor, field: "expert[\(logical)].tensors.\(name)")
                    let end = try ssdaiCheckedAdd(
                        tensor.offset, tensor.size,
                        field: "tensor.\(name).range")
                    guard end <= expert.size else {
                        throw TinyTitanFormatError.invalid(
                            field: "expert[\(logical)].tensors.\(name)",
                            reason: "range exceeds expert blob")
                    }
                    tensorRanges.append((tensor.offset, end, name))
                }
                let sortedRanges = tensorRanges.sorted {
                    $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start
                }
                for pair in zip(sortedRanges, sortedRanges.dropFirst())
                where pair.0.end > pair.1.start {
                    throw TinyTitanFormatError.invalid(
                        field: "expert[\(logical)].tensors",
                        reason: "overlapping ranges \(pair.0.name) and \(pair.1.name)")
                }
            }
        }
    }

    package static func crossValidate(
        manifest: SSDAIManifestV1,
        layout: SSDAIPackedExpertsLayoutV1
    ) throws {
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
        layout: SSDAIPackedExpertsLayoutV1
    ) throws {
        guard manifestNumLayers == layout.numLayers,
            manifestExpertsPerLayer == layout.expertsPerLayer,
            manifestExpertStride == layout.expertStride
        else {
            throw TinyTitanFormatError.invalid(
                field: "manifest/layout",
                reason: "dimension mismatch")
        }
        let expectedLayerSize = try ssdaiCheckedMultiply(
            UInt64(layout.expertsPerLayer),
            layout.expertStride,
            field: "layout.layerSize")
        // Nothing packed means nothing to size. A dense install's layout names
        // one file per layer with an empty expert list and the manifest lists
        // none of them, because the repacker wrote no such files -- with zero
        // experts there is no per-layer payload to stream. The dimension
        // agreement above is the real check for that case.
        guard expectedLayerSize > 0 else { return }
        for layer in layout.layers {
            // A mixed install (K2's three dense layers, then 45 sparse ones)
            // says per layer how many experts it carries. A layer with none has
            // no packed file either, so it is not a missing entry -- the
            // per-layer count is what makes that distinguishable from a sparse
            // layer whose file the manifest forgot.
            let layerCount = layer.expertsPerLayer ?? layout.expertsPerLayer
            let layerCapacity = try ssdaiCheckedMultiply(
                UInt64(layerCount), layout.expertStride,
                field: "layout.layerSize")
            guard layerCapacity > 0 else { continue }
            let path = "packed_experts/\(layer.file)"
            guard let size = manifestFileSizes[path] else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.files.\(path)",
                    reason: "missing or wrong layer size")
            }
            if layout.compression == nil {
                // A uniform layer fills the stride product exactly. A tiered
                // layer is compact: every expert's range must end inside the
                // file, and the file cannot exceed the uniform product it was
                // sized from.
                var layerEnd: UInt64 = 0
                for expert in layer.experts {
                    let (end, overflowed) = expert.offset.addingReportingOverflow(
                        expert.size)
                    guard !overflowed else {
                        throw TinyTitanFormatError.invalid(
                            field: "manifest.files.\(path)",
                            reason: "expert range overflows")
                    }
                    layerEnd = max(layerEnd, end)
                }
                guard size >= layerEnd, size <= layerCapacity else {
                    throw TinyTitanFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "missing or wrong layer size")
                }
                continue
            }
            // A compressed layer's size is the sum of its frames, so the raw
            // product is no longer the contract. What is still checkable from
            // the manifest alone: the layer is smaller than the raw product
            // (that is the entire point, and the writer refuses a layer that
            // grew), and the index that addresses it is listed and non-empty.
            // The frames themselves are checked against the file by the reader
            // (and by `--verify-install`), which is where they can be read.
            let indexName: String
            switch layout.compression {
            case .rans:
                indexName = ExpertRANSIndex.sidecarName(forLayerFile: layer.file)
            default:
                indexName = ExpertBlobIndex.sidecarName(forLayerFile: layer.file)
            }
            let indexPath = "packed_experts/\(indexName)"
            guard size > 0, size < layerCapacity else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.files.\(path)",
                    reason:
                        "compressed layer size \(size) is not smaller than the raw "
                        + "\(layerCapacity) bytes; the install is not compressed")
            }
            guard let indexSize = manifestFileSizes[indexPath], indexSize > 0 else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.files.\(indexPath)",
                    reason: "a compressed layer needs its index in the manifest")
            }
        }
    }
}
