import Foundation
import TinyTitanFormat
import TinyTitanCore

// The packed-expert half of `--verify-install`.
//
// Split out of `VerifiedInstallTool.swift` (2026-10-06, AUD-124) under the
// 500-line-per-file rule. `layout` and `validatePackedExpertLayout` moved as pure
// code motion — same bodies, same messages, `private` widened to internal because
// the caller stays in the other file — and the width cross-check below them is
// what the split was for: `packed_experts/layout.json` is the only place a MoE
// install records how its expert bytes are packed, so the file that reads it is
// the file that has to prove it.

extension VerifiedInstallTool {

    /// Read the packed-expert layout the manifest names.
    ///
    /// Decoded through the verifier's local mirror, which carries the fields this
    /// tool needs and none of the format layer's structural validation. That is
    /// what makes the arithmetic below the check that has to hold: nothing else
    /// here asks whether the described bytes add up.
    static func loadLayout(access: SSDAIDirectoryAccess) throws -> PackedExpertsLayout {
        do {
            let data = try loadMetadataJSON(
                access: access,
                relativePath: "packed_experts/layout.json")
            return try JSONDecoder().decode(PackedExpertsLayout.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(
                detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    static func validatePackedExpertLayout(
        access: SSDAIDirectoryAccess,
        manifest: Manifest
    ) throws {
        let layoutRelativePath = "packed_experts/layout.json"
        guard manifest.files[layoutRelativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(layoutRelativePath)")
        }
        let layout = try loadLayout(access: access)
        let alignment = SSDAIFormatV1.alignmentBytes
        guard layout.expertStride == manifest.expertStride,
            layout.numLayers == manifest.numLayers,
            layout.expertsPerLayer == manifest.expertsPerLayer
        else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layout dimensions mismatch manifest")
        }
        guard layout.expertStride % alignment == 0 else {
            throw RepackError.configurationInvalid(
                detail: "expertStride \(layout.expertStride) is not aligned to \(alignment) bytes")
        }
        guard layout.layers.count == layout.numLayers else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layout layer count mismatch")
        }
        if let compression = layout.compression,
            compression != "lzfse", compression != "rans"
        {
            throw RepackError.configurationInvalid(
                detail: "packed_experts/layout.json declares unknown compression "
                    + "\"\(compression)\"")
        }
        // The routed-expert width the expert bytes below have to agree with. A
        // dense install never reaches it: its layers hold no experts, and the
        // resident index is what describes its weights.
        let declaredExpertBits = manifest.quant?.routedExpert.weightBits
        for layer in layout.layers {
            let count = layer.expertsPerLayer ?? layout.expertsPerLayer
            let (expectedLayerSize, overflow) = UInt64(count)
                .multipliedReportingOverflow(by: layout.expertStride)
            guard !overflow else {
                throw RepackError.configurationInvalid(
                    detail: "packed expert layer size overflows")
            }
            try validateLayer(
                layer, access: access, manifest: manifest, layout: layout,
                expectedLayerSize: expectedLayerSize,
                declaredExpertBits: declaredExpertBits)
        }
    }

    /// One layer: its file size, and every expert's own arithmetic.
    ///
    /// Split out of `validatePackedExpertLayout` so the compressed and raw
    /// cases each read as one check rather than as branches inside a loop.
    private static func validateLayer(
        _ layer: Layer,
        access: SSDAIDirectoryAccess,
        manifest: Manifest,
        layout: PackedExpertsLayout,
        expectedLayerSize: UInt64,
        declaredExpertBits: Int?
    ) throws {
        guard layer.layer >= 0 && layer.layer < layout.numLayers else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layer index out of range")
        }
        let layerExperts = layer.expertsPerLayer ?? layout.expertsPerLayer
        guard layer.experts.count == layerExperts else {
            throw RepackError.configurationInvalid(
                detail: "packed_experts/\(layer.file) expert count mismatch")
        }
        try SSDAIPathValidator.validateBasename(
            layer.file, field: "packed_experts/layout.json layers[\(layer.layer)].file")
        // A layer with no routed experts has no file, and writing one empty
        // `layer_NN.bin` per layer to satisfy this loop would be worse than the
        // check: the dense Qwen 3.5 installs are exactly this shape, 24 layouts
        // and no packed experts at all. The expected size is what makes this
        // safe rather than a hole -- `expectedLayerSize` is 0 only when the
        // layer is empty, so a layer that should carry bytes still fails on a
        // missing file below, with a non-zero expected size to compare against.
        if expectedLayerSize == 0 {
            return
        }
        let relativePath = "packed_experts/\(layer.file)"
        guard let manifestEntry = manifest.files[relativePath] else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
        }
        let actualSize = try access.fileSize(relativePath)
        if layout.compression == nil {
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail:
                        "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)"
                )
            }
            guard actualSize == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedLayerSize)")
            }
        } else {
            try validateCompressedLayer(
                layer, access: access, manifest: manifest, layout: layout,
                relativePath: relativePath, manifestEntry: manifestEntry,
                actualSize: actualSize, expectedLayerSize: expectedLayerSize)
        }
        guard let declaredExpertBits else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no quant block")
        }
        var seenExperts = Set<Int>()
        for (index, expert) in layer.experts.enumerated() {
            let expertID = expert.expert ?? index
            guard expertID >= 0 && expertID < layout.expertsPerLayer else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) expert id out of range")
            }
            guard seenExperts.insert(expertID).inserted else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) duplicate expert \(expertID)")
            }
            guard expert.size == layout.expertStride else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) expert \(expertID) size mismatch")
            }
            guard expert.offset % SSDAIFormatV1.alignmentBytes == 0 else {
                throw RepackError.configurationInvalid(
                    detail:
                        "\(relativePath) expert \(expertID) offset is not aligned to \(SSDAIFormatV1.alignmentBytes) bytes"
                )
            }
            // A compressed layer's file does not contain the raw offsets the
            // layout still records -- the frames do -- so the range check is
            // the reader's (each frame's end), done by `validateCompressedLayer`.
            if layout.compression == nil {
                guard expert.offset <= actualSize,
                    expert.size <= actualSize - expert.offset
                else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
            try PackedExpertBytes.validate(
                expert,
                stride: layout.expertStride,
                declaredWeightBits: declaredExpertBits,
                in: relativePath,
                rank: expertID)
        }
    }

    /// A compressed layer: its index, every frame inside the file, and the sizes
    /// the manifest claims.
    ///
    /// The frames are not decompressed here. Their bytes are already covered by
    /// the file's SHA-256 in the manifest, and a decode that disagrees with the
    /// geometry is what the reader refuses; re-decoding 17 GB to prove what the
    /// hash already proved would make verification cost as much as the install.
    private static func validateCompressedLayer(
        _ layer: Layer,
        access: SSDAIDirectoryAccess,
        manifest: Manifest,
        layout: PackedExpertsLayout,
        relativePath: String,
        manifestEntry: ManifestFileEntry,
        actualSize: UInt64,
        expectedLayerSize: UInt64
    ) throws {
        guard manifestEntry.size == actualSize else {
            throw RepackError.configurationInvalid(
                detail: "\(relativePath) size \(actualSize) != manifest \(manifestEntry.size)")
        }
        guard actualSize > 0, actualSize < expectedLayerSize else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(relativePath) compressed size \(actualSize) is not smaller than the "
                    + "raw \(expectedLayerSize) bytes; the install is not compressed")
        }
        let rans = layout.compression == "rans"
        let indexPath =
            "packed_experts/"
            + (rans
                ? ExpertRANSIndex.sidecarName(forLayerFile: layer.file)
                : ExpertBlobIndex.sidecarName(forLayerFile: layer.file))
        guard let indexEntry = manifest.files[indexPath], indexEntry.size > 0 else {
            throw RepackError.configurationInvalid(
                detail: "manifest missing the index \(indexPath) of a compressed layer")
        }
        let indexSize = try access.fileSize(indexPath)
        guard indexSize == indexEntry.size else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) size \(indexSize) != manifest \(indexEntry.size)")
        }
        if rans {
            let cap = UInt64(
                ExpertRANSIndex.headerBytes
                    + Int(ExpertRANSIndex.maximumBlockCount) * 512
                    + layout.expertsPerLayer * Int(ExpertRANSIndex.maximumBlockCount)
                    * ExpertRANSIndex.blockBytes)
            guard indexSize <= cap else {
                throw RepackError.configurationInvalid(
                    detail: "\(indexPath) is \(indexSize) bytes; the cap is \(cap)")
            }
            let data = try access.readPrefix(indexPath, maxBytes: cap)
            do {
                _ = try ExpertRANSIndex.decode(
                    data, expectedExperts: layout.expertsPerLayer,
                    expertStride: layout.expertStride, payloadSize: actualSize)
            } catch {
                throw RepackError.configurationInvalid(detail: "\(indexPath): \(error)")
            }
            return
        }
        guard
            let cap = ExpertBlobIndex.maximumIndexBytes(
                expertsPerLayer: layout.expertsPerLayer)
        else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) cannot be sized for \(layout.expertsPerLayer) experts")
        }
        guard indexSize <= cap else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) is \(indexSize) bytes; an index for "
                    + "\(layout.expertsPerLayer) experts is at most \(cap)")
        }
        let indexData = try access.readPrefix(indexPath, maxBytes: cap)
        let split = ExpertBlobIndex.phase1Bytes(
            downOffset: layer.experts.first?.tensors["down"]?.offset,
            expertStride: layout.expertStride)
        let decoded: ExpertBlobIndex.Layout
        do {
            decoded = try ExpertBlobIndex.decodeLayout(
                indexData, expectedCount: layout.expertsPerLayer,
                phase0Bytes: split ?? layout.expertStride,
                phase1Bytes: split.map { layout.expertStride - $0 })
        } catch {
            throw RepackError.configurationInvalid(detail: "\(indexPath): \(error)")
        }
        let end = ExpertBlobIndex.maximumEnd(decoded.entries)
        guard end > 0, end <= actualSize else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) addresses \(end) bytes, past the "
                    + "\(actualSize)-byte layer file")
        }
    }
}

/// The arithmetic that ties one packed expert's bytes to the width it claims.
enum PackedExpertBytes {

    /// Prove `declaredWeightBits` against one expert's own slices.
    ///
    /// A packed-expert install keeps its routed-expert widths here rather than in
    /// the resident index, which is why `validateQuantAgainstResident` steps
    /// aside for it -- and until now nothing read what this file says. So the
    /// slot that becomes the `_<bits>-Bit` name in `/v1/models`, and that the
    /// dequantizer and the Metal pipeline constant are chosen from, was compared
    /// against nothing for exactly the shapes the product ships. The writer's own
    /// comment names the failure it invites: the word count changes, the strides
    /// still divide evenly, every shape check passes, and the model answers
    /// fluently and wrongly.
    ///
    /// Three arithmetic facts, none of which asks to be trusted:
    ///
    ///   - a U32 tensor of `n` elements in `bits`-bit packing is
    ///     `n * bits / 8` bytes, so its byte extent *implies* its width;
    ///   - that implied width has to equal the tensor's own `bits` annotation
    ///     where it carries one, and the declared slot for every one of them;
    ///   - the slices of an expert fill its stride once, with page padding
    ///     allowed at the end and nowhere else — so a missing, duplicated or
    ///     wrongly-sized slice shows up as a sum that does not round to the
    ///     stride rather than as a file that still divides evenly.
    ///
    /// A BF16 tensor is two bytes an element and has no width to agree with.
    ///
    /// The bound on a *derived* width is the format layer's own (1...32 bits),
    /// not the 4-or-8 the resident check insists on. That gap is deliberate:
    /// 6-bit was withdrawn as a format, and an install whose bytes match its
    /// description is a truthful install even when the runtime will not load it.
    /// Refusing it here would make the verifier stricter than the load path, with
    /// a message telling the user to re-download weights that are fine.
    static func validate(
        _ expert: Expert,
        stride: UInt64,
        declaredWeightBits: Int,
        in file: String,
        rank: Int
    ) throws {
        var blobBytes: UInt64 = 0
        // Sorted so the first complaint is the same on every run: dictionary
        // order is not, and a verifier whose message depends on it cannot be
        // compared between two machines.
        for (name, tensor) in expert.tensors.sorted(by: { $0.key < $1.key }) {
            let at = "\(file) expert \(rank) tensor \(name)"
            let elements = try elementCount(of: tensor, at: at)
            switch tensor.dtype {
            case "U32":
                let implied = try impliedBits(of: tensor, elements: elements, at: at)
                if let annotated = tensor.bits, annotated != implied {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): annotated \(annotated)-bit but its \(tensor.size) bytes "
                            + "at \(describe(tensor.shape)) are \(implied)-bit. Repack it "
                            + "from its source snapshot; the bytes are fine, the "
                            + "description of them is not")
                }
                guard implied == declaredWeightBits else {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): its \(tensor.size) bytes at \(describe(tensor.shape)) are "
                            + "\(implied)-bit, while the install declares a routed-expert "
                            + "width of \(declaredWeightBits). The dequantizer's word "
                            + "count and the Metal pipeline constant are both taken from "
                            + "that declaration, so this install would read these bytes "
                            + "wrongly")
                }
            case "BF16":
                let (expected, byteOverflow) = elements.multipliedReportingOverflow(by: 2)
                guard !byteOverflow else {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): \(describe(tensor.shape)) is too many bf16 values to count "
                            + "in bytes")
                }
                guard expected == tensor.size else {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): \(tensor.size) bytes at \(describe(tensor.shape)) is not "
                            + "the \(expected) bytes a bf16 tensor of that shape is")
                }
            default:
                throw RepackError.configurationInvalid(
                    detail: "\(at): unknown dtype \(tensor.dtype)")
            }
            let (sum, blobOverflow) = blobBytes.addingReportingOverflow(tensor.size)
            guard !blobOverflow else {
                throw RepackError.configurationInvalid(detail: "\(at): expert byte total overflows")
            }
            blobBytes = sum
        }
        let padded = try roundedToStride(blobBytes)
        guard padded == stride else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(file) expert \(rank): its tensors total \(blobBytes) bytes, which "
                    + "pads to \(padded) rather than the \(stride)-byte expert stride. "
                    + "Either a slice is missing, duplicated, or a different width than "
                    + "the one declared")
        }
    }

    /// The element count a tensor's own shape describes.
    private static func elementCount(of tensor: SSDAISubTensorV1, at: String) throws -> UInt64 {
        guard !tensor.shape.isEmpty, tensor.shape.allSatisfy({ $0 > 0 }) else {
            throw RepackError.configurationInvalid(
                detail: "\(at): shape \(describe(tensor.shape)) has no elements")
        }
        var elements: UInt64 = 1
        for dimension in tensor.shape {
            let (product, overflow) = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !overflow else {
                throw RepackError.configurationInvalid(
                    detail: "\(at): shape \(describe(tensor.shape)) overflows")
            }
            elements = product
        }
        return elements
    }

    /// The width a U32 tensor's byte extent implies, in bits per element.
    ///
    /// `size * 8 / elements`, and only when it divides: a byte extent that is not
    /// a whole number of packed values per element is a broken layout, not a width
    /// to guess at.
    private static func impliedBits(
        of tensor: SSDAISubTensorV1, elements: UInt64, at: String
    ) throws -> Int {
        guard tensor.size > 0 else {
            throw RepackError.configurationInvalid(detail: "\(at): zero-length tensor")
        }
        let (bits, bitsOverflow) = tensor.size.multipliedReportingOverflow(by: 8)
        guard !bitsOverflow else {
            throw RepackError.configurationInvalid(
                detail: "\(at): \(tensor.size) bytes overflow a bit count")
        }
        guard bits % elements == 0 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(at): \(tensor.size) bytes at \(describe(tensor.shape)) does not divide "
                    + "into a whole number of values per element")
        }
        let implied = bits / elements
        guard implied > 0, implied <= 32 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(at): implied width \(implied) is outside the 1...32 bits a packed "
                    + "tensor can hold")
        }
        return Int(implied)
    }

    /// The stride a blob of `bytes` is cut to: the next multiple of the page.
    ///
    /// The same rounding the planner applies when it lays an expert out, which is
    /// why a 35B expert whose bytes fill their stride exactly and a 125B expert
    /// padded by 4 KiB both pass, and a blob a page short or long does not.
    private static func roundedToStride(_ bytes: UInt64) throws -> UInt64 {
        let page = SSDAIFormatV1.alignmentBytes
        let (padded, overflow) = bytes.addingReportingOverflow(page - 1)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "expert byte total \(bytes) cannot be padded to a page")
        }
        return (padded / page) * page
    }

    private static func describe(_ shape: [UInt32]) -> String {
        shape.map(String.init).joined(separator: "x")
    }
}
