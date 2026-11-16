import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitan
@testable import TinyTitanRepackCore
@testable import TinyTitanCore

/// A tiny, runnable Laguna-S-2.1 install: layer 0 dense, layer 1 sparse with
/// four routed experts at top-2, the checkpoint's tensor spelling, and the
/// toy geometry. It is the smallest install that exercises
/// `PerLayerHeadGeometry`, the resident schema, the manifest/layout readers, the
/// loader's mapping path and the probe graph without a checkpoint.
enum LagunaToySynthetic {
    static let toy = ArchConfig(
        hiddenSize: 64, intermediateSize: 64, moeIntermediateSize: 64,
        numHeads: 2, numKVHeads: 1, numFullKVHeads: 1,
        headDim: 32, fullHeadDim: 32, vocabSize: 128,
        slidingWindow: 8, finalLogitSoftcap: 0,
        ropeTheta: 10_000, fullRopeTheta: 500_000,
        partialRotaryFactor: 0.5, numLayers: 2,
        numExperts: 4, topKExperts: 2,
        tieWordEmbeddings: false, attentionKEqV: false,
        fullAttentionLayerMask: [1, 0], hiddenActivation: "silu",
        family: "laguna", attnOutputGate: true,
        attentionScale: 1.0 / Double(32).squareRoot(),
        embeddingScaledBySqrtHidden: false, routerScaled: false,
        ffnSandwichNorms: false, sharedExpertGated: false,
        ropeNeoxSubdim: true, routerNormTopK: true, quantGroupSize: 64,
        normGroups: 1, attentionGate: .softplus, routerHasBias: true,
        routerScaling: 2.5, denseLayerIndices: [0],
        perLayerHeadGeometry: PerLayerHeadGeometry(
            headsPerLayer: [2, 4], qkNorm: true, gatePerHead: true,
            swaPartialRotaryFactor: 1.0, yarnFactor: 128,
            yarnOriginalMax: 8192, yarnBetaFast: 32, yarnBetaSlow: 1,
            yarnAttentionFactor: 1.4852030263919618))

    static func write() throws -> URL {
        let toy = Self.toy
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-laguna-toy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        let d = toy.hiddenSize
        let u16 = MemoryLayout<UInt16>.stride

        struct Spec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }
        func affine(_ name: String, rows: Int, cols: Int) -> Spec {
            let groups = cols / Quantization.groupSize
            let aux = UInt64(rows * groups * u16)
            return Spec(
                name: name, dtype: 0,
                shape: [UInt32(rows), UInt32(cols), 0, 0],
                weightBytes: UInt64(rows * cols * 4 / 8),
                scaleBytes: aux, biasBytes: aux)
        }
        func bf16(_ name: String, count: Int) -> Spec {
            Spec(
                name: name, dtype: 1,
                shape: [UInt32(count), 0, 0, 0],
                weightBytes: UInt64(count * u16),
                scaleBytes: 0, biasBytes: 0)
        }

        var specs: [Spec] = [
            affine(LagunaTensors.embedding, rows: toy.vocabSize, cols: d),
            affine(LagunaTensors.lmHead, rows: toy.vocabSize, cols: d),
            bf16(LagunaTensors.finalNorm, count: d),
        ]
        let kvDim = toy.numFullKVHeads * toy.fullHeadDim
        for layer in 0..<toy.numLayers {
            let heads = toy.perLayerHeadGeometry.headsPerLayer[layer]
            let queryDim = heads * toy.fullHeadDim
            specs.append(bf16(LagunaTensors.inputNorm(layer), count: d))
            specs.append(bf16(LagunaTensors.postAttnNorm(layer), count: d))
            specs.append(bf16(LagunaTensors.qNorm(layer), count: toy.fullHeadDim))
            specs.append(bf16(LagunaTensors.kNorm(layer), count: toy.fullHeadDim))
            specs.append(affine(LagunaTensors.qProj(layer), rows: queryDim, cols: d))
            specs.append(affine(LagunaTensors.kProj(layer), rows: kvDim, cols: d))
            specs.append(affine(LagunaTensors.vProj(layer), rows: kvDim, cols: d))
            specs.append(affine(LagunaTensors.oProj(layer), rows: d, cols: queryDim))
            specs.append(affine(LagunaTensors.attnGate(layer), rows: heads, cols: d))
            if toy.denseLayerIndices.contains(layer) {
                specs.append(
                    affine(LagunaTensors.denseGate(layer), rows: toy.intermediateSize, cols: d))
                specs.append(
                    affine(LagunaTensors.denseUp(layer), rows: toy.intermediateSize, cols: d))
                specs.append(
                    affine(LagunaTensors.denseDown(layer), rows: d, cols: toy.intermediateSize))
            } else {
                specs.append(bf16(LagunaTensors.moeRouter(layer), count: toy.numExperts * d))
                specs.append(bf16(LagunaTensors.moeRouterBias(layer), count: toy.numExperts))
                specs.append(
                    affine(LagunaTensors.sharedGate(layer), rows: toy.moeIntermediateSize, cols: d))
                specs.append(
                    affine(LagunaTensors.sharedUp(layer), rows: toy.moeIntermediateSize, cols: d))
                specs.append(
                    affine(LagunaTensors.sharedDown(layer), rows: d, cols: toy.moeIntermediateSize))
            }
        }

        let names = specs.map(\.name)
        let stringTable = Data(names.joined().utf8)
        let headerBytes = SSDAIBinary.indexHeaderBytes
        let entryBytes = SSDAIBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameOffsets: [UInt32] = []
        var cursor = 0
        for name in names {
            nameOffsets.append(UInt32(stringTableBase + cursor))
            cursor += name.utf8.count
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)
        let alignment = SSDAIFormatV1.alignmentBytes
        let alignedIndexBytes = (indexBytes + alignment - 1) & ~(alignment - 1)
        let wordAlignment = UInt64(MemoryLayout<UInt32>.alignment)

        var entries: [ResidentEntry] = []
        var payloadCursor = alignedIndexBytes
        for spec in specs {
            let weightOffset = (payloadCursor + wordAlignment - 1) & ~(wordAlignment - 1)
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(
                ResidentEntry(
                    name: spec.name, dtype: spec.dtype,
                    logicalShape4: spec.shape,
                    fileOffset: weightOffset, sizeBytes: spec.weightBytes,
                    scaleOffset: scaleOffset, scaleSize: spec.scaleBytes,
                    biasOffset: biasOffset, biasSize: spec.biasBytes,
                    quantSpec: nil,
                    sourceWeight: ModelLoaderTests.dummySource(spec.name),
                    sourceScales: nil, sourceBiases: nil))
            payloadCursor =
                weightOffset + spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize =
            ((payloadCursor + wordAlignment - 1) & ~(wordAlignment - 1)) - alignedIndexBytes
        let total = Int(alignedIndexBytes + residentSize)
        var file = [UInt8](repeating: 0, count: total)
        try file.withUnsafeMutableBytes { raw in
            let base = try #require(raw.baseAddress)
            SSDAIBinary.writeIndexHeader(
                into: base, indexSize: alignedIndexBytes,
                residentSize: residentSize, entryCount: UInt64(entries.count))
            for (index, entry) in entries.enumerated() {
                SSDAIBinary.writeIndexEntry(
                    into: base.advanced(by: entriesBase + index * entryBytes),
                    entry: entry, nameOffset: nameOffsets[index])
            }
            stringTable.withUnsafeBytes { table in
                if let source = table.baseAddress {
                    memcpy(
                        base.advanced(by: stringTableBase), source,
                        stringTable.count)
                }
            }
            for entry in entries where entry.dtype == 0 {
                memset(
                    base.advanced(by: Int(entry.fileOffset)), 0x11,
                    Int(entry.sizeBytes))
                if entry.scaleSize > 0 {
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / u16) {
                        scales[i] = Quantization.bf16Bits(0.01)
                    }
                }
            }
            for entry in entries where entry.dtype == 1 {
                let values = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    values[i] = Quantization.bf16Bits(1.0)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(file).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        func expertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]
            func projection(_ role: String, rows: Int, cols: Int, weightRow: Int) {
                let values = (0..<rows).map { row in
                    (0..<cols).map { column in
                        Float(expert + 1) * 0.001 + Float(weightRow + 1) * 0.003
                            + Float((row % 7) - 3) * 0.0004
                            + Float((column % 11) - 5) * 0.0002
                    }
                }
                let quantized = values.map { Quantization.quantizeInt4Affine($0) }
                let weightOffset = bytes.count
                for row in quantized { bytes.append(contentsOf: row.packed) }
                tensors[role] = [
                    "offset": weightOffset,
                    "size": bytes.count - weightOffset,
                    "dtype": "U32",
                    "shape": [rows, cols],
                    "bits": 4,
                ]
                let scalesOffset = bytes.count
                for row in quantized { appendU16(row.scales, to: &bytes) }
                tensors["\(role)_scales"] = [
                    "offset": scalesOffset,
                    "size": bytes.count - scalesOffset,
                    "dtype": "BF16",
                    "shape": [rows, cols / Quantization.groupSize],
                ]
                let biasesOffset = bytes.count
                for row in quantized { appendU16(row.biases, to: &bytes) }
                tensors["\(role)_biases"] = [
                    "offset": biasesOffset,
                    "size": bytes.count - biasesOffset,
                    "dtype": "BF16",
                    "shape": [rows, cols / Quantization.groupSize],
                ]
            }
            projection("gate", rows: toy.moeIntermediateSize, cols: d, weightRow: 0)
            projection("up", rows: toy.moeIntermediateSize, cols: d, weightRow: 1)
            projection("down", rows: d, cols: toy.moeIntermediateSize, weightRow: 2)
            return (bytes, tensors)
        }

        let expertStride = Int(SSDAIFormatV1.alignmentBytes) * 2
        let sparseLayer = 1
        let layerBytes = expertStride * toy.numExperts
        var payload = [UInt8](repeating: 0, count: layerBytes)
        var expertJSON: [[String: Any]] = []
        for expert in 0..<toy.numExperts {
            let blob = expertBlob(expert: expert)
            precondition(blob.bytes.count <= expertStride, "toy expert exceeds stride")
            let base = expert * expertStride
            for (offset, byte) in blob.bytes.enumerated() {
                payload[base + offset] = byte
            }
            expertJSON.append([
                "expert": expert,
                "offset": UInt64(base),
                "size": UInt64(expertStride),
                "tensors": blob.tensors,
            ])
        }
        let layerName = String(format: "layer_%02d.bin", sparseLayer)
        let layerURL = dir.appendingPathComponent("packed_experts/\(layerName)")
        try Data(payload).write(to: layerURL)
        let layout: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": (0..<toy.numLayers).map { layer in
                if layer == sparseLayer {
                    return [
                        "layer": layer,
                        "file": layerName,
                        "experts": expertJSON,
                        "expertsPerLayer": toy.numExperts,
                    ] as [String: Any]
                }
                return [
                    "layer": layer,
                    "file": String(format: "layer_%02d.bin", layer),
                    "experts": [],
                    "expertsPerLayer": 0,
                ] as [String: Any]
            },
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layout, options: [.sortedKeys])
        let layoutURL = dir.appendingPathComponent("packed_experts/layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)
        let layerSha = try Sha256Verifier.hashFile(at: layerURL)

        let manifest: [String: Any] = [
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": [
                "streamingPresent": false, "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            "modelID": "laguna-toy",
            "arch": archJSON(toy),
            "quant": [
                "embedding": slot(), "attention": slot(), "router": slot(),
                "sharedExpert": slot(), "routedExpert": slot(),
            ],
            "files": [
                "model_weights.bin": ["size": file.count, "sha256": weightsSha],
                "packed_experts/layout.json": [
                    "size": layoutData.count, "sha256": layoutSha,
                ],
                "packed_experts/\(layerName)": [
                    "size": layerBytes, "sha256": layerSha,
                ],
            ],
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
        ]
        try JSONSerialization.data(
            withJSONObject: manifest, options: [.sortedKeys]
        ).write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    private static func slot() -> [String: Any] {
        [
            "weightBits": 4, "scheme": "affine", "scaleType": "bf16",
            "biasType": "bf16", "groupSize": Quantization.groupSize,
        ]
    }

    private static func archJSON(_ toy: ArchConfig) -> [String: Any] {
        [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
            "family": toy.family,
            "attnOutputGate": toy.attnOutputGate,
            "attentionScale": toy.attentionScale,
            "embeddingScaledBySqrtHidden": toy.embeddingScaledBySqrtHidden,
            "routerScaled": toy.routerScaled,
            "ffnSandwichNorms": toy.ffnSandwichNorms,
            "sharedExpertGated": toy.sharedExpertGated,
            "ropeNeoxSubdim": toy.ropeNeoxSubdim,
            "linearNumKHeads": 0, "linearNumVHeads": 0,
            "linearKeyHeadDim": 0, "linearValueHeadDim": 0,
            "linearConvKernelSize": 0,
            "routerNormTopK": toy.routerNormTopK,
            "quantGroupSize": toy.quantGroupSize,
            "normGroups": toy.normGroups,
            "attentionGate": toy.attentionGate.rawValue,
            "routerHasBias": toy.routerHasBias,
            "routerScaling": toy.routerScaling,
            "denseLayerIndices": toy.denseLayerIndices,
        ]
    }
}
