import Foundation
import Testing
import TinyTitanFormat
@testable import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanRepackCore

/// A tiny, runnable-shaped K2-Horizon install: three dense layers, no routed
/// experts, the K2 tensor names and the K2 manifest block. It is not a
/// checkpoint reproduction -- the geometry is toy -- it is the smallest install
/// that exercises `ArchConfig.resolved`, the K2 resident schema, the manifest
/// and layout readers, and the loader's mapping path in CI.
///
/// A sparse layer and its packed experts are deliberately absent: K2's mixed
/// layout already has its own tests (`K2RoutedLayoutTests`), and a dense-only
/// toy keeps the fixture small enough to read.
enum K2ToySynthetic {
    /// The toy geometry. `sparse` adds a fourth layer carrying the MoE and
    /// MoVA tensors; the default is dense-only so the dense graph tests have a
    /// three-layer model to run end to end.
    static func toyConfig(sparse: Bool) -> ArchConfig {
        ArchConfig(
            hiddenSize: 64, intermediateSize: 128, moeIntermediateSize: 64,
            numHeads: 4, numKVHeads: 1, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 16, vocabSize: 128,
            slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 10_000_000, fullRopeTheta: 10_000_000,
            partialRotaryFactor: 1, numLayers: sparse ? 4 : 3,
            numExperts: 2, topKExperts: 1,
            tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [UInt8](repeating: 1, count: sparse ? 4 : 3),
            hiddenActivation: "silu",
            family: "k2_horizon", attnOutputGate: true, attentionScale: 0.25,
            embeddingScaledBySqrtHidden: false, routerScaled: false,
            ffnSandwichNorms: false, sharedExpertGated: true,
            ropeNeoxSubdim: true, routerNormTopK: true,
            normGroups: 2, attentionGate: .softplus,
            routerHasBias: true, routerScaling: 2.5,
            denseLayerIndices: [0, 1, 2],
            mova: MoVAConfig(
                numExperts: 4, topKExperts: 4,
                routerScaling: 2.5, routerHasBias: true, activation: "silu"))
    }

    static let toy = toyConfig(sparse: false)

    static func write(sparse: Bool = false) throws -> URL {
        let toy = toyConfig(sparse: sparse)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-k2-toy-\(UUID().uuidString)")
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
            affine(K2HorizonTensors.embedding, rows: toy.vocabSize, cols: d),
            affine(K2HorizonTensors.lmHead, rows: toy.vocabSize, cols: d),
            bf16(K2HorizonTensors.finalNorm, count: d),
        ]
        let queryDim = toy.numHeads * toy.fullHeadDim
        let kvDim = toy.numFullKVHeads * toy.fullHeadDim
        for layer in 0..<toy.numLayers {
            specs.append(bf16(K2HorizonTensors.inputNorm(layer), count: d))
            specs.append(bf16(K2HorizonTensors.postAttnNorm(layer), count: d))
            specs.append(affine(K2HorizonTensors.qProj(layer), rows: queryDim, cols: d))
            specs.append(affine(K2HorizonTensors.kProj(layer), rows: kvDim, cols: d))
            specs.append(affine(K2HorizonTensors.oProj(layer), rows: d, cols: queryDim))
            specs.append(affine(K2HorizonTensors.attnGate(layer), rows: queryDim, cols: d))
            if toy.denseLayerIndices.contains(layer) {
                specs.append(affine(K2HorizonTensors.vProj(layer), rows: kvDim, cols: d))
                specs.append(
                    affine(K2HorizonTensors.denseGate(layer), rows: toy.intermediateSize, cols: d))
                specs.append(
                    affine(K2HorizonTensors.denseUp(layer), rows: toy.intermediateSize, cols: d))
                specs.append(
                    affine(K2HorizonTensors.denseDown(layer), rows: d, cols: toy.intermediateSize))
            } else {
                specs.append(bf16(K2HorizonTensors.vRouter(layer), count: toy.mova.numExperts * d))
                specs.append(bf16(K2HorizonTensors.vRouterBias(layer), count: toy.mova.numExperts))
                for expert in 0..<toy.mova.numExperts {
                    specs.append(
                        affine(
                            K2HorizonTensors.vExpert(layer, expert),
                            rows: kvDim, cols: d))
                }
                specs.append(bf16(K2HorizonTensors.moeRouter(layer), count: toy.numExperts * d))
                specs.append(bf16(K2HorizonTensors.moeRouterBias(layer), count: toy.numExperts))
                specs.append(
                    affine(
                        K2HorizonTensors.sharedGate(layer), rows: toy.moeIntermediateSize, cols: d))
                specs.append(
                    affine(K2HorizonTensors.sharedUp(layer), rows: toy.moeIntermediateSize, cols: d)
                )
                specs.append(
                    affine(
                        K2HorizonTensors.sharedDown(layer), rows: d, cols: toy.moeIntermediateSize))
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

        // The sparse layer's routed experts, packed exactly as the runtime
        // reads them: one blob per expert, gate/up/down plus their BF16
        // companions, page-strided. The dense-only fixture writes the same
        // layout with every layer empty and no layer file.
        var layoutData: Data
        var layerFile: (name: String, size: Int, sha: String)?
        if sparse {
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
            let sparseLayer = toy.numLayers - 1
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
            layerFile = (
                layerName, layerBytes,
                try Sha256Verifier.hashFile(at: layerURL)
            )
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
            layoutData = try JSONSerialization.data(
                withJSONObject: layout, options: [.sortedKeys])
        } else {
            let layout: [String: Any] = [
                "expertStride": Int(SSDAIFormatV1.alignmentBytes),
                "numLayers": toy.numLayers,
                "expertsPerLayer": toy.numExperts,
                "layers": (0..<toy.numLayers).map { layer in
                    [
                        "layer": layer,
                        "file": String(format: "layer_%02d.bin", layer),
                        "experts": [],
                        "expertsPerLayer": 0,
                    ] as [String: Any]
                },
            ]
            layoutData = try JSONSerialization.data(
                withJSONObject: layout, options: [.sortedKeys])
        }
        let layoutURL = dir.appendingPathComponent("packed_experts/layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": file.count, "sha256": weightsSha],
            "packed_experts/layout.json": [
                "size": layoutData.count, "sha256": layoutSha,
            ],
        ]
        if let layerFile {
            files["packed_experts/\(layerFile.name)"] = [
                "size": layerFile.size, "sha256": layerFile.sha,
            ]
        }

        let manifest: [String: Any] = [
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": [
                "streamingPresent": false, "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            "modelID": "k2-horizon-toy",
            "arch": archJSON(toy),
            "quant": [
                "embedding": slot(), "attention": slot(), "router": slot(),
                "sharedExpert": slot(), "routedExpert": slot(),
            ],
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": sparse
                ? Int(SSDAIFormatV1.alignmentBytes) * 2
                : Int(SSDAIFormatV1.alignmentBytes),
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
            "normGroups": toy.normGroups,
            "attentionGate": toy.attentionGate.rawValue,
            "routerHasBias": toy.routerHasBias,
            "routerScaling": toy.routerScaling,
            "movaNumExperts": toy.mova.numExperts,
            "movaTopKExperts": toy.mova.topKExperts,
            "movaRouterScaling": toy.mova.routerScaling,
            "movaRouterHasBias": toy.mova.routerHasBias,
            "movaActivation": toy.mova.activation,
            "denseLayerIndices": toy.denseLayerIndices,
        ]
    }
}
