import Darwin
import Foundation
import TurboFieldfare
@testable import TurboFieldfareAppCore

func makeCompleteModelInstall(_ tag: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("turbofieldfare-complete-\(tag)-\(UUID().uuidString).gturbo")
    let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: experts.appendingPathComponent("layout.json"))

    let arch = ArchConfig.gemma4_26B_A4B
    var files: [String: Any] = [
        "model_weights.bin": ["size": 0, "sha256": String(repeating: "0", count: 64)],
        "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
    ]
    for layer in 0..<arch.numLayers {
        files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
            "size": 0,
            "sha256": String(repeating: "0", count: 64),
        ]
    }
    let manifest: [String: Any] = [
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": ["streamingPresent": true],
        "modelID": "test/gemma-4-26b-a4b",
        "sourceSnapshotHash": "sha256:" + AppModelInstallDescriptor.default.sourceIndexSHA256,
        "quant": [
            "embedding": quantSlot(4),
            "attention": quantSlot(4),
            "router": quantSlot(8),
            "sharedExpert": quantSlot(4),
            "routedExpert": quantSlot(4),
        ],
        "arch": [
            "hiddenSize": arch.hiddenSize,
            "ffnIntermediate": arch.intermediateSize,
            "moeIntermediateSize": arch.moeIntermediateSize,
            "numHeads": arch.numHeads,
            "numKVHeads": arch.numKVHeads,
            "numFullKVHeads": arch.numFullKVHeads,
            "headDim": arch.headDim,
            "fullHeadDim": arch.fullHeadDim,
            "vocabSize": arch.vocabSize,
            "slidingWindow": arch.slidingWindow,
            "finalLogitSoftcap": arch.finalLogitSoftcap,
            "ropeTheta": arch.ropeTheta,
            "fullRopeTheta": arch.fullRopeTheta,
            "partialRotaryFactor": arch.partialRotaryFactor,
            "numLayers": arch.numLayers,
            "numExperts": arch.numExperts,
            "topKExperts": arch.topKExperts,
            "tieWordEmbeddings": arch.tieWordEmbeddings,
            "attentionKEqV": arch.attentionKEqV,
            "hiddenActivation": arch.hiddenActivation,
            "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
        ],
        "files": files,
        "expertsPerLayer": arch.numExperts,
        "numLayers": arch.numLayers,
        "expertStride": UInt64(getpagesize()),
    ]
    let manifestData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
    let manifestURL = directory.appendingPathComponent("manifest.json")
    try manifestData.write(to: manifestURL)
    let manifestHash = Sha256Verifier.hashData(manifestData)
    let receipt: [String: Any] = [
        "schemaVersion": 1,
        "manifestSha256": manifestHash,
        "modelDirectoryPath": directory.standardizedFileURL.path,
        "verificationTimestamp": "2026-07-11T00:00:00Z",
        "toolVersion": "TurboFieldfareAppCoreTests",
        "files": [:],
    ]
    let receiptData = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
    try receiptData.write(to: directory.appendingPathComponent("verified-install.json"))
    return directory
}

private func quantSlot(_ weightBits: Int) -> [String: Any] {
    [
        "weightBits": weightBits,
        "scheme": "affine",
        "scaleType": "bf16",
        "biasType": "bf16",
        "groupSize": Quantization.groupSize,
    ]
}
