import EchoPlugin
import Foundation
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanPluginAPI
import TinyTitanVariants

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("PluginDemo: FAIL — \(message)\n".utf8))
    exit(1)
}

do {
    let registry = EngineFactories.pluginRegistry
    try registry.register(EchoPlugin.self)

    let scratch = FileManager.default.temporaryDirectory
        .appendingPathComponent("tinytitan-plugin-demo-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scratch) }

    let manifest = """
        {
          "magic": "SSDAI",
          "versionMajor": 1,
          "versionMinor": 1,
          "flags": {},
          "modelID": "example-echo-demo",
          "expertsPerLayer": 0,
          "numLayers": 1,
          "expertStride": 0,
          "files": {},
          "arch": {
            "family": "example-echo",
            "hiddenSize": 16,
            "ffnIntermediate": 32,
            "moeIntermediateSize": 0,
            "numHeads": 1,
            "numKVHeads": 1,
            "numFullKVHeads": 1,
            "headDim": 16,
            "fullHeadDim": 16,
            "vocabSize": 64,
            "slidingWindow": 0,
            "finalLogitSoftcap": 0,
            "ropeTheta": 10000,
            "fullRopeTheta": 10000,
            "partialRotaryFactor": 0,
            "numLayers": 1,
            "numExperts": 0,
            "topKExperts": 0,
            "tieWordEmbeddings": false,
            "attentionKEqV": false,
            "fullAttentionLayerMask": [1],
            "hiddenActivation": "none"
          }
        }
        """
    try manifest.write(
        to: scratch.appendingPathComponent("manifest.json"),
        atomically: true, encoding: .utf8)

    let resolved = try registry.resolve(scratch)
    guard resolved.pluginID == EchoPlugin.pluginID else {
        fail("expected \(EchoPlugin.pluginID), resolved \(resolved.pluginID)")
    }
    guard resolved.strength == .exact else {
        fail("expected .exact match, got \(resolved.strength)")
    }
    let metadata = try resolved.engineFactory().metadata(scratch)
    guard metadata.family == EchoPlugin.family else {
        fail("expected family \(EchoPlugin.family), got \(metadata.family)")
    }
    print(
        "PluginDemo: ok — registered and resolved \(resolved.pluginID) "
            + "(protocol \(resolved.plugin.protocolVersion), family \(metadata.family))")
} catch {
    fail("\(error)")
}
