import Foundation
import TinyTitanCore
import TinyTitanPluginAPI

/// Minimal out-of-tree `EnginePlugin` (TT-133, TPP/1).
///
/// Imports only `Foundation`, `TinyTitanCore`, and `TinyTitanPluginAPI`,
/// proving an external package can register and resolve an engine plugin
/// without touching TinyTitan's source tree.
public enum EchoPlugin: EnginePlugin {
    public static let pluginID = "com.example.tinytitan.echo"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "Example Echo Plugin"
    public static let family = "example-echo"

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        install.declaredFamily == family ? .exact : nil
    }

    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport(
            kvCachePrecision: .supported,
            hostKVWindow: .refused(reason: "example echo plugin has no host-KV cache"),
            concurrentSequences: .supported,
            multiTokenPrediction: .refused(reason: "example echo plugin has no draft head"),
            expertStreaming: .refused(reason: "example echo plugin has no routed experts"))
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
        .binaryThinking
    }

    public static func engineFactory(host: HostServices) -> EngineFactory {
        let architecture = ArchConfig(
            hiddenSize: 16, intermediateSize: 32, moeIntermediateSize: 0, numHeads: 1,
            numKVHeads: 1, numFullKVHeads: 1, headDim: 16, fullHeadDim: 16,
            vocabSize: 64, slidingWindow: 0, finalLogitSoftcap: 0, ropeTheta: 10_000,
            fullRopeTheta: 10_000, partialRotaryFactor: 0, numLayers: 1, numExperts: 0,
            topKExperts: 0, tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [1], hiddenActivation: "none", family: family)
        return EngineFactory(
            prepareLoad: { options in
                host.log("\(displayName): planned context \(options.maxContext)")
                return EngineLoadPlan(
                    architecture: architecture, expertCacheSlots: 0,
                    tunedExpertCacheBudgetBytes: 0, prefillConfiguration: .defaultChunked,
                    fp16RingEnabled: false, yarnContextTokens: 0, runtimeIdentity: pluginID,
                    engineConfiguration: options.maxContext)
            },
            metadata: { directory in
                EngineModelMetadata(
                    modelID: directory.lastPathComponent, family: family, weightBits: 16,
                    sampling: .house, reasoningControl: .binaryThinking)
            },
            resolveArchitecture: { _ in architecture },
            preview: { directory, chunk in
                EngineSessionPreview(
                    modelID: directory.lastPathComponent, prefillChunkTokens: chunk ?? 64)
            },
            buildModel: { _, _, _ in throw EchoPluginError.notRunnable },
            buildRunner: { _, _ in throw EchoPluginError.notRunnable },
            runRawCompletion: { _ in throw EchoPluginError.notRunnable })
    }
}

public enum EchoPluginError: Error, CustomStringConvertible {
    case notRunnable

    public var description: String {
        "example echo plugin is a registration fixture and does not run inference"
    }
}
