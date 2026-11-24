import Foundation
import TinyTitanCore

/// What the dense CPU backend needs to know about an install before it can
/// choose an engine, read without the 6.x runtime (TT-134).
///
/// The hidden size names the model (2B / 4B / 9B) and the weight width its
/// quant twin; together they select the variant engine. Both shapes the CPU
/// path serves are read: a `.ssdai` install from its manifest, an affine
/// safetensors snapshot from its `config.json`. Nothing here interprets the
/// architecture -- the selected variant refuses one it does not implement.
public struct DenseCPUInstall: Sendable, Equatable {
    public let directory: URL
    /// A `.ssdai` declares itself with a manifest; a snapshot does not.
    public let isSSDAI: Bool
    public let hiddenSize: Int
    /// The width the install declares: the manifest's routed-expert slot
    /// (4 when absent, the rule the catalog applies), or the snapshot's
    /// `quantization.bits`.
    public let weightBits: Int
    /// The manifest's model id; nil for a snapshot, which carries none.
    public let modelID: String?
    /// The declared family (`.ssdai`) or `model_type` (snapshot).
    public let family: String?

    public init(directory: URL) throws {
        self.directory = directory
        isSSDAI = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("manifest.json").path)
        if isSSDAI {
            let manifest = try SSDAIManifestReader.read(directoryURL: directory)
            hiddenSize = manifest.arch.hiddenSize
            weightBits = manifest.quant?.routedExpert.weightBits ?? 4
            modelID = manifest.modelID.isEmpty ? nil : manifest.modelID
            family = try SSDAIManifestReader.declaredFamily(directoryURL: directory)
        } else {
            let data = try BoundedMetadataRead.read(
                fileAt: directory.appendingPathComponent("config.json"),
                maxBytes: SSDAIManifestReader.defaultMaxBytes)
            guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                throw ServerInferenceErrorText.malformed("config.json is not an object")
            }
            guard let hidden = config["hidden_size"] as? Int else {
                throw ServerInferenceErrorText.malformed("config.json lacks hidden_size")
            }
            guard let quantization = config["quantization"] as? [String: Any],
                let bits = quantization["bits"] as? Int
            else {
                throw ServerInferenceErrorText.malformed("config.json lacks a quantization block")
            }
            hiddenSize = hidden
            weightBits = bits
            modelID = nil
            family = config["model_type"] as? String
        }
    }
}

/// The reader's refusal, in the same words the snapshot loader used.
enum ServerInferenceErrorText: Error, CustomStringConvertible {
    case malformed(String)

    var description: String {
        switch self {
        case .malformed(let detail): "malformed snapshot: \(detail)"
        }
    }
}
