import Foundation

/// Checks a source repository's architecture before any weights are fetched.
///
/// The full `ManifestReader` gates still run at load time; this exists purely
/// so an unsupported model costs the user seconds instead of a multi-gigabyte
/// download. Matching is on the coarse `model_type` rather than the full shape
/// because that is all `config.json` reliably exposes before repacking.
public enum ArchPreflight {
    public struct MalformedConfig: Error, Equatable {
        public let reason: String
    }

    public enum Result: Equatable, Sendable {
        case supported(modelType: String)
        case unsupported(modelType: String)
    }

    /// The Gemma 4 26B-A4B checkpoint reports `gemma4` at the top level. Add an
    /// entry here only alongside the kernels that can execute it.
    public static let supportedModelTypes: Set<String> = [
        "gemma4",
        "Gemma4ForConditionalGeneration",
    ]

    private struct SourceConfig: Decodable {
        let modelType: String?
        let architectures: [String]?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case architectures
        }
    }

    public static func evaluate(configJSON: Data) throws -> Result {
        let config = try JSONDecoder().decode(SourceConfig.self, from: configJSON)
        guard let identifier = config.modelType ?? config.architectures?.first else {
            throw MalformedConfig(
                reason: "config.json has neither model_type nor architectures.")
        }
        return supportedModelTypes.contains(identifier)
            ? .supported(modelType: identifier)
            : .unsupported(modelType: identifier)
    }
}
