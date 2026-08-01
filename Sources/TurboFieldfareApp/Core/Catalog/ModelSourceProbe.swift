import CryptoKit
import Foundation

/// Fetches the two small files needed to vet a repository before committing to
/// a multi-gigabyte download: `config.json` for the architecture pre-flight,
/// and the weight index whose hash is the trust-on-first-use fingerprint.
public enum ModelSourceProbe {
    public struct ProbeFailure: Error, Equatable, Sendable {
        public let reason: String
        /// Set when the repository resolved but the file was absent, which is
        /// the case worth diagnosing further.
        public var isMissingFile = false

        public init(reason: String, isMissingFile: Bool = false) {
            self.reason = reason
            self.isMissingFile = isMissingFile
        }
    }

    public struct SourceFacts: Equatable, Sendable {
        public let architecture: ArchPreflight.Result
        public let indexSHA256: String
    }

    static let indexFileName = "model.safetensors.index.json"

    /// Why a repository has no `config.json`.
    ///
    /// Reporting the missing file is accurate but useless: the user pasted a
    /// repository, not a file, and what they need to know is which kind of
    /// repository they landed on and what to look for instead.
    public enum RepositoryKind: Equatable, Sendable {
        /// llama.cpp weights. A different format entirely — not convertible here.
        case ggufOnly
        /// Transformers-style weights whose config was not published.
        case safetensorsWithoutConfig
        case unrecognised
    }

    public static func classify(fileNames: [String]) -> RepositoryKind {
        if fileNames.contains(where: { $0.lowercased().hasSuffix(".gguf") }) {
            return .ggufOnly
        }
        if fileNames.contains(where: { $0.lowercased().hasSuffix(".safetensors") }) {
            return .safetensorsWithoutConfig
        }
        return .unrecognised
    }

    public static func explanation(for kind: RepositoryKind, repoID: String) -> String {
        switch kind {
        case .ggufOnly:
            return "\(repoID) is a GGUF repository, which is llama.cpp's format. "
                + "TurboFieldfare needs MLX weights quantized affine 4-bit at group size 64. "
                + "Look for a conversion of the same model whose name ends in -mlx-4bit "
                + "or -4bit, or convert it yourself with mlx_lm.convert."
        case .safetensorsWithoutConfig:
            return "\(repoID) has safetensors weights but no config.json, so its "
                + "architecture cannot be checked. Pick a repository that publishes "
                + "config.json alongside its weights."
        case .unrecognised:
            return "\(repoID) does not contain model weights this build recognises. "
                + "TurboFieldfare needs an MLX 4-bit conversion of Gemma 4 26B-A4B."
        }
    }

    public static func repositoryInfoURL(repoID: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models/\(repoID)"
        return components.url
    }

    /// Hugging Face serves file contents from `/resolve/<revision>/<path>`.
    public static func fileURL(repoID: String,
                               revision: String,
                               fileName: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repoID)/resolve/\(revision)/\(fileName)"
        return components.url
    }

    public static func probe(repoID: String,
                             revision: String,
                             token: String?,
                             session: URLSession = .shared) async throws -> SourceFacts {
        let config: Data
        do {
            config = try await fetch(repoID: repoID,
                                     revision: revision,
                                     fileName: "config.json",
                                     token: token,
                                     session: session)
        } catch let failure as ProbeFailure where failure.isMissingFile {
            // Turn "no config.json" into an explanation of what the repository
            // actually is, which is what the user needs in order to fix it.
            let kind = await repositoryKind(repoID: repoID, token: token, session: session)
            throw ProbeFailure(reason: explanation(for: kind, repoID: repoID))
        }
        let architecture = try ArchPreflight.evaluate(configJSON: config)
        // Only fetched once the architecture is known good — no point hashing
        // the index of a model that cannot run.
        guard case .supported = architecture else {
            return SourceFacts(architecture: architecture, indexSHA256: "")
        }
        let index = try await fetch(repoID: repoID,
                                    revision: revision,
                                    fileName: indexFileName,
                                    token: token,
                                    session: session)
        let digest = SHA256.hash(data: index)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return SourceFacts(architecture: architecture, indexSHA256: hex)
    }

    /// Best-effort: if the listing cannot be fetched we still say something
    /// useful rather than failing the diagnosis itself.
    private static func repositoryKind(repoID: String,
                                       token: String?,
                                       session: URLSession) async -> RepositoryKind {
        struct RepositoryInfo: Decodable {
            struct Sibling: Decodable { let rfilename: String }
            let siblings: [Sibling]?
        }
        guard let url = repositoryInfoURL(repoID: repoID) else { return .unrecognised }
        var request = URLRequest(url: url)
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        guard let (data, _) = try? await session.data(for: request),
              let info = try? JSONDecoder().decode(RepositoryInfo.self, from: data)
        else { return .unrecognised }
        return classify(fileNames: (info.siblings ?? []).map(\.rfilename))
    }

    private static func fetch(repoID: String,
                              revision: String,
                              fileName: String,
                              token: String?,
                              session: URLSession) async throws -> Data {
        guard let url = fileURL(repoID: repoID, revision: revision, fileName: fileName) else {
            throw ProbeFailure(reason: "Could not build a URL for \(repoID).")
        }
        var request = URLRequest(url: url)
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ProbeFailure(reason: "Unexpected response fetching \(fileName).")
        }
        switch http.statusCode {
        case 200:
            return data
        case 401, 403:
            throw ProbeFailure(
                reason: "\(repoID) is gated or private. Add a Hugging Face token and retry.")
        case 404:
            throw ProbeFailure(
                reason: "\(repoID) has no \(fileName) at revision \(revision).",
                isMissingFile: true)
        default:
            throw ProbeFailure(
                reason: "Fetching \(fileName) failed with HTTP \(http.statusCode).")
        }
    }
}
