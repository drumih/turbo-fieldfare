import CryptoKit
import Foundation

/// Fetches the two small files needed to vet a repository before committing to
/// a multi-gigabyte download: `config.json` for the architecture pre-flight,
/// and the weight index whose hash is the trust-on-first-use fingerprint.
public enum ModelSourceProbe {
    public struct ProbeFailure: Error, Equatable, Sendable {
        public let reason: String
    }

    public struct SourceFacts: Equatable, Sendable {
        public let architecture: ArchPreflight.Result
        public let indexSHA256: String
    }

    static let indexFileName = "model.safetensors.index.json"

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
        let config = try await fetch(repoID: repoID,
                                     revision: revision,
                                     fileName: "config.json",
                                     token: token,
                                     session: session)
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
                reason: "\(repoID) has no \(fileName) at revision \(revision).")
        default:
            throw ProbeFailure(
                reason: "Fetching \(fileName) failed with HTTP \(http.statusCode).")
        }
    }
}
