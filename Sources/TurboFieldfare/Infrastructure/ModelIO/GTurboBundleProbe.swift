import Foundation

/// One bundle's on-disk format, probed from `manifest.json` without touching
/// weights. The CLI and server use this to pick the Gemma (v1) or Kimi K3
/// (v2) runtime path.
public struct GTurboBundleInfo: Equatable, Sendable {
    public let versionMajor: Int
    public let versionMinor: Int
    public let modelID: String

    public init(versionMajor: Int, versionMinor: Int, modelID: String) {
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.modelID = modelID
    }

    /// Kimi K3 bundles are `.gturbo` format v2.
    public var isK3: Bool { versionMajor == 2 }
}

public enum GTurboBundleProbe {
    /// Read `manifest.json` and report the wire format. Only version 1 and 2
    /// bundles are recognized; anything else throws a ModelError-style error
    /// (`partialInstall` / `notAGTurboDirectory` / `unsupportedVersion`).
    public static func probe(bundleURL: URL) throws -> GTurboBundleInfo {
        let root = bundleURL.standardizedFileURL
        let manifestURL = root.appendingPathComponent("manifest.json")
        let data: Data
        do {
            data = try Data(contentsOf: manifestURL, options: [.mappedIfSafe])
        } catch {
            throw ModelError.partialInstall(path: root.path)
        }
        let object: [String: Any]
        do {
            guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw ModelError.notAGTurboDirectory
            }
            object = parsed
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        guard object["magic"] as? String == "GTURBO",
              let major = object["versionMajor"] as? Int,
              let minor = object["versionMinor"] as? Int,
              let modelID = object["modelID"] as? String else {
            throw ModelError.notAGTurboDirectory
        }
        guard major == 1 || major == 2 else {
            throw ModelError.unsupportedVersion(major: major, minor: minor)
        }
        return GTurboBundleInfo(versionMajor: major, versionMinor: minor, modelID: modelID)
    }
}
