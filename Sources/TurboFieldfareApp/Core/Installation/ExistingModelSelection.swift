import Foundation
import TurboFieldfare

public struct ExistingModelCandidate: Identifiable, Equatable, Sendable {
    public var id: String { textURL.path }
    public let textURL: URL
    public let textStatus: AppModelInstallationStatus
    public let visionURL: URL
    public let visionStatus: AppVisionPackInstallationStatus
    public var canSelect: Bool { textStatus == .complete }

    public var textSummary: String {
        switch textStatus {
        case .complete: "Text model ready"
        case .missing: "Incomplete text model: manifest.json is missing."
        case .partial(let reason): "Text model unavailable: \(reason)"
        }
    }

    public var visionSummary: String {
        switch visionStatus {
        case .complete: "Image support ready"
        case .missing: "Text only — no matching vision pack found"
        case .partial(let reason): "Text only — vision pack is invalid or incompatible: \(reason)"
        case .unsupportedLayout: "Text only — this folder name cannot locate a vision pack"
        }
    }
}

public struct ExistingModelReview: Identifiable, Sendable {
    public let id = UUID()
    public let folder: URL
    public let candidates: [ExistingModelCandidate]
    public var initialSelection: String? {
        candidates.count == 1 && candidates[0].canSelect ? candidates[0].id : nil
    }
}

public enum ExistingModelSelection {
    public static func inspect(_ folder: URL) throws -> ExistingModelReview {
        let folder = AppModelLocation.canonicalURL(folder)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw AppModelLocation.LocationError("The selected folder is unavailable: \(folder.path)")
        }
        if folder.lastPathComponent.hasSuffix(".vision.gturbo") {
            throw AppModelLocation.LocationError("This is a vision pack. Choose its parent folder or the text model folder beside it.")
        }
        let urls: [URL]
        if folder.pathExtension == "gturbo" {
            urls = [folder]
        } else {
            urls = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey])
                .filter { url in
                    guard url.pathExtension == "gturbo", !url.lastPathComponent.hasSuffix(".vision.gturbo") else { return false }
                    let target = AppModelLocation.canonicalURL(url)
                    guard target.pathExtension == "gturbo",
                          !target.lastPathComponent.hasSuffix(".vision.gturbo") else { return false }
                    return try target.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true
                }
        }
        // Aliases to the same pack must not manufacture an ambiguous choice.
        let unique = Set(urls.map(AppModelLocation.canonicalURL)).sorted { $0.path < $1.path }
        guard !unique.isEmpty else {
            throw AppModelLocation.LocationError("No text model folders found directly inside \(folder.path). Choose a folder containing a completed .gturbo text model; a vision pack alone is not enough.")
        }
        return ExistingModelReview(folder: folder, candidates: try unique.map(candidate))
    }

    public static func candidate(at textURL: URL) throws -> ExistingModelCandidate {
        let textURL = AppModelLocation.canonicalURL(textURL)
        return ExistingModelCandidate(textURL: textURL,
                                      textStatus: AppModelInstallationProbe.status(at: textURL),
                                      visionURL: try VisionPackLocation.companionURL(forTextModel: textURL),
                                      visionStatus: AppVisionPackInstallationProbe.status(at: textURL))
    }
}
