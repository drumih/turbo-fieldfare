import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

public enum AppVisionPackInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
    /// This model layout cannot host a companion pack at all — the directory is
    /// not named `*.gturbo`, so there is no path to put one at.
    ///
    /// Distinct from `.partial`, which means a pack is there and damaged and
    /// offers Repair. Reporting this as `.partial` printed a raw
    /// `invalidTextModelPath(...)` under "Needs repair" and offered to download
    /// a pack that could never be located.
    case unsupportedLayout
}

public enum AppVisionPackInstallationProbe {
    public static func status(at textModelDirectory: URL) -> AppVisionPackInstallationStatus {
        let textModelDirectory = textModelDirectory.standardizedFileURL
        let companion: URL
        do {
            companion = try VisionPackLocation.companionURL(forTextModel: textModelDirectory)
        } catch {
            // The only failure here is a directory name that cannot carry a
            // companion, which is a property of the layout rather than of any
            // pack on disk.
            return .unsupportedLayout
        }
        guard FileManager.default.fileExists(atPath: companion.path) else {
            return .missing
        }
        do {
            _ = try VisionPackVerifier.verify(
                directory: companion,
                textModelDirectory: textModelDirectory,
                verifyWeights: false)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
