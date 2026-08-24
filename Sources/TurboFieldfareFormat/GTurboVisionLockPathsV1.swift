import Darwin
import Foundation

/// Lock-file path derivation shared by every process that synchronizes on a
/// vision companion pack. The reader's shared use lease and a writer's
/// exclusive mutation lock must lock the same file, and a derivation mismatch
/// does not fail loudly — it silently leaves the two sides locking different
/// files — so the derivation lives here, next to the format it protects,
/// instead of being duplicated per module.
package enum GTurboVisionLockPathsV1 {
    /// `.<companion-basename>.use.lock` beside the companion, with the parent
    /// resolved physically so a symlinked path and its target derive the same
    /// lock file.
    package static func useLockURL(forCompanion companionURL: URL) -> URL {
        let companion = companionURL.standardizedFileURL
        let parent = companion.deletingLastPathComponent()
        let resolvedParent: URL
        if let resolved = realpath(parent.path, nil) {
            defer { free(resolved) }
            resolvedParent = URL(fileURLWithPath: String(cString: resolved),
                                 isDirectory: true)
        } else {
            // The parent does not exist yet; the caller's `open` will fail
            // and say so.
            resolvedParent = parent.resolvingSymlinksInPath()
        }
        return resolvedParent
            .appendingPathComponent(".\(companion.lastPathComponent).use.lock")
    }
}
