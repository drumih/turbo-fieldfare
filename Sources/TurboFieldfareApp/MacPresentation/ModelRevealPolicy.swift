import Foundation

public enum ModelRevealTarget: Equatable, Sendable {
    case selectItem(URL)
    case openContainer(URL)
    case unavailable
}

public enum ModelRevealPolicy {
    // The model directory is user-editable and may not exist yet: an install
    // that never ran, or a hand-typed path. Falling back to the containing
    // directory still answers "where would the model go", which is the question
    // the menu item exists to answer.
    public static func target(forModelPath path: String,
                              fileExists: (String) -> Bool) -> ModelRevealTarget {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .unavailable }
        let url = URL(fileURLWithPath: trimmed, isDirectory: true).standardizedFileURL
        if fileExists(url.path) { return .selectItem(url) }
        let container = url.deletingLastPathComponent().standardizedFileURL
        if container.path != url.path, fileExists(container.path) {
            return .openContainer(container)
        }
        return .unavailable
    }
}
