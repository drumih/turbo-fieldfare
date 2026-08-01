import Foundation

/// Converts a Hugging Face repository ID into a filesystem-safe directory name.
///
/// The repository ID is user input that becomes a path component, so this is
/// the traversal boundary: anything that could escape the models directory is
/// rejected here rather than sanitised, because a silently-rewritten slug would
/// point a reinstall at a different directory than the original install.
public enum ModelSlug {
    public struct InvalidRepositoryID: Error, Equatable, Sendable {
        public let reason: String

        public init(reason: String) {
            self.reason = reason
        }
    }

    public static func make(repoID: String) throws -> String {
        guard !repoID.isEmpty else {
            throw InvalidRepositoryID(reason: "Repository ID is empty.")
        }
        guard !repoID.hasPrefix("/") else {
            throw InvalidRepositoryID(reason: "Repository ID must not be an absolute path.")
        }
        let components = repoID.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2 else {
            throw InvalidRepositoryID(
                reason: "Expected the form owner/name, got \"\(repoID)\".")
        }
        for component in components {
            guard !component.isEmpty else {
                throw InvalidRepositoryID(reason: "Repository ID has an empty component.")
            }
            guard component != "." && component != ".." else {
                throw InvalidRepositoryID(
                    reason: "Repository ID must not contain relative path components.")
            }
            guard component.allSatisfy(isAllowed) else {
                throw InvalidRepositoryID(
                    reason: "Repository ID may only contain letters, digits, "
                        + "hyphen, underscore, and period.")
            }
        }
        return components.joined(separator: "--")
    }

    private static func isAllowed(_ character: Character) -> Bool {
        guard character.isASCII else { return false }
        return character.isLetter
            || character.isNumber
            || character == "-"
            || character == "_"
            || character == "."
    }
}
