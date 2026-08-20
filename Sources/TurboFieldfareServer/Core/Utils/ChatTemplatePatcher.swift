import Foundation

/// `ChatTemplatePatcher` provides in-memory replacements for the Gemma 4 chat template
/// to prevent crashes/errors when calling the `upper` filter on non-string values.
public enum ChatTemplatePatcher {
    private static let replacements: [(old: String, new: String)] = [
        ("value['type'] | upper", "(value['type'] or '') | upper"),
        ("params['type'] | upper", "(params['type'] or '') | upper"),
        ("response_declaration['type'] | upper", "(response_declaration['type'] or '') | upper"),
        ("item_value | upper", "(item_value or '') | upper"),
    ]

    /// Patches the provided chat template string.
    ///
    /// If patching fails for any reason, it returns the original string and logs a warning.
    public static func patch(_ template: String) -> String {
        var patched = template
        for (old, new) in replacements {
            patched = patched.replacingOccurrences(of: old, with: new)
        }
        return patched
    }
}
