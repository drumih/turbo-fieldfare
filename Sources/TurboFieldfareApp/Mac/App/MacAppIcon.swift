import AppKit

// Bundle.module resolves only inside this target, so the icon cannot be loaded
// from the presentation library.
enum MacAppIcon {
    static func load() -> NSImage? {
        guard let url = Bundle.module.url(
            forResource: "turbofieldfare-app-icon",
            withExtension: "png"
        ) else { return nil }
        return NSImage(contentsOf: url)
    }
}
