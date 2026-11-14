import Foundation
import TurboFieldfare

public enum AppBundledResources {
    /// A model-free relocation probe for the app and its independently launched helper.
    public static func verify() throws {
        guard AppPromptPreset.all.count == 7 else {
            throw AppInferenceError.invalidRequest("Bundled prompt examples could not be read.")
        }
        _ = try MetalContext()
    }
}
