import Foundation

/// Whether the inspector shows its Image Support section.
///
/// This lived as a private computed property on the inspector view, where the
/// only way to check it was to launch the app and look. It decides four distinct
/// situations, so it is worth being able to test all four.
public enum VisionSectionVisibility {
    /// Shown whenever this build exposes the vision runtime.
    ///
    /// An installed text model is deliberately *not* required. Requiring one hid
    /// image support behind a 14.62 GB download, so the screen where someone
    /// decides whether this app does what they need never mentioned that it
    /// handles images at all. A healthy installed pack remains visible because
    /// its Remove action is part of the supported lifecycle.
    public static func shows(
        visionRuntimeEnabled: Bool,
        visionRuntimeSupported: Bool = true,
        isModelInstalled: Bool,
        isVisionPackInstalled: Bool,
        isCompanionOperationInProgress: Bool,
        installState: AppModelInstallState
    ) -> Bool {
        visionRuntimeEnabled
    }
}
