import Foundation

public enum TurboFieldfareAppVersion {
    public static let fallback = "0.4.1"

    public static func resolve(
        bundle: Bundle = .main,
        fallback: String = Self.fallback
    ) -> String {
        if let version = bundle.infoDictionary?["CFBundleShortVersionString"] as? String {
            return version
        }
        return fallback
    }
}
