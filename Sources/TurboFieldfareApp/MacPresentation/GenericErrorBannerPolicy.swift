import TurboFieldfareAppCore

public enum GenericErrorBannerPolicy {
    public static func shouldShow(error: AppInferenceError?, loadState: AppModelLoadState)
        -> Bool {
        guard let error, error != .cancelled else { return false }
        if case .failed = loadState { return false }
        return true
    }
}
