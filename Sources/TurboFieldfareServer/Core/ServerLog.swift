import Foundation

enum ServerLog {
    static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    static func prepared(id: String, promptTokens: Int?) {
        let count = promptTokens.map(String.init) ?? "backend-managed"
        write("request \(id) prepared prompt=\(count)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        let usage = completion.usage
        write("request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "finish=\(completion.finishReason)")
    }

    /// The session is an actor, so generation is serialized: this line always
    /// belongs to the request between the preceding `generating` and the
    /// following `completed`. Carries a reason code only, never prompt content.

    static func visionPackInvalid(at url: URL, error: Error) {
        write("vision pack at \(url.path) is invalid: \(String(reflecting: error))")
    }

    static func visionRuntimeUnsupported(at url: URL, error: Error) {
        write("vision runtime for pack at \(url.path) is unsupported: "
            + String(reflecting: error))
    }

    static func visionRuntimeUnsupported() {
        write("vision runtime is unsupported: the image tower requires an M2 or newer Mac")
    }

    /// A prefix that could not be continued because its bridge failed to
    /// render. Carries the underlying error, because this miss means the
    /// template and the cached turn disagree, which no other miss does.
    static func promptCacheBridgeFailed(error: Error) {
        write("prompt cache miss reason=bridge-render-failed "
            + "error=\(String(reflecting: error))")
    }

    /// A request whose client stopped listening. Kept distinct from `failed`
    /// because a cancel is not a server fault and should not read as one, and
    /// added because excluding it from logging altogether left the request's last
    /// line as `generating` forever: an operator could not tell a running request
    /// from an abandoned one, or from a crashed one. Carries no content.
    static func cancelled(id: String, phase: String, duration: Duration) {
        write(cancelledMessage(id: id, phase: phase, duration: duration))
    }

    /// Split out so the wording can be checked without a log sink: this line
    /// must stay distinguishable from `failed`, must name the phase, and must
    /// carry no prompt or generated content.
    static func cancelledMessage(id: String, phase: String, duration: Duration) -> String {
        "request \(id) cancelled by client in \(format(duration)) phase=\(phase)"
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.3fs", seconds)
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
