/// The lifecycle of the child `TurboFieldfareServer` process the app can
/// spawn from the Server section of the Inspector.
public enum AppServerState: Equatable, Sendable {
    case stopped
    case starting
    case running(port: Int)
    case stopping
    case failed(message: String)
}
