/// Spawns and stops the `TurboFieldfareServer` child process. Every outcome —
/// a successful start, a startup failure, a requested stop, or an
/// unexpected crash — is reported through `onStateChange`, so `AppModel` has
/// exactly one path to update its state instead of a separate one for each
/// case.
public protocol AppServerController: Sendable {
    func start(arguments: [String],
              onStateChange: @escaping @Sendable (AppServerState) -> Void)
    func stop()
}
