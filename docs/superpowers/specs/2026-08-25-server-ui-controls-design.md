# Server UI controls — design

## Problem

`TurboFieldfareServer` can only be started from a terminal today. Add a
"Server" section to the Mac app's Inspector sidebar with start/stop controls,
so a user can run the local OpenAI-compatible server without leaving the app.

## Architecture

`TurboFieldfareServer` is a standalone executable that loads its own model
process — it does not share the model the app itself may have loaded via
`DecodeServiceInferenceClient`. Starting the server is therefore "spawn a
sibling process," the same shape as how the app already launches
`TurboFieldfareDecodeService`, not a change to how the app's own chat model
works.

### `AppServerController`

A new protocol, mirroring the existing `AppModelInstallerClient` /
`AppModelLifecycleClient` pattern so `AppModel` can be tested against a fake:

```swift
public protocol AppServerController: Sendable {
    func start(arguments: [String],
               onStateChange: @escaping @Sendable (AppServerState) -> Void) throws
    func stop()
}
```

`ProcessServerController` is the real implementation:

- Locates the binary next to the app's own executable:
  `Bundle.main.executableURL!.deletingLastPathComponent()
    .appendingPathComponent("TurboFieldfareServer")` — the same resolution
  `DecodeServiceInferenceClient.defaultServiceURL()` already uses, and it
  works unmodified for both `swift build` output (all products land in one
  `.build/release` directory) and a packaged `.app` bundle.
- Spawns a plain `Foundation.Process` (no launchd). Confirmed in
  `Command/main.swift`: the server installs a `ServerTerminationSignals`
  handler for `SIGINT`/`SIGTERM` and shuts down gracefully, so
  `Process.terminate()` is sufficient to stop it.
- Watches stdout line-by-line for the `TurboFieldfareServer ready at
  http://127.0.0.1:<port> ...` line to detect success.
- Sets `Process.terminationHandler` once, before running. It fires for every
  exit path — a requested stop, a startup failure, or a mid-run crash — and
  is the single place that flips state back out of `.running`/`.starting`.
  If the process exits before printing "ready", the accumulated stderr
  (trimmed of the `error: ` prefix the CLI already writes) becomes the
  `.failed` message.

### State

```swift
public enum AppServerState: Equatable, Sendable {
    case stopped
    case starting
    case running(port: Int)
    case stopping
    case failed(message: String)
}
```

Transitions: `stopped -> starting -> running` (success) or `starting ->
failed` (bad flags, port in use, model failed to load — the server's own
stderr explains which). `running -> stopping -> stopped` (user-requested).
`running -> failed` (crash). `failed` behaves like `stopped` for gating
purposes (Start is available again) but keeps the message visible until the
next Start.

### Arguments built from existing app state

No new configuration duplicates what the sidebar already controls. The
argument list is built from:

- `--model` — `modelPathText`
- `--max-context` — `maxContextTokens`
- `--expert-cache-slots`, `--expert-cache-policy`, `--prefill`,
  `--prefill-chunk-tokens`, `--rdadvise` — from `model.runtimeOptions`
  (`AppRuntimeOptions`), the same struct already bound to the Memory/Runtime
  sections
- `--vision-pack <path>` — added automatically when `isVisionPackInstalled`
  is true; omitted otherwise. No separate toggle.
- `--port`, `--queue-limit` — the two new server-only settings described
  below; these have no app-wide equivalent to inherit from.

`--model-id` and `--prompt-cache-mode` are left at the server's own
defaults; nothing in the app UI maps to them and the user did not ask for
control over them.

### Concurrency with the app's own model

Starting the server does not require unloading the app's own model. If both
are loaded, the Server section shows a caption warning about double memory
use, but the Start button stays enabled (per the user's explicit choice —
"warn but allow").

### App-quit cleanup

`ForegroundAppDelegate.applicationWillTerminate` (`TurboFieldfareMacApp.swift`),
which already calls `model.releaseAllAttachments()`, also calls
`model.stopServerForTermination()`. This sends `Process.terminate()`
synchronously; it does not wait for full graceful shutdown since the app is
already quitting.

## UI — `InspectorView.swift`

A new `serverSection`, placed after `runtimeSection`:

```
Server
  State          Stopped
                 Starting…
                 Running · http://127.0.0.1:8080
                 Error: <trimmed stderr message>
  Port           [8080______]      (TextField, editable only while stopped)
  Queue limit    [4_________]      (TextField, editable only while stopped)
  [Start Server] / [Stop Server]
  (caption, shown only when model.loadState.isReady while stopped/starting)
  "The app's model is also loaded — running both uses memory for two copies."
```

- Start is disabled unless `model.isModelInstalled` and not
  `model.isInstallingModel` / `model.isVisionCompanionOperationInProgress` —
  the same gating pattern the other sections already use via `.disabled(...)`.
- Stop is enabled only in `.starting` / `.running`.
- Port and queue-limit fields lock once starting/running (the server's own
  settings are "fixed for the life of the process"; changing them requires a
  restart, so the UI reflects that instead of accepting an edit that would
  be silently ignored).

## Settings persistence — `MacAppSettings.swift`

Bump `currentVersion` to `3`. Add:

```swift
var serverPort: Int = 8080
var serverQueueLimit: Int = 4
```

Decoded with `decodeIfPresent(..., forKey:) ?? <default>`, matching how
`visionResidencyPolicy`, `rdadvisePolicy`, and `loadModelOnLaunch` were added
in the v2 migration — an older settings file on disk keeps loading under the
new build.

## Testing (TDD)

- **Argument building**: pure function
  `AppServerArguments.build(modelPath:maxContextTokens:runtimeOptions:visionPackPath:port:queueLimit:) -> [String]`
  — test it directly against expected flag arrays, no process involved.
- **`AppModel` state transitions**: drive `AppModel` against a fake
  `AppServerController` (records `start`/`stop` calls, lets the test fire
  `onStateChange` manually) covering: start → running, start → failed
  (bad config), running → stop → stopped, running → crash → failed, and that
  Start is disabled when the model isn't installed.
- **`MacAppSettings` migration**: covering the version 2 → 3 migration and
  defaulting when the two new keys are absent, mirroring the existing
  migration tests in `Tests/TurboFieldfareApp` for the v1 → v2 case.
- **`ProcessServerController`**: a real-process integration test analogous to
  `Tests/TurboFieldfareRepack/Core/Command/RepackCLITests.swift`, which
  already spawns a real CLI binary and asserts on its behavior — same
  approach here, spawning the actual built `TurboFieldfareServer` and
  asserting on the `.running`/`.failed` transition, with a fast-failing case
  (e.g. an invalid `--model` path) so the test doesn't need a real model.

## Out of scope

- No in-app log viewer — the status line only shows state and the bound
  address.
- No `--model-id` or `--prompt-cache-mode` controls.
- No launchd-based persistence across app quit.
- No changes to `TurboFieldfareServer` itself.
