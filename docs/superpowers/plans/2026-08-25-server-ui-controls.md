# Server UI Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Server" section to the Mac app's Inspector sidebar with start/stop controls for `TurboFieldfareServer`, configured from the app's existing model/runtime settings plus two new server-only settings (port, queue limit).

**Architecture:** `TurboFieldfareServer` is spawned as a sibling `Process`, the same shape the app already uses to launch `TurboFieldfareDecodeService`, but simpler — a plain child process (no launchd) that is told to stop via `SIGTERM` (which the server already handles gracefully) and is torn down on app quit. `AppModel` gains a small state machine (`AppServerState`) driven through a testable `AppServerController` protocol, mirroring the existing `AppModelInstallerClient` / `AppModelLifecycleClient` dependency-injection pattern.

**Tech Stack:** Swift 6.2, SwiftUI (`@Observable`/`@MainActor`), Foundation `Process`/`Pipe`, `Synchronization.Mutex`, Swift Testing (`@Test`/`#expect`).

## Global Constraints

- The `TurboFieldfareServer` binary is located the same way `DecodeServiceInferenceClient.defaultServiceURL()` already finds `TurboFieldfareDecodeService`: `Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("TurboFieldfareServer")`.
- The server is stopped with `Process.terminate()` (`SIGTERM`); `ServerTerminationSignals` (`Sources/TurboFieldfareServer/Core/ServerTerminationSignals.swift`) already makes the server shut down gracefully and exit with status 0 on that signal — no launchd, no custom protocol.
- Server arguments are built entirely from state the app already has: `modelPathText`, `maxContextTokens`, `runtimeOptions` (`AppRuntimeOptions`), plus two new settings, `serverPort` (default `8080`) and `serverQueueLimit` (default `4`). No new controls duplicate anything already in the Memory/Runtime sections.
- `--vision-pack` is added automatically when `isVisionPackInstalled` is true (via `VisionPackLocation.companionURL(forTextModel:)`, `Sources/TurboFieldfare/Runtime/Vision/VisionWeightStore.swift:39`), never a separate toggle.
- Starting the server does not require unloading the app's own model; the UI shows a warning caption instead of blocking.
- `MacAppSettings.currentVersion` moves from `2` to `3`. The two new keys decode with `decodeIfPresent(...) ?? <default>`, matching how `visionResidencyPolicy`/`rdadvisePolicy`/`loadModelOnLaunch` were added for the v1→v2 migration — an older settings file on disk must keep loading.
- Full design context: `docs/superpowers/specs/2026-08-25-server-ui-controls-design.md`.

---

### Task 1: `AppServerState` and `AppServerController`

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Server/AppServerState.swift`
- Create: `Sources/TurboFieldfareApp/Core/Server/AppServerController.swift`
- Test: `Tests/TurboFieldfareApp/Core/Server/AppServerStateTests.swift`

**Interfaces:**
- Produces: `public enum AppServerState: Equatable, Sendable { case stopped, starting, running(port: Int), stopping, failed(message: String) }`
- Produces: `public protocol AppServerController: Sendable { func start(arguments: [String], onStateChange: @escaping @Sendable (AppServerState) -> Void); func stop() }`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Server/AppServerStateTests.swift`:

```swift
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppServerStateTests {
    @Test func runningStatesWithSamePortAreEqual() {
        #expect(AppServerState.running(port: 8080) == .running(port: 8080))
    }

    @Test func runningStatesWithDifferentPortsAreNotEqual() {
        #expect(AppServerState.running(port: 8080) != .running(port: 9090))
    }

    @Test func failedStatesWithDifferentMessagesAreNotEqual() {
        #expect(AppServerState.failed(message: "a") != .failed(message: "b"))
    }

    @Test func stoppedIsNotStarting() {
        #expect(AppServerState.stopped != .starting)
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Scripts/test.sh --filter AppServerStateTests`
Expected: FAIL — `cannot find type 'AppServerState' in scope` (the type does not exist yet).

- [ ] **Step 3: Create `AppServerState.swift`**

```swift
/// The lifecycle of the child `TurboFieldfareServer` process the app can
/// spawn from the Server section of the Inspector.
public enum AppServerState: Equatable, Sendable {
    case stopped
    case starting
    case running(port: Int)
    case stopping
    case failed(message: String)
}
```

- [ ] **Step 4: Create `AppServerController.swift`**

```swift
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `Scripts/test.sh --filter AppServerStateTests`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Server/AppServerState.swift \
        Sources/TurboFieldfareApp/Core/Server/AppServerController.swift \
        Tests/TurboFieldfareApp/Core/Server/AppServerStateTests.swift
git commit -m "Add AppServerState and AppServerController"
```

---

### Task 2: `AppServerArguments` — pure flag builder

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Server/AppServerArguments.swift`
- Test: `Tests/TurboFieldfareApp/Core/Server/AppServerArgumentsTests.swift`

**Interfaces:**
- Consumes: `AppRuntimeOptions` (`Sources/TurboFieldfareApp/Core/Configuration/AppRuntimeOptions.swift`) — `expertCacheSlots: Int`, `expertCachePolicy: AppExpertCachePolicy` (`.rawValue` is `"lfu"`/`"lru"`), `prefillEnabled: Bool`, `prefillChunkTokens: Int`, `rdadvisePolicy: AppRDAdvicePolicy` (`.rawValue` is `"off"`/`"default"`/`"bounded"`/`"adaptive"`).
- Produces: `public enum AppServerArguments { public static func build(modelPath: String, maxContextTokens: Int, runtimeOptions: AppRuntimeOptions, visionPackPath: String?, port: Int, queueLimit: Int) -> [String] }`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Server/AppServerArgumentsTests.swift`:

```swift
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppServerArgumentsTests {
    @Test func buildsFlagsFromRuntimeOptionsWithoutVision() {
        let options = AppRuntimeOptions(
            expertCacheSlots: 24,
            expertCachePolicy: .lru,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: .bounded)

        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 32_768,
            runtimeOptions: options,
            visionPackPath: nil,
            port: 9090,
            queueLimit: 8)

        #expect(arguments == [
            "--model", "/tmp/gemma4.gturbo",
            "--max-context", "32768",
            "--expert-cache-slots", "24",
            "--expert-cache-policy", "lru",
            "--prefill", "off",
            "--prefill-chunk-tokens", "64",
            "--rdadvise", "bounded",
            "--port", "9090",
            "--queue-limit", "8",
        ])
    }

    @Test func prefillOnMapsToOn() {
        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 16_384,
            runtimeOptions: AppRuntimeOptions(prefillEnabled: true),
            visionPackPath: nil,
            port: 8080,
            queueLimit: 4)

        #expect(arguments.contains("--prefill"))
        let index = arguments.firstIndex(of: "--prefill")!
        #expect(arguments[index + 1] == "on")
    }

    @Test func appendsVisionPackFlagWhenPathProvided() {
        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 16_384,
            runtimeOptions: AppRuntimeOptions(),
            visionPackPath: "/tmp/gemma4.vision.gturbo",
            port: 8080,
            queueLimit: 4)

        #expect(arguments.suffix(2) == ["--vision-pack", "/tmp/gemma4.vision.gturbo"])
    }

    @Test func omitsVisionPackFlagWhenPathIsNil() {
        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 16_384,
            runtimeOptions: AppRuntimeOptions(),
            visionPackPath: nil,
            port: 8080,
            queueLimit: 4)

        #expect(!arguments.contains("--vision-pack"))
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Scripts/test.sh --filter AppServerArgumentsTests`
Expected: FAIL — `cannot find 'AppServerArguments' in scope`.

- [ ] **Step 3: Implement `AppServerArguments.swift`**

```swift
/// Builds the `TurboFieldfareServer` command-line flags from state the app
/// already tracks, so the server is always configured with exactly what the
/// Memory/Runtime sections show — never a separate, driftable copy.
public enum AppServerArguments {
    public static func build(modelPath: String,
                             maxContextTokens: Int,
                             runtimeOptions: AppRuntimeOptions,
                             visionPackPath: String?,
                             port: Int,
                             queueLimit: Int) -> [String] {
        var arguments = [
            "--model", modelPath,
            "--max-context", String(maxContextTokens),
            "--expert-cache-slots", String(runtimeOptions.expertCacheSlots),
            "--expert-cache-policy", runtimeOptions.expertCachePolicy.rawValue,
            "--prefill", runtimeOptions.prefillEnabled ? "on" : "off",
            "--prefill-chunk-tokens", String(runtimeOptions.prefillChunkTokens),
            "--rdadvise", runtimeOptions.rdadvisePolicy.rawValue,
            "--port", String(port),
            "--queue-limit", String(queueLimit),
        ]
        if let visionPackPath {
            arguments += ["--vision-pack", visionPackPath]
        }
        return arguments
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `Scripts/test.sh --filter AppServerArgumentsTests`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Server/AppServerArguments.swift \
        Tests/TurboFieldfareApp/Core/Server/AppServerArgumentsTests.swift
git commit -m "Add AppServerArguments flag builder"
```

---

### Task 3: `MacAppSettings` v3 — `serverPort` and `serverQueueLimit`

**Files:**
- Modify: `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift`
- Test: `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift` (append)

**Interfaces:**
- Produces: `MacAppSettings.serverPort: Int` (default `8080`), `MacAppSettings.serverQueueLimit: Int` (default `4`).

Note: this task's third test, `serverPortPersistsImmediately`, calls `AppModel.setServerPort(_:)`, which is only added in Task 5. It will fail to compile until Task 5 lands — expected, and re-verified at the end of Task 5.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift`, inside `struct MacAppSettingsTests`, just before the final closing `}`:

```swift
    @Test func serverPortAndQueueLimitRoundTrip() throws {
        let initial = MacAppSettings(serverPort: 9090, serverQueueLimit: 8)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func serverPortAndQueueLimitDefaultWhenAbsentFromAnOlderFile() throws {
        let data = Data("""
        {
          "version": 2,
          "contextTokens": 8192,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true
        }
        """.utf8)

        let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)

        #expect(settings.serverPort == 8080)
        #expect(settings.serverQueueLimit == 4)
    }

    @MainActor
    @Test func serverPortPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = AppModel(modelDirectory: modelDirectory, settingsPersistenceEnabled: true)

        model.setServerPort(9090)

        let saved = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: modelDirectory)
        #expect(saved.serverPort == 9090)
    }
```

- [ ] **Step 2: Run the settings-only tests to verify they fail**

Run: `Scripts/test.sh --filter MacAppSettingsTests/serverPortAndQueueLimitRoundTrip`
Expected: FAIL — `MacAppSettings` has no member `serverPort`/`serverQueueLimit`.

- [ ] **Step 3: Add the fields to `MacAppSettings`**

In `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift`, change:

```swift
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 2
```

to:

```swift
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 3
```

Change:

```swift
    var rdadvisePolicy: AppRDAdvicePolicy = .off
    var loadModelOnLaunch: Bool = false

    private enum CodingKeys: String, CodingKey {
```

to:

```swift
    var rdadvisePolicy: AppRDAdvicePolicy = .off
    var loadModelOnLaunch: Bool = false
    var serverPort: Int = 8080
    var serverQueueLimit: Int = 4

    private enum CodingKeys: String, CodingKey {
```

Change:

```swift
        case rdadvisePolicy
        case loadModelOnLaunch
    }
```

to:

```swift
        case rdadvisePolicy
        case loadModelOnLaunch
        case serverPort
        case serverQueueLimit
    }
```

Change the memberwise `init`'s signature and body — from:

```swift
         rdadvisePolicy: AppRDAdvicePolicy = .off,
         loadModelOnLaunch: Bool = false) {
        self.version = version
```

to:

```swift
         rdadvisePolicy: AppRDAdvicePolicy = .off,
         loadModelOnLaunch: Bool = false,
         serverPort: Int = 8080,
         serverQueueLimit: Int = 4) {
        self.version = version
```

and, further down in the same `init`, from:

```swift
        self.rdadvisePolicy = rdadvisePolicy
        self.loadModelOnLaunch = loadModelOnLaunch
    }

    init(from decoder: Decoder) throws {
```

to:

```swift
        self.rdadvisePolicy = rdadvisePolicy
        self.loadModelOnLaunch = loadModelOnLaunch
        self.serverPort = serverPort
        self.serverQueueLimit = serverQueueLimit
    }

    init(from decoder: Decoder) throws {
```

Change the decoding `init(from:)` — from:

```swift
        loadModelOnLaunch = try container.decodeIfPresent(
            Bool.self,
            forKey: .loadModelOnLaunch) ?? false
    }

    func isValid() -> Bool {
        AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
    }
```

to:

```swift
        loadModelOnLaunch = try container.decodeIfPresent(
            Bool.self,
            forKey: .loadModelOnLaunch) ?? false
        serverPort = try container.decodeIfPresent(
            Int.self,
            forKey: .serverPort) ?? 8080
        serverQueueLimit = try container.decodeIfPresent(
            Int.self,
            forKey: .serverQueueLimit) ?? 4
    }

    func isValid() -> Bool {
        AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
            && (1...65_535).contains(serverPort)
            && serverQueueLimit > 0
    }
```

- [ ] **Step 4: Run the settings-only tests to verify they pass**

Run: `Scripts/test.sh --filter MacAppSettingsTests/serverPortAndQueueLimitRoundTrip`
Run: `Scripts/test.sh --filter MacAppSettingsTests/serverPortAndQueueLimitDefaultWhenAbsentFromAnOlderFile`
Expected: PASS. (`serverPortPersistsImmediately` still fails to compile — expected until Task 5.)

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift \
        Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift
git commit -m "Add serverPort and serverQueueLimit to MacAppSettings (v3)"
```

---

### Task 4: `ProcessServerController`

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Server/ProcessServerController.swift`
- Test: `Tests/TurboFieldfareApp/Core/Server/ProcessServerControllerTests.swift`

**Interfaces:**
- Consumes: `AppServerState`, `AppServerController` (Task 1).
- Produces: `public final class ProcessServerController: AppServerController, @unchecked Sendable { public init(executableURL: URL = ProcessServerController.defaultExecutableURL()); public static func defaultExecutableURL() -> URL }`

- [ ] **Step 1: Write the failing tests**

Create `Tests/TurboFieldfareApp/Core/Server/ProcessServerControllerTests.swift`:

```swift
import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct ProcessServerControllerTests {
    @Test func anOutOfRangePortReportsAFailedStateQuickly() async throws {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/TurboFieldfareServer")
        let controller = ProcessServerController(executableURL: executable)
        let states = StateCollector()

        controller.start(arguments: [
            "--model", "/tmp/turbofieldfare-arguments-only-\(UUID().uuidString).gturbo",
            "--port", "0",
        ]) { state in
            states.append(state)
        }

        let deadline = Date().addingTimeInterval(10)
        while !states.containsFailure(), Date() < deadline {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }

        #expect(states.containsFailure(), "expected a failed state before the deadline")
        #expect(states.failureMessages().contains {
            $0.contains("--port must be between 1 and 65535")
        })
    }

    @Test func aMissingExecutableReportsAFailedStateImmediately() {
        let executable = URL(fileURLWithPath: "/tmp/turbofieldfare-no-such-binary")
        let controller = ProcessServerController(executableURL: executable)
        let states = StateCollector()

        controller.start(arguments: ["--model", "/tmp/anything"]) { state in
            states.append(state)
        }

        #expect(states.containsFailure())
    }
}

private final class StateCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppServerState] = []

    func append(_ state: AppServerState) {
        lock.lock()
        states.append(state)
        lock.unlock()
    }

    func containsFailure() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return states.contains { if case .failed = $0 { return true }; return false }
    }

    func failureMessages() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return states.compactMap { state in
            if case .failed(let message) = state { return message }
            return nil
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `Scripts/test.sh --filter ProcessServerControllerTests`
Expected: FAIL — `cannot find 'ProcessServerController' in scope`.

- [ ] **Step 3: Implement `ProcessServerController.swift`**

```swift
import Foundation
import Synchronization

/// Spawns `TurboFieldfareServer` as a plain child process — no launchd, since
/// unlike the decode service this does not need to survive the app quitting
/// (`AppModel.stopServerForTermination()` stops it explicitly). Stopping
/// sends `SIGTERM`, which `ServerTerminationSignals` already turns into a
/// graceful shutdown that exits with status 0; any other exit status is
/// reported as `.failed`, whether that is a startup argument error or a
/// mid-run crash — one path for both, the same way `terminationHandler`
/// fires for either.
public final class ProcessServerController: AppServerController, @unchecked Sendable {
    private let executableURL: URL
    private let lock = NSLock()
    private var process: Process?

    public init(executableURL: URL = ProcessServerController.defaultExecutableURL()) {
        self.executableURL = executableURL
    }

    public func start(arguments: [String],
                      onStateChange: @escaping @Sendable (AppServerState) -> Void) {
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
            onStateChange(.failed(message:
                "server executable is missing at \(executableURL.path); run swift build -c release before starting the server"))
            return
        }

        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.executableURL = executableURL
        process.arguments = arguments
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let stdoutBuffer = Mutex(Data())
        let stderrBuffer = Mutex(Data())
        let reportedPort = Mutex<Int?>(nil)

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text: String? = stdoutBuffer.withLock { buffer in
                buffer.append(data)
                return String(data: buffer, encoding: .utf8)
            }
            guard let text, let port = Self.port(fromReadyOutput: text) else { return }
            let isNewReport = reportedPort.withLock { existing -> Bool in
                guard existing == nil else { return false }
                existing = port
                return true
            }
            if isNewReport { onStateChange(.running(port: port)) }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            stderrBuffer.withLock { $0.append(data) }
        }

        process.terminationHandler = { finished in
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            if finished.terminationStatus == 0 {
                onStateChange(.stopped)
                return
            }
            let stderrText = stderrBuffer.withLock {
                String(decoding: $0, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            onStateChange(.failed(message: Self.errorMessage(
                stderr: stderrText, status: finished.terminationStatus)))
        }

        lock.lock()
        self.process = process
        lock.unlock()

        do {
            try process.run()
        } catch {
            lock.lock()
            self.process = nil
            lock.unlock()
            onStateChange(.failed(message: "failed to launch server: \(error)"))
        }
    }

    public func stop() {
        lock.lock()
        let process = self.process
        lock.unlock()
        process?.terminate()
    }

    private static func port(fromReadyOutput text: String) -> Int? {
        guard let readyRange = text.range(of: "TurboFieldfareServer ready") else { return nil }
        let tail = text[readyRange.lowerBound...]
        guard let markerRange = tail.range(of: "http://127.0.0.1:") else { return nil }
        let digits = tail[markerRange.upperBound...].prefix { $0.isNumber }
        return digits.isEmpty ? nil : Int(digits)
    }

    private static func errorMessage(stderr: String, status: Int32) -> String {
        guard !stderr.isEmpty else { return "server exited with status \(status)" }
        return stderr.hasPrefix("error: ") ? String(stderr.dropFirst(7)) : stderr
    }

    public static func defaultExecutableURL() -> URL {
        Bundle.main.executableURL!
            .deletingLastPathComponent()
            .appendingPathComponent("TurboFieldfareServer")
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `Scripts/test.sh --filter ProcessServerControllerTests`
Expected: PASS (2 tests). `swift test` builds every product in the package before running tests (including `TurboFieldfareServer`), so `.build/debug/TurboFieldfareServer` exists by this point — the same precondition `Tests/TurboFieldfareRepack/Core/Command/RepackCLITests.swift` already relies on for `.build/debug/TurboFieldfareRepack`.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Server/ProcessServerController.swift \
        Tests/TurboFieldfareApp/Core/Server/ProcessServerControllerTests.swift
git commit -m "Add ProcessServerController"
```

---

### Task 5: `AppModel` server control and settings wiring

**Files:**
- Create: `Tests/TurboFieldfareApp/Core/Support/MockServerController.swift`
- Create: `Tests/TurboFieldfareApp/Core/State/AppModelServerControlTests.swift`
- Modify: `Sources/TurboFieldfareApp/Core/State/AppModel.swift`

**Interfaces:**
- Consumes: `AppServerState`, `AppServerController` (Task 1), `AppServerArguments.build(...)` (Task 2), `MacAppSettings.serverPort`/`serverQueueLimit` (Task 3), `ProcessServerController()` (Task 4), `VisionPackLocation.companionURL(forTextModel:) throws -> URL` (`Sources/TurboFieldfare/Runtime/Vision/VisionWeightStore.swift:39`, already reachable — `AppModel.swift` already `import TurboFieldfare`).
- Produces on `AppModel`: `public var serverState: AppServerState`, `public var serverPort: Int`, `public var serverQueueLimit: Int`, `public var canStartServer: Bool`, `public var canStopServer: Bool`, `public var canEditServerSettings: Bool`, `public func startServer()`, `public func stopServer()`, `public func stopServerForTermination()`, `public func setServerPort(_ port: Int)`, `public func setServerQueueLimit(_ limit: Int)`, `func applyServerState(_ state: AppServerState, generation: UInt64)` (internal, mirrors the existing internal `applyLoadState`), and a new `serverController: any AppServerController = ProcessServerController()` init parameter.

- [ ] **Step 1: Add `MockServerController`**

Create `Tests/TurboFieldfareApp/Core/Support/MockServerController.swift`:

```swift
import Foundation
@testable import TurboFieldfareAppCore

final class MockServerController: AppServerController, @unchecked Sendable {
    private let lock = NSLock()
    private var _startCalls: [[String]] = []
    private var _stopCallCount = 0
    private var latestHandler: (@Sendable (AppServerState) -> Void)?

    var startCalls: [[String]] {
        lock.lock()
        defer { lock.unlock() }
        return _startCalls
    }

    var stopCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _stopCallCount
    }

    func start(arguments: [String],
              onStateChange: @escaping @Sendable (AppServerState) -> Void) {
        lock.lock()
        _startCalls.append(arguments)
        latestHandler = onStateChange
        lock.unlock()
    }

    func stop() {
        lock.lock()
        _stopCallCount += 1
        lock.unlock()
    }

    /// Simulates the controller reporting a new state, the way the real
    /// `ProcessServerController` does from its stdout/termination handlers.
    func emit(_ state: AppServerState) {
        let handler: (@Sendable (AppServerState) -> Void)?
        lock.lock()
        handler = latestHandler
        lock.unlock()
        handler?(state)
    }
}
```

- [ ] **Step 2: Write the failing `AppModel` tests**

Create `Tests/TurboFieldfareApp/Core/State/AppModelServerControlTests.swift`:

```swift
import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppModelServerControlTests {
    @MainActor
    @Test func startServerDisabledWithoutInstalledModel() {
        let model = AppModel(client: MockLifecycleInferenceClient(),
                             serverController: MockServerController())
        model.modelPathText = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing.gturbo").path

        #expect(!model.canStartServer)
        model.startServer()
        #expect(model.serverState == .stopped)
    }

    @MainActor
    @Test func startServerSendsArgumentsBuiltFromCurrentSettings() throws {
        let directory = try makeCompleteModelInstall("server-control-arguments")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)
        model.maxContextTokens = 32_768
        model.runtimeOptions.expertCacheSlots = 24
        model.setServerPort(9090)
        model.setServerQueueLimit(8)

        model.startServer()

        #expect(model.serverState == .starting)
        let expected = AppServerArguments.build(
            modelPath: model.modelPathText,
            maxContextTokens: 32_768,
            runtimeOptions: model.runtimeOptions,
            visionPackPath: nil,
            port: 9090,
            queueLimit: 8)
        #expect(controller.startCalls == [expected])
    }

    @MainActor
    @Test func startServerIncludesVisionPackPathWhenInstalled() throws {
        let directory = try makeVisionReadyModelInstall("server-control-vision")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)
        #expect(model.isVisionPackInstalled)

        model.startServer()

        let companion = try VisionPackLocation.companionURL(forTextModel: directory)
        #expect(controller.startCalls.last?.suffix(2) == ["--vision-pack", companion.path])
    }

    @MainActor
    @Test func applyServerStateIgnoresAStaleGeneration() throws {
        let directory = try makeCompleteModelInstall("server-control-stale")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: MockServerController())

        model.startServer()
        #expect(model.serverState == .starting)
        model.applyServerState(.running(port: 1234), generation: 0)

        #expect(model.serverState == .starting,
                "a callback from a previous start overwrote the current one")
    }

    @MainActor
    @Test func runningStateEnablesStopAndDisablesStart() async throws {
        let directory = try makeCompleteModelInstall("server-control-running")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        #expect(model.canStopServer)
        #expect(!model.canStartServer)
    }

    @MainActor
    @Test func stopServerSendsStopAndReturnsToStoppedOnGracefulExit() async throws {
        let directory = try makeCompleteModelInstall("server-control-stop")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        model.stopServer()
        #expect(model.serverState == .stopping)
        #expect(controller.stopCallCount == 1)

        controller.emit(.stopped)
        try await waitUntil(deadline: 5) { model.serverState == .stopped }
        #expect(model.canStartServer)
    }

    @MainActor
    @Test func aCrashWhileRunningIsReportedAsFailedAndCanBeRestarted() async throws {
        let directory = try makeCompleteModelInstall("server-control-crash")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        controller.emit(.failed(message: "server exited unexpectedly with status 134"))
        try await waitUntil(deadline: 5) {
            model.serverState == .failed(message: "server exited unexpectedly with status 134")
        }
        #expect(model.canStartServer)
    }

    @MainActor
    @Test func stopServerForTerminationStopsARunningServer() async throws {
        let directory = try makeCompleteModelInstall("server-control-termination")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        model.stopServerForTermination()

        #expect(controller.stopCallCount == 1)
    }

    @MainActor
    @Test func stopServerForTerminationDoesNothingWhenAlreadyStopped() {
        let controller = MockServerController()
        let model = AppModel(client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.stopServerForTermination()

        #expect(controller.stopCallCount == 0)
    }

    private func waitUntil(deadline seconds: TimeInterval,
                           _ condition: @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition(), Date() < deadline {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `Scripts/test.sh --filter AppModelServerControlTests`
Expected: FAIL to compile — `AppModel` has no `serverController` parameter, and no `serverState`/`serverPort`/`serverQueueLimit`/`canStartServer`/`canStopServer`/`canEditServerSettings`/`startServer`/`stopServer`/`stopServerForTermination`/`applyServerState` members.

- [ ] **Step 4: Add stored properties to `AppModel`**

In `Sources/TurboFieldfareApp/Core/State/AppModel.swift`, change:

```swift
    public private(set) var loadModelOnLaunch: Bool = false
    /// Whether launching the app should load the model straight away. Off by
    /// default, because loading takes minutes and holds gigabytes.
    public var diagnostics: AppDiagnostics?
```

to:

```swift
    public private(set) var loadModelOnLaunch: Bool = false
    /// Whether launching the app should load the model straight away. Off by
    /// default, because loading takes minutes and holds gigabytes.
    public var serverState: AppServerState = .stopped
    public var serverPort: Int = 8080
    public var serverQueueLimit: Int = 4
    public var diagnostics: AppDiagnostics?
```

Change:

```swift
    private let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private let visionInstaller: any AppVisionPackInstallerClient
    private var runTask: Task<Void, Never>?
```

to:

```swift
    private let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private let visionInstaller: any AppVisionPackInstallerClient
    private let serverController: any AppServerController
    private var serverGeneration: UInt64 = 0
    private var runTask: Task<Void, Never>?
```

- [ ] **Step 5: Wire the new init parameter and load persisted server settings**

Change the `init` signature and body — from:

```swift
    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(),
                visionInstaller: any AppVisionPackInstallerClient = RepackVisionPackInstallerClient(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                attachmentStore: AppImageAttachmentStore = AppImageAttachmentStore(),
                visionRuntimeSupported: Bool = true,
                settingsPersistenceEnabled: Bool = false) {
```

to:

```swift
    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(),
                visionInstaller: any AppVisionPackInstallerClient = RepackVisionPackInstallerClient(),
                serverController: any AppServerController = ProcessServerController(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                attachmentStore: AppImageAttachmentStore = AppImageAttachmentStore(),
                visionRuntimeSupported: Bool = true,
                settingsPersistenceEnabled: Bool = false) {
```

Change:

```swift
        self.loadModelOnLaunch = settings.loadModelOnLaunch
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.visionInstallationStatus = AppVisionPackInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.visionInstaller = visionInstaller
        self.memorySampler = memorySampler
```

to:

```swift
        self.loadModelOnLaunch = settings.loadModelOnLaunch
        self.serverPort = settings.serverPort
        self.serverQueueLimit = settings.serverQueueLimit
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.visionInstallationStatus = AppVisionPackInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.visionInstaller = visionInstaller
        self.serverController = serverController
        self.memorySampler = memorySampler
```

- [ ] **Step 6: Wire persisted-settings load/save**

Change `applyPersistedSettings` — from:

```swift
        sentPromptBehavior = settings.sentPromptBehavior
        loadModelOnLaunch = settings.loadModelOnLaunch
    }

    private func persistSettings() {
```

to:

```swift
        sentPromptBehavior = settings.sentPromptBehavior
        loadModelOnLaunch = settings.loadModelOnLaunch
        serverPort = settings.serverPort
        serverQueueLimit = settings.serverQueueLimit
    }

    private func persistSettings() {
```

Change the `MacAppSettings(...)` construction inside `persistSettings()` — from:

```swift
            rdadvisePolicy: runtimeOptions.rdadvisePolicy,
            loadModelOnLaunch: loadModelOnLaunch)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
```

to:

```swift
            rdadvisePolicy: runtimeOptions.rdadvisePolicy,
            loadModelOnLaunch: loadModelOnLaunch,
            serverPort: serverPort,
            serverQueueLimit: serverQueueLimit)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
```

- [ ] **Step 7: Add the gating computed properties**

Change:

```swift
    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && loadState.isReady
    }

    public var isModelInstalled: Bool { installationStatus == .complete }
```

to:

```swift
    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && loadState.isReady
    }

    public var canStartServer: Bool {
        guard isModelInstalled, !isInstallingModel, !isVisionCompanionOperationInProgress else {
            return false
        }
        switch serverState {
        case .stopped, .failed: return true
        case .starting, .running, .stopping: return false
        }
    }

    public var canStopServer: Bool {
        switch serverState {
        case .starting, .running: return true
        case .stopped, .stopping, .failed: return false
        }
    }

    public var canEditServerSettings: Bool {
        switch serverState {
        case .stopped, .failed: return true
        case .starting, .running, .stopping: return false
        }
    }

    public var isModelInstalled: Bool { installationStatus == .complete }
```

- [ ] **Step 8: Add the server control methods**

Change:

```swift
    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func installModel() {
```

to:

```swift
    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func startServer() {
        guard canStartServer else { return }
        serverGeneration &+= 1
        let generation = serverGeneration
        serverState = .starting
        let arguments = AppServerArguments.build(
            modelPath: modelPathText,
            maxContextTokens: maxContextTokens,
            runtimeOptions: runtimeOptions,
            visionPackPath: serverVisionPackPath,
            port: serverPort,
            queueLimit: serverQueueLimit)
        serverController.start(arguments: arguments) { [weak self] state in
            Task { @MainActor in
                self?.applyServerState(state, generation: generation)
            }
        }
    }

    public func stopServer() {
        guard canStopServer else { return }
        serverState = .stopping
        serverController.stop()
    }

    /// Called from `applicationWillTerminate`. Sends the stop signal without
    /// waiting for the graceful shutdown the app is about to stop observing.
    public func stopServerForTermination() {
        switch serverState {
        case .starting, .running: serverController.stop()
        case .stopped, .stopping, .failed: break
        }
    }

    public func setServerPort(_ port: Int) {
        guard serverPort != port else { return }
        serverPort = port
        persistSettings()
    }

    public func setServerQueueLimit(_ limit: Int) {
        guard serverQueueLimit != limit else { return }
        serverQueueLimit = limit
        persistSettings()
    }

    private var serverVisionPackPath: String? {
        guard isVisionPackInstalled,
              let companion = try? VisionPackLocation.companionURL(
                forTextModel: URL(fileURLWithPath: modelPathText)) else {
            return nil
        }
        return companion.path
    }

    public func installModel() {
```

- [ ] **Step 9: Add the internal `applyServerState`**

Change:

```swift
    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }
```

to:

```swift
    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    /// Ignores a callback from a start this model has since moved past — a
    /// stop, a crash, or a newer start already replaced the generation the
    /// callback was registered under.
    func applyServerState(_ state: AppServerState, generation: UInt64) {
        guard generation == serverGeneration else { return }
        serverState = state
    }
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `Scripts/test.sh --filter AppModelServerControlTests`
Run: `Scripts/test.sh --filter MacAppSettingsTests/serverPortPersistsImmediately`
Expected: PASS (all 9 `AppModelServerControlTests` tests, plus `serverPortPersistsImmediately` from Task 3, which now compiles).

- [ ] **Step 11: Run the full suite**

Run: `Scripts/test.sh`
Expected: PASS — nothing from Tasks 1-4 regressed.

- [ ] **Step 12: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/State/AppModel.swift \
        Tests/TurboFieldfareApp/Core/Support/MockServerController.swift \
        Tests/TurboFieldfareApp/Core/State/AppModelServerControlTests.swift
git commit -m "Add AppModel server control state machine and settings"
```

---

### Task 6: Inspector "Server" section

**Files:**
- Modify: `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift`

**Interfaces:**
- Consumes: `model.serverState`, `model.serverPort`, `model.serverQueueLimit`, `model.canStartServer`, `model.canStopServer`, `model.canEditServerSettings`, `model.startServer()`, `model.stopServer()`, `model.setServerPort(_:)`, `model.setServerQueueLimit(_:)`, `model.loadState.isReady`, `model.isInstallingModel`, `model.isVisionCompanionOperationInProgress` (all from Task 5).

No `AppModel`/business-logic test applies to this file — there is no existing test target covering SwiftUI views in this codebase (`InspectorView.swift` has no corresponding test file). Verification is manual, in Step 3.

- [ ] **Step 1: Add the section to `body`**

In `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift`, change:

```swift
    var body: some View {
        Form {
            modelSection
            // Second, beside Model: image support is an install concern, not a
            // diagnostic. Last put it under the runner diagnostics and below the
            // fold, where the one screen that must mention it - the empty state
            // before any model exists - could not.
            if showsVisionSection {
                visionSection
            }
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
```

to:

```swift
    var body: some View {
        Form {
            modelSection
            // Second, beside Model: image support is an install concern, not a
            // diagnostic. Last put it under the runner diagnostics and below the
            // fold, where the one screen that must mention it - the empty state
            // before any model exists - could not.
            if showsVisionSection {
                visionSection
            }
            memorySection
            generationSection
            runtimeSection
            serverSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
```

- [ ] **Step 2: Add the `serverSection` and its helpers**

Change the end of the file — from:

```swift
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

}
```

to:

```swift
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

    private var serverSection: some View {
        Section("Server") {
            LabeledContent("State") {
                Text(serverStateLabel)
                    .font(.caption)
                    .foregroundStyle(serverStateColor)
            }
            LabeledContent("Port") {
                TextField("Port", value: serverPortBinding, format: .number.grouping(.never))
                    .labelsHidden()
                    .frame(width: 80)
                    .multilineTextAlignment(.trailing)
                    .disabled(!model.canEditServerSettings)
            }
            LabeledContent("Queue limit") {
                TextField("Queue limit", value: serverQueueLimitBinding,
                          format: .number.grouping(.never))
                    .labelsHidden()
                    .frame(width: 80)
                    .multilineTextAlignment(.trailing)
                    .disabled(!model.canEditServerSettings)
            }
            if model.loadState.isReady && model.canStartServer {
                Text("The app's model is also loaded — running both uses memory for two copies.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if model.canStopServer {
                Button("Stop Server", role: .destructive, action: model.stopServer)
            } else {
                Button("Start Server", action: model.startServer)
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canStartServer)
            }
        }
        .disabled(model.isInstallingModel || model.isVisionCompanionOperationInProgress)
    }

    private var serverStateLabel: String {
        switch model.serverState {
        case .stopped: return "Stopped"
        case .starting: return "Starting…"
        case .running(let port): return "Running · http://127.0.0.1:\(port)"
        case .stopping: return "Stopping…"
        case .failed(let message): return "Error: \(message)"
        }
    }

    private var serverStateColor: Color {
        switch model.serverState {
        case .failed: return .red
        case .running: return .green
        case .stopped, .starting, .stopping: return .secondary
        }
    }

    private var serverPortBinding: Binding<Int> {
        Binding(get: { model.serverPort }, set: { model.setServerPort($0) })
    }

    private var serverQueueLimitBinding: Binding<Int> {
        Binding(get: { model.serverQueueLimit }, set: { model.setServerQueueLimit($0) })
    }

}
```

- [ ] **Step 3: Build and manually verify**

Run: `swift build -c debug`
Expected: builds with no errors.

Run: `swift run TurboFieldfareMac`, open the Inspector sidebar, and confirm:
- A "Server" section appears after Runtime, with State/Port/Queue limit rows and a Start Server button.
- With no model installed, Start Server is disabled.
- After installing/loading a model (or pointing the app's model path at an existing `.gturbo` directory), Start Server is enabled; clicking it flips State to "Starting…" then "Running · http://127.0.0.1:8080"; `curl http://127.0.0.1:8080/health` succeeds while running.
- Stop Server flips State to "Stopping…" then "Stopped", and the port/queue-limit fields become editable again.
- Quitting the app while the server is running leaves no orphaned `TurboFieldfareServer` process (`pgrep -fl TurboFieldfareServer` after quitting the app returns nothing).

- [ ] **Step 4: Commit**

```bash
git add Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift
git commit -m "Add Server section to the Inspector sidebar"
```

---

### Task 7: Stop the server when the app quits

**Files:**
- Modify: `Sources/TurboFieldfareApp/Mac/App/TurboFieldfareMacApp.swift`

**Interfaces:**
- Consumes: `model.stopServerForTermination()` (Task 5, already unit-tested by `AppModelServerControlTests.stopServerForTerminationStopsARunningServer` / `...DoesNothingWhenAlreadyStopped`).

No new automated test — `TurboFieldfareMacApp.swift` is part of the `TurboFieldfareMac` executable target, which has no test target in `Package.swift`, and the behavior it calls is already covered at the `AppModel` level in Task 5.

- [ ] **Step 1: Call `stopServerForTermination()` on quit**

In `Sources/TurboFieldfareApp/Mac/App/TurboFieldfareMacApp.swift`, change:

```swift
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { Self.model?.releaseAllAttachments() }
    }
```

to:

```swift
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            Self.model?.releaseAllAttachments()
            Self.model?.stopServerForTermination()
        }
    }
```

- [ ] **Step 2: Build**

Run: `swift build -c debug`
Expected: builds with no errors.

- [ ] **Step 3: Manually verify**

Start the app, click Start Server, confirm it reaches "Running" (per Task 6 Step 3), then quit the app (Cmd+Q). Run `pgrep -fl TurboFieldfareServer` — expect no output.

- [ ] **Step 4: Commit**

```bash
git add Sources/TurboFieldfareApp/Mac/App/TurboFieldfareMacApp.swift
git commit -m "Stop the server process when the app quits"
```

---

## Self-Review Notes

- **Spec coverage:** Process-per-server-spawn (Tasks 1, 4), argument inheritance from app settings (Task 2, 5), concurrency "warn but allow" (Task 6 caption), terminate-with-app (Task 7), editable port/queue-limit defaulting to 8080/4 (Task 3, 6), vision auto-inherit (Task 5's `startServerIncludesVisionPackPathWhenInstalled`), status-line-only UI (Task 6), settings persistence version 3 (Task 3) — all covered.
- **Placeholder scan:** no TBD/TODO; every step has complete code.
- **Type consistency:** `AppServerState`, `AppServerController`, `AppServerArguments.build(...)`, `ProcessServerController`, and every `AppModel` member name/signature used in Task 6/7 match exactly what Tasks 1-5 define.
- **Dependency order:** Task 4 (`ProcessServerController`) depends only on Task 1, so it now lands *before* Task 5 (`AppModel` wiring, whose init needs the `ProcessServerController` type to compile) — no task is left uncompilable pending a later one.
