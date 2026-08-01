# Model Picker and Switcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user browse, install, delete, and switch between multiple models — including custom Hugging Face repositories — where every supported model is Gemma 4 26B-A4B or a finetune of it.

**Architecture:** A compiled-in curated catalog merged with user-added custom entries persisted to `catalog.json`. Each model installs into its own slug-keyed directory. Custom repositories are gated by a consent prompt and trust-on-first-use fingerprint recording. Switching is `unload` then `load` over the existing decode protocol — no process restart. Conversations move from per-model files to one global store so chat history survives a switch.

**Tech Stack:** Swift 6.2, SwiftUI, Swift Testing (`import Testing`, `@Suite` / `@Test` / `#expect`), Metal 4, macOS 26.

## Global Constraints

- Swift 6.2, macOS 26 or later. Strict concurrency: every new public type is `Sendable`.
- Tests use Swift Testing, never XCTest. Existing style: `@Suite struct NameTests { @Test func behavior() throws {} }`.
- App-layer code lives in target `TurboFieldfareAppCore` (path `Sources/TurboFieldfareApp`), tested by `TurboFieldfareAppCoreTests` (path `Tests/TurboFieldfareApp/Core`).
- Runtime code lives in target `TurboFieldfare` (path `Sources/TurboFieldfare`), tested by `TurboFieldfareTestsCore` (path `Tests/TurboFieldfare/Core`).
- Repack code lives in target `TurboFieldfareRepackCore` (path `Sources/TurboFieldfareRepack`).
- Only supported architecture: Gemma 4 26B-A4B. Its `config.json` reports `model_type: "gemma4"`, `architectures: ["Gemma4ForConditionalGeneration"]`, `text_config.model_type: "gemma4_text"`.
- Memory ceiling formula, exact: `min(4 GB, max(1.5 GB, 0.25 × installedRAM))`.
- Run a single suite with `swift test --filter <SuiteName>`. Build with `swift build`.
- Commit after every task. Do not push.
- `AGENTS.md` forbids changing runtime defaults beyond what this plan specifies.

## File Structure

**Create — `Sources/TurboFieldfareApp/Core/Catalog/`**

| File | Responsibility |
|---|---|
| `ModelSlug.swift` | Repository ID → filesystem-safe slug; rejects path escapes |
| `ModelCatalogEntry.swift` | Entry value type and trust tier |
| `ModelCatalogFile.swift` | Codable persisted shape for custom entries |
| `ModelCatalogStore.swift` | Load/save `catalog.json` with corrupt-file fallback |
| `ModelCatalog.swift` | Curated + custom merge, collision rule, lookup |
| `ModelTrustPolicy.swift` | Trust-on-first-use decisions |
| `ArchPreflight.swift` | Source `config.json` architecture check before download |
| `AppMemoryBudget.swift` | Installed-RAM-aware ceiling |

**Create — `Sources/TurboFieldfareApp/Mac/Catalog/`**

| File | Responsibility |
|---|---|
| `ModelPickerView.swift` | Catalog list with per-entry state and actions |
| `AddCustomModelSheet.swift` | Repo entry, preflight result, consent |

**Modify**

| File | Change |
|---|---|
| `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift` | Per-slug install directories |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallDescriptor.swift` | Build descriptors from catalog entries |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallerClient.swift` | Install any entry, not just the default |
| `Sources/TurboFieldfareApp/Core/Installation/RepackModelInstallerClient.swift` | Per-entry repack options and trust flag |
| `Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift` | Options computed from `ArchConfig` + budget |
| `Sources/TurboFieldfareApp/Core/State/ConversationModels.swift` | `ChatTurn.modelID` |
| `Sources/TurboFieldfareApp/Core/State/ConversationFileStore.swift` | Global store + migration |
| `Sources/TurboFieldfareApp/Core/State/AppModel.swift` | Per-model install state, switching |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift` | Validate against a list of architectures |
| `Sources/TurboFieldfareRepack/Core/Verification/SourceFingerprint.swift` | Expose curated fingerprints for lookup |

---

### Task 1: Model slug derivation

Pure function, no dependencies. The repository ID is user input that becomes a directory name, so this is the security boundary for path traversal.

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ModelSlug.swift`
- Test: `Tests/TurboFieldfareApp/Core/Catalog/ModelSlugTests.swift`

**Interfaces:**
- Consumes: nothing
- Produces: `ModelSlug.make(repoID: String) throws -> String`, `ModelSlug.InvalidRepositoryID(reason: String)`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Catalog/ModelSlugTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelSlugTests {
    @Test func derivesStableSlugFromRepositoryID() throws {
        let slug = try ModelSlug.make(repoID: "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(slug == "mlx-community--gemma-4-26b-a4b-it-4bit")
    }

    @Test func isDeterministic() throws {
        let first = try ModelSlug.make(repoID: "someone/gemma-4-26b-a4b-uncensored")
        let second = try ModelSlug.make(repoID: "someone/gemma-4-26b-a4b-uncensored")
        #expect(first == second)
    }

    @Test func rejectsParentDirectoryComponents() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "../etc")
        }
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "owner/..")
        }
    }

    @Test func rejectsAbsolutePaths() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "/etc/passwd")
        }
    }

    @Test func rejectsWrongComponentCount() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "no-owner")
        }
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "a/b/c")
        }
    }

    @Test func rejectsEmptyAndSeparatorOnlyInput() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "")
        }
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "owner/")
        }
    }

    @Test func rejectsShellAndPathMetacharacters() {
        for hostile in ["own er/name", "owner/na\\me", "owner/na:me", "owner/na\0me"] {
            #expect(throws: ModelSlug.InvalidRepositoryID.self) {
                try ModelSlug.make(repoID: hostile)
            }
        }
    }

    @Test func slugStaysInsideParentDirectory() throws {
        let parent = URL(fileURLWithPath: "/tmp/models", isDirectory: true)
        let slug = try ModelSlug.make(repoID: "mlx-community/gemma-4-26b-a4b-it-4bit")
        let resolved = parent.appendingPathComponent(slug, isDirectory: true).standardizedFileURL
        #expect(resolved.path.hasPrefix("/tmp/models/"))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelSlugTests`
Expected: FAIL — cannot find `ModelSlug` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/ModelSlug.swift`:

```swift
import Foundation

/// Converts a Hugging Face repository ID into a filesystem-safe directory name.
///
/// The repository ID is user input that becomes a path component, so this is
/// the traversal boundary: anything that could escape the models directory is
/// rejected here rather than sanitised, because a silently-rewritten slug would
/// point a reinstall at a different directory than the original install.
public enum ModelSlug {
    public struct InvalidRepositoryID: Error, Equatable {
        public let reason: String

        public init(reason: String) {
            self.reason = reason
        }
    }

    public static func make(repoID: String) throws -> String {
        guard !repoID.isEmpty else {
            throw InvalidRepositoryID(reason: "Repository ID is empty.")
        }
        guard !repoID.hasPrefix("/") else {
            throw InvalidRepositoryID(reason: "Repository ID must not be an absolute path.")
        }
        let components = repoID.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2 else {
            throw InvalidRepositoryID(
                reason: "Expected the form owner/name, got \"\(repoID)\".")
        }
        for component in components {
            guard !component.isEmpty else {
                throw InvalidRepositoryID(reason: "Repository ID has an empty component.")
            }
            guard component != "." && component != ".." else {
                throw InvalidRepositoryID(
                    reason: "Repository ID must not contain relative path components.")
            }
            guard component.allSatisfy(isAllowed) else {
                throw InvalidRepositoryID(
                    reason: "Repository ID may only contain letters, digits, "
                        + "hyphen, underscore, and period.")
            }
        }
        return components.joined(separator: "--")
    }

    private static func isAllowed(_ character: Character) -> Bool {
        guard character.isASCII else { return false }
        return character.isLetter
            || character.isNumber
            || character == "-"
            || character == "_"
            || character == "."
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ModelSlugTests`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/ModelSlug.swift \
        Tests/TurboFieldfareApp/Core/Catalog/ModelSlugTests.swift
git commit -m "feat: add model slug derivation with path-escape rejection"
```

---

### Task 2: Catalog entry and persisted catalog file

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogEntry.swift`
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogFile.swift`
- Test: `Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogFileTests.swift`

**Interfaces:**
- Consumes: nothing
- Produces: `ModelTrustTier` (`.curated`, `.custom`), `ModelCatalogEntry` with fields `displayName`, `repoID`, `revision`, `trustTier`, `recordedIndexSHA256: String?`, `approximateDownloadBytes`, `installedBytes`, `reserveBytes`, and computed `id: String`. `ModelCatalogFile` with `version`, `customEntries`, `isValid() -> Bool`, `static let fileName = "catalog.json"`, `static let currentVersion = 1`.

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogFileTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelCatalogFileTests {
    private func makeEntry(repoID: String,
                           tier: ModelTrustTier = .custom,
                           sha: String? = "abc123") -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: tier,
            recordedIndexSHA256: sha,
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func roundTripsThroughJSON() throws {
        let file = ModelCatalogFile(customEntries: [makeEntry(repoID: "owner/model")])
        let encoded = try JSONEncoder().encode(file)
        let decoded = try JSONDecoder().decode(ModelCatalogFile.self, from: encoded)
        #expect(decoded == file)
    }

    @Test func identityIsRepositoryID() {
        #expect(makeEntry(repoID: "owner/model").id == "owner/model")
    }

    @Test func rejectsDuplicateRepositoryIDs() {
        let file = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model"),
            makeEntry(repoID: "owner/model"),
        ])
        #expect(file.isValid() == false)
    }

    @Test func rejectsCuratedEntriesInTheCustomList() {
        let file = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model", tier: .curated),
        ])
        #expect(file.isValid() == false)
    }

    @Test func rejectsUnknownVersion() {
        var file = ModelCatalogFile(customEntries: [])
        file.version = 99
        #expect(file.isValid() == false)
    }

    @Test func acceptsWellFormedFile() {
        let file = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/one"),
            makeEntry(repoID: "owner/two"),
        ])
        #expect(file.isValid())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelCatalogFileTests`
Expected: FAIL — cannot find `ModelCatalogEntry` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogEntry.swift`:

```swift
import Foundation

/// Whether a catalog entry's source was validated by the project or added by
/// the user. Determines both the install-time fingerprint policy and the badge
/// shown in the picker.
public enum ModelTrustTier: String, Codable, Equatable, Sendable {
    case curated
    case custom
}

/// One model the user can install and load.
///
/// `recordedIndexSHA256` carries different meaning per tier: for curated
/// entries it is the fingerprint pinned by the project, for custom entries it
/// is the trust-on-first-use value observed at the first install (nil until
/// then).
public struct ModelCatalogEntry: Codable, Equatable, Sendable, Identifiable {
    public var id: String { repoID }

    public let displayName: String
    public let repoID: String
    public let revision: String
    public let trustTier: ModelTrustTier
    public var recordedIndexSHA256: String?
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public init(displayName: String,
                repoID: String,
                revision: String,
                trustTier: ModelTrustTier,
                recordedIndexSHA256: String?,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.trustTier = trustTier
        self.recordedIndexSHA256 = recordedIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }
}
```

Create `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogFile.swift`:

```swift
import Foundation

/// On-disk shape of the user-added part of the catalog. Curated entries are
/// compiled in and never written here, so a user editing this file by hand can
/// only affect their own additions.
public struct ModelCatalogFile: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public static let fileName = "catalog.json"

    public var version: Int
    public var customEntries: [ModelCatalogEntry]

    public init(version: Int = ModelCatalogFile.currentVersion,
                customEntries: [ModelCatalogEntry] = []) {
        self.version = version
        self.customEntries = customEntries
    }

    public func isValid() -> Bool {
        guard version == Self.currentVersion else { return false }
        guard customEntries.allSatisfy({ $0.trustTier == .custom }) else { return false }
        let ids = customEntries.map(\.repoID)
        return Set(ids).count == ids.count
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ModelCatalogFileTests`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogEntry.swift \
        Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogFile.swift \
        Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogFileTests.swift
git commit -m "feat: add model catalog entry and persisted catalog file"
```

---

### Task 3: Catalog store with corrupt-file fallback

Mirrors the discipline in `ConversationFileStore` and `MacAppSettingsFileStore`: atomic write, and a decode failure never wedges the app. Differs in one way — the corrupt file is *backed up* rather than deleted, because unlike chat history a hand-edited catalog may hold entries the user wants to recover.

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogStore.swift`
- Test: `Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogStoreTests.swift`

**Interfaces:**
- Consumes: `ModelCatalogFile`, `ModelCatalogEntry` from Task 2
- Produces: `ModelCatalogStore.fileURL(inSupportDirectory:) -> URL`, `.load(inSupportDirectory:fileManager:) -> ModelCatalogFile`, `.save(_:inSupportDirectory:fileManager:) throws`, `.corruptFileName = "catalog.corrupt.json"`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogStoreTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelCatalogStoreTests {
    private func makeSupportDirectory(_ name: String) throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catalog-store-\(name)-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func makeEntry(repoID: String) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: .custom,
            recordedIndexSHA256: "abc123",
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func returnsEmptyCatalogWhenFileMissing() throws {
        let directory = try makeSupportDirectory("missing")
        defer { try? FileManager.default.removeItem(at: directory) }

        let loaded = ModelCatalogStore.load(inSupportDirectory: directory)
        #expect(loaded.customEntries.isEmpty)
    }

    @Test func roundTripsThroughDisk() throws {
        let directory = try makeSupportDirectory("round-trip")
        defer { try? FileManager.default.removeItem(at: directory) }

        let file = ModelCatalogFile(customEntries: [makeEntry(repoID: "owner/model")])
        try ModelCatalogStore.save(file, inSupportDirectory: directory)
        #expect(ModelCatalogStore.load(inSupportDirectory: directory) == file)
    }

    @Test func backsUpCorruptFileAndFallsBackToEmpty() throws {
        let directory = try makeSupportDirectory("corrupt")
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = ModelCatalogStore.fileURL(inSupportDirectory: directory)
        try Data("{ not json".utf8).write(to: url)

        let loaded = ModelCatalogStore.load(inSupportDirectory: directory)
        #expect(loaded.customEntries.isEmpty)

        let backup = directory.appendingPathComponent(ModelCatalogStore.corruptFileName)
        #expect(FileManager.default.fileExists(atPath: backup.path))
        #expect(FileManager.default.fileExists(atPath: url.path) == false)
    }

    @Test func backsUpStructurallyInvalidFile() throws {
        let directory = try makeSupportDirectory("invalid")
        defer { try? FileManager.default.removeItem(at: directory) }

        let duplicates = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model"),
            makeEntry(repoID: "owner/model"),
        ])
        let url = ModelCatalogStore.fileURL(inSupportDirectory: directory)
        try JSONEncoder().encode(duplicates).write(to: url)

        #expect(ModelCatalogStore.load(inSupportDirectory: directory).customEntries.isEmpty)
        let backup = directory.appendingPathComponent(ModelCatalogStore.corruptFileName)
        #expect(FileManager.default.fileExists(atPath: backup.path))
    }

    @Test func refusesToSaveInvalidCatalog() throws {
        let directory = try makeSupportDirectory("refuse")
        defer { try? FileManager.default.removeItem(at: directory) }

        let duplicates = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model"),
            makeEntry(repoID: "owner/model"),
        ])
        #expect(throws: (any Error).self) {
            try ModelCatalogStore.save(duplicates, inSupportDirectory: directory)
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelCatalogStoreTests`
Expected: FAIL — cannot find `ModelCatalogStore` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogStore.swift`:

```swift
import Foundation

/// Persists user-added catalog entries.
///
/// A corrupt file is renamed rather than deleted: unlike chat history, the
/// catalog can hold entries the user typed by hand and would want back, and a
/// silent delete would look like the app forgot them.
public enum ModelCatalogStore {
    public static let corruptFileName = "catalog.corrupt.json"

    public static func fileURL(inSupportDirectory supportDirectory: URL) -> URL {
        supportDirectory.standardizedFileURL
            .appendingPathComponent(ModelCatalogFile.fileName, isDirectory: false)
    }

    public static func load(inSupportDirectory supportDirectory: URL,
                            fileManager: FileManager = .default) -> ModelCatalogFile {
        let url = fileURL(inSupportDirectory: supportDirectory)
        guard fileManager.fileExists(atPath: url.path) else {
            return ModelCatalogFile()
        }
        do {
            let file = try JSONDecoder().decode(
                ModelCatalogFile.self,
                from: try Data(contentsOf: url))
            guard file.isValid() else { throw InvalidCatalog() }
            return file
        } catch {
            quarantine(url, in: supportDirectory, fileManager: fileManager)
            return ModelCatalogFile()
        }
    }

    public static func save(_ file: ModelCatalogFile,
                            inSupportDirectory supportDirectory: URL,
                            fileManager: FileManager = .default) throws {
        guard file.isValid() else { throw InvalidCatalog() }
        let url = fileURL(inSupportDirectory: supportDirectory)
        try fileManager.createDirectory(at: supportDirectory,
                                        withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(file)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }

    private static func quarantine(_ url: URL,
                                   in supportDirectory: URL,
                                   fileManager: FileManager) {
        let backup = supportDirectory
            .appendingPathComponent(corruptFileName, isDirectory: false)
        try? fileManager.removeItem(at: backup)
        try? fileManager.moveItem(at: url, to: backup)
        try? fileManager.removeItem(at: url)
    }

    struct InvalidCatalog: Error {}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ModelCatalogStoreTests`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/ModelCatalogStore.swift \
        Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogStoreTests.swift
git commit -m "feat: add catalog store with corrupt-file quarantine"
```

---

### Task 4: Catalog merge with curated-wins collision rule

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalog.swift`
- Test: `Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogTests.swift`

**Interfaces:**
- Consumes: `ModelCatalogEntry`, `ModelTrustTier` from Task 2
- Produces: `ModelCatalog.curated: [ModelCatalogEntry]`, `ModelCatalog(curated:custom:)`, `.entries: [ModelCatalogEntry]`, `.entry(forRepoID:) -> ModelCatalogEntry?`, `.addingCustom(_:) throws -> ModelCatalog`, `ModelCatalog.DuplicateEntry(repoID:)`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelCatalogTests {
    private func makeEntry(repoID: String,
                           tier: ModelTrustTier = .custom) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: tier,
            recordedIndexSHA256: nil,
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func curatedListContainsThePinnedGemmaModel() {
        let pinned = ModelCatalog.curated.first { $0.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit" }
        #expect(pinned != nil)
        #expect(pinned?.trustTier == .curated)
        #expect(pinned?.recordedIndexSHA256?.isEmpty == false)
    }

    @Test func mergesCuratedAndCustomEntries() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/curated", tier: .curated)],
                                   custom: [makeEntry(repoID: "owner/custom")])
        #expect(catalog.entries.count == 2)
    }

    @Test func curatedWinsOnRepositoryIDCollision() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/shared", tier: .curated)],
                                   custom: [makeEntry(repoID: "owner/shared")])
        #expect(catalog.entries.count == 1)
        #expect(catalog.entry(forRepoID: "owner/shared")?.trustTier == .curated)
    }

    @Test func listsCuratedEntriesBeforeCustomOnes() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/curated", tier: .curated)],
                                   custom: [makeEntry(repoID: "owner/aaa-custom")])
        #expect(catalog.entries.first?.trustTier == .curated)
    }

    @Test func addingCustomRejectsCuratedCollision() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/shared", tier: .curated)],
                                   custom: [])
        #expect(throws: ModelCatalog.DuplicateEntry.self) {
            try catalog.addingCustom(makeEntry(repoID: "owner/shared"))
        }
    }

    @Test func addingCustomRejectsExistingCustomEntry() {
        let catalog = ModelCatalog(curated: [], custom: [makeEntry(repoID: "owner/model")])
        #expect(throws: ModelCatalog.DuplicateEntry.self) {
            try catalog.addingCustom(makeEntry(repoID: "owner/model"))
        }
    }

    @Test func addingCustomReturnsCatalogContainingTheNewEntry() throws {
        let catalog = try ModelCatalog(curated: [], custom: [])
            .addingCustom(makeEntry(repoID: "owner/model"))
        #expect(catalog.entry(forRepoID: "owner/model")?.trustTier == .custom)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelCatalogTests`
Expected: FAIL — cannot find `ModelCatalog` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/ModelCatalog.swift`:

```swift
import Foundation
import TurboFieldfareRepackCore

/// The set of models the picker offers.
///
/// Curated entries are compiled in, so a new architecture always arrives with
/// the kernels that can run it. On a repository-ID collision the curated entry
/// wins: otherwise re-adding a pinned model as "custom" would silently
/// downgrade it out of the fingerprint-checked tier.
public struct ModelCatalog: Equatable, Sendable {
    public struct DuplicateEntry: Error, Equatable {
        public let repoID: String
    }

    public static let curated: [ModelCatalogEntry] = [
        ModelCatalogEntry(
            displayName: SupportedModelSource.displayName,
            repoID: SupportedModelSource.repoID,
            revision: SupportedModelSource.revision,
            trustTier: .curated,
            recordedIndexSHA256: SupportedModelSource.sourceIndexSHA256,
            approximateDownloadBytes: SupportedModelSource.approximateDownloadBytes,
            installedBytes: SupportedModelSource.installedBytes,
            reserveBytes: SupportedModelSource.reserveBytes),
    ]

    public let entries: [ModelCatalogEntry]

    public init(curated: [ModelCatalogEntry] = ModelCatalog.curated,
                custom: [ModelCatalogEntry]) {
        let curatedIDs = Set(curated.map(\.repoID))
        self.entries = curated + custom.filter { !curatedIDs.contains($0.repoID) }
    }

    public func entry(forRepoID repoID: String) -> ModelCatalogEntry? {
        entries.first { $0.repoID == repoID }
    }

    public var customEntries: [ModelCatalogEntry] {
        entries.filter { $0.trustTier == .custom }
    }

    public func addingCustom(_ entry: ModelCatalogEntry) throws -> ModelCatalog {
        guard self.entry(forRepoID: entry.repoID) == nil else {
            throw DuplicateEntry(repoID: entry.repoID)
        }
        return ModelCatalog(curated: entries.filter { $0.trustTier == .curated },
                            custom: customEntries + [entry])
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ModelCatalogTests`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/ModelCatalog.swift \
        Tests/TurboFieldfareApp/Core/Catalog/ModelCatalogTests.swift
git commit -m "feat: merge curated and custom catalog entries, curated wins collisions"
```

---

### Task 5: Trust-on-first-use policy

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ModelTrustPolicy.swift`
- Test: `Tests/TurboFieldfareApp/Core/Catalog/ModelTrustPolicyTests.swift`

**Interfaces:**
- Consumes: `ModelCatalogEntry`, `ModelTrustTier` from Task 2
- Produces: `ModelTrustDecision` enum with cases `.allowCurated`, `.curatedFingerprintMismatch(expected:observed:)`, `.needsConsent(observed:)`, `.allowCustom`, `.sourceChanged(recorded:observed:)`; `ModelTrustPolicy.decide(entry:observedIndexSHA256:) -> ModelTrustDecision`; `ModelTrustPolicy.requiresKnownSource(for:) -> Bool`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Catalog/ModelTrustPolicyTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelTrustPolicyTests {
    private func makeEntry(tier: ModelTrustTier,
                           sha: String?) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: "owner/model",
            revision: "main",
            trustTier: tier,
            recordedIndexSHA256: sha,
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func curatedWithMatchingFingerprintIsAllowed() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .curated, sha: "aaa"),
            observedIndexSHA256: "aaa")
        #expect(decision == .allowCurated)
    }

    @Test func curatedWithMismatchedFingerprintIsRejected() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .curated, sha: "aaa"),
            observedIndexSHA256: "bbb")
        #expect(decision == .curatedFingerprintMismatch(expected: "aaa", observed: "bbb"))
    }

    @Test func customWithNoRecordedFingerprintNeedsConsent() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .custom, sha: nil),
            observedIndexSHA256: "bbb")
        #expect(decision == .needsConsent(observed: "bbb"))
    }

    @Test func customWithMatchingRecordedFingerprintIsAllowed() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .custom, sha: "bbb"),
            observedIndexSHA256: "bbb")
        #expect(decision == .allowCustom)
    }

    @Test func customWithChangedFingerprintIsBlocked() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .custom, sha: "bbb"),
            observedIndexSHA256: "ccc")
        #expect(decision == .sourceChanged(recorded: "bbb", observed: "ccc"))
    }

    @Test func onlyCuratedEntriesRequireAKnownSource() {
        #expect(ModelTrustPolicy.requiresKnownSource(for: makeEntry(tier: .curated, sha: "a")))
        #expect(ModelTrustPolicy.requiresKnownSource(for: makeEntry(tier: .custom, sha: "a")) == false)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelTrustPolicyTests`
Expected: FAIL — cannot find `ModelTrustPolicy` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/ModelTrustPolicy.swift`:

```swift
import Foundation

/// What the installer should do about a source's provenance.
///
/// Local integrity is a separate concern handled by `verified-install.json`,
/// which is written and validated for every install regardless of tier. This
/// type only decides whether the *source* is trusted enough to repack.
public enum ModelTrustDecision: Equatable, Sendable {
    /// Curated entry whose observed index matches the pinned fingerprint.
    case allowCurated
    /// Curated entry whose upstream contents no longer match the pin. Always a
    /// hard stop: either the project's pin is stale or the repository moved.
    case curatedFingerprintMismatch(expected: String, observed: String)
    /// Custom entry being installed for the first time. Needs explicit user
    /// consent, after which the observed value is recorded.
    case needsConsent(observed: String)
    /// Custom entry whose observed index matches what was recorded on the first
    /// install.
    case allowCustom
    /// Custom entry whose upstream contents changed since it was added.
    case sourceChanged(recorded: String, observed: String)
}

public enum ModelTrustPolicy {
    public static func decide(entry: ModelCatalogEntry,
                              observedIndexSHA256: String) -> ModelTrustDecision {
        switch entry.trustTier {
        case .curated:
            guard let expected = entry.recordedIndexSHA256 else {
                return .curatedFingerprintMismatch(expected: "", observed: observedIndexSHA256)
            }
            return expected == observedIndexSHA256
                ? .allowCurated
                : .curatedFingerprintMismatch(expected: expected, observed: observedIndexSHA256)
        case .custom:
            guard let recorded = entry.recordedIndexSHA256 else {
                return .needsConsent(observed: observedIndexSHA256)
            }
            return recorded == observedIndexSHA256
                ? .allowCustom
                : .sourceChanged(recorded: recorded, observed: observedIndexSHA256)
        }
    }

    public static func requiresKnownSource(for entry: ModelCatalogEntry) -> Bool {
        entry.trustTier == .curated
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ModelTrustPolicyTests`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/ModelTrustPolicy.swift \
        Tests/TurboFieldfareApp/Core/Catalog/ModelTrustPolicyTests.swift
git commit -m "feat: add trust-on-first-use policy for custom model sources"
```

---

### Task 6: Architecture pre-flight

Runs after the metadata fetch and before any payload transfer, so pasting a Qwen repository fails in seconds instead of after a multi-gigabyte download. Real values from the pinned repository's `config.json`: `model_type: "gemma4"`, `architectures: ["Gemma4ForConditionalGeneration"]`, `text_config.model_type: "gemma4_text"`.

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ArchPreflight.swift`
- Test: `Tests/TurboFieldfareApp/Core/Catalog/ArchPreflightTests.swift`

**Interfaces:**
- Consumes: nothing
- Produces: `ArchPreflight.Result` enum with `.supported(modelType: String)` and `.unsupported(modelType: String)`; `ArchPreflight.evaluate(configJSON: Data) throws -> Result`; `ArchPreflight.supportedModelTypes: Set<String>`; `ArchPreflight.MalformedConfig`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Catalog/ArchPreflightTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ArchPreflightTests {
    private let gemmaConfig = Data("""
    {
      "model_type": "gemma4",
      "architectures": ["Gemma4ForConditionalGeneration"],
      "text_config": { "model_type": "gemma4_text", "num_hidden_layers": 30 }
    }
    """.utf8)

    private let qwenNextConfig = Data("""
    {
      "model_type": "qwen3_next",
      "architectures": ["Qwen3NextForCausalLM"],
      "num_hidden_layers": 48
    }
    """.utf8)

    private let qwen36Config = Data("""
    {
      "model_type": "qwen3_6_moe",
      "architectures": ["Qwen36MoeForConditionalGeneration"],
      "num_hidden_layers": 40
    }
    """.utf8)

    @Test func acceptsTheSupportedGemmaArchitecture() throws {
        #expect(try ArchPreflight.evaluate(configJSON: gemmaConfig) == .supported(modelType: "gemma4"))
    }

    @Test func rejectsQwenNext() throws {
        #expect(try ArchPreflight.evaluate(configJSON: qwenNextConfig)
            == .unsupported(modelType: "qwen3_next"))
    }

    @Test func rejectsQwen36() throws {
        #expect(try ArchPreflight.evaluate(configJSON: qwen36Config)
            == .unsupported(modelType: "qwen3_6_moe"))
    }

    @Test func fallsBackToArchitecturesWhenModelTypeIsAbsent() throws {
        let config = Data("""
        { "architectures": ["Gemma4ForConditionalGeneration"] }
        """.utf8)
        #expect(try ArchPreflight.evaluate(configJSON: config)
            == .supported(modelType: "Gemma4ForConditionalGeneration"))
    }

    @Test func throwsWhenNeitherFieldIsPresent() {
        let config = Data("{ \"num_hidden_layers\": 30 }".utf8)
        #expect(throws: ArchPreflight.MalformedConfig.self) {
            try ArchPreflight.evaluate(configJSON: config)
        }
    }

    @Test func throwsOnUndecodableJSON() {
        #expect(throws: (any Error).self) {
            try ArchPreflight.evaluate(configJSON: Data("{ not json".utf8))
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ArchPreflightTests`
Expected: FAIL — cannot find `ArchPreflight` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/ArchPreflight.swift`:

```swift
import Foundation

/// Checks a source repository's architecture before any weights are fetched.
///
/// The full `ManifestReader` gates still run at load time; this exists purely
/// so an unsupported model costs the user seconds instead of a multi-gigabyte
/// download. Matching is on the coarse `model_type` rather than the full shape
/// because that is all `config.json` reliably exposes before repacking.
public enum ArchPreflight {
    public struct MalformedConfig: Error, Equatable {
        public let reason: String
    }

    public enum Result: Equatable, Sendable {
        case supported(modelType: String)
        case unsupported(modelType: String)
    }

    /// The Gemma 4 26B-A4B checkpoint reports `gemma4` at the top level. Add an
    /// entry here only alongside the kernels that can execute it.
    public static let supportedModelTypes: Set<String> = [
        "gemma4",
        "Gemma4ForConditionalGeneration",
    ]

    private struct SourceConfig: Decodable {
        let modelType: String?
        let architectures: [String]?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case architectures
        }
    }

    public static func evaluate(configJSON: Data) throws -> Result {
        let config = try JSONDecoder().decode(SourceConfig.self, from: configJSON)
        guard let identifier = config.modelType ?? config.architectures?.first else {
            throw MalformedConfig(
                reason: "config.json has neither model_type nor architectures.")
        }
        return supportedModelTypes.contains(identifier)
            ? .supported(modelType: identifier)
            : .unsupported(modelType: identifier)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ArchPreflightTests`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/ArchPreflight.swift \
        Tests/TurboFieldfareApp/Core/Catalog/ArchPreflightTests.swift
git commit -m "feat: reject unsupported architectures before download"
```

---

### Task 7: Per-model install directories

`AppModelLocation` currently hardcodes `gemma4.gturbo` at three sites (`:31`, `:36`, `:40`). Each becomes slug-keyed. The package-root branch used by dev builds is preserved so `swift run TurboFieldfareMac` from a checkout still installs into `scratch/`.

**Files:**
- Modify: `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift`
- Test: `Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift`

**Interfaces:**
- Consumes: `ModelSlug.make(repoID:)` from Task 1
- Produces: `AppModelLocation.defaultURL(forRepoID:) throws -> URL`, `AppModelLocation.resolve(repoID:explicitURL:executableURL:currentDirectoryURL:applicationSupportURL:fileExists:) throws -> URL`, `AppModelLocation.supportDirectory(applicationSupportURL:) -> URL`

- [ ] **Step 1: Write the failing test**

Append to `Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift` (keep existing tests; add this suite alongside them):

```swift
@Suite struct AppModelLocationSlugTests {
    private let gemma = "mlx-community/gemma-4-26b-a4b-it-4bit"

    @Test func packageRootInstallsUnderScratchModelsSlug() throws {
        let root = URL(fileURLWithPath: "/repo", isDirectory: true)
        let resolved = try AppModelLocation.resolve(
            repoID: gemma,
            explicitURL: nil,
            executableURL: root.appendingPathComponent(".build/debug/TurboFieldfareMac"),
            currentDirectoryURL: root,
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { path in
                path == "/repo/Package.swift" || path == "/repo/Sources/TurboFieldfareApp/Mac"
            })
        #expect(resolved.path
            == "/repo/scratch/models/mlx-community--gemma-4-26b-a4b-it-4bit/model.gturbo")
    }

    @Test func applicationSupportFallbackUsesModelsSlug() throws {
        let resolved = try AppModelLocation.resolve(
            repoID: gemma,
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { _ in false })
        #expect(resolved.path
            == "/support/TurboFieldfare/models/mlx-community--gemma-4-26b-a4b-it-4bit/model.gturbo")
    }

    @Test func distinctRepositoriesGetDistinctDirectories() throws {
        let support = URL(fileURLWithPath: "/support", isDirectory: true)
        let first = try AppModelLocation.resolve(
            repoID: gemma, explicitURL: nil, executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: support, fileExists: { _ in false })
        let second = try AppModelLocation.resolve(
            repoID: "someone/gemma-4-26b-a4b-uncensored", explicitURL: nil, executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: support, fileExists: { _ in false })
        #expect(first != second)
    }

    @Test func explicitURLStillWins() throws {
        let resolved = try AppModelLocation.resolve(
            repoID: gemma,
            explicitURL: URL(fileURLWithPath: "/custom/path", isDirectory: true),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { _ in false })
        #expect(resolved.path == "/custom/path")
    }

    @Test func hostileRepositoryIDCannotEscapeModelsDirectory() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try AppModelLocation.resolve(
                repoID: "../../etc",
                explicitURL: nil,
                executableURL: nil,
                currentDirectoryURL: URL(fileURLWithPath: "/elsewhere", isDirectory: true),
                applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
                fileExists: { _ in false })
        }
    }

    @Test func supportDirectoryIsTheModelsParent() {
        let support = AppModelLocation.supportDirectory(
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true))
        #expect(support.path == "/support/TurboFieldfare")
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter AppModelLocationSlugTests`
Expected: FAIL — `resolve` has no `repoID:` parameter.

- [ ] **Step 3: Write minimal implementation**

Replace the body of `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift` with:

```swift
import Foundation

enum AppModelLocation {
    static let modelDirectoryName = "model.gturbo"

    static func defaultURL(forRepoID repoID: String) throws -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return try resolve(
            repoID: repoID,
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:))
    }

    /// The directory holding `catalog.json`, `conversations.json`, and the
    /// per-model `models/` tree.
    static func supportDirectory(applicationSupportURL: URL) -> URL {
        applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .standardizedFileURL
    }

    static func resolve(repoID: String,
                        explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool) throws -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        // Derived before any path is built, so a hostile repository ID cannot
        // reach `appendingPathComponent`.
        let slug = try ModelSlug.make(repoID: repoID)
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return modelURL(under: root.appendingPathComponent("scratch", isDirectory: true),
                            slug: slug)
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return modelURL(under: root.appendingPathComponent("scratch", isDirectory: true),
                            slug: slug)
        }
        return modelURL(under: supportDirectory(applicationSupportURL: applicationSupportURL),
                        slug: slug)
    }

    private static func modelURL(under base: URL, slug: String) -> URL {
        base.appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(slug, isDirectory: true)
            .appendingPathComponent(modelDirectoryName, isDirectory: true)
            .standardizedFileURL
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/TurboFieldfareApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
```

- [ ] **Step 4: Fix existing callers and run the full app suite**

`AppModelLocation.defaultURL()` gained a parameter and `resolve` now throws. Update every call site to pass `SupportedModelSource.repoID` for now; Task 11 replaces that with the selected model.

Run: `swift build && swift test --filter TurboFieldfareAppCoreTests`
Expected: PASS. Pre-existing `AppModelLocationTests` cases that call the old signature must be updated to the new one, not deleted.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift \
        Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift
git commit -m "feat: give each model its own slug-keyed install directory"
```

---

### Task 8: Validate manifests against a list of architectures

Runtime change. `ManifestReader.validateArch` compares against the single constant `ArchConfig.gemma4_26B_A4B`; it becomes a list with one entry, and the rejection message names every architecture tried. This is the seam a future architecture plugs into.

**Files:**
- Modify: `Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift:131-137,174-184`
- Modify: `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift`
- Test: `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestArchListTests.swift`

**Interfaces:**
- Consumes: existing `ArchConfig`, `ManifestArch`, `ModelError`
- Produces: `ArchConfig.supported: [ArchConfig]` (one entry, `gemma4_26B_A4B`); `ManifestReader.matchArch(_ manifestArch: ManifestArch, against candidates: [ArchConfig]) -> ArchConfig?`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestArchListTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfare

@Suite struct ManifestArchListTests {
    @Test func supportedListContainsExactlyGemma4() {
        #expect(ArchConfig.supported.count == 1)
        #expect(ArchConfig.supported.first?.numLayers == ArchConfig.gemma4_26B_A4B.numLayers)
        #expect(ArchConfig.supported.first?.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize)
    }

    @Test func matchesTheGemmaArchitecture() {
        let manifestArch = ManifestArch(from: ArchConfig.gemma4_26B_A4B)
        #expect(ManifestReader.matchArch(manifestArch, against: ArchConfig.supported) != nil)
    }

    @Test func returnsNilForAnUnknownArchitecture() {
        var manifestArch = ManifestArch(from: ArchConfig.gemma4_26B_A4B)
        manifestArch.hiddenSize = 2048
        #expect(ManifestReader.matchArch(manifestArch, against: ArchConfig.supported) == nil)
    }
}
```

Note: if `ManifestArch` has no `init(from: ArchConfig)` and no mutable fields, add that initialiser to `ModelTypes.swift` in Step 3 and make the stored properties `var`. The initialiser copies every field `validateArch` compares.

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ManifestArchListTests`
Expected: FAIL — `ArchConfig.supported` does not exist.

- [ ] **Step 3: Write minimal implementation**

In `ModelTypes.swift`, after the `gemma4_26B_A4B` declaration, add:

```swift
    /// Every architecture this build can execute. An entry here without the
    /// matching Metal kernels would let a model install and then trap at the
    /// first attention dispatch, so entries are added only alongside kernels.
    public static let supported: [ArchConfig] = [gemma4_26B_A4B]
```

In `ManifestReader.swift`, replace the single-config validation with list matching. Change the `validate` body at line 131 from `try validateArch(m.arch, expected: expected)` to:

```swift
        guard let matched = Self.matchArch(m.arch, against: ArchConfig.supported) else {
            try validateArch(m.arch, expected: expected)
            return
        }
        _ = matched
```

and add:

```swift
    /// Returns the first supported architecture that matches the manifest, or
    /// nil. Callers that need a field-level reason fall through to
    /// `validateArch`, which throws `archMismatch` naming the first difference.
    static func matchArch(_ manifestArch: ManifestArch,
                          against candidates: [ArchConfig]) -> ArchConfig? {
        candidates.first { candidate in
            (try? validateArch(manifestArch, expected: candidate)) != nil
        }
    }
```

`validateArch` stays exactly as it is — it is now the reason-reporting path rather than the only path.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ManifestArchListTests && swift test --filter TurboFieldfareTestsCore`
Expected: PASS, including all pre-existing manifest tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift \
        Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift \
        Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestArchListTests.swift
git commit -m "feat: validate manifests against a list of supported architectures"
```

---

### Task 9: Memory budget and computed context options

`AppContextLengthOption` is a fixed enum with hardcoded labels (`"8K, +85 MB"`) and `fp16KVBytes` hardcoding `ArchConfig.gemma4_26B_A4B` at `:18`. It becomes a computed list driven by the loaded architecture and a RAM-aware ceiling.

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/AppMemoryBudget.swift`
- Modify: `Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift`
- Test: `Tests/TurboFieldfareApp/Core/Configuration/AppMemoryBudgetTests.swift`

**Interfaces:**
- Consumes: `ArchConfig` from `TurboFieldfare`
- Produces: `AppMemoryBudget.ceilingBytes(installedRAMBytes:) -> UInt64`; `AppContextLengthOption.availableOptions(architecture:residentWeightBytes:installedRAMBytes:) -> [AppContextOption]` where `AppContextOption` has `tokens: Int`, `kvBytes: UInt64`, `isEnabled: Bool`, `disabledReason: String?`, `label: String`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Configuration/AppMemoryBudgetTests.swift`:

```swift
import Foundation
import Testing
import TurboFieldfare

@testable import TurboFieldfareAppCore

@Suite struct AppMemoryBudgetTests {
    private let gigabyte: UInt64 = 1_073_741_824

    @Test func eightGigabyteMachineGetsTwoGigabyteCeiling() {
        #expect(AppMemoryBudget.ceilingBytes(installedRAMBytes: 8 * gigabyte) == 2 * gigabyte)
    }

    @Test func sixteenGigabyteMachineGetsTheFourGigabyteCap() {
        #expect(AppMemoryBudget.ceilingBytes(installedRAMBytes: 16 * gigabyte) == 4 * gigabyte)
    }

    @Test func largeMachineIsStillCappedAtFourGigabytes() {
        #expect(AppMemoryBudget.ceilingBytes(installedRAMBytes: 128 * gigabyte) == 4 * gigabyte)
    }

    @Test func tinyMachineGetsTheFloorNotAProportionalShare() {
        let ceiling = AppMemoryBudget.ceilingBytes(installedRAMBytes: 4 * gigabyte)
        #expect(ceiling == UInt64(1.5 * Double(gigabyte)))
    }

    @Test func optionsBelowTheCeilingAreEnabled() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: gigabyte,
            installedRAMBytes: 16 * gigabyte)
        let fourK = options.first { $0.tokens == 4_096 }
        #expect(fourK?.isEnabled == true)
    }

    @Test func optionsAboveTheCeilingAreDisabledNotHidden() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: 3 * gigabyte,
            installedRAMBytes: 8 * gigabyte)
        let largest = options.last
        #expect(largest?.isEnabled == false)
        #expect(largest?.disabledReason != nil)
        #expect(options.contains { $0.tokens == 65_536 })
    }

    @Test func kvBytesGrowMonotonicallyWithContext() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: gigabyte,
            installedRAMBytes: 32 * gigabyte)
        let sorted = options.sorted { $0.tokens < $1.tokens }
        for (smaller, larger) in zip(sorted, sorted.dropFirst()) {
            #expect(larger.kvBytes > smaller.kvBytes)
        }
    }

    @Test func offersContextBeyondSixtyFourKOnLargeMachines() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: gigabyte,
            installedRAMBytes: 32 * gigabyte)
        #expect(options.contains { $0.tokens == 131_072 && $0.isEnabled })
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter AppMemoryBudgetTests`
Expected: FAIL — cannot find `AppMemoryBudget` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/AppMemoryBudget.swift`:

```swift
import Foundation

/// How much RAM the app is willing to hold in resident weights plus KV cache.
///
/// The cap is not just about fitting: resident state competes with the page
/// cache that keeps streamed experts warm, so spending the machine's whole
/// budget on KV makes decode slower even when it fits.
public enum AppMemoryBudget {
    public static let absoluteCapBytes: UInt64 = 4 * 1_073_741_824
    public static let floorBytes: UInt64 = UInt64(1.5 * 1_073_741_824)

    public static func ceilingBytes(installedRAMBytes: UInt64) -> UInt64 {
        let proportional = installedRAMBytes / 4
        return min(absoluteCapBytes, max(floorBytes, proportional))
    }

    public static var installedRAMBytes: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }
}
```

Replace `Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift` with:

```swift
import Foundation
import TurboFieldfare

/// One selectable context length with its measured cost.
public struct AppContextOption: Equatable, Sendable, Identifiable {
    public var id: Int { tokens }

    public let tokens: Int
    public let kvBytes: UInt64
    public let isEnabled: Bool
    public let disabledReason: String?

    public var shortLabel: String { "\(tokens / 1_024)K" }

    public var label: String {
        let megabytes = Double(kvBytes) / (1_024 * 1_024)
        let cost = megabytes >= 1_024
            ? String(format: "%.2f GB", megabytes / 1_024)
            : String(format: "%.0f MB", megabytes)
        return "\(shortLabel), \(cost)"
    }
}

public enum AppContextLengthOption {
    /// Candidate lengths offered to the user. Entries beyond 64K exist for
    /// machines whose budget can hold them; they render disabled elsewhere.
    public static let candidateTokens = [4_096, 8_192, 16_384, 32_768,
                                         65_536, 131_072, 262_144]

    public static let defaultTokens = 4_096

    public static func availableOptions(architecture: ArchConfig,
                                        residentWeightBytes: UInt64,
                                        installedRAMBytes: UInt64) -> [AppContextOption] {
        let ceiling = AppMemoryBudget.ceilingBytes(installedRAMBytes: installedRAMBytes)
        return candidateTokens.map { tokens in
            let kvBytes = fp16KVBytes(architecture: architecture, tokens: tokens)
            let total = residentWeightBytes + kvBytes
            let fits = total <= ceiling
            return AppContextOption(
                tokens: tokens,
                kvBytes: kvBytes,
                isEnabled: fits,
                disabledReason: fits
                    ? nil
                    : "Needs \(formatted(total)) of \(formatted(ceiling)) available.")
        }
    }

    /// Sliding-window layers stop growing once the window plus one prefill
    /// chunk is covered; full-attention layers grow with the whole context.
    public static func fp16KVBytes(architecture: ArchConfig, tokens: Int) -> UInt64 {
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 0 : 1)
        }
        let slidingLayers = architecture.numLayers - fullLayers
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + PrefillRuntimeConfig.defaultChunked.chunkTokens)
        let slidingBytesPerRow = architecture.numKVHeads
            * architecture.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = architecture.numFullKVHeads
            * architecture.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    private static func formatted(_ bytes: UInt64) -> String {
        let gigabytes = Double(bytes) / (1_024 * 1_024 * 1_024)
        return String(format: "%.1f GB", gigabytes)
    }
}
```

- [ ] **Step 4: Fix callers and run tests**

`MacAppSettings.contextTokens` defaults to `AppContextLengthOption.fourK.tokens`; change to `AppContextLengthOption.defaultTokens`. Any UI reading `AppContextLengthOption.allCases` now reads `availableOptions(...)`.

Run: `swift build && swift test --filter AppMemoryBudgetTests && swift test --filter TurboFieldfareAppCoreTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/AppMemoryBudget.swift \
        Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift \
        Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift \
        Tests/TurboFieldfareApp/Core/Configuration/AppMemoryBudgetTests.swift
git commit -m "feat: compute context options from architecture and RAM budget"
```

---

### Task 10: Model-tagged turns and the global conversation store

`ConversationFileStore.fileURL(forModelDirectory:)` writes beside the model directory, so with Task 7's layout every model would keep its own chat list and deleting a model would delete its history. Conversations move to one global file and turns record which model produced them.

**Files:**
- Modify: `Sources/TurboFieldfareApp/Core/State/ConversationModels.swift`
- Modify: `Sources/TurboFieldfareApp/Core/State/ConversationFileStore.swift`
- Test: `Tests/TurboFieldfareApp/Core/State/ConversationMigrationTests.swift`

**Interfaces:**
- Consumes: `ModelCatalogEntry` (for `repoID` as `modelID`), `AppModelLocation.supportDirectory(applicationSupportURL:)` from Task 7
- Produces: `ChatTurn.modelID: String?`; `ConversationFileStore.globalFileURL(inSupportDirectory:) -> URL`, `.loadGlobal(inSupportDirectory:fileManager:) -> ConversationStoreFile`, `.saveGlobal(_:inSupportDirectory:fileManager:) throws`, `.migrate(fromModelDirectories:modelIDsByDirectory:inSupportDirectory:fileManager:) throws -> Int` returning the number of conversations imported, and `.migratedFileName = "conversations.migrated.json"`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/State/ConversationMigrationTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ConversationMigrationTests {
    private func makeTree(_ name: String) throws -> (support: URL, modelDirectory: URL) {
        let support = FileManager.default.temporaryDirectory
            .appendingPathComponent("conv-migrate-\(name)-\(UUID().uuidString)",
                                    isDirectory: true)
        let modelDirectory = support
            .appendingPathComponent("models/owner--model/model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory,
                                                withIntermediateDirectories: true)
        return (support, modelDirectory)
    }

    private func seedLegacyStore(at modelDirectory: URL, title: String) throws {
        let conversation = Conversation(
            title: title,
            turns: [
                ChatTurn(role: .user, content: "Hello"),
                ChatTurn(role: .assistant, content: "Hi"),
            ])
        try ConversationFileStore.save(
            ConversationStoreFile(conversations: [conversation]),
            forModelDirectory: modelDirectory)
    }

    @Test func chatTurnCarriesAnOptionalModelID() {
        let turn = ChatTurn(role: .assistant, content: "Hi", modelID: "owner/model")
        #expect(turn.modelID == "owner/model")
        #expect(ChatTurn(role: .user, content: "Hello").modelID == nil)
    }

    @Test func globalStoreRoundTripsThroughDisk() throws {
        let (support, _) = try makeTree("round-trip")
        defer { try? FileManager.default.removeItem(at: support) }

        let store = ConversationStoreFile(conversations: [
            Conversation(title: "Kept", turns: [
                ChatTurn(role: .user, content: "Hello", modelID: "owner/model"),
            ]),
        ])
        try ConversationFileStore.saveGlobal(store, inSupportDirectory: support)
        #expect(ConversationFileStore.loadGlobal(inSupportDirectory: support) == store)
    }

    @Test func migrationImportsLegacyStoreAndTagsTurns() throws {
        let (support, modelDirectory) = try makeTree("import")
        defer { try? FileManager.default.removeItem(at: support) }
        try seedLegacyStore(at: modelDirectory, title: "Legacy")

        let imported = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)
        #expect(imported == 1)

        let global = ConversationFileStore.loadGlobal(inSupportDirectory: support)
        #expect(global.conversations.count == 1)
        #expect(global.conversations[0].title == "Legacy")
        #expect(global.conversations[0].turns.allSatisfy { $0.modelID == "owner/model" })
    }

    @Test func migrationIsIdempotent() throws {
        let (support, modelDirectory) = try makeTree("idempotent")
        defer { try? FileManager.default.removeItem(at: support) }
        try seedLegacyStore(at: modelDirectory, title: "Legacy")

        let first = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)
        let second = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)

        #expect(first == 1)
        #expect(second == 0)
        #expect(ConversationFileStore.loadGlobal(inSupportDirectory: support)
            .conversations.count == 1)
    }

    @Test func migrationRenamesTheLegacyFile() throws {
        let (support, modelDirectory) = try makeTree("rename")
        defer { try? FileManager.default.removeItem(at: support) }
        try seedLegacyStore(at: modelDirectory, title: "Legacy")

        let legacyURL = ConversationFileStore.fileURL(forModelDirectory: modelDirectory)
        _ = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)

        #expect(FileManager.default.fileExists(atPath: legacyURL.path) == false)
        let renamed = legacyURL.deletingLastPathComponent()
            .appendingPathComponent(ConversationFileStore.migratedFileName)
        #expect(FileManager.default.fileExists(atPath: renamed.path))
    }

    @Test func migrationMergesMultipleModels() throws {
        let (support, firstModel) = try makeTree("merge")
        defer { try? FileManager.default.removeItem(at: support) }
        let secondModel = support
            .appendingPathComponent("models/owner--other/model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: secondModel,
                                                withIntermediateDirectories: true)
        try seedLegacyStore(at: firstModel, title: "From first")
        try seedLegacyStore(at: secondModel, title: "From second")

        let imported = try ConversationFileStore.migrate(
            fromModelDirectories: [firstModel, secondModel],
            modelIDsByDirectory: [firstModel: "owner/model", secondModel: "owner/other"],
            inSupportDirectory: support)

        #expect(imported == 2)
        let titles = Set(ConversationFileStore.loadGlobal(inSupportDirectory: support)
            .conversations.map(\.title))
        #expect(titles == ["From first", "From second"])
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ConversationMigrationTests`
Expected: FAIL — `ChatTurn` has no `modelID` parameter.

- [ ] **Step 3: Write minimal implementation**

In `ConversationModels.swift`, add the field to `ChatTurn`. Keep it optional and decode-tolerant so existing stores load unchanged:

```swift
public struct ChatTurn: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var role: ChatRole
    public var content: String
    public var timestamp: Date
    /// Repository ID of the model that produced this turn. Nil for turns
    /// written before model tagging, and for user turns recorded without a
    /// loaded model.
    public var modelID: String?

    public init(id: UUID = UUID(),
                role: ChatRole,
                content: String,
                timestamp: Date = Date(),
                modelID: String? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.modelID = modelID
    }
}
```

Preserve the existing `decodeMessage` computed property exactly as it is — history keeps flowing to the decode service as structured `DecodeChatMessage` values, so the loaded model re-applies its own chat template.

In `ConversationFileStore.swift`, keep `fileURL(forModelDirectory:)`, `load(forModelDirectory:)` and `save(_:forModelDirectory:)` — migration reads through them — and add:

```swift
    public static let migratedFileName = "conversations.migrated.json"

    public static func globalFileURL(inSupportDirectory supportDirectory: URL) -> URL {
        supportDirectory.standardizedFileURL
            .appendingPathComponent(ConversationStoreFile.fileName, isDirectory: false)
    }

    public static func loadGlobal(inSupportDirectory supportDirectory: URL,
                                  fileManager: FileManager = .default) -> ConversationStoreFile {
        let url = globalFileURL(inSupportDirectory: supportDirectory)
        guard fileManager.fileExists(atPath: url.path) else {
            return ConversationStoreFile()
        }
        do {
            let store = try JSONDecoder().decode(
                ConversationStoreFile.self,
                from: try Data(contentsOf: url))
            guard store.isValid() else { throw InvalidStore() }
            return store
        } catch {
            try? fileManager.removeItem(at: url)
            return ConversationStoreFile()
        }
    }

    public static func saveGlobal(_ store: ConversationStoreFile,
                                  inSupportDirectory supportDirectory: URL,
                                  fileManager: FileManager = .default) throws {
        guard store.isValid() else { throw InvalidStore() }
        let url = globalFileURL(inSupportDirectory: supportDirectory)
        try fileManager.createDirectory(at: supportDirectory,
                                        withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(store)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }

    /// Folds per-model stores into the global one, tagging every imported turn.
    ///
    /// Renaming the source file is what makes this idempotent: a second run
    /// finds nothing to read. Renaming rather than deleting keeps the original
    /// recoverable if the merge is ever wrong.
    @discardableResult
    public static func migrate(fromModelDirectories directories: [URL],
                               modelIDsByDirectory: [URL: String],
                               inSupportDirectory supportDirectory: URL,
                               fileManager: FileManager = .default) throws -> Int {
        var global = loadGlobal(inSupportDirectory: supportDirectory,
                                fileManager: fileManager)
        var imported = 0
        for directory in directories {
            let legacyURL = fileURL(forModelDirectory: directory)
            guard fileManager.fileExists(atPath: legacyURL.path) else { continue }
            let legacy = load(forModelDirectory: directory, fileManager: fileManager)
            let modelID = modelIDsByDirectory[directory]
            for var conversation in legacy.conversations where !conversation.isEmpty {
                for index in conversation.turns.indices where conversation.turns[index].modelID == nil {
                    conversation.turns[index].modelID = modelID
                }
                global.conversations.append(conversation)
                imported += 1
            }
            let renamed = legacyURL.deletingLastPathComponent()
                .appendingPathComponent(migratedFileName, isDirectory: false)
            try? fileManager.removeItem(at: renamed)
            try fileManager.moveItem(at: legacyURL, to: renamed)
        }
        if imported > 0 {
            try saveGlobal(global, inSupportDirectory: supportDirectory,
                           fileManager: fileManager)
        }
        return imported
    }
```

- [ ] **Step 4: Run tests**

Run: `swift test --filter ConversationMigrationTests && swift test --filter ConversationStoreTests`
Expected: PASS. Existing `ConversationStoreTests` must still pass — the per-model API is unchanged.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/State/ConversationModels.swift \
        Sources/TurboFieldfareApp/Core/State/ConversationFileStore.swift \
        Tests/TurboFieldfareApp/Core/State/ConversationMigrationTests.swift
git commit -m "feat: tag turns with model and move conversations to a global store"
```

---

### Task 11: Per-model install state and serialized queue

`AppModel` holds one `AppModelInstallState`. It becomes a map keyed by repository ID, and installs run one at a time because `requiredFreeBytes` assumes exclusive use of the reserve.

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Installation/ModelInstallQueue.swift`
- Modify: `Sources/TurboFieldfareApp/Core/State/AppModel.swift`
- Test: `Tests/TurboFieldfareApp/Core/Installation/ModelInstallQueueTests.swift`

**Interfaces:**
- Consumes: `AppModelInstallState`, `ModelCatalogEntry`
- Produces: `ModelInstallQueue` actor with `enqueue(repoID:) -> Bool`, `activeRepoID: String?`, `pendingRepoIDs: [String]`, `finishActive()`, `cancel(repoID:)`; `ModelInstallStates` struct with `state(for:) -> AppModelInstallState`, `setState(_:for:)`, `installedRepoIDs: Set<String>`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Installation/ModelInstallQueueTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelInstallQueueTests {
    @Test func firstEnqueuedRepositoryBecomesActive() async {
        let queue = ModelInstallQueue()
        let started = await queue.enqueue(repoID: "owner/one")
        #expect(started)
        #expect(await queue.activeRepoID == "owner/one")
    }

    @Test func secondEnqueuedRepositoryWaits() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        let started = await queue.enqueue(repoID: "owner/two")
        #expect(started == false)
        #expect(await queue.activeRepoID == "owner/one")
        #expect(await queue.pendingRepoIDs == ["owner/two"])
    }

    @Test func finishingActivePromotesTheNextRepository() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        let promoted = await queue.finishActive()
        #expect(promoted == "owner/two")
        #expect(await queue.activeRepoID == "owner/two")
        #expect(await queue.pendingRepoIDs.isEmpty)
    }

    @Test func enqueueingTheSameRepositoryTwiceIsIgnored() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        _ = await queue.enqueue(repoID: "owner/two")
        #expect(await queue.pendingRepoIDs == ["owner/two"])
    }

    @Test func cancellingAPendingRepositoryRemovesItWithoutTouchingActive() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        await queue.cancel(repoID: "owner/two")
        #expect(await queue.activeRepoID == "owner/one")
        #expect(await queue.pendingRepoIDs.isEmpty)
    }

    @Test func cancellingTheActiveRepositoryPromotesTheNext() async {
        let queue = ModelInstallQueue()
        _ = await queue.enqueue(repoID: "owner/one")
        _ = await queue.enqueue(repoID: "owner/two")
        await queue.cancel(repoID: "owner/one")
        #expect(await queue.activeRepoID == "owner/two")
    }

    @Test func statesAreTrackedPerRepository() {
        var states = ModelInstallStates()
        #expect(states.state(for: "owner/one") == .idle)
        states.setState(.installed(modelDirectory: URL(fileURLWithPath: "/tmp/a")),
                        for: "owner/one")
        states.setState(.checking, for: "owner/two")
        #expect(states.state(for: "owner/two") == .checking)
        #expect(states.installedRepoIDs == ["owner/one"])
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelInstallQueueTests`
Expected: FAIL — cannot find `ModelInstallQueue` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Installation/ModelInstallQueue.swift`:

```swift
import Foundation

/// Serialises installs.
///
/// Two concurrent repacks would contend for the same SSD bandwidth and, worse,
/// each would size its free-space reserve as if it were alone — so both could
/// pass the readiness check and then run the volume dry.
public actor ModelInstallQueue {
    private(set) public var activeRepoID: String?
    private var pending: [String] = []

    public init() {}

    public var pendingRepoIDs: [String] { pending }

    /// Returns true when the caller may start immediately.
    @discardableResult
    public func enqueue(repoID: String) -> Bool {
        if activeRepoID == repoID || pending.contains(repoID) { return false }
        guard activeRepoID != nil else {
            activeRepoID = repoID
            return true
        }
        pending.append(repoID)
        return false
    }

    /// Clears the active slot and promotes the next repository, if any.
    @discardableResult
    public func finishActive() -> String? {
        activeRepoID = pending.isEmpty ? nil : pending.removeFirst()
        return activeRepoID
    }

    public func cancel(repoID: String) {
        if activeRepoID == repoID {
            _ = finishActive()
            return
        }
        pending.removeAll { $0 == repoID }
    }
}

/// Install state for every known model, so the picker can render each row's
/// progress without the single-model assumption baked into `AppModel` today.
public struct ModelInstallStates: Equatable, Sendable {
    private var states: [String: AppModelInstallState] = [:]

    public init() {}

    public func state(for repoID: String) -> AppModelInstallState {
        states[repoID] ?? .idle
    }

    public mutating func setState(_ state: AppModelInstallState, for repoID: String) {
        states[repoID] = state
    }

    public var installedRepoIDs: Set<String> {
        Set(states.compactMap { key, value in
            if case .installed = value { return key }
            return nil
        })
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ModelInstallQueueTests`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Installation/ModelInstallQueue.swift \
        Tests/TurboFieldfareApp/Core/Installation/ModelInstallQueueTests.swift
git commit -m "feat: track install state per model and serialise installs"
```

---

### Task 12: Per-model install descriptors and installer client

Nothing yet turns a catalog entry into an actual install. `AppModelInstallDescriptor.default` is a single hardcoded value and `AppModelInstallerClient.installDefaultModel(outputDirectory:)` names the assumption in its signature. Both become per-entry.

**Files:**
- Modify: `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallDescriptor.swift:36-44`
- Modify: `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallerClient.swift`
- Modify: `Sources/TurboFieldfareApp/Core/Installation/RepackModelInstallerClient.swift:36`
- Test: `Tests/TurboFieldfareApp/Core/Installation/ModelInstallDescriptorTests.swift`

**Interfaces:**
- Consumes: `ModelCatalogEntry` from Task 2, `ModelTrustPolicy.requiresKnownSource(for:)` from Task 5
- Produces: `AppModelInstallDescriptor.init(entry: ModelCatalogEntry)`; `AppModelInstallerClient.install(entry:outputDirectory:) -> AsyncThrowingStream<AppModelInstallEvent, Error>` and `.checkInstallRequirement(entry:outputDirectory:) throws -> AppModelInstallRequirement`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Installation/ModelInstallDescriptorTests.swift`:

```swift
import Foundation
import Testing
import TurboFieldfareRepackCore

@testable import TurboFieldfareAppCore

@Suite struct ModelInstallDescriptorTests {
    private func makeEntry(tier: ModelTrustTier) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Custom Gemma",
            repoID: "owner/gemma-4-26b-a4b-uncensored",
            revision: "abc123",
            trustTier: tier,
            recordedIndexSHA256: "deadbeef",
            approximateDownloadBytes: 2_000,
            installedBytes: 1_500,
            reserveBytes: 500)
    }

    @Test func copiesEveryFieldFromTheCatalogEntry() {
        let descriptor = AppModelInstallDescriptor(entry: makeEntry(tier: .custom))
        #expect(descriptor.displayName == "Custom Gemma")
        #expect(descriptor.repoID == "owner/gemma-4-26b-a4b-uncensored")
        #expect(descriptor.revision == "abc123")
        #expect(descriptor.sourceIndexSHA256 == "deadbeef")
        #expect(descriptor.approximateDownloadBytes == 2_000)
        #expect(descriptor.installedBytes == 1_500)
        #expect(descriptor.reserveBytes == 500)
    }

    @Test func requiredFreeBytesIncludesStagingAndReserve() {
        let descriptor = AppModelInstallDescriptor(entry: makeEntry(tier: .custom))
        let expected = UInt64(1_500) + UInt64(RemoteChunkPolicy.defaultBytes) + UInt64(500)
        #expect(descriptor.requiredFreeBytes == expected)
    }

    @Test func curatedEntriesRequireAKnownSource() {
        #expect(ModelTrustPolicy.requiresKnownSource(for: makeEntry(tier: .curated)))
    }

    @Test func customEntriesDoNotRequireAKnownSource() {
        #expect(ModelTrustPolicy.requiresKnownSource(for: makeEntry(tier: .custom)) == false)
    }

    @Test func theCuratedGemmaEntryMatchesTheLegacyDefaultDescriptor() throws {
        let curated = try #require(ModelCatalog.curated.first)
        let descriptor = AppModelInstallDescriptor(entry: curated)
        #expect(descriptor.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(descriptor.installedBytes == 14_291_921_884)
        #expect(descriptor.approximateDownloadBytes == 14_620_479_420)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelInstallDescriptorTests`
Expected: FAIL — `AppModelInstallDescriptor` has no `init(entry:)`.

- [ ] **Step 3: Write minimal implementation**

In `AppModelInstallDescriptor.swift`, keep the existing memberwise initialiser and `requiredFreeBytes`, and add:

```swift
    public init(entry: ModelCatalogEntry) {
        self.init(
            displayName: entry.displayName,
            repoID: entry.repoID,
            revision: entry.revision,
            // Empty for a custom entry on its first install; the repacker is
            // told not to require a known source in that case, and the observed
            // value is recorded afterwards.
            sourceIndexSHA256: entry.recordedIndexSHA256 ?? "",
            approximateDownloadBytes: entry.approximateDownloadBytes,
            installedBytes: entry.installedBytes,
            rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
            reserveBytes: entry.reserveBytes)
    }
```

Leave `AppModelInstallDescriptor.default` in place; it is the fallback used before a selection exists.

In `AppModelInstallerClient.swift`, widen the protocol:

```swift
import Foundation

public protocol AppModelInstallerClient: Sendable {
    var descriptor: AppModelInstallDescriptor { get }
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement
    func checkInstallRequirement(entry: ModelCatalogEntry,
                                 outputDirectory: URL) throws -> AppModelInstallRequirement
    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func install(entry: ModelCatalogEntry,
                 outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func discardPartialInstall(outputDirectory: URL) async throws
    func cancel()
}
```

In `RepackModelInstallerClient.swift`, implement `install(entry:outputDirectory:)` by building `RemoteStreamingRepackOptions` from the entry rather than from `SupportedModelSource`, and replace the hardcoded `requireKnownSource: true` at line 36 with `ModelTrustPolicy.requiresKnownSource(for: entry)`. Keep `installDefaultModel` delegating to `install(entry:)` with `ModelCatalog.curated[0]` so existing call sites and tests keep working.

- [ ] **Step 4: Run tests**

Run: `swift build && swift test --filter ModelInstallDescriptorTests && swift test --filter RepackModelInstallerClientTests`
Expected: PASS. Any test double conforming to `AppModelInstallerClient` needs the two new methods; forward them to the existing ones.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Installation/AppModelInstallDescriptor.swift \
        Sources/TurboFieldfareApp/Core/Installation/AppModelInstallerClient.swift \
        Sources/TurboFieldfareApp/Core/Installation/RepackModelInstallerClient.swift \
        Tests/TurboFieldfareApp/Core/Installation/ModelInstallDescriptorTests.swift
git commit -m "feat: install any catalog entry, not just the pinned model"
```

---

### Task 13: Model switching in AppModel

Wires the catalog into `AppModel`: selection, switch, guards, and last-good persistence. The decode protocol already carries `load(DecodeLoadRequest)` with `modelPath` and `unload`, so no process restart is involved.

**Files:**
- Create: `Sources/TurboFieldfareApp/Core/Catalog/ModelSwitchGuard.swift`
- Modify: `Sources/TurboFieldfareApp/Core/State/AppModel.swift:397-521`
- Modify: `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift`
- Test: `Tests/TurboFieldfareApp/Core/Catalog/ModelSwitchGuardTests.swift`

**Interfaces:**
- Consumes: `AppModelLoadState`, `AppGenerationPhase`, `ModelCatalogEntry`, `ModelInstallStates` from Task 11
- Produces: `ModelSwitchGuard.evaluate(target:currentRepoID:loadState:isGenerating:installStates:) -> ModelSwitchVerdict` with cases `.allowed`, `.alreadyLoaded`, `.blockedByGeneration`, `.notInstalled`, `.busy`; `MacAppSettings.selectedRepoID: String?` and `.lastGoodRepoID: String?`

- [ ] **Step 1: Write the failing test**

Create `Tests/TurboFieldfareApp/Core/Catalog/ModelSwitchGuardTests.swift`:

```swift
import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelSwitchGuardTests {
    private let installedDirectory = URL(fileURLWithPath: "/tmp/model.gturbo")

    private func makeEntry(repoID: String) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: .custom,
            recordedIndexSHA256: "abc",
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    private func installedStates(_ repoIDs: [String]) -> ModelInstallStates {
        var states = ModelInstallStates()
        for repoID in repoIDs {
            states.setState(.installed(modelDirectory: installedDirectory), for: repoID)
        }
        return states
    }

    @Test func allowsSwitchToAnInstalledModel() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: false,
            installStates: installedStates(["owner/one", "owner/two"]))
        #expect(verdict == .allowed)
    }

    @Test func reportsAlreadyLoaded() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/one"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: false,
            installStates: installedStates(["owner/one"]))
        #expect(verdict == .alreadyLoaded)
    }

    @Test func blocksSwitchWhileGenerating() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: true,
            installStates: installedStates(["owner/one", "owner/two"]))
        #expect(verdict == .blockedByGeneration)
    }

    @Test func blocksSwitchToAnUninstalledModel() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: "owner/one",
            loadState: .ready(modelDirectory: installedDirectory, loadSeconds: 1),
            isGenerating: false,
            installStates: installedStates(["owner/one"]))
        #expect(verdict == .notInstalled)
    }

    @Test func blocksSwitchWhileAlreadyLoadingOrUnloading() {
        for busyState in [AppModelLoadState.loading(.tokenizer),
                          .unloading,
                          .cancelling] {
            let verdict = ModelSwitchGuard.evaluate(
                target: makeEntry(repoID: "owner/two"),
                currentRepoID: "owner/one",
                loadState: busyState,
                isGenerating: false,
                installStates: installedStates(["owner/one", "owner/two"]))
            #expect(verdict == .busy)
        }
    }

    @Test func allowsSwitchWhenNothingIsLoaded() {
        let verdict = ModelSwitchGuard.evaluate(
            target: makeEntry(repoID: "owner/two"),
            currentRepoID: nil,
            loadState: .notLoaded,
            isGenerating: false,
            installStates: installedStates(["owner/two"]))
        #expect(verdict == .allowed)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ModelSwitchGuardTests`
Expected: FAIL — cannot find `ModelSwitchGuard` in scope.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/TurboFieldfareApp/Core/Catalog/ModelSwitchGuard.swift`:

```swift
import Foundation

public enum ModelSwitchVerdict: Equatable, Sendable {
    case allowed
    case alreadyLoaded
    case blockedByGeneration
    case notInstalled
    case busy
}

/// Decides whether a switch may proceed.
///
/// Pure so the rules are testable without a runner: `AppModel` performs the
/// unload/load, this only says whether it may.
public enum ModelSwitchGuard {
    public static func evaluate(target: ModelCatalogEntry,
                                currentRepoID: String?,
                                loadState: AppModelLoadState,
                                isGenerating: Bool,
                                installStates: ModelInstallStates) -> ModelSwitchVerdict {
        if isGenerating { return .blockedByGeneration }
        if loadState.isLoading { return .busy }
        if target.repoID == currentRepoID, loadState.isReady { return .alreadyLoaded }
        guard installStates.installedRepoIDs.contains(target.repoID) else {
            return .notInstalled
        }
        return .allowed
    }
}
```

`AppModelLoadState` already exposes `isLoading` — which is true for `.loading`, `.cancelling`, *and* `.unloading` despite the name — plus `isReady`. Use them rather than re-deriving.

In `MacAppSettings`, add:

```swift
    /// Repository ID the user last selected in the picker.
    var selectedRepoID: String?
    /// Repository ID of the last model that loaded successfully. Used to
    /// recover when a switch fails, and to avoid booting into a broken
    /// selection after relaunch.
    var lastGoodRepoID: String?
```

In `AppModel`, add the switch flow:

```swift
    /// Switches the loaded model. Callers must have a verdict of `.allowed`.
    ///
    /// The prompt cache is dropped explicitly rather than relying on
    /// `AppPromptCacheDomain` failing to match: the domain check would keep a
    /// dead entry alive in memory for the rest of the session.
    public func switchModel(to entry: ModelCatalogEntry) {
        let verdict = ModelSwitchGuard.evaluate(
            target: entry,
            currentRepoID: selectedRepoID,
            loadState: loadState,
            isGenerating: isGenerating,
            installStates: installStates)
        guard verdict == .allowed else {
            presentSwitchVerdict(verdict, for: entry)
            return
        }
        let previousRepoID = selectedRepoID
        promptCache.invalidate()
        unloadModel()
        selectedRepoID = entry.repoID
        modelPathText = (try? AppModelLocation.defaultURL(forRepoID: entry.repoID).path) ?? ""
        loadModel()
        onLoadFailure = { [weak self] in
            guard let self, let previousRepoID else { return }
            self.selectedRepoID = previousRepoID
            self.modelPathText =
                (try? AppModelLocation.defaultURL(forRepoID: previousRepoID).path) ?? ""
        }
    }
```

Add the observable state and the install/delete entry points the picker calls. These are the exact names Task 14's view uses:

```swift
    /// Curated entries merged with whatever the user has added.
    public private(set) var catalog: ModelCatalog
    /// Install progress for every known model.
    public private(set) var installStates: ModelInstallStates
    /// Repository ID of the selected model, loaded or not.
    public private(set) var selectedRepoID: String?

    public func startInstall(for entry: ModelCatalogEntry) {
        Task {
            guard await installQueue.enqueue(repoID: entry.repoID) else {
                installStates.setState(.checking, for: entry.repoID)
                return
            }
            await runInstall(for: entry)
            if let next = await installQueue.finishActive(),
               let nextEntry = catalog.entry(forRepoID: next) {
                await runInstall(for: nextEntry)
            }
        }
    }

    public func cancelInstall(for entry: ModelCatalogEntry) {
        installer.cancel()
        Task { await installQueue.cancel(repoID: entry.repoID) }
        installStates.setState(.cancelled, for: entry.repoID)
    }

    /// Removes the installed weights but keeps the catalog entry, so the model
    /// can be re-downloaded without retyping the repository.
    public func deleteInstall(for entry: ModelCatalogEntry) {
        guard !(selectedRepoID == entry.repoID && loadState.isReady) else {
            errorMessage = "Unload \(entry.displayName) before deleting it."
            return
        }
        guard let directory = try? AppModelLocation.defaultURL(forRepoID: entry.repoID) else { return }
        try? FileManager.default.removeItem(at: directory.deletingLastPathComponent())
        installStates.setState(.idle, for: entry.repoID)
    }
```

`runInstall(for:)` drives `installer.install(entry:outputDirectory:)` from Task 12, mapping each `AppModelInstallEvent` onto `installStates.setState(_:for:)` exactly as the current single-model path maps onto `installState`.

On a successful load, set `settings.lastGoodRepoID = entry.repoID` and persist. On launch, prefer `settings.selectedRepoID`, falling back to `settings.lastGoodRepoID`, then to `SupportedModelSource.repoID`.

Replace the `loadConversations(forModelDirectory:)` call with `ConversationFileStore.loadGlobal(inSupportDirectory:)`, and `persistConversations` with `saveGlobal`. Run the Task 10 migration once at launch before the first load, passing every installed model directory and its repository ID.

- [ ] **Step 4: Run tests**

Run: `swift build && swift test --filter ModelSwitchGuardTests && swift test --filter TurboFieldfareAppCoreTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Core/Catalog/ModelSwitchGuard.swift \
        Sources/TurboFieldfareApp/Core/State/AppModel.swift \
        Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift \
        Tests/TurboFieldfareApp/Core/Catalog/ModelSwitchGuardTests.swift
git commit -m "feat: switch the loaded model without restarting the decode service"
```

---

### Task 14: Picker UI and custom-model sheet

The only task with no unit tests — SwiftUI views here are thin over the tested types. Verification is by running the app.

**Files:**
- Create: `Sources/TurboFieldfareApp/Mac/Catalog/ModelPickerView.swift`
- Create: `Sources/TurboFieldfareApp/Mac/Catalog/AddCustomModelSheet.swift`
- Modify: `Sources/TurboFieldfareApp/Mac/App/RootView.swift`
- Modify: `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift`

**Interfaces:**
- Consumes: `ModelCatalog`, `ModelCatalogEntry`, `ModelTrustTier`, `ModelInstallStates`, `ModelSwitchVerdict`, `ArchPreflight.Result`, `AppContextOption`
- Produces: `ModelPickerView(model: AppModel)`, `AddCustomModelSheet(model: AppModel, isPresented: Binding<Bool>)`

- [ ] **Step 1: Build the picker list**

Create `Sources/TurboFieldfareApp/Mac/Catalog/ModelPickerView.swift`:

```swift
import SwiftUI

struct ModelPickerView: View {
    @Bindable var model: AppModel
    @State private var isAddingCustomModel = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Models").font(.headline)
                Spacer()
                Button("Add Custom Model…") { isAddingCustomModel = true }
            }
            ForEach(model.catalog.entries) { entry in
                ModelPickerRow(
                    entry: entry,
                    installState: model.installStates.state(for: entry.repoID),
                    isLoaded: model.selectedRepoID == entry.repoID && model.loadState.isReady,
                    onDownload: { model.startInstall(for: entry) },
                    onCancel: { model.cancelInstall(for: entry) },
                    onLoad: { model.switchModel(to: entry) },
                    onDelete: { model.deleteInstall(for: entry) })
            }
        }
        .padding()
        .sheet(isPresented: $isAddingCustomModel) {
            AddCustomModelSheet(model: model, isPresented: $isAddingCustomModel)
        }
    }
}

private struct ModelPickerRow: View {
    let entry: ModelCatalogEntry
    let installState: AppModelInstallState
    let isLoaded: Bool
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onLoad: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(entry.displayName).font(.body.weight(.medium))
                    TrustBadge(tier: entry.trustTier)
                    if isLoaded {
                        Text("Loaded").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Text(entry.repoID).font(.caption).foregroundStyle(.secondary)
                statusLine
            }
            Spacer()
            actions
        }
        .padding(.vertical, 6)
    }

    @ViewBuilder private var statusLine: some View {
        switch installState {
        case .idle, .cancelled:
            Text("Not installed").font(.caption).foregroundStyle(.secondary)
        case .installed:
            Text("Installed").font(.caption).foregroundStyle(.secondary)
        case .failed(let message), .recoverable(let message):
            Text(message).font(.caption).foregroundStyle(.red)
        default:
            ProgressView().controlSize(.small)
        }
    }

    @ViewBuilder private var actions: some View {
        switch installState {
        case .installed:
            Button("Load", action: onLoad).disabled(isLoaded)
            Button("Delete", action: onDelete)
                .disabled(isLoaded)
                .help(isLoaded ? "Unload this model before deleting it." : "")
        case .idle, .cancelled, .failed:
            Button("Download", action: onDownload)
        case .recoverable:
            Button("Resume", action: onDownload)
        default:
            Button("Cancel", action: onCancel)
        }
    }
}

private struct TrustBadge: View {
    let tier: ModelTrustTier

    var body: some View {
        Text(tier == .curated ? "Verified" : "Unverified")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tier == .curated ? Color.green.opacity(0.15) : Color.orange.opacity(0.15))
            .clipShape(Capsule())
    }
}
```

Match the surrounding look by reusing `TurboFieldfareMacTheme` colors instead of the literal `.green` / `.orange` above if that theme defines equivalents. Do not introduce a new styling system.

- [ ] **Step 2: Build the custom-model sheet**

Create `AddCustomModelSheet.swift` with a text field for the repository ID and an optional Hugging Face token field. On submit:

1. Validate with `ModelSlug.make(repoID:)`; show `InvalidRepositoryID.reason` inline on failure.
2. Fetch `config.json` and run `ArchPreflight.evaluate(configJSON:)`. On `.unsupported(modelType:)` show: "This model's architecture is `<modelType>`, which this build does not support. Only Gemma 4 26B-A4B and its finetunes can run." Stop here — nothing is downloaded.
3. On `.supported`, fetch index metadata and run `ModelTrustPolicy.decide(entry:observedIndexSHA256:)`.
4. On `.needsConsent(observed:)` show the consent panel with repository, revision, observed SHA-256, and download size, and the note that custom models are not verified by the project. On confirm, record the SHA into the entry and save via `ModelCatalogStore.save`.
5. On `.sourceChanged(recorded:observed:)` show both values and require explicit re-consent before continuing.

- [ ] **Step 3: Wire the context menu to computed options**

In `ModelInstallView.swift` and wherever context length is chosen, replace `AppContextLengthOption.allCases` with `AppContextLengthOption.availableOptions(architecture:residentWeightBytes:installedRAMBytes:)`. Render disabled options greyed out with `disabledReason` as the tooltip — do not filter them out, so the user can see what more RAM would buy.

- [ ] **Step 4: Verify by running the app**

```bash
swift build -c release
.build/release/TurboFieldfareMac
```

Check by hand:
1. Picker lists the curated Gemma entry with a "Verified" badge.
2. Adding `Qwen/Qwen3.6-35B-A3B` fails within seconds naming `qwen3_6_moe`, and no download starts.
3. Adding a Gemma 4 26B-A4B finetune reaches the consent panel showing a SHA-256.
4. With two models installed, switching loads the second without restarting the app, and the conversation list is unchanged across the switch.
5. Attempting a switch mid-generation is refused with a clear message.
6. Deleting the loaded model is refused; deleting an unloaded one frees its directory.

- [ ] **Step 5: Commit**

```bash
git add Sources/TurboFieldfareApp/Mac/Catalog/ModelPickerView.swift \
        Sources/TurboFieldfareApp/Mac/Catalog/AddCustomModelSheet.swift \
        Sources/TurboFieldfareApp/Mac/App/RootView.swift \
        Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift
git commit -m "feat: add model picker and custom-model sheet"
```

---

## Verification

After Task 14, run the whole suite and confirm nothing regressed:

```bash
swift build
swift test
```

Expected: all suites pass, including the pre-existing `TurboFieldfareTestsCore`, `TurboFieldfareRepackTests`, and `TurboFieldfareAppCoreTests`.

## Notes for the implementer

- **`AGENTS.md` restricts this checkout.** It says not to change runtime defaults or start optimization work unless asked. This plan was explicitly requested, and the only runtime-target change is Task 8. Do not take the opportunity to tune anything else.
- **Branch state.** This work builds on `feat/mac-app-chat-history`, which has uncommitted changes including `ConversationFileStore`, `ConversationModels`, and `AppPromptCache`. Task 10 modifies two of those files. Confirm with the user before rebasing or reordering.
- **`AppPromptCacheDomain` already keys on `modelID` and `sourceSnapshotHash`.** Do not add a second invalidation mechanism; Task 12 only makes the existing one eager.
- **Do not add architectures to `ArchConfig.supported` or `ArchPreflight.supportedModelTypes`** without the Metal kernels to execute them. An entry in either list without kernels turns a clean rejection into a runtime trap — `Attention.swift:177` and `:206` are `precondition`s, not throws.
