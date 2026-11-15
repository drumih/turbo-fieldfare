import Foundation
import Tokenizers

// Loading a tokenizer: the sidecar resolution, the public factories, and the
// coordinator that de-duplicates concurrent loads.
//
// Split out of `Tokenizer.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion; the load-source enum and
// coordinator moved with their only user, so they stayed `private`.
extension GFTokenizer {

    public static func load(
        from folder: URL,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil
    ) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(
            .local(folder.standardizedFileURL.path, thinkingMode, reasoningEffort))
    }

    public static func load(
        forModelDirectory modelDirectory: URL,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) async throws -> GFTokenizer {
        _ = try requireModelDirectory(modelDirectory)
        guard
            let folder = tokenizerFolder(
                forModelDirectory: modelDirectory, environment: environment)
        else {
            throw GFTokenizerError.missingToolTemplate
        }
        return try await load(
            from: folder,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
    }

    /// The model directory a load is about to read, checked before anything
    /// else. A path that is not a directory is the caller's mistake, not a
    /// damaged install, and it used to surface as "installed tokenizer is
    /// missing chat_template.jinja; reinstall the model" -- which sends the
    /// reader to reinstall a model that is simply not at that path (TT-036).
    public static func requireModelDirectory(
        _ url: URL,
        fileManager: FileManager = .default
    ) throws -> URL {
        let standardized = url.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard
            fileManager.fileExists(atPath: standardized.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw GFTokenizerError.modelDirectoryNotFound(standardized.path)
        }
        return standardized
    }

    public static func tokenizerFolder(
        forModelDirectory modelDirectory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["TURBO_FIELDFARE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    /// The folder to hand to `load(from:)` for a model directory, in either
    /// shape this project ships: a safetensors snapshot keeps `tokenizer.json`
    /// at the directory root, a `.ssdai` install keeps it under `tokenizer/`.
    ///
    /// `tokenizerFolder(forModelDirectory:)` answers only the second shape --
    /// it is about locating a *sidecar* -- so a caller that has a model
    /// directory and needs the folder `load(from:)` can actually read must use
    /// this instead. Passing the model directory to `load(from:)` works for a
    /// snapshot and fails for every install, which is the bug this exists to
    /// keep from being written twice.
    public static func resolvedTokenizerFolder(
        forModelDirectory modelDirectory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL? {
        let root = modelDirectory.standardizedFileURL
        if hasTokenizerJSON(in: root, fileManager: fileManager) {
            return root
        }
        return tokenizerFolder(
            forModelDirectory: root,
            environment: environment,
            fileManager: fileManager)
    }

    static func loadUncached(
        from folder: URL,
        thinkingMode: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort? = nil
    ) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(modelFolder: folder)
        let decoder = try GFByteLevelDecoderConfiguration.load(
            from: folder.appendingPathComponent("tokenizer.json"),
            tokenizer: underlying)
        return try GFTokenizer(
            tokenizer: underlying,
            byteLevelDecoderConfiguration: decoder,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }
}

private enum GFTokenizerLoadSource: Hashable {
    case local(String, ModelThinkingMode, ModelReasoningEffort?)
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            switch source {
            case .local(let path, let thinkingMode, let reasoningEffort):
                return try await GFTokenizer.loadUncached(
                    from: URL(fileURLWithPath: path),
                    thinkingMode: thinkingMode,
                    reasoningEffort: reasoningEffort)
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
