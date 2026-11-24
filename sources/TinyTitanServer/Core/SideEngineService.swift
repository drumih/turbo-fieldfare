import Foundation
import TinyTitanCore
import TinyTitanLib
import TinyTitanMemory
import TinyTitanVariants

/// The resident CPU side-engine, wearing memory's port.
///
/// The engine judges in its own vocabulary (`SideEngineJudgement`); memory
/// asks in its own (`MemoryFact`). This is the whole translation, and every
/// failure — a shut-down engine, a model that will not load, a completion the
/// parser refused — becomes `nil`, which the memory path reads as "no
/// decision" and answers with the deterministic behaviour it had before the
/// engine existed.
struct SideEngineMemoryAdapter: MemorySideEngine, Sendable {
    let engine: SideEngine

    func isDurable(_ fact: MemoryFact) async -> Bool? {
        await yesNo(.durability(key: fact.key, value: fact.value))
    }

    /// The stored fact is `A` and the incoming one `B`, which is the order the
    /// prompts were measured in. On the 4B the same pair answers YES in that
    /// order and NO reversed, so the mapping is part of the contract.
    func duplicates(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? {
        await yesNo(
            .duplication(
                aKey: stored.key, aValue: stored.value,
                bKey: new.key, bValue: new.value))
    }

    func contradicts(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? {
        await yesNo(
            .contradiction(
                aKey: stored.key, aValue: stored.value,
                bKey: new.key, bValue: new.value))
    }

    /// T7. The caller is the background hinter, which owns the idle gate and
    /// the budget; the adapter only translates the question and the fact.
    func couldAnswer(_ question: String, _ fact: MemoryFact) async -> Bool? {
        await yesNo(.retrieval(question: question, key: fact.key, value: fact.value))
    }

    /// The rule is data the caller found; without one the engine's answer is a
    /// guess about whether the change was allowed, so a `nil` rule answers
    /// `nil` rather than pretending.
    func supersedes(
        _ stored: MemoryFact, _ new: MemoryFact,
        rule: String?
    ) async -> MemorySupersession? {
        guard let rule else { return nil }
        guard
            let answer = try? await engine.judge(
                .supersession(
                    key: stored.key, earlier: stored.value,
                    now: new.value, rule: rule))
        else { return nil }
        switch answer {
        case .update: return .update
        case .conflict: return .conflict
        // Neither is a kind of change.
        case .yes, .no: return nil
        }
    }

    func shutdown() async {
        await engine.shutdown()
    }

    private func yesNo(_ judgement: SideEngineJudgement) async -> Bool? {
        guard let answer = try? await engine.judge(judgement) else { return nil }
        switch answer {
        case .yes: return true
        case .no: return false
        // Only the two-way tasks are asked through this port, and neither
        // UPDATE nor CONFLICT is a yes or a no.
        case .update, .conflict: return nil
        }
    }
}

/// Builds the resident side-engine, or nothing.
public enum ServerSideEngineFactory {
    /// The install the engine loads when none is named.
    ///
    /// The 4B is the smallest that decides contradiction, duplication and
    /// retrieval on the shipped prompts; the 2B is not the verification
    /// instrument, and the reply check needs the 9B
    /// (`docs/side-engine-tasks.md`).
    public static let defaultInstall = "qwen3.5_4B_4Bit"

    /// Names a directory, an install under the models directory, or `0`/`off`.
    public static let environmentKey = "TINYTITAN_SIDE_ENGINE"

    /// The engine, or nil when none is installed or it was turned off.
    ///
    /// The weights load on the first judgement rather than here, so a server
    /// that is never asked a question never pays for one.
    public static func make(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        modelsDirectory: String?,
        isClientGenerating: (@Sendable () -> Bool)?
    ) -> SideEngine? {
        if isExplicitlyOff(setting(in: environment)) { return nil }
        guard
            let directory = resolve(
                environment: environment,
                modelsDirectory: modelsDirectory)
        else {
            ServerLog.memory(
                "side-engine off: no \(defaultInstall) install under "
                    + "\(modelsDirectory ?? "the models directory")")
            return nil
        }
        return SideEngine(isClientGenerating: isClientGenerating) {
            try await DenseSideEngineModel(
                snapshotDirectory: URL(fileURLWithPath: directory))
        }
    }

    /// The directory to load, or nil.
    ///
    /// A value with a slash — or one that already names an existing directory
    /// — is a path and is used as given. Anything else is an install name
    /// under the models directory, which is how the server resolves a model
    /// everywhere else.
    static func resolve(
        environment: [String: String],
        modelsDirectory: String?
    ) -> String? {
        let setting = setting(in: environment)
        if isExplicitlyOff(setting) { return nil }
        let name = setting ?? defaultInstall
        let fileManager = FileManager.default
        if name.contains("/") || fileManager.fileExists(atPath: name) {
            return fileManager.fileExists(atPath: name) ? name : nil
        }
        guard let modelsDirectory, !modelsDirectory.isEmpty else { return nil }
        let candidate = URL(fileURLWithPath: modelsDirectory)
            .appendingPathComponent(name, isDirectory: true).path
        return fileManager.fileExists(atPath: candidate) ? candidate : nil
    }

    static func setting(in environment: [String: String]) -> String? {
        let raw = environment[environmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return raw?.isEmpty == false ? raw : nil
    }

    static func isExplicitlyOff(_ setting: String?) -> Bool {
        guard let setting else { return false }
        return ["0", "off", "false"].contains(setting.lowercased())
    }
}
