import Foundation
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib

/// A dense CPU variant engine plus the tokenizer from the same directory,
/// backing `SideEngine` without the 6.x runtime (TT-134).
///
/// The install's hidden size and quant width select the matching per-width
/// dense CPU variant through `DenseCPUInstall` and
/// `EngineFactories.variantCPUEngine`. For the shipped prompts the 4B is the
/// verified floor and the 9B decides more (`docs/side-engine-tasks.md`); the
/// 2B is not the verification instrument.
///
/// unchecked-invariant: every member is touched only from the `SideEngine`
/// actor that owns this wrapper, which serialises `reset`, `encode`, and
/// `generate`. The underlying engine is a class with one KV state, so that
/// serialisation is the correctness argument, not a convenience.
public final class DenseSideEngineModel: SideEngineModel, @unchecked Sendable {
    private let engine: any CPUEngineRunner
    private let tokenizer: GFTokenizer

    public init(snapshotDirectory: URL, threads: Int? = nil) async throws {
        let install = try DenseCPUInstall(directory: snapshotDirectory)
        let factory = try EngineFactories.variantCPUEngine(
            hiddenSize: install.hiddenSize,
            weightBits: install.weightBits,
            snapshotDirectory: install.directory,
            isSSDAI: install.isSSDAI,
            threads: threads)
        guard
            let folder = GFTokenizer.resolvedTokenizerFolder(
                forModelDirectory: snapshotDirectory)
        else {
            throw SideEngineError.missingTokenizer(snapshotDirectory.path)
        }
        self.engine = try factory.build()
        self.tokenizer = try await GFTokenizer.load(from: folder)
    }

    public func reset() { engine.reset() }

    public func encode(system: String, user: String) throws -> [Int] {
        let rendered = try tokenizer.applyChatTemplate([
            GFTokenizer.Message(role: .system, content: system),
            GFTokenizer.Message(role: .user, content: user),
        ])
        return tokenizer.encode(rendered, addBOS: false).map(Int.init)
    }

    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens.map(Int32.init))
    }

    public func generate(
        prompt: [Int], maximumTokens: Int,
        stopping: Set<Int>
    ) throws -> [Int] {
        try engine.generate(
            prompt: prompt, maximumTokens: maximumTokens,
            stopping: stopping)
    }

    public var endOfSequence: Int { Int(tokenizer.eosID) }

    public var contention: (@Sendable () -> Bool)? {
        get { engine.contention }
        set { engine.contention = newValue }
    }

    public var busyThreads: Int {
        get { engine.busyThreads }
        set { engine.busyThreads = newValue }
    }

    public var idleThreads: Int {
        get { engine.idleThreads }
        set { engine.idleThreads = newValue }
    }

    public var threads: Int { engine.threads }
}
