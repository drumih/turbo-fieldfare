import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

/// Every model a models directory holds that this server can actually serve,
/// GPU installs and CPU snapshots alike.
///
/// Identity is read, never parsed out of a directory name: a GPU install's id,
/// family and width come from its manifest through the same reader the loader
/// uses, and a CPU snapshot's from its own config.json. A directory the scan
/// cannot read is skipped with a reason rather than failing startup, because
/// one half-copied model must not take the other nine offline.
public struct ModelCatalog: Sendable {

    /// The engine that serves an install. The concept now lives in
    /// `TinyTitanLib` (2026-10-02, phase A1 of `docs/plan-embedded-library.md`)
    /// because the generation path consults it, and only the two cases travel;
    /// the catalog and everything else about it stay here. This spelling is
    /// kept so the catalog's own callers read unchanged.
    package typealias Backend = ModelBackend

    public enum Kind: Sendable, Equatable {
        case gpu(ModelFamily)
        case cpu(CPUModelFamily)

        package var backend: Backend {
            switch self {
            case .gpu: .gpu
            case .cpu: .cpu
            }
        }

        /// The kind that serves the same install on `backend`, where the
        /// runtime implements the family there too.
        package func kind(forBackend backend: Backend) -> Kind? {
            switch (self, backend) {
            case (.gpu, .gpu), (.cpu, .cpu):
                return self
            case (.gpu(.qwen35Dense), .cpu), (.cpu(.qwen35Dense), .gpu):
                // The dense family is the one both engines implement, and its
                // payload is identical for either.
                return backend == .cpu ? .cpu(.qwen35Dense) : .gpu(.qwen35Dense)
            default:
                return nil
            }
        }

        public var familyName: String {
            switch self {
            case .gpu(let family): family.rawValue
            case .cpu(let family): family.rawValue
            }
        }

        public var supportedReasoningLevels: [ReasoningLevel] {
            switch self {
            case .gpu(let family): family.supportedReasoningLevels
            case .cpu(let family): family.supportedReasoningLevels
            }
        }

        public var levelWhenOn: ReasoningLevel {
            switch self {
            case .gpu(let family): family.reasoningControl.levelWhenOn
            case .cpu(let family): family.levelWhenOn
            }
        }

        public func runtimeReasoning(
            for level: ReasoningLevel
        ) throws -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
            switch self {
            case .gpu(let family): try family.runtimeReasoning(for: level)
            case .cpu(let family): try family.runtimeReasoning(for: level)
            }
        }
    }

    public struct Entry: Sendable, Equatable {
        public let id: String
        public let name: String
        public let kind: Kind
        /// Weight width in bits: the routed experts for a GPU install, the
        /// base affine width for a CPU snapshot.
        public let quant: Int
        public let path: URL
        /// The sampling the model will really use for anything a request
        /// omits: the tuning profile's for a GPU install, the family's for a
        /// CPU snapshot.
        public let sampling: GenerationDefaults.Sampling
        /// The context the checkpoint claims, where the engine honours less
        /// than the server's --max-context. Nil for GPU installs, which serve
        /// whatever the server was configured for.
        public let contextLimit: Int?
        public let sizeBytes: Int64
        /// Every engine that can serve this install, most-preferred first.
        ///
        /// One for almost everything -- a MoE family is GPU-only and a snapshot
        /// is CPU-only -- but the dense Qwen 3.5 models run on both, and that is
        /// what makes `cpu` or `gpu` a request-level choice rather than a
        /// property of the model. `kind` is the default; the others are named by
        /// an `@cpu` / `@gpu` suffix on the id.
        package let engines: [Backend]

        package init(
            id: String, name: String, kind: Kind, quant: Int, path: URL,
            sampling: GenerationDefaults.Sampling, contextLimit: Int? = nil,
            sizeBytes: Int64 = 0, engines: [Backend]? = nil
        ) {
            self.id = id
            self.name = name
            self.kind = kind
            self.quant = quant
            self.path = path
            self.sampling = sampling
            self.contextLimit = contextLimit
            self.sizeBytes = sizeBytes
            // Defaulting to the entry's own engine keeps every existing caller
            // (and every earlier catalog) meaning what it meant.
            self.engines = engines ?? [kind.backend]
        }

        package var backend: Backend { kind.backend }

        /// The same install served by `backend`, or nil when that engine cannot
        /// serve it. Only the dense Qwen 3.5 family has two engines today: its
        /// `.ssdai` payload is the same file for either, and the runtime picks
        /// the engine.
        package func served(by backend: Backend, id aliasID: String) -> Entry? {
            guard engines.contains(backend),
                let kind = kind.kind(forBackend: backend)
            else { return nil }
            return Entry(
                id: aliasID,
                name: "\(name) (\(backend.rawValue.uppercased()))",
                kind: kind, quant: quant, path: path, sampling: sampling,
                contextLimit: contextLimit, sizeBytes: sizeBytes,
                engines: engines)
        }
    }

    public struct Skipped: Sendable, Equatable {
        public let path: URL
        public let reason: String
    }

    public let directory: URL
    public private(set) var entries: [Entry]
    public private(set) var skipped: [Skipped]

    public init(directory: URL, entries: [Entry], skipped: [Skipped] = []) {
        self.directory = directory
        self.entries = entries
        self.skipped = skipped
    }

    // MARK: - Scanning

    public static func scan(
        directory: URL,
        fileManager: FileManager = .default
    ) -> ModelCatalog {
        let root = directory.standardizedFileURL
        var catalog = ModelCatalog(directory: root, entries: [])
        let children: [URL]
        do {
            children = try fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        } catch {
            catalog.skipped.append(Skipped(path: root, reason: "cannot list: \(error)"))
            return catalog
        }
        for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            // A model linked in from another disk is served from where it
            // lives; the link itself does not pass as a directory.
            let target = child.resolvingSymlinksInPath()
            // Lock files and other loose files sit beside the installs; they
            // are not models and not worth a warning.
            guard (try? target.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            else {
                continue
            }
            catalog.add(probing: target)
        }
        catalog.entries.sort { lhs, rhs in
            lhs.backend == rhs.backend ? lhs.id < rhs.id : lhs.backend == .gpu
        }
        return catalog
    }

    /// Probes one directory and adds it, or records why it was skipped.
    /// Returns the entry when one was added.
    @discardableResult
    public mutating func add(probing directory: URL) -> Entry? {
        let path = directory.standardizedFileURL
        switch Self.probe(path) {
        case .success(let entry):
            if let existing = entries.first(where: { $0.id == entry.id }) {
                skipped.append(
                    Skipped(
                        path: path,
                        reason:
                            "duplicate id \(entry.id), already served from \(existing.path.path)"))
                return nil
            }
            entries.append(entry)
            return entry
        case .failure(let failure):
            skipped.append(Skipped(path: path, reason: failure.reason))
            return nil
        }
    }

    struct ProbeFailure: Error {
        let reason: String
    }

    static func probe(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: directory.appendingPathComponent("manifest.json").path) {
            return probeInstall(directory)
        }
        if fileManager.fileExists(atPath: directory.appendingPathComponent("config.json").path) {
            return probeSnapshot(directory)
        }
        return .failure(
            ProbeFailure(
                reason: "neither manifest.json (a GPU install) nor config.json (a CPU snapshot)"))
    }

    private static func probeInstall(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let identity: ManifestIdentity
        do {
            identity = try ManifestReader.peekIdentity(directoryURL: directory)
        } catch {
            return .failure(ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        switch identity.family {
        case .qwen36MTP, .qwen38flashMTP, .qwen4flashMTP:
            // A draft head has no tokenizer and no layers of its own to run;
            // listing it would offer a model that fails on first use.
            return .failure(
                ProbeFailure(
                    reason:
                        "an MTP draft head (\(identity.family.rawValue)), served only beside its target"
                ))
        case .qwen36, .qwen38flash, .qwen4flash, .qwen35Dense, .k2Horizon:
            // K2-Horizon is served by `K2ProbeBackend`, not by the runner, and
            // that is a property of the loader rather than of the catalogue:
            // the entry is honest, and `ModelRouter` picks the backend by
            // family.
            break
        case .laguna:
            return .failure(
                ProbeFailure(
                    reason:
                        "Laguna-S-2.1 (\(identity.family.rawValue)) is not served yet: "
                        + "its decode probe exists, the server graph does not "
                        + "(docs/plan-laguna-s-2.1.md)"
                ))
        }
        guard GFTokenizer.tokenizerFolder(forModelDirectory: directory) != nil else {
            return .failure(
                ProbeFailure(reason: "no tokenizer/tokenizer.json; the install is incomplete"))
        }
        let id = ServerModelIdentity.apiModelID(
            manifestModelID: identity.modelID,
            family: identity.family,
            weightBits: identity.weightBits)
        let base = ServerModelIdentity.base(
            manifestModelID: identity.modelID,
            family: identity.family)
        return .success(
            Entry(
                id: id,
                name: displayNames[base] ?? base,
                // A dense install is served by the GPU engine by default now that
                // the family is implemented there, and by the CPU engine when the
                // request (or the launch) names it -- `engines` is what says both
                // are available, and `ModelRouter` derives the `@cpu` alias from it.
                kind: .gpu(identity.family),
                quant: identity.weightBits,
                path: directory,
                sampling: ModelProfile.resolve(identity: identity).sampling,
                // The probe's KV caches are what cap it, not the checkpoint.
                contextLimit:
                    identity.family == .k2Horizon
                    ? K2ProbeBackend.contextCeiling : nil,
                sizeBytes: sizeOnDisk(directory),
                engines: identity.family == .qwen35Dense ? [.gpu, .cpu] : nil))
    }

    /// A scanned directory's `config.json`, read under the manifest's bound.
    /// A probe walks every directory the scan finds, so the file in front of it
    /// is not a document this server was told to load, and it can be far larger
    /// than the bound before the bound is applied: a bound that checks the bytes
    /// *after* the read has already paid for them. K17: the size is `fstat`'d on
    /// the descriptor being read, so the cap fires before the allocation and
    /// still belongs to the file actually read.
    static func readSnapshotConfig(
        _ directory: URL,
        maxBytes: UInt64 = ManifestReader.defaultMaxBytes
    ) throws -> Data {
        do {
            return try BoundedMetadataRead.read(
                fileAt: directory.appendingPathComponent("config.json"),
                maxBytes: maxBytes)
        } catch ModelError.metadataOverBound(let document, let bytes, let cap) {
            // The catalog's own error type, because the probe below catches it
            // and turns it into the skip reason a scan reports.
            throw CPUModelBackend.CPUBackendError.unsupported(
                "\(document) is \(bytes) bytes, over the \(cap)-byte metadata bound")
        }
    }

    private static func probeSnapshot(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let config: [String: Any]
        do {
            let data = try readSnapshotConfig(directory)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return .failure(ProbeFailure(reason: "config.json is not a JSON object"))
            }
            config = object
        } catch let error as CPUModelBackend.CPUBackendError {
            return .failure(ProbeFailure(reason: "\(error)"))
        } catch {
            return .failure(ProbeFailure(reason: "unreadable config.json: \(error)"))
        }
        let modelType = config["model_type"] as? String
        guard let family = CPUModelFamily.resolve(modelType: modelType) else {
            return .failure(ProbeFailure(reason: CPUModelFamily.refusal(modelType: modelType)))
        }
        guard let quantization = config["quantization"] as? [String: Any],
            let bits = quantization["bits"] as? Int
        else {
            return .failure(
                ProbeFailure(
                    reason:
                        "config.json has no quantization block; the CPU engine serves affine snapshots"
                ))
        }
        // Snapshots are written in place by a converter, so a config can
        // exist before the rest of the directory does.
        for required in ["model.safetensors.index.json"]
        where !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(required).path)
        {
            return .failure(ProbeFailure(reason: "incomplete snapshot: no \(required)"))
        }
        // A snapshot may keep its tokenizer at the root or under `tokenizer/`;
        // `resolvedTokenizerFolder` is the one reader that knows both.
        guard GFTokenizer.resolvedTokenizerFolder(forModelDirectory: directory) != nil else {
            return .failure(ProbeFailure(reason: "incomplete snapshot: no tokenizer.json"))
        }
        let declared = (config["model_id"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let id = declared ?? directory.lastPathComponent
        let name = (config["display_name"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? id
        return .success(
            Entry(
                id: id,
                name: name,
                kind: .cpu(family),
                quant: bits,
                path: directory,
                sampling: family.samplingDefaults,
                contextLimit: config["max_position_embeddings"] as? Int,
                sizeBytes: sizeOnDisk(directory)))
    }

    /// A single-model CPU server's family, for fitting `--reasoning` to it
    /// before the model itself is opened.
    ///
    /// Both shapes ship: a safetensors snapshot declares its architecture in
    /// `config.json`, and a `.ssdai` install declares it in `manifest.json`.
    /// Reading only the first is how `--cpu` against an installed dense model
    /// died with a raw `NSCocoaErrorDomain` "config.json couldn't be opened"
    /// instead of naming the family -- and a `.ssdai` install is exactly the
    /// shape the three dense Qwen 3.5 models are installed as.
    static func snapshotFamily(_ directory: URL) throws -> CPUModelFamily {
        if FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("manifest.json").path)
        {
            let family = try ManifestReader.peekFamily(directoryURL: directory)
            guard family == .qwen35Dense else {
                throw CPUModelBackend.CPUBackendError.unsupported(
                    CPUModelFamily.refusal(modelType: family.rawValue))
            }
            return .qwen35Dense
        }
        let data = try readSnapshotConfig(directory)
        let config = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let modelType = config?["model_type"] as? String
        guard let family = CPUModelFamily.resolve(modelType: modelType) else {
            throw CPUModelBackend.CPUBackendError.unsupported(
                CPUModelFamily.refusal(modelType: modelType))
        }
        return family
    }

    /// Allocated bytes, so an APFS clone or a sparse file reports what it
    /// really occupies rather than its logical length.
    static func sizeOnDisk(_ directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileSizeKey, .isRegularFileKey]
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: keys)
        else { return 0 }
        var total: Int64 = 0
        for case let file as URL in walker {
            guard let values = try? file.resourceValues(forKeys: Set(keys)),
                values.isRegularFile == true
            else { continue }
            total += Int64(values.totalFileAllocatedSize ?? values.fileSize ?? 0)
        }
        return total
    }

    // MARK: - Lookup

    public func entry(id: String) -> Entry? {
        entries.first { $0.id == id }
    }

    /// The entry `--model` names: a catalog id, or the path of a directory
    /// the catalog holds.
    public func entry(idOrPath value: String) -> Entry? {
        if let byID = entry(id: value) { return byID }
        let path = URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath().path
        return entries.first { $0.path.resolvingSymlinksInPath().path == path }
    }

    // MARK: - Reporting

    /// One notice for everything skipped, written once at startup.
    public func reportSkipped(to handle: FileHandle = .standardError) {
        guard !skipped.isEmpty else { return }
        var lines = [
            "catalog: skipped \(skipped.count) of \(skipped.count + entries.count) "
                + "directories in \(directory.path):"
        ]
        for item in skipped {
            lines.append("  \(item.path.lastPathComponent): \(item.reason)")
        }
        handle.write(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    /// `TinyTitanServer --catalog` output, which a launcher script parses.
    ///
    /// Assembled by hand because JSONEncoder does not keep key order, and the
    /// document should read in the order the launcher documents its fields.
    /// Strings still go through the encoder, so escaping is not hand-rolled.
    public func jsonData() throws -> Data {
        let models = try entries.map(Self.json(for:))
        return Data(("{\"models\":[" + models.joined(separator: ",") + "]}").utf8)
    }

    private static func json(for entry: Entry) throws -> String {
        let thinking = try entry.kind.supportedReasoningLevels.map { try quoted($0.rawValue) }
        let sampling =
            "{\"temperature\":\(decimal(entry.sampling.temperature, places: 4)),"
            + "\"top_p\":\(decimal(entry.sampling.topP, places: 4)),"
            + "\"top_k\":\(entry.sampling.topK)}"
        let fields: [(key: String, value: String)] = [
            ("id", try quoted(entry.id)),
            ("name", try quoted(entry.name)),
            ("family", try quoted(entry.kind.familyName)),
            ("quant", String(entry.quant)),
            ("backend", try quoted(entry.backend.rawValue)),
            ("engines", try quoted(entry.engines.map(\.rawValue).joined(separator: ","))),
            ("path", try quoted(entry.path.path)),
            ("thinking", "[" + thinking.joined(separator: ",") + "]"),
            ("sampling", sampling),
            ("size_gb", decimal(Double(entry.sizeBytes) / 1e9, places: 1)),
        ]
        return "{" + fields.map { "\"\($0.key)\":\($0.value)" }.joined(separator: ",") + "}"
    }

    private static func quoted(_ value: String) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return try encoder.encode(value).lossyUTF8String
    }

    /// Float 0.6 widens to 0.6000000238; a launcher showing the defaults
    /// should read the value the model card states.
    private static func decimal(_ value: some BinaryFloatingPoint, places: Int) -> String {
        let scale = pow(10.0, Double(places))
        return String((Double(value) * scale).rounded() / scale)
    }
}
