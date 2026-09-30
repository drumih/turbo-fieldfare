import Foundation

/// Small durable subset of the OpenAI Files API used by the local Batch API.
public actor BatchFileStore {
    public struct File: Codable, Sendable {
        public let id: String
        public let object = "file"
        public let bytes: Int
        public let createdAt: Int
        public let filename: String
        public let purpose: String
        public let status = "processed"
        public let expiresAt: Int?

        enum CodingKeys: String, CodingKey {
            case id, object, bytes, filename, purpose, status
            case createdAt = "created_at"
            case expiresAt = "expires_at"
        }
    }

    private let directory: URL
    private let maximumInputBytes: Int
    private var files: [String: File] = [:]

    public init(directory: URL,
                maximumInputBytes: Int = TurboFieldfareHTTPServer.maximumBatchFileBytes) {
        self.directory = directory
        self.maximumInputBytes = maximumInputBytes
        // The loopback server deliberately has no cross-process persistence.
        // Starting a new server instance discards Files and Batch artifacts.
        try? FileManager.default.removeItem(at: directory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    public func create(filename: String, purpose: String, contents: Data) throws -> File {
        guard purpose == "batch" else {
            throw ServerRequestError.invalid(message: "only purpose=batch is supported", param: "purpose", code: "unsupported_value")
        }
        guard filename.lowercased().hasSuffix(".jsonl") else {
            throw ServerRequestError.invalid(message: "batch files must use the .jsonl extension", param: "file", code: "invalid_value")
        }
        guard contents.count <= maximumInputBytes else {
            throw ServerRequestError.invalid(message: "batch files may not exceed 200 MiB", param: "file", code: "invalid_value")
        }
        return try write(filename: filename, purpose: purpose, contents: contents)
    }

    public func createOutput(filename: String) throws -> File {
        try write(filename: filename, purpose: "batch", contents: Data())
    }

    /// Registers a result file produced by BatchRegistry, whose historical path is `<id>.jsonl`.
    public func registerBatchOutput(_ id: String, expiresAt: Int?) throws -> File {
        if let file = files[id] { return file }
        let url = directory.appendingPathComponent(id).appendingPathExtension("jsonl")
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let file = File(id: id, bytes: bytes, createdAt: Int(Date().timeIntervalSince1970),
                        filename: id + ".jsonl", purpose: "batch_output", expiresAt: expiresAt)
        files[id] = file
        try persist()
        if let expiresAt {
            Task { [weak self] in
                let delay = max(0, expiresAt - Int(Date().timeIntervalSince1970))
                try? await Task.sleep(for: .seconds(delay))
                await self?.expire(id)
            }
        }
        return file
    }

    public func get(_ id: String) -> File? {
        if let file = files[id] {
            if let expiresAt = file.expiresAt, expiresAt <= Int(Date().timeIntervalSince1970) {
                return nil
            }
            let output = url(for: id).appendingPathExtension("jsonl")
            if FileManager.default.fileExists(atPath: output.path) {
                return generatedFile(id: id, url: output, fallback: file)
            }
            return file
        }
        let output = url(for: id).appendingPathExtension("jsonl")
        guard FileManager.default.fileExists(atPath: output.path) else { return nil }
        return generatedFile(id: id, url: output, fallback: nil)
    }

    public func list(order: String = "desc", purpose: String? = nil) -> [File] {
        var ids = Set(files.keys)
        if let urls = try? FileManager.default.contentsOfDirectory(at: directory,
                                                                     includingPropertiesForKeys: nil) {
            for url in urls where url.pathExtension == "jsonl" {
                ids.insert(url.deletingPathExtension().lastPathComponent)
            }
        }
        return ids.compactMap(get)
            .filter { purpose == nil || $0.purpose == purpose }
            .sorted {
                if $0.createdAt != $1.createdAt {
                    return order == "asc" ? $0.createdAt < $1.createdAt : $0.createdAt > $1.createdAt
                }
                return order == "asc" ? $0.id < $1.id : $0.id > $1.id
            }
    }

    public func delete(_ id: String) throws -> Bool {
        let file = files.removeValue(forKey: id)
        let direct = url(for: id)
        let output = direct.appendingPathExtension("jsonl")
        guard file != nil || FileManager.default.fileExists(atPath: output.path) else { return false }
        try? FileManager.default.removeItem(at: direct)
        try? FileManager.default.removeItem(at: output)
        try persist()
        return true
    }

    public func contents(_ id: String) throws -> Data? {
        guard get(id) != nil else { return nil }
        let direct = url(for: id)
        let output = direct.appendingPathExtension("jsonl")
        return try Data(contentsOf: FileManager.default.fileExists(atPath: direct.path) ? direct : output)
    }

    public func appendJSONL(_ object: [String: Any], to id: String) throws {
        guard var metadata = files[id] else { throw CocoaError(.fileNoSuchFile) }
        var line = try JSONSerialization.data(withJSONObject: object)
        line.append(0x0A)
        let handle = try FileHandle(forWritingTo: url(for: id))
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: line)
        metadata = File(id: metadata.id, bytes: metadata.bytes + line.count,
                        createdAt: metadata.createdAt, filename: metadata.filename,
                        purpose: metadata.purpose, expiresAt: metadata.expiresAt)
        files[id] = metadata
        try persist()
    }

    private func write(filename: String, purpose: String, contents: Data) throws -> File {
        let id = "file-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
        let file = File(id: id, bytes: contents.count, createdAt: Int(Date().timeIntervalSince1970), filename: filename, purpose: purpose, expiresAt: nil)
        try contents.write(to: url(for: id), options: .atomic)
        files[id] = file
        try persist()
        return file
    }

    private func url(for id: String) -> URL { directory.appendingPathComponent(id) }

    private func generatedFile(id: String, url: URL, fallback: File?) -> File {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .creationDateKey])
        let bytes = values?.fileSize ?? fallback?.bytes ?? 0
        let createdAt = Int((values?.creationDate ?? fallback.map { Date(timeIntervalSince1970: TimeInterval($0.createdAt)) } ?? Date()).timeIntervalSince1970)
        return File(id: id, bytes: bytes, createdAt: createdAt,
                    filename: id + ".jsonl", purpose: "batch_output", expiresAt: fallback?.expiresAt)
    }

    private func expire(_ id: String) {
        files.removeValue(forKey: id)
        try? FileManager.default.removeItem(at: url(for: id).appendingPathExtension("jsonl"))
        try? persist()
    }

    private func persist() throws {
        let data = try JSONEncoder().encode(files)
        try data.write(to: directory.appendingPathComponent("files.json"), options: .atomic)
    }
}
