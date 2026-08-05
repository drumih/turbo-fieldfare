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

        enum CodingKeys: String, CodingKey {
            case id, object, bytes, filename, purpose, status
            case createdAt = "created_at"
        }
    }

    private let directory: URL
    private var files: [String: File] = [:]

    public init(directory: URL) {
        self.directory = directory
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
        guard contents.count <= TurboFieldfareHTTPServer.maximumBatchFileBytes else {
            throw ServerRequestError.invalid(message: "batch files may not exceed 200 MiB", param: "file", code: "invalid_value")
        }
        return try write(filename: filename, purpose: purpose, contents: contents)
    }

    public func createOutput(filename: String) throws -> File {
        try write(filename: filename, purpose: "batch", contents: Data())
    }

    /// Registers a result file produced by BatchRegistry, whose historical path is `<id>.jsonl`.
    public func registerBatchOutput(_ id: String) throws -> File {
        if let file = files[id] { return file }
        let url = directory.appendingPathComponent(id).appendingPathExtension("jsonl")
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let file = File(id: id, bytes: bytes, createdAt: Int(Date().timeIntervalSince1970),
                        filename: id + ".jsonl", purpose: "batch")
        files[id] = file
        try persist()
        return file
    }

    public func get(_ id: String) -> File? {
        if let file = files[id] {
            let output = url(for: id).appendingPathExtension("jsonl")
            if FileManager.default.fileExists(atPath: output.path) {
                let bytes = (try? output.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? file.bytes
                return File(id: file.id, bytes: bytes, createdAt: file.createdAt,
                            filename: file.filename, purpose: file.purpose)
            }
            return file
        }
        let output = url(for: id).appendingPathExtension("jsonl")
        guard FileManager.default.fileExists(atPath: output.path) else { return nil }
        let bytes = (try? output.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return File(id: id, bytes: bytes, createdAt: Int(Date().timeIntervalSince1970),
                    filename: id + ".jsonl", purpose: "batch")
    }

    public func list() -> [File] { files.keys.sorted().compactMap(get) }

    public func delete(_ id: String) throws -> Bool {
        guard let file = files.removeValue(forKey: id) else { return false }
        try? FileManager.default.removeItem(at: url(for: file.id))
        try? FileManager.default.removeItem(at: url(for: file.id).appendingPathExtension("jsonl"))
        try persist()
        return true
    }

    public func contents(_ id: String) throws -> Data? {
        guard files[id] != nil || FileManager.default.fileExists(atPath: url(for: id).appendingPathExtension("jsonl").path) else { return nil }
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
                        purpose: metadata.purpose)
        files[id] = metadata
        try persist()
    }

    private func write(filename: String, purpose: String, contents: Data) throws -> File {
        let id = "file-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
        let file = File(id: id, bytes: contents.count, createdAt: Int(Date().timeIntervalSince1970), filename: filename, purpose: purpose)
        try contents.write(to: url(for: id), options: .atomic)
        files[id] = file
        try persist()
        return file
    }

    private func url(for id: String) -> URL { directory.appendingPathComponent(id) }

    private func persist() throws {
        let data = try JSONEncoder().encode(files)
        try data.write(to: directory.appendingPathComponent("files.json"), options: .atomic)
    }
}
