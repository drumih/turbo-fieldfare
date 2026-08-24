import CryptoKit
import Darwin
import Foundation

public final class ServerAttachmentLease: @unchecked Sendable {
    let directoryURL: URL

    init(directoryURL: URL) {
        self.directoryURL = directoryURL
    }

    deinit {
        try? FileManager.default.removeItem(at: directoryURL)
    }
}

/// Where inline images are staged, and how directories orphaned by a crash are
/// reclaimed.
///
/// A lease removes its own directory when it is released, but that only runs
/// when the process lives long enough to release it. A crash or a `SIGKILL`
/// mid-request leaves the staged bytes behind forever, so every run is scoped
/// under its own `pid-<n>` directory and each start reclaims the directories
/// whose process is gone. A live process — a second server sharing this
/// `TMPDIR` — keeps its own directory, so the sweep never races a peer.
public enum ServerAttachmentDirectory {
    public static let rootName = "TurboFieldfare-Server-Attachments"

    public static var root: URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(rootName, isDirectory: true)
    }

    public static var processRoot: URL {
        root.appendingPathComponent("pid-\(getpid())", isDirectory: true)
    }

    public static func makeLease(
        in root: URL = ServerAttachmentDirectory.root
    ) throws -> ServerAttachmentLease {
        let directory = root
            .appendingPathComponent("pid-\(getpid())", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return ServerAttachmentLease(directoryURL: directory)
    }

    /// Removes staging directories that no live process owns. Returns the names
    /// it reclaimed so a caller can log or assert on them.
    @discardableResult
    public static func sweepAbandoned(
        in root: URL = ServerAttachmentDirectory.root
    ) -> [String] {
        let manager = FileManager.default
        guard let entries = try? manager.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil) else { return [] }
        var reclaimed: [String] = []
        for entry in entries {
            let name = entry.lastPathComponent
            if let pid = ownerProcess(of: name), isAlive(pid) { continue }
            guard (try? manager.removeItem(at: entry)) != nil else { continue }
            reclaimed.append(name)
        }
        return reclaimed
    }

    private static func ownerProcess(of name: String) -> pid_t? {
        guard name.hasPrefix("pid-"), let value = pid_t(name.dropFirst(4)),
              value > 0 else { return nil }
        return value
    }

    private static func isAlive(_ pid: pid_t) -> Bool {
        // EPERM means the process exists but belongs to another user; only
        // ESRCH proves it is gone.
        kill(pid, 0) == 0 || errno == EPERM
    }
}

struct ServerStagedImage: Sendable {
    let id: UUID
    let fileURL: URL
    let encodedBytes: Int
    let sha256: String
}

struct ServerAttachmentStore {
    static let maximumImageBytes = 16 * 1_024 * 1_024
    static let maximumRequestImageBytes = 64 * 1_024 * 1_024
    /// Staging happens before the context budget can reject anything, so the
    /// byte caps alone leave the file count unbounded: each image costs about
    /// 105 JSON bytes, so 5 MiB of body can create tens of thousands of temp
    /// files and stall the single event loop for seconds.
    static let maximumRequestImages = 64
    static let allowedDataURLHeaders: Set<String> = [
        "data:image/jpeg;base64",
        "data:image/png;base64",
        "data:image/heic;base64",
        "data:image/heif;base64",
    ]

    let lease: ServerAttachmentLease
    private(set) var decodedBytes = 0
    private(set) var stagedCount = 0

    init(attachmentRoot: URL = ServerAttachmentDirectory.root) throws {
        lease = try ServerAttachmentDirectory.makeLease(in: attachmentRoot)
    }

    /// Stages a data URL that reached the validator directly. Wire requests
    /// never take this path — the streaming parser stages every acceptable
    /// message image before the validator runs — so this exists for callers
    /// of `OpenAIRequestValidator.validate` outside the HTTP server. The
    /// policy pieces (allowlist, caps, writer) are shared with the parser so
    /// the two entries cannot drift.
    mutating func stage(dataURL: String) throws -> ServerStagedImage {
        guard stagedCount < Self.maximumRequestImages else {
            throw ServerRequestError.invalid(
                message: "at most \(Self.maximumRequestImages) images per request",
                param: "messages", code: "too_many_images")
        }
        guard let comma = dataURL.firstIndex(of: ",") else {
            throw ServerRequestError.invalid(
                message: "image_url must be a base64 data URL",
                param: "messages", code: "invalid_image_url")
        }
        let header = String(dataURL[..<comma]).lowercased()
        guard Self.allowedDataURLHeaders.contains(header) else {
            throw ServerRequestError.invalid(
                message: "image_url must be a JPEG, PNG, HEIC, or HEIF base64 data URL",
                param: "messages", code: "unsupported_image")
        }
        let payload = dataURL[dataURL.index(after: comma)...].utf8
        let writer = try StreamingBase64FileWriter(directoryURL: lease.directoryURL)
        // Bounded spans, not the whole payload at once: the writer's pending
        // buffer grows by what one feed hands it before it can flush.
        let span = 64 * 1_024
        let fed: Void? = try payload.withContiguousStorageIfAvailable { buffer in
            var index = 0
            while index < buffer.count {
                let end = min(index + span, buffer.count)
                try writer.feed(
                    contentsOf: UnsafeBufferPointer(rebasing: buffer[index..<end]))
                index = end
            }
        }
        if fed == nil {
            var chunk: [UInt8] = []
            chunk.reserveCapacity(span)
            for byte in payload {
                chunk.append(byte)
                if chunk.count == span {
                    try writer.feed(contentsOf: chunk)
                    chunk.removeAll(keepingCapacity: true)
                }
            }
            try writer.feed(contentsOf: chunk)
        }
        let staged = try writer.finish()
        guard decodedBytes + staged.encodedBytes <= Self.maximumRequestImageBytes else {
            try? FileManager.default.removeItem(at: staged.fileURL)
            throw ServerRequestError.invalid(
                message: "inline image bytes exceed the server limit",
                param: "messages", code: "image_too_large")
        }
        decodedBytes += staged.encodedBytes
        stagedCount += 1
        return staged
    }
}

final class StreamingBase64FileWriter {
    private let id = UUID()
    private let fileURL: URL
    private let handle: FileHandle
    private var pending: [UInt8] = []
    private var hasher = SHA256()
    private var decodedBytes = 0
    /// Padding may only terminate the payload. Enforcing that here makes the
    /// stream's validity independent of where flush boundaries fall: without
    /// it, mid-stream padding decodes or fails depending on how the wire
    /// happened to chunk the same bytes.
    private var paddingBytes = 0
    private var finished = false

    init(directoryURL: URL) throws {
        fileURL = directoryURL.appendingPathComponent(id.uuidString)
        FileManager.default.createFile(
            atPath: fileURL.path,
            contents: nil,
            attributes: [.posixPermissions: 0o600])
        handle = try FileHandle(forWritingTo: fileURL)
        pending.reserveCapacity(64 * 1_024)
    }

    deinit {
        if !finished {
            try? handle.close()
            try? FileManager.default.removeItem(at: fileURL)
        }
    }

    func feed(contentsOf bytes: some Sequence<UInt8>) throws {
        for byte in bytes {
            let padded = byte == UInt8(ascii: "=")
            if padded { paddingBytes += 1 }
            guard Self.isBase64(byte), paddingBytes <= 2,
                  padded || paddingBytes == 0 else {
                throw ServerRequestError.invalid(
                    message: "image_url contains invalid base64",
                    param: "messages", code: "invalid_image")
            }
        }
        pending.append(contentsOf: bytes)
        if pending.count >= 64 * 1_024 { try flush(final: false) }
    }

    func finish() throws -> ServerStagedImage {
        try flush(final: true)
        try handle.close()
        guard decodedBytes > 0, chmod(fileURL.path, S_IRUSR) == 0 else {
            throw ServerRequestError.invalid(
                message: "image_url contains no image bytes",
                param: "messages", code: "invalid_image")
        }
        finished = true
        return ServerStagedImage(
            id: id,
            fileURL: fileURL,
            encodedBytes: decodedBytes,
            sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined())
    }

    private static func isBase64(_ byte: UInt8) -> Bool {
        (65...90).contains(byte) || (97...122).contains(byte)
            || (48...57).contains(byte) || byte == 43 || byte == 47 || byte == 61
    }

    private func flush(final: Bool) throws {
        let count = final ? pending.count : pending.count - pending.count % 4
        guard count > 0 else { return }
        if final, count % 4 != 0 {
            throw ServerRequestError.invalid(
                message: "image_url contains invalid base64",
                param: "messages", code: "invalid_image")
        }
        guard let decoded = Data(base64Encoded: Data(pending.prefix(count))) else {
            throw ServerRequestError.invalid(
                message: "image_url contains invalid base64",
                param: "messages", code: "invalid_image")
        }
        decodedBytes += decoded.count
        guard decodedBytes <= ServerAttachmentStore.maximumImageBytes else {
            throw ServerRequestError.invalid(
                message: "inline image bytes exceed the server limit",
                param: "messages", code: "image_too_large")
        }
        try handle.write(contentsOf: decoded)
        hasher.update(data: decoded)
        pending.removeFirst(count)
    }
}
