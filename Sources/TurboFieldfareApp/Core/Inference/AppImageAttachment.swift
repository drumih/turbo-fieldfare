import Darwin
import Foundation
import TurboFieldfare

public struct AppImageAttachment: Sendable, Equatable {
    public let id: UUID
    public let fileURL: URL
    public let displayName: String
    public let encodedBytes: Int
    public let sha256: String

    public init(id: UUID = UUID(), fileURL: URL, displayName: String,
                encodedBytes: Int, sha256: String) {
        self.id = id
        self.fileURL = fileURL
        self.displayName = displayName
        self.encodedBytes = encodedBytes
        self.sha256 = sha256
    }
}

public struct AppImageAttachmentStore: Sendable {
    public static let rootName = "TurboFieldfare-Attachments"

    public static var root: URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(rootName, isDirectory: true)
    }

    /// Each run stages under its own `pid-<n>` directory so a run that was
    /// killed can be told apart from one that is still going. Without that,
    /// every session left up to four full-size image copies behind forever.
    public static func defaultDirectory() -> URL {
        root.appendingPathComponent("pid-\(getpid())", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
    }

    /// Deletes the staging directories of processes that are gone. A live
    /// peer — a second copy of the app — keeps its own.
    @discardableResult
    public static func sweepAbandoned(in root: URL = AppImageAttachmentStore.root)
        -> [String] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil) else { return [] }
        var reclaimed: [String] = []
        for entry in entries {
            let name = entry.lastPathComponent
            if name.hasPrefix("pid-"), let pid = pid_t(name.dropFirst(4)), pid > 0,
               kill(pid, 0) == 0 || errno == EPERM {
                continue
            }
            guard (try? FileManager.default.removeItem(at: entry)) != nil else { continue }
            reclaimed.append(name)
        }
        return reclaimed
    }

    /// Whether a path names a file this app staged.
    ///
    /// The decode service receives attachment paths over a socket and opens
    /// them, so it needs a rule that does not depend on which process staged
    /// the file: the app and the service are separate processes with separate
    /// stores, but both live under the same root. Symlinks are resolved first,
    /// so a link planted inside the root cannot point out of it.
    public static func contains(_ fileURL: URL) -> Bool {
        guard fileURL.isFileURL else { return false }
        let resolved = fileURL.standardizedFileURL.resolvingSymlinksInPath()
        let fileComponents = resolved.pathComponents
        return allowedRoots.contains { root in
            let rootComponents = root.pathComponents
            guard fileComponents.count > rootComponents.count else { return false }
            return Array(fileComponents.prefix(rootComponents.count)) == rootComponents
        }
    }

    /// The attachment roots this process will accept a path under.
    ///
    /// The app and the decode service are separate processes and resolve their
    /// temporary directory independently, so the per-user Darwin temp
    /// directory is included explicitly: a service launched without `TMPDIR`
    /// must still recognise the app's staged files, and rejecting them all
    /// would silently break every image.
    private static var allowedRoots: [URL] {
        var roots = [FileManager.default.temporaryDirectory]
        if let userTemp = darwinUserTemporaryDirectory { roots.append(userTemp) }
        return roots.map {
            $0.appendingPathComponent(rootName, isDirectory: true)
                .standardizedFileURL.resolvingSymlinksInPath()
        }
    }

    private static var darwinUserTemporaryDirectory: URL? {
        var buffer = [UInt8](repeating: 0, count: Int(PATH_MAX))
        let length = buffer.withUnsafeMutableBytes {
            confstr(_CS_DARWIN_USER_TEMP_DIR,
                    $0.baseAddress!.assumingMemoryBound(to: CChar.self), $0.count)
        }
        // confstr counts the terminating NUL.
        guard length > 1, length <= buffer.count else { return nil }
        let path = String(decoding: buffer[..<(length - 1)], as: UTF8.self)
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    public let directoryURL: URL

    public init(directoryURL: URL = AppImageAttachmentStore.defaultDirectory()) {
        self.directoryURL = directoryURL
    }

    /// A second reference to an already staged file, with an independent
    /// lifetime.
    ///
    /// The transcript renders from the same files the composer holds, so
    /// clearing or replacing the composer's attachments deleted the images a
    /// finished answer was still showing. A hard link rather than a copy: the
    /// files are sealed read-only and can be up to 64 MB each.
    public func retain(_ attachment: AppImageAttachment) throws -> AppImageAttachment {
        let directory = directoryURL.appendingPathComponent(
            "retained", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(UUID().uuidString)
        guard link(attachment.fileURL.path, destination.path) == 0 else {
            throw VisionImageError.invalidSource(
                "could not retain \(attachment.displayName): errno \(errno)")
        }
        return AppImageAttachment(
            id: attachment.id,
            fileURL: destination,
            displayName: attachment.displayName,
            encodedBytes: attachment.encodedBytes,
            sha256: attachment.sha256)
    }

    public func stage(_ sourceURL: URL) throws -> AppImageAttachment {
        let accessed = sourceURL.startAccessingSecurityScopedResource()
        defer { if accessed { sourceURL.stopAccessingSecurityScopedResource() } }

        let source = sourceURL.standardizedFileURL
        let sourceFD = source.path.withCString {
            Darwin.open($0, O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW)
        }
        guard sourceFD >= 0 else {
            // The open is deliberately O_NOFOLLOW, so ELOOP here means the path
            // was a symlink and was refused on purpose. Without errno that is
            // indistinguishable from a missing file, a sandbox denial or a
            // descriptor exhaustion, and none of them can be acted on.
            throw VisionImageError.invalidSource(
                "could not open \(source.lastPathComponent): errno \(errno)")
        }
        defer { Darwin.close(sourceFD) }
        var sourceStatus = stat()
        guard fstat(sourceFD, &sourceStatus) == 0 else {
            throw VisionImageError.invalidSource(
                "could not read \(source.lastPathComponent): errno \(errno)")
        }
        guard (sourceStatus.st_mode & S_IFMT) == S_IFREG,
              sourceStatus.st_size >= 0,
              let encodedBytes = Int(exactly: sourceStatus.st_size) else {
            throw VisionImageError.invalidSource("a regular file is required")
        }
        let limit = VisionImageLimits().maximumEncodedBytes
        guard encodedBytes <= limit else {
            throw VisionImageError.sourceTooLarge(bytes: encodedBytes, limit: limit)
        }

        return try stage(
            displayName: source.lastPathComponent,
            encodedBytes: encodedBytes
        ) { destinationFD in
            var copied = 0
            var buffer = [UInt8](repeating: 0, count: 256 * 1_024)
            while copied < encodedBytes {
                let readCount = buffer.withUnsafeMutableBytes {
                    Darwin.read(
                        sourceFD, $0.baseAddress!,
                        min($0.count, encodedBytes - copied))
                }
                if readCount < 0, errno == EINTR { continue }
                // Only a zero-length read means the file shrank under us. A
                // negative one is a read error — EIO, EACCES, a network volume
                // going away — and saying the source changed sends the reader
                // after the wrong thing.
                if readCount < 0 {
                    throw VisionImageError.invalidSource(
                        "could not read \(source.lastPathComponent): errno \(errno)")
                }
                guard readCount > 0 else {
                    throw VisionImageError.invalidSource(
                        "source changed while copying")
                }
                try buffer.withUnsafeBytes {
                    try Self.writeAll(
                        UnsafeRawBufferPointer(
                            start: $0.baseAddress, count: readCount),
                        to: destinationFD)
                }
                copied += readCount
            }
        }
    }

    /// Stages bytes that never existed as a file — an image copied from another
    /// app arrives on the pasteboard as data, with no URL to open.
    public func stage(data: Data, displayName: String) throws -> AppImageAttachment {
        let limit = VisionImageLimits().maximumEncodedBytes
        guard !data.isEmpty else {
            throw VisionImageError.invalidSource("the pasteboard image was empty")
        }
        guard data.count <= limit else {
            throw VisionImageError.sourceTooLarge(bytes: data.count, limit: limit)
        }
        return try stage(displayName: displayName, encodedBytes: data.count) { fd in
            try data.withUnsafeBytes { try Self.writeAll($0, to: fd) }
        }
    }

    private func stage(
        displayName: String,
        encodedBytes: Int,
        writeContents: (Int32) throws -> Void
    ) throws -> AppImageAttachment {
        try FileManager.default.createDirectory(
            at: directoryURL, withIntermediateDirectories: true)
        let id = UUID()
        let destination = directoryURL.appendingPathComponent(id.uuidString)
        let temporary = directoryURL.appendingPathComponent(".\(id.uuidString).partial")
        let destinationFD = temporary.path.withCString {
            Darwin.open($0, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR)
        }
        guard destinationFD >= 0 else {
            throw VisionImageError.invalidSource("could not create attachment storage")
        }
        var didClose = false
        defer {
            if !didClose { Darwin.close(destinationFD) }
            try? FileManager.default.removeItem(at: temporary)
        }
        try writeContents(destinationFD)
        guard Darwin.fsync(destinationFD) == 0 else {
            throw VisionImageError.invalidSource("attachment sync failed")
        }
        // Before the guard, not after it: close() releases the descriptor even
        // when it reports EINTR or EIO, so throwing first left the `defer`
        // closing a number the kernel had already handed to something else —
        // the decode-service socket or a mapped weight file, staged off the
        // main actor while both are open.
        didClose = true
        guard Darwin.close(destinationFD) == 0 else {
            throw VisionImageError.invalidSource("attachment close failed")
        }
        try FileManager.default.moveItem(at: temporary, to: destination)
        guard chmod(destination.path, S_IRUSR) == 0 else {
            try? FileManager.default.removeItem(at: destination)
            throw VisionImageError.invalidSource("could not seal attachment")
        }
        let digest: String
        do {
            digest = try Sha256Verifier.hashFile(
                at: destination, chunkBytes: 256 * 1_024)
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
        return AppImageAttachment(
            id: id,
            fileURL: destination,
            displayName: displayName,
            encodedBytes: encodedBytes,
            sha256: digest)
    }

    private static func writeAll(
        _ bytes: UnsafeRawBufferPointer,
        to fileDescriptor: Int32
    ) throws {
        var written = 0
        while written < bytes.count {
            let count = Darwin.write(
                fileDescriptor,
                bytes.baseAddress!.advanced(by: written),
                bytes.count - written)
            if count < 0, errno == EINTR { continue }
            guard count > 0 else {
                throw VisionImageError.invalidSource("attachment copy failed")
            }
            written += count
        }
    }

    public func remove(_ attachment: AppImageAttachment) {
        try? FileManager.default.removeItem(at: attachment.fileURL)
    }

    public func removeAll() {
        try? FileManager.default.removeItem(at: directoryURL)
    }
}
