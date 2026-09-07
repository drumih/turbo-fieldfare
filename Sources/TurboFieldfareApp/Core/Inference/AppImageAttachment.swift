import Darwin
import Foundation
import TurboFieldfare

/// A picture this session staged, which the attachment store owns and may
/// delete.
///
/// The composer holds these, a run's retained hard links are these, and
/// `AppImageAttachmentStore.remove` accepts nothing else. One type with an
/// `ownership` tag used to stand for both this and `StoredImage`, checked at
/// run time inside `remove` — and once a reopened conversation's turns carried
/// attachments pointing into the conversation store, opening any other chat
/// deleted the pictures out of the one just left. The files were gone from
/// disk, not merely absent from the window.
public struct StagedImage: Sendable, Equatable, Identifiable {
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

/// A picture the conversation store owns, kept for as long as the conversation
/// exists and released only by deleting it.
///
/// There is deliberately no path from one of these to `remove`: the compiler
/// refuses it, so a release path that finishes a turn cannot reach a stored
/// conversation's files however it is written.
public struct StoredImage: Sendable, Equatable, Identifiable {
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

/// A picture a turn is showing, and therefore who may delete it.
///
/// The transcript draws both kinds the same way, which is why they share a
/// type here; everything that deletes takes `StagedImage` and cannot be handed
/// one of these.
public enum ChatImage: Sendable, Equatable, Identifiable {
    case staged(StagedImage)
    case stored(StoredImage)

    public var id: UUID {
        switch self {
        case .staged(let image): return image.id
        case .stored(let image): return image.id
        }
    }

    public var fileURL: URL {
        switch self {
        case .staged(let image): return image.fileURL
        case .stored(let image): return image.fileURL
        }
    }

    public var displayName: String {
        switch self {
        case .staged(let image): return image.displayName
        case .stored(let image): return image.displayName
        }
    }

    public var encodedBytes: Int {
        switch self {
        case .staged(let image): return image.encodedBytes
        case .stored(let image): return image.encodedBytes
        }
    }

    /// The digest of the file the user attached. Also the thumbnail cache's
    /// key, so a picture already decoded for the composer is not decoded again
    /// for the transcript.
    public var sha256: String {
        switch self {
        case .staged(let image): return image.sha256
        case .stored(let image): return image.sha256
        }
    }

    /// The staged file behind this picture, or nil when the conversation store
    /// owns it. The only route from a drawn picture to a deletable one.
    public var staged: StagedImage? {
        switch self {
        case .staged(let image): return image
        case .stored: return nil
        }
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
    public func retain(_ attachment: StagedImage) throws -> StagedImage {
        let directory = directoryURL.appendingPathComponent(
            "retained", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(UUID().uuidString)
        guard link(attachment.fileURL.path, destination.path) == 0 else {
            throw VisionImageError.invalidSource(
                "could not retain \(attachment.displayName): errno \(errno)")
        }
        return StagedImage(
            id: attachment.id,
            fileURL: destination,
            displayName: attachment.displayName,
            encodedBytes: attachment.encodedBytes,
            sha256: attachment.sha256)
    }

    public func stage(_ sourceURL: URL) throws -> StagedImage {
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
    public func stage(data: Data, displayName: String) throws -> StagedImage {
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
    ) throws -> StagedImage {
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
        return StagedImage(
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

    /// Deletes a file this store staged, and nothing else.
    ///
    /// The type is the first condition and the compiler checks it: a
    /// `StoredImage` cannot be handed to this at all, where the tag it
    /// replaces was a run-time check on a value anybody could set.
    public func remove(_ attachment: StagedImage) {
        // This store's own directory, not the shared root: a store only
        // removes what it staged, and a test store points somewhere else
        // entirely.
        // Resolved the same way `contains` resolves, or the two answer
        // differently under a symlinked root (`/var` against `/private/var`):
        // a link `contains` accepted was refused here and never freed.
        let owned = directoryURL.standardizedFileURL.resolvingSymlinksInPath().path
        let file = attachment.fileURL.standardizedFileURL.resolvingSymlinksInPath().path
        guard file.hasPrefix(owned + "/") else { return }
        try? FileManager.default.removeItem(at: attachment.fileURL)
    }

    public func removeAll() {
        try? FileManager.default.removeItem(at: directoryURL)
    }
}
