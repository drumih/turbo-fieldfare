import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

public enum ConversationStoreError: Error, Equatable, CustomStringConvertible {
    case readOnly(reason: String)
    case notFound(id: UUID)
    case lockUnavailable(path: String, errno: Int32)
    case headerMismatch(directory: String, header: UUID)
    case missingHeader(id: UUID)
    case tornRecord(id: UUID, line: Int)
    case newerVersion(id: UUID, version: Int)
    case writeFailed(path: String, reason: String)
    /// The transcript carries none of the creation facts and there is no
    /// readable meta to lift them out of, so there is no identity to check a
    /// replay against.
    case missingOrigin(id: UUID)

    public var description: String {
        switch self {
        case .readOnly(let reason):
            return "the conversation store is read-only: \(reason)"
        case .notFound(let id):
            return "no conversation \(id.uuidString) in the store"
        case .lockUnavailable(let path, let code):
            return "could not open the store lock at \(path): errno \(code)"
        case .headerMismatch(let directory, let header):
            return "conversation directory \(directory) holds a transcript "
                + "headed \(header.uuidString)"
        case .missingHeader(let id):
            return "conversation \(id.uuidString) has a transcript with no header"
        case .tornRecord(let id, let line):
            return "conversation \(id.uuidString) has an unreadable record at "
                + "line \(line); only a torn final line is recoverable"
        case .newerVersion(let id, let version):
            return "conversation \(id.uuidString) was written by a newer build "
                + "(version \(version))"
        case .writeFailed(let path, let reason):
            return "could not write \(path): \(reason)"
        case .missingOrigin(let id):
            return "conversation \(id.uuidString) records neither the model it "
                + "was written against nor a settings snapshot, and its "
                + "conversation.json cannot be read"
        }
    }
}

/// What `open` found, beside the records.
public struct ConversationOpenResult: Sendable {
    public let meta: ConversationMeta
    public let records: [TranscriptRecord]
    /// A newer build wrote this conversation. It is readable and must never be
    /// rewritten: an older build replacing a newer file with its own schema is
    /// the migration loss every client that has done it still hears about.
    public let isReadOnly: Bool
    /// A final line that was cut off mid-write, dropped and truncated away
    /// before the append handle opens. One lost turn, never a lost file.
    public let droppedTornFinalLine: Bool
    /// Why the cached meta had to be rebuilt from the records, or nil when it
    /// was already what the projection says.
    ///
    /// Reported rather than discarded: a `conversation.json` that could not be
    /// read is a real fault of the store even though the conversation survives
    /// it, and silently healing one leaves nobody able to say it ever happened.
    /// An ordinary disagreement — a stale copy, a hand edit — is not reported;
    /// it is simply replaced.
    public let metaRebuildReason: String?

    public init(meta: ConversationMeta,
                records: [TranscriptRecord],
                isReadOnly: Bool,
                droppedTornFinalLine: Bool,
                metaRebuildReason: String? = nil) {
        self.meta = meta
        self.records = records
        self.isReadOnly = isReadOnly
        self.droppedTornFinalLine = droppedTornFinalLine
        self.metaRebuildReason = metaRebuildReason
    }
}

struct ConversationLegacyTrashOperations: Sendable {
    let list: @Sendable (String) throws -> [String]
    let remove: @Sendable (URL) throws -> Void
}

struct ConversationStoreAppendOperations: Sendable {
    let beforeBatch: @Sendable () async throws -> Void
}

struct ConversationStoreListOperations: Sendable {
    let beforeList: @Sendable () async -> Void
}

/// The conversation store: a directory of plain-text conversations beside the
/// settings and the model.
///
/// One directory per conversation, because that makes delete an `rm -rf`, makes
/// a corrupt conversation one skippable directory rather than a broken store,
/// and lets images be sidecar files instead of base64 inside the transcript.
/// `conversation.json` is small and rewritten atomically; `transcript.jsonl` is
/// append-only and written one whole line at a time, so the worst a crash can
/// cost is the line that was in flight.
public actor ConversationStore {
    public let rootURL: URL
    private let fileManager: FileManager
    private let legacyTrashOperations: ConversationLegacyTrashOperations?
    private let appendOperations: ConversationStoreAppendOperations?
    private let listOperations: ConversationStoreListOperations?
    private var lock: ConversationStoreLock?
    /// Conversations a newer build wrote. Held so an append can refuse before
    /// it touches the file rather than after.
    private var readOnlyConversations: Set<UUID> = []

    public init(rootURL: URL, fileManager: FileManager = .default) {
        self.rootURL = rootURL
        self.fileManager = fileManager
        self.legacyTrashOperations = nil
        self.appendOperations = nil
        self.listOperations = nil
    }

    init(rootURL: URL,
         fileManager: FileManager = .default,
         legacyTrashOperations: ConversationLegacyTrashOperations) {
        self.rootURL = rootURL
        self.fileManager = fileManager
        self.legacyTrashOperations = legacyTrashOperations
        self.appendOperations = nil
        self.listOperations = nil
    }

    init(rootURL: URL,
         fileManager: FileManager = .default,
         appendOperations: ConversationStoreAppendOperations) {
        self.rootURL = rootURL
        self.fileManager = fileManager
        self.legacyTrashOperations = nil
        self.appendOperations = appendOperations
        self.listOperations = nil
    }

    init(rootURL: URL,
         fileManager: FileManager = .default,
         listOperations: ConversationStoreListOperations) {
        self.rootURL = rootURL
        self.fileManager = fileManager
        self.legacyTrashOperations = nil
        self.appendOperations = nil
        self.listOperations = listOperations
    }

    /// Takes the writer lock. A store that could not take it still lists and
    /// opens conversations; it just cannot change them.
    public func activate() throws {
        guard lock == nil else { return }
        lock = try ConversationStoreLock.acquire(storeRoot: rootURL)
        do {
            try discardLegacyTrash()
        } catch {
            lock = nil
            throw error
        }
    }

    /// Removes the `.trash` an older build left behind.
    ///
    /// Delete used to park a conversation there so an undo could restore it,
    /// and nothing ever emptied it — so every chat anyone deleted is still on
    /// disk, transcript and images, with the code that could have collected it
    /// now gone. Deleting means deleting, including retroactively. Once: after
    /// this the directory does not exist and there is nothing to do.
    private func discardLegacyTrash() throws {
        // Only the instance holding the writer lock. A second window is here to
        // read, and must not delete anything on the first one's behalf.
        guard lock != nil else { return }
        let trash = rootURL.appendingPathComponent(".trash", isDirectory: true)
        guard fileManager.fileExists(atPath: trash.path) else { return }
        let contents: [String]
        if let legacyTrashOperations {
            contents = try legacyTrashOperations.list(trash.path)
            try legacyTrashOperations.remove(trash)
        } else {
            contents = try fileManager.contentsOfDirectory(atPath: trash.path)
            try fileManager.removeItem(at: trash)
        }
        discardedLegacyTrashCount = contents.count
    }

    /// How many conversations the legacy `.trash` held when it was removed, so
    /// the app can say so rather than silently reclaiming someone's chats.
    public private(set) var discardedLegacyTrashCount = 0

    /// Tries again for a lock this store could not take at activation, and says
    /// why when it could not for a reason worth hearing.
    ///
    /// Never releases one it already holds. `EWOULDBLOCK` is the ordinary
    /// answer while another window is open, and returns nil; anything else — a
    /// read-only volume, a permission change, a store root that is now a file —
    /// used to read as "another instance holds it" for the rest of the session,
    /// with the window offering a retry that could never succeed.
    @discardableResult
    public func retryLock() -> String? {
        guard lock == nil else { return nil }
        do {
            lock = try ConversationStoreLock.acquire(storeRoot: rootURL)
        } catch {
            return "\(error)"
        }
        if lock != nil {
            do {
                try discardLegacyTrash()
            } catch {
                lock = nil
                return "\(error)"
            }
        }
        return nil
    }

    public var isReadOnly: Bool { lock == nil }

    public func directoryURL(for id: UUID) -> URL {
        rootURL.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    // MARK: - Listing

    /// Every conversation the store holds, newest first.
    ///
    /// Reads only `conversation.json`: a sidebar that parsed transcripts would
    /// scale with the size of the conversations rather than their number. A
    /// directory whose meta will not decode is skipped rather than failing the
    /// list, because one bad conversation must not hide the rest.
    public func list() async throws -> [ConversationMeta] {
        await listOperations?.beforeList()
        skippedByLastList = []
        skipReasons = [:]
        guard fileManager.fileExists(atPath: rootURL.path) else { return [] }
        let entries = try fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles])
        var result: [ConversationMeta] = []
        var skipped: [UUID] = []
        var reasons: [UUID: String] = [:]
        for entry in entries {
            guard let id = UUID(uuidString: entry.lastPathComponent) else { continue }
            // A directory whose meta names a different conversation is not that
            // conversation: it is a copy someone made, or a restore from a
            // recovery tool. Listed, its id collides with the original's, and
            // the sidebar — which keys rows by id — drew one of the two as a
            // blank row taking up space and doing nothing.
            var reason = "its conversation.json names another conversation"
            do {
                let meta = try readMeta(id: id)
                if meta.id == id {
                    result.append(meta)
                    continue
                }
            } catch {
                reason = "\(error)"
            }
            // The cache is missing, damaged, or names somebody else. It is only
            // a cache, so the records can rebuild it — one parse for this
            // directory, and none for the healthy ones. Only under the writer
            // lock: a second window must not repair the first one's store, and
            // a conversation whose transcript is genuinely unreadable still
            // ends up skipped.
            //
            // Skipping is right — one unreadable conversation must not hide the
            // rest — but it is not nothing, and a chat that disappears with no
            // trace is worse than one that shows as broken. The reason travels
            // with the id now, so the sidebar's "could not be read" can say what
            // stopped it instead of discarding the cause a second time.
            guard lock != nil else {
                skipped.append(id)
                reasons[id] = reason + "; this window cannot repair it because "
                    + "another one holds the writer lock"
                continue
            }
            do {
                let rebuilt = try open(id: id).meta
                guard rebuilt.id == id else {
                    skipped.append(id)
                    reasons[id] = "its records are headed by another conversation"
                    continue
                }
                result.append(rebuilt)
            } catch {
                skipped.append(id)
                reasons[id] = "\(error)"
            }
        }
        skippedByLastList = skipped
        skipReasons = reasons
        return result.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// The conversations the last `list` could not read, and therefore left
    /// out. Reported rather than counted at the call site, because only this
    /// knows which directories it walked past.
    public private(set) var skippedByLastList: [UUID] = []

    /// Why each of them was left out.
    ///
    /// The sidebar counts the rows it could not read; without this it could say
    /// how many and nothing about why, which is the one question the person
    /// looking at that count has.
    public private(set) var skipReasons: [UUID: String] = [:]

    // MARK: - Lifecycle

    /// Starts a conversation: a directory, a header carrying what it was made
    /// under, and the name it starts with.
    ///
    /// `title` is written as a `title` record rather than into the meta. The
    /// meta is a projection of the records, so a name assigned only there would
    /// be gone at the next open — and the row would read "New Chat" for the
    /// length of the first reply, which is exactly when the user is looking at
    /// it.
    public func create(id: UUID = UUID(),
                       title: String,
                       identity: ConversationIdentity,
                       session: ConversationSessionSettings,
                       sampling: ConversationSampling,
                       now: Date = Date()) throws -> ConversationMeta {
        try requireWritable()
        let directory = directoryURL(for: id)
        try fileManager.createDirectory(
            at: directory.appendingPathComponent("images", isDirectory: true),
            withIntermediateDirectories: true)
        var records: [TranscriptRecord] = [
            .header(ConversationHeaderRecord(
                id: id, createdAt: now, identity: identity, session: session,
                sampling: sampling)),
        ]
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            records.append(.title(ConversationTitleRecord(
                title: trimmed, source: .firstMessage, at: now)))
        }
        for record in records { try appendLine(record, to: id) }
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        try writeMeta(meta)
        return meta
    }

    /// Reads a conversation's meta and every transcript record.
    ///
    /// A final line cut off mid-write is dropped and truncated away, so the
    /// append that follows starts on a clean boundary. A line that fails to
    /// parse anywhere else is an error: it means the file was written by two
    /// writers or damaged in place, and appending to it would bury the evidence
    /// under more records.
    public func open(id: UUID) throws -> ConversationOpenResult {
        try open(id: id, recoverIncompleteExchange: true)
    }

    private func open(id: UUID,
                      recoverIncompleteExchange: Bool) throws -> ConversationOpenResult {
        // The cached meta is read for two things only: the version stamp, which
        // decides read-only, and the creation facts of a store written before
        // they moved into the header. Everything else comes from the records.
        // A failure here is not fatal and not silent either — it is the reason
        // the cache is about to be rebuilt, and it travels with the result.
        var legacy: ConversationMeta?
        var metaFailure: String?
        do {
            legacy = try readMeta(id: id)
        } catch {
            metaFailure = "\(error)"
        }
        // `readMeta` has already recorded a newer major it could not decode
        // past the version stamp, and that one must not be rewritten either.
        var isReadOnly = readOnlyConversations.contains(id)
            || (legacy?.version ?? ConversationMeta.currentVersion)
                > ConversationMeta.currentVersion
        if isReadOnly { readOnlyConversations.insert(id) }

        let transcript = transcriptURL(for: id)
        guard let data = fileManager.contents(atPath: transcript.path) else {
            throw ConversationStoreError.missingHeader(id: id)
        }

        var droppedTornFinalLine = false
        var complete = data
        if let lastNewline = data.lastIndex(of: 0x0A) {
            if lastNewline + 1 < data.count {
                complete = data.prefix(upTo: lastNewline + 1)
                droppedTornFinalLine = true
            }
        } else if !data.isEmpty {
            complete = Data()
            droppedTornFinalLine = true
        }

        struct PositionedRecord {
            let record: TranscriptRecord
            let byteOffset: Int
        }

        var positioned: [PositionedRecord] = []
        var line = 0
        let decoder = Self.makeDecoder()
        var lineStart = complete.startIndex
        while lineStart < complete.endIndex {
            guard let newline = complete[lineStart...].firstIndex(of: 0x0A) else {
                break
            }
            let chunk = complete[lineStart..<newline]
            defer { lineStart = complete.index(after: newline) }
            guard !chunk.isEmpty else { continue }
            line += 1
            guard let record = try? decoder.decode(
                TranscriptRecord.self, from: Data(chunk)) else {
                throw ConversationStoreError.tornRecord(id: id, line: line)
            }
            positioned.append(PositionedRecord(
                record: record,
                byteOffset: complete.distance(from: complete.startIndex, to: lineStart)))
        }

        // Metadata is disposable. Its loss must not let this build repair or
        // append to a newer writer's authoritative transcript.
        if case .header(let header)? = positioned.first?.record,
           header.version > ConversationMeta.currentVersion {
            isReadOnly = true
            readOnlyConversations.insert(id)
        }

        var incompleteExchangeOffset: Int?
        var pendingUserIndex: Int?
        for index in positioned.indices {
            guard case .turn(let turn) = positioned[index].record else { continue }
            switch turn.role {
            case .user:
                if pendingUserIndex == nil { pendingUserIndex = index }
            case .assistant:
                pendingUserIndex = nil
            }
        }
        if recoverIncompleteExchange, let pendingUserIndex {
            incompleteExchangeOffset = positioned[pendingUserIndex].byteOffset
            positioned.removeSubrange(pendingUserIndex...)
        }

        var records = positioned.map(\.record)
        if !isReadOnly, lock != nil {
            if let incompleteExchangeOffset {
                try truncate(transcript, to: incompleteExchangeOffset)
            } else if droppedTornFinalLine {
                try truncate(transcript, to: complete.count)
            }
        }

        let isWritable = !isReadOnly && lock != nil
        // A transcript that predates the header's creation facts gets them
        // once, lifted out of the meta that still has them. After this the
        // conversation projects from its own records alone, and the cache it
        // came from can be deleted without losing anything.
        if isWritable,
           let hoisted = ConversationMetaProjection.hoist(
            records: records, legacy: legacy) {
            try appendLine(.origin(hoisted), to: id)
            records.append(.origin(hoisted))
        }

        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: legacy)
        // Healed under the writer lock only. A read-only store — a second
        // window, or a conversation a newer build wrote — leaves the bytes
        // exactly as their owner wrote them.
        if isWritable, legacy != meta {
            try writeMeta(meta)
        }
        return ConversationOpenResult(
            meta: meta, records: records, isReadOnly: isReadOnly,
            droppedTornFinalLine: droppedTornFinalLine,
            metaRebuildReason: metaFailure)
    }

    /// Appends records and returns the meta they project to.
    ///
    /// A batch rather than one call per record because the meta is written once
    /// for the group: a turn is two records, and rebuilding the cache between
    /// them would publish a conversation with a question and no answer. Each
    /// record is still its own `write(2)` and `fsync`, so a crash still costs
    /// at most the line in flight.
    @discardableResult
    public func append(_ records: [TranscriptRecord],
                       to id: UUID) async throws -> ConversationMeta {
        try requireWritable()
        try requireConversationWritable(id)
        try await appendOperations?.beforeBatch()
        try Task.checkCancellation()
        // Read first, with the repair every open makes: the torn line a crash
        // left. Only `open` used to make it, and a rename or a turn on a
        // conversation that had been listed but not opened since the crash
        // landed its record behind the fragment — a torn record in the middle,
        // which nothing recovers, so the conversation could never be opened
        // again. Not the unanswered-question repair: a caller may append the
        // answer on its own, and that repair would truncate the question it
        // is about to answer.
        let opened = try open(id: id, recoverIncompleteExchange: false)
        // Opening can discover a newer version that was not in the cache when
        // this append began. Its records and metadata must both stay untouched.
        try requireConversationWritable(id)
        for record in records {
            try Task.checkCancellation()
            try appendLine(record, to: id)
        }
        // Projected from the records just read plus the ones just written,
        // so the transcript is decoded once per append rather than read again
        // for the cache.
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: opened.records + records, legacy: opened.meta)
        try writeMeta(meta)
        return meta
    }

    /// Renames a conversation without touching `updatedAt`.
    ///
    /// The list is ordered by when a conversation was last *used*, and renaming
    /// one is not using it: bumping the timestamp jumped the row to the top of
    /// the sidebar, which is not what anyone tidying up their titles is asking
    /// for. The title record carries its own timestamp, so nothing is lost —
    /// and because the meta is projected from the records, the rule holds by
    /// construction rather than by remembering not to assign the field.
    public func rename(id: UUID, to title: String, now: Date = Date()) async throws {
        try requireWritable()
        try requireConversationWritable(id)
        // A rename reaches a conversation the sidebar listed but nobody opened
        // since a crash. The question that crash left unanswered is dropped
        // now, before the title lands: behind it, the next recovering open
        // truncated the question and the title together, and the rename
        // silently reverted.
        _ = try open(id: id, recoverIncompleteExchange: true)
        try await append(
            [.title(ConversationTitleRecord(title: title, source: .user, at: now))],
            to: id)
    }

    // MARK: - Delete

    /// Deletes a conversation and everything in it, for good.
    ///
    /// No staging area. A conversation used to be moved to `.trash` so an undo
    /// could bring it back, which meant a deleted chat kept its transcript and
    /// every image it held for the life of the store and nothing ever collected
    /// them. Delete means delete; the protection against deleting the wrong one
    /// is asking first, which is where it belongs.
    public func delete(id: UUID) throws {
        try requireWritable()
        let directory = directoryURL(for: id)
        guard fileManager.fileExists(atPath: directory.path) else {
            throw ConversationStoreError.notFound(id: id)
        }
        try fileManager.removeItem(at: directory)
        readOnlyConversations.remove(id)
    }

    /// Removes image files no surviving turn refers to.
    ///
    /// A turn that was rewound after its images were written leaves them
    /// behind, and nothing else would ever collect them: the transcript is
    /// append-only and the files are named by digest, so they are invisible to
    /// every other pass.
    @discardableResult
    public func sweepOrphanImages(in id: UUID) throws -> Int {
        try requireWritable()
        try requireConversationWritable(id)
        let opened = try open(id: id)
        try requireConversationWritable(id)
        var referenced: Set<String> = []
        for record in opened.records {
            let turn: ConversationTurnRecord
            switch record {
            case .turn(let value), .partial(let value): turn = value
            default: continue
            }
            for image in turn.images {
                referenced.insert(image.pixelsFile)
                referenced.insert(image.thumbnailFile)
            }
        }
        let imagesDirectory = directoryURL(for: id)
            .appendingPathComponent("images", isDirectory: true)
        guard fileManager.fileExists(atPath: imagesDirectory.path) else { return 0 }
        var removed = 0
        for file in try fileManager.contentsOfDirectory(
            at: imagesDirectory, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]) {
            let relative = "images/" + file.lastPathComponent
            guard !referenced.contains(relative) else { continue }
            try fileManager.removeItem(at: file)
            removed += 1
        }
        return removed
    }

    // MARK: - Files

    public func metaURL(for id: UUID) -> URL {
        directoryURL(for: id).appendingPathComponent(
            ConversationMeta.fileName, isDirectory: false)
    }

    public func transcriptURL(for id: UUID) -> URL {
        directoryURL(for: id).appendingPathComponent(
            "transcript.jsonl", isDirectory: false)
    }

    public func imagesURL(for id: UUID) -> URL {
        directoryURL(for: id).appendingPathComponent("images", isDirectory: true)
    }

    // MARK: - Internals

    private func requireWritable() throws {
        guard lock != nil else {
            throw ConversationStoreError.readOnly(
                reason: "another instance of the app holds the writer lock")
        }
    }

    private func requireConversationWritable(_ id: UUID) throws {
        guard !readOnlyConversations.contains(id) else {
            throw ConversationStoreError.readOnly(
                reason: "conversation \(id.uuidString) was written by a newer build")
        }
    }

    /// Reads the version on its own before decoding the rest.
    ///
    /// The same shape `MacAppSettings` uses, and for the same reason: every
    /// other field decodes strictly, so a newer build that moved any of them
    /// would throw here — and a throw is what an older build turns into
    /// "replace the file with my defaults".
    private func readMeta(id: UUID) throws -> ConversationMeta {
        let url = metaURL(for: id)
        guard let data = fileManager.contents(atPath: url.path) else {
            throw ConversationStoreError.notFound(id: id)
        }
        let decoder = Self.makeDecoder()
        let stamp = try decoder.decode(VersionStamp.self, from: data)
        guard stamp.version <= ConversationMeta.currentVersion else {
            // Enough to list and open it read-only. Decoding the rest could
            // throw on a field this build does not know, and the conversation
            // would then be invisible instead of merely uneditable.
            readOnlyConversations.insert(id)
            var placeholder = try? decoder.decode(ConversationMeta.self, from: data)
            placeholder?.version = stamp.version
            guard let placeholder else {
                throw ConversationStoreError.newerVersion(
                    id: id, version: stamp.version)
            }
            return placeholder
        }
        return try decoder.decode(ConversationMeta.self, from: data)
    }

    private func writeMeta(_ meta: ConversationMeta) throws {
        let encoder = Self.makeEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(meta)
        data.append(0x0A)
        let url = metaURL(for: meta.id)
        // Temp file, fsync, rename, fsync the parent. A half-written
        // conversation.json is a conversation the sidebar cannot list, and the
        // repack target already owns the only implementation of this the tree
        // needs.
        try Posix.atomicWrite(
            data, to: url.path,
            durableIn: url.deletingLastPathComponent().path)
    }

    /// One `write(2)` of the whole line plus its newline, then `fsync`.
    ///
    /// Whole-line writes are what keep a crash to one lost turn: a partial line
    /// can only ever be the last one, which `open` drops and truncates. Two
    /// writes per record would let a crash split a record in the middle of the
    /// file, which is not recoverable.
    private func appendLine(_ record: TranscriptRecord, to id: UUID) throws {
        let url = transcriptURL(for: id)
        var data = try Self.makeEncoder().encode(record)
        data.append(0x0A)
        let descriptor = Darwin.open(url.path, O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw ConversationStoreError.writeFailed(
                path: url.path, reason: "open failed with errno \(errno)")
        }
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw ConversationStoreError.writeFailed(
                path: url.path, reason: "fstat failed with errno \(errno)")
        }
        let lengthBefore = status.st_size
        let written = data.withUnsafeBytes { raw -> Int in
            guard let base = raw.baseAddress else { return 0 }
            return Darwin.write(descriptor, base, raw.count)
        }
        guard written == data.count else {
            // A short write is a torn line that is not the last one for long:
            // the next append would land behind it. Cut back to the record
            // boundary so the file holds whole lines or nothing new.
            _ = Darwin.ftruncate(descriptor, lengthBefore)
            throw ConversationStoreError.writeFailed(
                path: url.path,
                reason: "wrote \(written) of \(data.count) bytes")
        }
        guard Darwin.fsync(descriptor) == 0 else {
            throw ConversationStoreError.writeFailed(
                path: url.path, reason: "fsync failed with errno \(errno)")
        }
    }

    private func truncate(_ url: URL, to length: Int) throws {
        let descriptor = Darwin.open(url.path, O_WRONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw ConversationStoreError.writeFailed(
                path: url.path, reason: "open failed with errno \(errno)")
        }
        defer { close(descriptor) }
        guard Darwin.ftruncate(descriptor, off_t(length)) == 0 else {
            throw ConversationStoreError.writeFailed(
                path: url.path, reason: "ftruncate failed with errno \(errno)")
        }
        // A repair that did not reach the disk is a repair a crash undoes: the
        // torn line would be back under the record appended after it.
        guard Darwin.fsync(descriptor) == 0 else {
            throw ConversationStoreError.writeFailed(
                path: url.path, reason: "fsync failed with errno \(errno)")
        }
    }

    private struct VersionStamp: Decodable {
        let version: Int
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
