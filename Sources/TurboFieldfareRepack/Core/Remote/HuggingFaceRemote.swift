import Foundation

public struct RemoteFileInfo: Sendable, Hashable {
    public let filename: String
    public let resolvedCommit: String
    public let size: UInt64
    public let etag: String?
    public let xetHash: String?
    public let acceptsRanges: Bool
}

public struct TemporaryRangeFile: Sendable {
    public let path: String
    public let byteCount: UInt64
}

public struct HuggingFaceRemoteSource: Sendable {
    public let repoID: String
    public let requestedRevision: String
    public let resolvedCommit: String?
    public let token: String?
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let tempDirectory: String
    public let retryPolicy: RemoteRetryPolicy

    public init(repoID: String,
                requestedRevision: String,
                resolvedCommit: String? = nil,
                token: String? = nil,
                downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                baseURL: URL = URL(string: "https://huggingface.co")!,
                tempDirectory: String = NSTemporaryDirectory(),
                retryPolicy: RemoteRetryPolicy = RemoteRetryPolicy()) {
        self.repoID = repoID
        self.requestedRevision = requestedRevision
        self.resolvedCommit = resolvedCommit
        self.token = token
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.tempDirectory = tempDirectory
        self.retryPolicy = retryPolicy
    }

    public func pinned(commit: String) -> HuggingFaceRemoteSource {
        HuggingFaceRemoteSource(repoID: repoID,
                                requestedRevision: requestedRevision,
                                resolvedCommit: commit,
                                token: token,
                                downloadSession: downloadSession,
                                baseURL: baseURL,
                                tempDirectory: tempDirectory,
                                retryPolicy: retryPolicy)
    }

    public func fileURL(filename: String) throws -> URL {
        try validateRepoID(repoID)
        try validateFilename(filename)
        let revision = resolvedCommit ?? requestedRevision
        var url = baseURL
        for part in repoID.split(separator: "/") {
            url.appendPathComponent(String(part))
        }
        url.appendPathComponent("resolve")
        url.appendPathComponent(revision)
        for part in filename.split(separator: "/") {
            url.appendPathComponent(String(part))
        }
        return url
    }

    public func resolveFileInfo(filename: String,
                                audit: RepackAudit? = nil) async throws -> RemoteFileInfo {
        try await withRemoteRetries(retryPolicy,
                                    label: "resolveFileInfo:\(filename)",
                                    audit: audit) {
            try await resolveFileInfoOnce(filename: filename)
        }
    }

    private func resolveFileInfoOnce(filename: String) async throws -> RemoteFileInfo {
        let url = try fileURL(filename: filename)
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        applyHeaders(to: &request)
        let (_, response) = try await downloadSession.response(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw RepackError.remoteProtocolInvalid(detail: "missing HTTP response for \(filename)")
        }
        guard (200..<400).contains(http.statusCode) else {
            throw RepackError.remoteHTTPResponse(
                url: redactRemoteURL(url),
                status: http.statusCode,
                retryAfter: remoteHeader(http, "Retry-After"))
        }
        let commit = remoteHeader(http, "X-Repo-Commit") ?? resolvedCommit
        guard let commit, commit.count == 40 else {
            throw RepackError.remoteProtocolInvalid(detail: "missing full X-Repo-Commit for \(filename)")
        }
        let size = try remoteSize(http: http, filename: filename)
        let etag = remoteHeader(http, "X-Linked-ETag") ?? remoteHeader(http, "ETag")
        return RemoteFileInfo(filename: filename,
                              resolvedCommit: commit,
                              size: size,
                              etag: etag,
                              xetHash: remoteHeader(http, "X-Xet-Hash"),
                              acceptsRanges: (remoteHeader(http, "Accept-Ranges") ?? "")
                                  .lowercased().contains("bytes"))
    }

    public func fetchSmallFile(filename: String,
                               info: RemoteFileInfo,
                               capBytes: UInt64,
                               outputPath: String,
                               audit: RepackAudit? = nil) async throws {
        guard info.size <= capBytes else {
            throw RepackError.remoteFileTooLarge(path: filename, size: info.size, cap: capBytes)
        }
        try await withRemoteRetries(retryPolicy,
                                    label: "fetchSmallFile:\(filename)",
                                    audit: audit) {
            let tmp = try await downloadRangeToTempFileOnce(filename: filename,
                                                           info: info,
                                                           offset: 0,
                                                           length: Int(info.size))
            do {
                try Posix.mkdirP((outputPath as NSString).deletingLastPathComponent)
                let kind = try Posix.entryKind(outputPath)
                guard kind == .absent || kind == .regular else {
                    throw RepackError.installPathUnsafe(
                        path: outputPath,
                        detail: "metadata destination has the wrong entry type")
                }
                try Posix.rename(from: tmp.path, to: outputPath)
            } catch {
                try? FileManager.default.removeItem(atPath: tmp.path)
                throw error
            }
        }
    }

    public func downloadRangeToTempFile(filename: String,
                                        info: RemoteFileInfo,
                                        offset: UInt64,
                                        length: Int,
                                        targetPath: String? = nil,
                                        progress: @escaping @Sendable (UInt64) -> Void = { _ in },
                                        audit: RepackAudit? = nil) async throws -> TemporaryRangeFile {
        try await withRemoteRetries(retryPolicy,
                                    label: "downloadRangeToTempFile:\(filename)",
                                    audit: audit) {
            progress(0)
            return try await downloadRangeToTempFileOnce(
                filename: filename,
                info: info,
                offset: offset,
                length: length,
                targetPath: targetPath,
                progress: progress)
        }
    }

    private func downloadRangeToTempFileOnce(filename: String,
                                             info: RemoteFileInfo,
                                             offset: UInt64,
                                             length: Int,
                                             targetPath: String? = nil,
                                             progress: @escaping @Sendable (UInt64) -> Void = { _ in })
        async throws -> TemporaryRangeFile {
        guard length >= 0 else {
            throw RepackError.remoteProtocolInvalid(detail: "negative range length")
        }
        if length == 0 {
            let path = targetPath ?? (tempDirectory as NSString)
                .appendingPathComponent("turbofieldfare-range-\(UUID().uuidString).tmp")
            FileManager.default.createFile(atPath: path, contents: Data())
            return TemporaryRangeFile(path: path,
                                      byteCount: 0)
        }
        let end = offset + UInt64(length) - 1
        guard end < info.size else {
            throw RepackError.remoteProtocolInvalid(detail: "range \(offset)-\(end) exceeds \(info.filename)")
        }
        let url = try pinned(commit: info.resolvedCommit).fileURL(filename: filename)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("bytes=\(offset)-\(end)", forHTTPHeaderField: "Range")
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        applyHeaders(to: &request)
        try Posix.mkdirP(tempDirectory)
        let target = targetPath ?? (tempDirectory as NSString)
            .appendingPathComponent("turbofieldfare-range-\(UUID().uuidString).tmp")
        let result = try await downloadSession.transfer(
            request: request,
            targetPath: target,
            expectation: RemoteRangeExpectation(
                filename: filename,
                offset: offset,
                length: UInt64(length),
                totalSize: info.size,
                xetHash: info.xetHash),
            progress: progress)
        return TemporaryRangeFile(path: result.path,
                                  byteCount: result.byteCount)
    }

    private func applyHeaders(to request: inout URLRequest) {
        if let token, request.url?.host == baseURL.host {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
    }
}

private func validateRepoID(_ repoID: String) throws {
    let parts = repoID.split(separator: "/")
    guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && !$0.contains("..") }) else {
        throw RepackError.configurationInvalid(detail: "invalid repo id \(repoID)")
    }
}

private func validateFilename(_ filename: String) throws {
    guard !filename.isEmpty,
          !filename.hasPrefix("/"),
          !filename.contains(".."),
          !filename.contains("?"),
          !filename.contains("#") else {
        throw RepackError.configurationInvalid(detail: "invalid remote filename \(filename)")
    }
}

private func remoteSize(http: HTTPURLResponse, filename: String) throws -> UInt64 {
    if let linked = remoteHeader(http, "X-Linked-Size"), let size = UInt64(linked) {
        return size
    }
    if (200..<300).contains(http.statusCode),
       let contentLength = remoteHeader(http, "Content-Length"),
       let size = UInt64(contentLength) {
        return size
    }
    throw RepackError.remoteProtocolInvalid(detail: "missing remote size for \(filename)")
}
