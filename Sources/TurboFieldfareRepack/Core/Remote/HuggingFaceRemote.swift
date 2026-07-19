import Foundation

public struct RemoteFileInfo: Sendable, Hashable {
    public let filename: String
    public let resolvedCommit: String
    public let size: UInt64
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
    public let session: URLSession
    public let baseURL: URL
    public let tempDirectory: String
    public let retryPolicy: RemoteRetryPolicy

    public init(repoID: String,
                requestedRevision: String,
                resolvedCommit: String? = nil,
                token: String? = nil,
                session: URLSession = .shared,
                baseURL: URL = URL(string: "https://huggingface.co")!,
                tempDirectory: String = NSTemporaryDirectory(),
                retryPolicy: RemoteRetryPolicy = RemoteRetryPolicy()) {
        self.repoID = repoID
        self.requestedRevision = requestedRevision
        self.resolvedCommit = resolvedCommit
        self.token = token
        self.session = session
        self.baseURL = baseURL
        self.tempDirectory = tempDirectory
        self.retryPolicy = retryPolicy
    }

    public func pinned(commit: String) -> HuggingFaceRemoteSource {
        HuggingFaceRemoteSource(repoID: repoID,
                                requestedRevision: requestedRevision,
                                resolvedCommit: commit,
                                token: token,
                                session: session,
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
                                    audit: audit) {
            try await resolveFileInfoOnce(filename: filename)
        }
    }

    private func resolveFileInfoOnce(filename: String) async throws -> RemoteFileInfo {
        let url = try fileURL(filename: filename)
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        applyHeaders(to: &request)
        let (_, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw RepackError.remoteProtocolInvalid(detail: "missing HTTP response for \(filename)")
        }
        guard (200..<400).contains(http.statusCode) else {
            throw RepackError.remoteHTTPStatus(url: redact(url), status: http.statusCode)
        }
        let commit = header(http, "X-Repo-Commit") ?? resolvedCommit
        guard let commit, commit.count == 40 else {
            throw RepackError.remoteProtocolInvalid(detail: "missing full X-Repo-Commit for \(filename)")
        }
        let size = try remoteSize(http: http, filename: filename)
        return RemoteFileInfo(filename: filename,
                              resolvedCommit: commit,
                              size: size,
                              acceptsRanges: (header(http, "Accept-Ranges") ?? "").lowercased().contains("bytes"))
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
                                    audit: audit) {
            let tmp = try await downloadRangeToTempFileOnce(filename: filename,
                                                           info: info,
                                                           offset: 0,
                                                           length: Int(info.size))
            do {
                try Posix.mkdirP((outputPath as NSString).deletingLastPathComponent)
                if FileManager.default.fileExists(atPath: outputPath) {
                    try FileManager.default.removeItem(atPath: outputPath)
                }
                try FileManager.default.moveItem(atPath: tmp.path, toPath: outputPath)
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
                                        audit: RepackAudit? = nil) async throws -> TemporaryRangeFile {
        try await withRemoteRetries(retryPolicy,
                                    audit: audit) {
            try await downloadRangeToTempFileOnce(filename: filename,
                                                  info: info,
                                                  offset: offset,
                                                  length: length)
        }
    }

    private func downloadRangeToTempFileOnce(filename: String,
                                             info: RemoteFileInfo,
                                             offset: UInt64,
                                             length: Int) async throws -> TemporaryRangeFile {
        guard length >= 0 else {
            throw RepackError.remoteProtocolInvalid(detail: "negative range length")
        }
        if length == 0 {
            let path = (tempDirectory as NSString)
                .appendingPathComponent("turbofieldfare-range-\(UUID().uuidString).tmp")
            FileManager.default.createFile(atPath: path, contents: Data())
            return TemporaryRangeFile(path: path, byteCount: 0)
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
        let (downloadedURL, response) = try await session.download(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw RepackError.remoteProtocolInvalid(detail: "missing HTTP response for range \(filename)")
        }
        guard http.statusCode == 206 else {
            throw RepackError.remoteHTTPStatus(url: redact(url), status: http.statusCode)
        }
        try validateRangeResponse(http: http,
                                  filename: filename,
                                  offset: offset,
                                  end: end,
                                  total: info.size,
                                  length: UInt64(length))

        try Posix.mkdirP(tempDirectory)
        let target = (tempDirectory as NSString)
            .appendingPathComponent("turbofieldfare-range-\(UUID().uuidString).tmp")
        if FileManager.default.fileExists(atPath: target) {
            try FileManager.default.removeItem(atPath: target)
        }
        do {
            try FileManager.default.moveItem(at: downloadedURL, to: URL(fileURLWithPath: target))
        } catch {
            try? FileManager.default.removeItem(at: downloadedURL)
            throw error
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: target)
        let actual = (attrs[.size] as? NSNumber)?.uint64Value ?? 0
        guard actual == UInt64(length) else {
            try? FileManager.default.removeItem(atPath: target)
            throw RepackError.remoteProtocolInvalid(detail:
                "range \(filename) wrote \(actual), expected \(length)")
        }
        return TemporaryRangeFile(path: target, byteCount: actual)
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

private func header(_ response: HTTPURLResponse, _ name: String) -> String? {
    for (k, v) in response.allHeaderFields {
        if String(describing: k).caseInsensitiveCompare(name) == .orderedSame {
            return String(describing: v)
        }
    }
    return nil
}

private func remoteSize(http: HTTPURLResponse, filename: String) throws -> UInt64 {
    if let linked = header(http, "X-Linked-Size"), let size = UInt64(linked) {
        return size
    }
    if let contentLength = header(http, "Content-Length"), let size = UInt64(contentLength) {
        return size
    }
    throw RepackError.remoteProtocolInvalid(detail: "missing remote size for \(filename)")
}

private func validateRangeResponse(http: HTTPURLResponse,
                                   filename: String,
                                   offset: UInt64,
                                   end: UInt64,
                                   total: UInt64,
                                   length: UInt64) throws {
    guard (header(http, "Content-Encoding") ?? "identity").lowercased() == "identity" else {
        throw RepackError.remoteProtocolInvalid(detail: "compressed range response for \(filename)")
    }
    guard let contentLength = header(http, "Content-Length"),
          UInt64(contentLength) == length else {
        throw RepackError.remoteProtocolInvalid(detail: "wrong Content-Length for \(filename)")
    }
    guard let range = header(http, "Content-Range") else {
        throw RepackError.remoteProtocolInvalid(detail: "missing Content-Range for \(filename)")
    }
    let prefix = "bytes "
    guard range.hasPrefix(prefix) else {
        throw RepackError.remoteProtocolInvalid(detail: "malformed Content-Range \(range)")
    }
    let rest = range.dropFirst(prefix.count)
    let parts = rest.split(separator: "/", maxSplits: 1)
    guard parts.count == 2,
          UInt64(parts[1]) == total else {
        throw RepackError.remoteProtocolInvalid(detail: "wrong Content-Range total \(range)")
    }
    let bounds = parts[0].split(separator: "-", maxSplits: 1)
    guard bounds.count == 2,
          UInt64(bounds[0]) == offset,
          UInt64(bounds[1]) == end else {
        throw RepackError.remoteProtocolInvalid(detail: "wrong Content-Range bounds \(range)")
    }
}

private func redact(_ url: URL) -> String {
    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
    components?.query = nil
    return components?.url?.absoluteString ?? url.absoluteString
}
