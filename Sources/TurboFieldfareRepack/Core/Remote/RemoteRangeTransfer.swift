import Darwin
import Foundation

public struct RemoteRangeExpectation: Sendable, Equatable {
    public let filename: String
    public let offset: UInt64
    public let length: UInt64
    public let totalSize: UInt64
    public let xetHash: String?

    public init(filename: String,
                offset: UInt64,
                length: UInt64,
                totalSize: UInt64,
                xetHash: String?) {
        self.filename = filename
        self.offset = offset
        self.length = length
        self.totalSize = totalSize
        self.xetHash = xetHash
    }
}

public struct RemoteRangeTransferResult: Sendable, Equatable {
    public let path: String
    public let byteCount: UInt64
}

public enum RemoteRangeTransfer {
    public static func run(configuration: URLSessionConfiguration,
                           request: URLRequest,
                           targetPath: String,
                           expectation: RemoteRangeExpectation,
                           maximumRedirects: Int,
                           progress: @escaping @Sendable (UInt64) -> Void = { _ in })
        async throws -> RemoteRangeTransferResult {
        try Posix.mkdirP((targetPath as NSString).deletingLastPathComponent)
        let delegate = try RemoteRangeTransferDelegate(
            targetPath: targetPath,
            expectation: expectation,
            maximumRedirects: maximumRedirects,
            originalRequest: request,
            progress: progress)
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .utility
        let session = URLSession(configuration: configuration,
                                 delegate: delegate,
                                 delegateQueue: queue)
        return try await withTaskCancellationHandler {
            defer { session.finishTasksAndInvalidate() }
            return try await delegate.start(session: session, request: request)
        } onCancel: {
            delegate.cancel()
        }
    }
}

private final class RemoteRangeTransferDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let targetPath: String
    private let expectation: RemoteRangeExpectation
    private let fd: Int32
    private let progress: @Sendable (UInt64) -> Void
    private let lock = NSLock()

    private var continuation: CheckedContinuation<RemoteRangeTransferResult, Error>?
    private var task: URLSessionDataTask?
    private var terminal = false
    private var rejection: Error?
    private var receivedBytes: UInt64 = 0
    private var redirectPolicy: RemoteRedirectPolicy
    private var responseAccepted = false
    private var lastProgressNanoseconds: UInt64?

    init(targetPath: String,
         expectation: RemoteRangeExpectation,
         maximumRedirects: Int,
         originalRequest: URLRequest,
        progress: @escaping @Sendable (UInt64) -> Void) throws {
        self.targetPath = targetPath
        self.expectation = expectation
        self.redirectPolicy = RemoteRedirectPolicy(
            originalRequest: originalRequest,
            maximumRedirects: maximumRedirects)
        self.progress = progress
        let fd = open(targetPath, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            throw RepackError.fileOpenFailed(path: targetPath, errno: errno)
        }
        self.fd = fd
    }

    func start(session: URLSession, request: URLRequest) async throws -> RemoteRangeTransferResult {
        try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            self.continuation = continuation
            let task = session.dataTask(with: request)
            self.task = task
            let shouldCancel = terminal || rejection != nil
            lock.unlock()
            if shouldCancel {
                task.cancel()
            } else {
                task.resume()
            }
        }
    }

    func cancel() {
        lock.lock()
        let task = task
        if rejection == nil {
            rejection = CancellationError()
        }
        lock.unlock()
        task?.cancel()
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        do {
            completionHandler(try redirectPolicy.request(
                response: response,
                proposedRequest: request))
        } catch {
            reject(error)
            completionHandler(nil)
        }
    }

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        do {
            if let cancellation = rejectionSnapshot() {
                throw cancellation
            }
            guard let http = response as? HTTPURLResponse else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "missing HTTP response for range \(expectation.filename)")
            }
            try validate(http)
            lock.lock()
            responseAccepted = true
            lock.unlock()
            completionHandler(.allow)
        } catch {
            reject(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        lock.lock()
        let mayReceive = responseAccepted && rejection == nil
        lock.unlock()
        guard mayReceive else { return }
        let attempted = receivedBytes + UInt64(data.count)
        guard attempted <= expectation.length else {
            reject(RepackError.remoteBodyExceeded(
                path: expectation.filename,
                limit: expectation.length,
                attempted: attempted))
            dataTask.cancel()
            return
        }
        do {
            try data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                try Posix.pwriteAll(fd: fd,
                                    path: targetPath,
                                    buf: base,
                                    count: raw.count,
                                    offset: receivedBytes)
            }
            receivedBytes = attempted
            emitProgressIfNeeded(force: receivedBytes == expectation.length)
        } catch {
            reject(error)
            dataTask.cancel()
        }
    }

    private func emitProgressIfNeeded(force: Bool) {
        let now = DispatchTime.now().uptimeNanoseconds
        if force
            || lastProgressNanoseconds == nil
            || now &- (lastProgressNanoseconds ?? now) >= 1_000_000_000 {
            lastProgressNanoseconds = now
            progress(receivedBytes)
        }
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        let result: Result<RemoteRangeTransferResult, Error>
        if let rejection = rejectionSnapshot() {
            result = .failure(rejection)
        } else if let error {
            result = .failure(error)
        } else if !responseAccepted {
            result = .failure(RepackError.remoteProtocolInvalid(
                detail: "range \(expectation.filename) completed without an accepted response"))
        } else if receivedBytes != expectation.length {
            result = .failure(RepackError.remoteBodyTruncated(
                path: expectation.filename,
                expected: expectation.length,
                actual: receivedBytes))
        } else {
            result = .success(RemoteRangeTransferResult(
                path: targetPath,
                byteCount: receivedBytes))
        }
        finish(result)
    }

    private func validate(_ response: HTTPURLResponse) throws {
        guard response.statusCode == 206 else {
            throw RepackError.remoteHTTPResponse(
                url: redactRemoteURL(response.url),
                status: response.statusCode,
                retryAfter: remoteHeader(response, "Retry-After"))
        }
        guard (remoteHeader(response, "Content-Encoding") ?? "identity").lowercased() == "identity" else {
            throw RepackError.remoteProtocolInvalid(
                detail: "compressed range response for \(expectation.filename)")
        }
        guard let contentLength = remoteHeader(response, "Content-Length"),
              UInt64(contentLength) == expectation.length else {
            throw RepackError.remoteProtocolInvalid(
                detail: "wrong Content-Length for \(expectation.filename)")
        }
        guard let range = remoteHeader(response, "Content-Range") else {
            throw RepackError.remoteProtocolInvalid(
                detail: "missing Content-Range for \(expectation.filename)")
        }
        let prefix = "bytes "
        let end = expectation.offset + expectation.length - 1
        guard range.hasPrefix(prefix) else {
            throw RepackError.remoteProtocolInvalid(detail: "malformed Content-Range \(range)")
        }
        let parts = range.dropFirst(prefix.count).split(separator: "/", maxSplits: 1)
        let bounds = parts.first?.split(separator: "-", maxSplits: 1) ?? []
        guard parts.count == 2,
              bounds.count == 2,
              UInt64(bounds[0]) == expectation.offset,
              UInt64(bounds[1]) == end,
              UInt64(parts[1]) == expectation.totalSize else {
            throw RepackError.remoteProtocolInvalid(detail: "wrong Content-Range \(range)")
        }
        if let xetHash = expectation.xetHash {
            guard normalizedStrongETag(remoteHeader(response, "ETag")) == xetHash.lowercased() else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "final ranged validator differs for \(expectation.filename)")
            }
        }
    }

    private func reject(_ error: Error) {
        lock.lock()
        if rejection == nil {
            rejection = error
        }
        lock.unlock()
    }

    private func rejectionSnapshot() -> Error? {
        lock.lock()
        defer { lock.unlock() }
        return rejection
    }

    private func finish(_ result: Result<RemoteRangeTransferResult, Error>) {
        lock.lock()
        guard !terminal else {
            lock.unlock()
            return
        }
        terminal = true
        let continuation = continuation
        self.continuation = nil
        self.task = nil
        lock.unlock()

        close(fd)
        if case .failure = result {
            try? FileManager.default.removeItem(atPath: targetPath)
        }
        continuation?.resume(with: result)
    }
}

struct RemoteRedirectPolicy {
    private let originalRange: String?
    private let originalAuthorization: String?
    private let sourceHost: String?
    private let maximumRedirects: Int
    private var redirectCount = 0
    private var authorizationMayBeForwarded = true

    init(originalRequest: URLRequest, maximumRedirects: Int) {
        self.originalRange = originalRequest.value(forHTTPHeaderField: "Range")
        self.originalAuthorization = originalRequest.value(
            forHTTPHeaderField: "Authorization")
        self.sourceHost = originalRequest.url?.host?.lowercased()
        self.maximumRedirects = maximumRedirects
    }

    mutating func request(
        response: HTTPURLResponse,
        proposedRequest: URLRequest
    ) throws -> URLRequest {
        redirectCount += 1
        guard redirectCount <= maximumRedirects else {
            throw RepackError.remoteRedirectRejected(
                url: redactRemoteURL(response.url),
                detail: "more than \(maximumRedirects) redirects")
        }
        guard proposedRequest.url?.scheme?.lowercased() == "https" else {
            throw RepackError.remoteRedirectRejected(
                url: redactRemoteURL(proposedRequest.url),
                detail: "only HTTPS redirects are allowed")
        }
        let targetHost = proposedRequest.url?.host?.lowercased()
        if targetHost != sourceHost {
            authorizationMayBeForwarded = false
        }
        var redirected = proposedRequest
        redirected.httpMethod = "GET"
        redirected.setValue(originalRange, forHTTPHeaderField: "Range")
        redirected.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        redirected.setValue(
            authorizationMayBeForwarded && targetHost == sourceHost
                ? originalAuthorization
                : nil,
            forHTTPHeaderField: "Authorization")
        return redirected
    }
}

func remoteHeader(_ response: HTTPURLResponse, _ name: String) -> String? {
    for (key, value) in response.allHeaderFields
    where String(describing: key).caseInsensitiveCompare(name) == .orderedSame {
        return String(describing: value)
    }
    return nil
}

func normalizedStrongETag(_ value: String?) -> String? {
    guard var value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
          !value.isEmpty,
          !value.lowercased().hasPrefix("w/") else {
        return nil
    }
    if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
        value.removeFirst()
        value.removeLast()
    }
    return value.lowercased()
}

func redactRemoteURL(_ url: URL?) -> String {
    guard let url else { return "<unknown>" }
    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
    components?.query = nil
    components?.fragment = nil
    return components?.url?.absoluteString ?? "<redacted>"
}
