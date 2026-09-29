import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
  @Suite
  struct RemoteRangeTransferTests {
    @Test func exact206IsAcceptedAndWritten() async throws {
        resetFakeHF()
        FakeHFURLProtocol.files["file.bin"] = Data([1, 2, 3, 4])
        let target = tmpPathForRemote("range-exact")
        defer { cleanUpRemote([target]) }

        let result = try await fakeHFSession().transfer(
            request: rangeRequest(length: 4),
            targetPath: target,
            expectation: RemoteRangeExpectation(
                filename: "file.bin",
                offset: 0,
                length: 4,
                totalSize: 4,
                xetHash: nil))

        #expect(result.byteCount == 4)
        #expect(try Data(contentsOf: URL(fileURLWithPath: target)) == Data([1, 2, 3, 4]))
    }

    @Test func oversized200IsRejectedBeforeTargetGrowth() async throws {
        resetFakeHF()
        let body = Data(repeating: 0x7f, count: 2 * 1024 * 1024)
        FakeHFURLProtocol.files["file.bin"] = Data([1])
        FakeHFURLProtocol.failures["GET:file.bin"] = [
            .response(
                status: 200,
                headers: ["Content-Length": "\(body.count)"],
                body: body),
        ]
        let target = tmpPathForRemote("range-200")
        defer { cleanUpRemote([target]) }

        await #expect(throws: RepackError.self) {
            _ = try await fakeHFSession().transfer(
                request: rangeRequest(length: 1),
                targetPath: target,
                expectation: RemoteRangeExpectation(
                    filename: "file.bin",
                    offset: 0,
                    length: 1,
                    totalSize: 1,
                    xetHash: nil))
        }
        #expect(!FileManager.default.fileExists(atPath: target))
    }

    @Test func lying206BodyCannotExceedRequestedRange() async throws {
        resetFakeHF()
        let body = Data(repeating: 0x55, count: 2 * 1024 * 1024)
        FakeHFURLProtocol.files["file.bin"] = Data([1])
        FakeHFURLProtocol.failures["GET:file.bin"] = [
            .response(
                status: 206,
                headers: [
                    "Content-Length": "1",
                    "Content-Range": "bytes 0-0/1",
                    "Content-Encoding": "identity",
                ],
                body: body),
        ]
        let target = tmpPathForRemote("range-oversize")
        defer { cleanUpRemote([target]) }

        await #expect(throws: RepackError.self) {
            _ = try await fakeHFSession().transfer(
                request: rangeRequest(length: 1),
                targetPath: target,
                expectation: RemoteRangeExpectation(
                    filename: "file.bin",
                    offset: 0,
                    length: 1,
                    totalSize: 1,
                    xetHash: nil))
        }
        #expect(!FileManager.default.fileExists(atPath: target))
    }

    @Test(arguments: [
        "content-length",
        "content-range",
        "content-encoding",
        "validator",
    ])
    func mismatchedRangeResponseIsRejectedBeforeWriting(_ mismatch: String) async throws {
        resetFakeHF()
        FakeHFURLProtocol.files["file.bin"] = Data([1])
        var headers = [
            "Content-Length": "1",
            "Content-Range": "bytes 0-0/1",
            "Content-Encoding": "identity",
            "ETag": "\"expected\"",
        ]
        switch mismatch {
        case "content-length":
            headers["Content-Length"] = "2"
        case "content-range":
            headers["Content-Range"] = "bytes 1-1/2"
        case "content-encoding":
            headers["Content-Encoding"] = "gzip"
        case "validator":
            headers["ETag"] = "\"different\""
        default:
            Issue.record("unknown mismatch \(mismatch)")
        }
        FakeHFURLProtocol.failures["GET:file.bin"] = [
            .response(status: 206, headers: headers, body: Data([1])),
        ]
        let target = tmpPathForRemote("range-mismatch-\(mismatch)")
        defer { cleanUpRemote([target]) }

        await #expect(throws: RepackError.self) {
            _ = try await fakeHFSession().transfer(
                request: rangeRequest(length: 1),
                targetPath: target,
                expectation: RemoteRangeExpectation(
                    filename: "file.bin",
                    offset: 0,
                    length: 1,
                    totalSize: 1,
                    xetHash: "expected"))
        }
        #expect(!FileManager.default.fileExists(atPath: target))
    }

    @Test func retryAfterParsesSecondsAndHTTPDateAndRejectsCap() throws {
        let policy = RemoteRetryPolicy(maxServerDelayNs: 10 * 1_000_000_000)
        let error = RepackError.remoteHTTPResponse(
            url: "https://hf.test/file",
            status: 429,
            retryAfter: "5")
        #expect(try policy.retryDelayNs(error: error, localDelayNs: 1) == 5_000_000_000)

        let now = Date(timeIntervalSince1970: 0)
        let dateError = RepackError.remoteHTTPResponse(
            url: "https://hf.test/file",
            status: 503,
            retryAfter: "Thu, 01 Jan 1970 00:00:08 GMT")
        #expect(try policy.retryDelayNs(
            error: dateError,
            localDelayNs: 1,
            now: now) == 8_000_000_000)

        let capped = RepackError.remoteHTTPResponse(
            url: "https://hf.test/file",
            status: 429,
            retryAfter: "11")
        #expect(throws: RepackError.self) {
            _ = try policy.retryDelayNs(error: capped, localDelayNs: 1)
        }

        let obsolete = RepackError.remoteHTTPResponse(
            url: "https://hf.test/file",
            status: 503,
            retryAfter: "Sunday, 06-Nov-94 08:49:37 GMT")
        let obsoleteNow = Date(timeIntervalSince1970: 784_111_770)
        #expect(try policy.retryDelayNs(
            error: obsolete,
            localDelayNs: 1,
            now: obsoleteNow) == 7_000_000_000)

        let asctime = RepackError.remoteHTTPResponse(
            url: "https://hf.test/file",
            status: 503,
            retryAfter: "Sun Nov 6 08:49:37 1994")
        #expect(try policy.retryDelayNs(
            error: asctime,
            localDelayNs: 1,
            now: obsoleteNow) == 7_000_000_000)
    }

    @Test func retryBackoffUsesInjectedJitterAndSleeper() async throws {
        let attempts = Mutex(0)
        let sleeps = Mutex<[UInt64]>([])
        let value: String = try await withRemoteRetries(
            RemoteRetryPolicy(
                attempts: 3,
                baseDelayNs: 100,
                maxDelayNs: 500,
                maxServerDelayNs: 1_000),
            label: "test",
            audit: nil,
            jitter: { 10 },
            sleeper: { delay in
                sleeps.withLock { $0.append(delay) }
            }) {
                let attempt = attempts.withLock { value -> Int in
                    value += 1
                    return value
                }
                if attempt < 3 {
                    throw URLError(.timedOut)
                }
                return "done"
            }

        #expect(value == "done")
        #expect(sleeps.withLock { $0 } == [110, 210])
    }

    @Test func retryStatusMatrixIsExplicit() {
        for status in [408, 429, 500, 502, 503, 504] {
            #expect(RemoteRetryPolicy.isRetryable(
                RepackError.remoteHTTPResponse(
                    url: "https://hf.test/file",
                    status: status,
                    retryAfter: nil)))
        }
        for status in [200, 400, 401, 403, 404, 409, 412, 416, 501, 505] {
            #expect(!RemoteRetryPolicy.isRetryable(
                RepackError.remoteHTTPResponse(
                    url: "https://hf.test/file",
                    status: status,
                    retryAfter: nil)))
        }
    }

    @Test func cancellationDuringBackoffStopsBeforeAnotherAttempt() async {
        let attempts = Mutex(0)
        await #expect(throws: CancellationError.self) {
            let _: Int = try await withRemoteRetries(
                RemoteRetryPolicy(
                    attempts: 4,
                    baseDelayNs: 1,
                    maxDelayNs: 8),
                label: "cancel",
                audit: nil,
                jitter: { 0 },
                sleeper: { _ in throw CancellationError() }) {
                    attempts.withLock { $0 += 1 }
                    throw URLError(.timedOut)
                }
        }
        #expect(attempts.withLock { $0 } == 1)
    }

    @Test func redirectPolicyPermanentlyStripsAuthorizationAfterHostChange() throws {
        var original = rangeRequest(length: 4)
        original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        var policy = RemoteRedirectPolicy(
            originalRequest: original,
            maximumRedirects: 3)
        let sourceResponse = HTTPURLResponse(
            url: original.url!,
            statusCode: 302,
            httpVersion: nil,
            headerFields: nil)!

        var storageRequest = URLRequest(
            url: URL(string: "https://storage.test/signed?token=private")!)
        storageRequest.httpMethod = "GET"
        let first = try policy.request(
            response: sourceResponse,
            proposedRequest: storageRequest)
        #expect(first.value(forHTTPHeaderField: "Authorization") == nil)
        #expect(first.value(forHTTPHeaderField: "Range") == "bytes=0-3")
        #expect(first.value(forHTTPHeaderField: "Accept-Encoding") == "identity")

        let storageResponse = HTTPURLResponse(
            url: storageRequest.url!,
            statusCode: 302,
            httpVersion: nil,
            headerFields: nil)!
        var backToSource = URLRequest(url: original.url!)
        backToSource.setValue("Bearer leaked", forHTTPHeaderField: "Authorization")
        let second = try policy.request(
            response: storageResponse,
            proposedRequest: backToSource)
        #expect(second.value(forHTTPHeaderField: "Authorization") == nil)
        #expect(second.value(forHTTPHeaderField: "Range") == "bytes=0-3")
    }

    @Test func redirectPolicyForwardsAuthorizationToAllowedCDNHost() throws {
        var original = rangeRequest(length: 4)
        original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        var policy = RemoteRedirectPolicy(
            originalRequest: original,
            maximumRedirects: 3)
        let sourceResponse = HTTPURLResponse(
            url: original.url!,
            statusCode: 302,
            httpVersion: nil,
            headerFields: nil)!

        var cdnRequest = URLRequest(
            url: URL(string: "https://cdn-lfs.huggingface.co/signed?token=private")!)
        cdnRequest.httpMethod = "GET"
        let cdnRedirect = try policy.request(
            response: sourceResponse,
            proposedRequest: cdnRequest)
        #expect(cdnRedirect?.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        #expect(cdnRedirect?.value(forHTTPHeaderField: "Range") == "bytes=0-3")

        let unknownResponse = HTTPURLResponse(
            url: cdnRequest.url!,
            statusCode: 302,
            httpVersion: nil,
            headerFields: nil)!
        var unknownRequest = URLRequest(
            url: URL(string: "https://evil.test/stolen")!)
        unknownRequest.httpMethod = "GET"
        let unknownRedirect = try policy.request(
            response: unknownResponse,
            proposedRequest: unknownRequest)
        #expect(unknownRedirect?.value(forHTTPHeaderField: "Authorization") == nil)
    }

    @Test func redirectPolicyStripsAuthAfterAllowedCDNThenUnknownHost() throws {
        var original = rangeRequest(length: 4)
        original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        var policy = RemoteRedirectPolicy(
            originalRequest: original,
            maximumRedirects: 3)
        let sourceResponse = HTTPURLResponse(
            url: original.url!,
            statusCode: 302,
            httpVersion: nil,
            headerFields: nil)!

        var cdnRequest = URLRequest(
            url: URL(string: "https://cdn-lfs.huggingface.co/signed")!)
        cdnRequest.httpMethod = "GET"
        _ = try policy.request(
            response: sourceResponse,
            proposedRequest: cdnRequest)

        let cdnResponse = HTTPURLResponse(
            url: cdnRequest.url!,
            statusCode: 302,
            httpVersion: nil,
            headerFields: nil)!
        var unknownRequest = URLRequest(
            url: URL(string: "https://evil.test/stolen")!)
        unknownRequest.httpMethod = "GET"
        let result = try policy.request(
            response: cdnResponse,
            proposedRequest: unknownRequest)
        #expect(result?.value(forHTTPHeaderField: "Authorization") == nil)
    }

    @Test func redirectPolicyRejectsNonHTTPSAndExcessHops() throws {
        let original = rangeRequest(length: 1)
        let response = HTTPURLResponse(
            url: original.url!,
            statusCode: 302,
            httpVersion: nil,
            headerFields: nil)!

        var insecurePolicy = RemoteRedirectPolicy(
            originalRequest: original,
            maximumRedirects: 2)
        #expect(throws: RepackError.self) {
            _ = try insecurePolicy.request(
                response: response,
                proposedRequest: URLRequest(url: URL(string: "http://storage.test/file")!))
        }

        var boundedPolicy = RemoteRedirectPolicy(
            originalRequest: original,
            maximumRedirects: 1)
        _ = try boundedPolicy.request(
            response: response,
            proposedRequest: URLRequest(url: URL(string: "https://storage.test/file")!))
        #expect(throws: RepackError.self) {
            _ = try boundedPolicy.request(
                response: response,
                proposedRequest: URLRequest(url: URL(string: "https://storage-2.test/file")!))
        }
    }

    private func rangeRequest(length: Int) -> URLRequest {
        let commit = String(repeating: "a", count: 40)
        let url = URL(string: "https://hf.test/owner/model/resolve/\(commit)/file.bin")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("bytes=0-\(length - 1)", forHTTPHeaderField: "Range")
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        return request
    }
  }
}
