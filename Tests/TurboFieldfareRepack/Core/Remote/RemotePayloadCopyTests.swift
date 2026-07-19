import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareRepackCore

final class FakeHFURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var files: [String: Data] = [:]
    nonisolated(unsafe) static var commit = "cc499c86a958ea7f05cffaa91c7e7243240dabbe"
    nonisolated(unsafe) static var failures: [String: [FakeFailure]] = [:]
    nonisolated(unsafe) static var requestCounts: [String: Int] = [:]

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "hf.test"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = request.url,
              let filename = Self.filename(from: url) else {
            let response = HTTPURLResponse(url: request.url!,
                                           statusCode: 404,
                                           httpVersion: nil,
                                           headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let method = request.httpMethod ?? "GET"
        let key = "\(method):\(filename)"
        Self.requestCounts[key, default: 0] += 1
        let failure = Self.nextFailure(for: key)
        switch failure {
        case .url(let code):
            client?.urlProtocol(self, didFailWithError: URLError(code))
            return
        case .http(let status):
            let response = HTTPURLResponse(url: url,
                                           statusCode: status,
                                           httpVersion: nil,
                                           headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        case .truncatedBody, nil:
            break
        }

        guard let data = Self.files[filename] else {
            let response = HTTPURLResponse(url: request.url!,
                                           statusCode: 404,
                                           httpVersion: nil,
                                           headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        if method == "HEAD" {
            let headers = baseHeaders(data: data,
                                      contentLength: data.count)
            let response = HTTPURLResponse(url: url,
                                           statusCode: 200,
                                           httpVersion: nil,
                                           headerFields: headers)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        let range = request.value(forHTTPHeaderField: "Range")
        guard let (start, end) = parseRange(range, fileSize: data.count) else {
            let response = HTTPURLResponse(url: url,
                                           statusCode: 416,
                                           httpVersion: nil,
                                           headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let expectedLength = end - start + 1
        let body: Data
        if failure == .truncatedBody {
            body = start < end ? Data(data[start..<end]) : Data()
        } else {
            body = Data(data[start...end])
        }
        var headers = baseHeaders(data: data,
                                  contentLength: expectedLength)
        headers["Content-Range"] = "bytes \(start)-\(end)/\(data.count)"
        let response = HTTPURLResponse(url: url,
                                       statusCode: 206,
                                       httpVersion: nil,
                                       headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    func baseHeaders(data: Data,
                             contentLength: Int) -> [String: String] {
        [
            "X-Repo-Commit": Self.commit,
            "X-Linked-Size": "\(data.count)",
            "Accept-Ranges": "bytes",
            "Content-Length": "\(contentLength)",
            "Content-Encoding": "identity",
        ]
    }

    static func filename(from url: URL) -> String? {
        let parts = url.path.split(separator: "/").map(String.init)
        guard let resolveIndex = parts.firstIndex(of: "resolve"),
              parts.count > resolveIndex + 2 else {
            return nil
        }
        return parts[(resolveIndex + 2)...].joined(separator: "/")
    }

    static func nextFailure(for key: String) -> FakeFailure? {
        guard var queue = failures[key], !queue.isEmpty else { return nil }
        let failure = queue.removeFirst()
        failures[key] = queue
        return failure
    }

    func parseRange(_ value: String?, fileSize: Int) -> (Int, Int)? {
        guard let value, value.hasPrefix("bytes=") else { return nil }
        let body = value.dropFirst("bytes=".count)
        let parts = body.split(separator: "-", maxSplits: 1)
        guard parts.count == 2,
              let start = Int(parts[0]),
              let end = Int(parts[1]),
              start >= 0,
              end >= start,
              end < fileSize else {
            return nil
        }
        return (start, end)
    }
}

enum FakeFailure: Equatable {
    case url(URLError.Code)
    case http(Int)
    case truncatedBody
}

let remoteTokenizerJSON = Data(#"{"model":{"type":"BPE"}}"#.utf8)
let remoteTokenizerConfigJSON = Data(#"{"tokenizer_class":"PreTrainedTokenizerFast"}"#.utf8)
let remoteSpecialTokensMapJSON = Data(#"{"eos_token":"<eos>"}"#.utf8)

@Suite(.serialized)
struct RemotePayloadCopyTests {

}

func remoteFiles(snapshotDir: String,
                         snap: SyntheticSnapshot.Snapshot,
                         includeRequiredTokenizer: Bool,
                         includeOptionalTokenizer: Bool) throws -> [String: Data] {
    var files = [
        "config.json": try Data(contentsOf: URL(fileURLWithPath:
            (snapshotDir as NSString).appendingPathComponent("config.json"))),
        "model.safetensors.index.json": try Data(contentsOf: URL(fileURLWithPath:
            (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json"))),
        "model-00001-of-00001.safetensors": try Data(contentsOf: URL(fileURLWithPath: snap.shardPath)),
    ]
    if includeRequiredTokenizer {
        files["tokenizer.json"] = remoteTokenizerJSON
        files["tokenizer_config.json"] = remoteTokenizerConfigJSON
    }
    if includeOptionalTokenizer {
        files["special_tokens_map.json"] = remoteSpecialTokensMapJSON
    }
    return files
}

func resetFakeHF() {
    FakeHFURLProtocol.files = [:]
    FakeHFURLProtocol.failures = [:]
    FakeHFURLProtocol.requestCounts = [:]
}

func fakeHFSession() -> URLSession {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [FakeHFURLProtocol.self]
    return URLSession(configuration: config)
}

func remoteOptions(outputDir: String,
                           session: URLSession,
                           rangeRetryAttempts: Int = 4,
                           retainPartialOnFailure: Bool = true) -> RemoteStreamingRepackOptions {
    RemoteStreamingRepackOptions(
        outputDir: outputDir,
        overwrite: true,
        repoID: "owner/model",
        revision: "main",
        requireKnownSource: false,
        rangeChunkBytes: 4096,
        minFreeReserveBytes: 0,
        retainPartialOnFailure: retainPartialOnFailure,
        session: session,
        baseURL: URL(string: "https://hf.test")!,
        rangeRetryAttempts: rangeRetryAttempts,
        retryBaseDelayNs: 0)
}

final class InstallProgressRecorder: Sendable {
    let storage = Mutex<[ModelInstallProgress]>([])
    var values: [ModelInstallProgress] { storage.withLock { $0 } }
    func append(_ value: ModelInstallProgress) { storage.withLock { $0.append(value) } }
}

func assertRemoteTokenizerFilesRecorded(outputDir: String,
                                                expectsOptionalSpecialTokens: Bool) throws {
    let tokenizerDir = (outputDir as NSString).appendingPathComponent("tokenizer")
    #expect(try Data(contentsOf: URL(fileURLWithPath:
        (tokenizerDir as NSString).appendingPathComponent("tokenizer.json"))) == remoteTokenizerJSON)
    #expect(try Data(contentsOf: URL(fileURLWithPath:
        (tokenizerDir as NSString).appendingPathComponent("tokenizer_config.json"))) == remoteTokenizerConfigJSON)
    let specialTokensPath = (tokenizerDir as NSString).appendingPathComponent("special_tokens_map.json")
    #expect(FileManager.default.fileExists(atPath: specialTokensPath) == expectsOptionalSpecialTokens)

    let manifestData = try Data(contentsOf: URL(fileURLWithPath:
        (outputDir as NSString).appendingPathComponent("manifest.json")))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
    let manifestFiles = manifest["files"] as! [String: Any]
    #expect(manifestFiles["tokenizer/config.json"] != nil)
    #expect(manifestFiles["tokenizer/tokenizer.json"] != nil)
    #expect(manifestFiles["tokenizer/tokenizer_config.json"] != nil)
    #expect((manifestFiles["tokenizer/special_tokens_map.json"] != nil) == expectsOptionalSpecialTokens)

    let receiptData = try Data(contentsOf: URL(fileURLWithPath:
        (outputDir as NSString).appendingPathComponent(VerifiedInstallReceiptWriter.fileName)))
    let receipt = try JSONSerialization.jsonObject(with: receiptData) as! [String: Any]
    let receiptFiles = receipt["files"] as! [String: Any]
    #expect(receiptFiles["tokenizer/config.json"] != nil)
    #expect(receiptFiles["tokenizer/tokenizer.json"] != nil)
    #expect(receiptFiles["tokenizer/tokenizer_config.json"] != nil)
    #expect((receiptFiles["tokenizer/special_tokens_map.json"] != nil) == expectsOptionalSpecialTokens)
}

func assertNoInternalRemoteDirs(outputDir: String) throws {
    let entries = try FileManager.default.contentsOfDirectory(atPath: outputDir)
    #expect(!entries.contains(".range-tmp"))
    #expect(!entries.contains(".remote-metadata"))
}

func tmpDirForRemote(_ tag: String) -> String {
    let path = (NSTemporaryDirectory() as NSString)
        .appendingPathComponent("turbofieldfare-remote-\(tag)-\(UUID().uuidString)")
    try? FileManager.default.removeItem(atPath: path)
    try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
    return path
}

func tmpPathForRemote(_ tag: String) -> String {
    let path = (NSTemporaryDirectory() as NSString)
        .appendingPathComponent("turbofieldfare-remote-\(tag)-\(UUID().uuidString)")
    try? FileManager.default.removeItem(atPath: path)
    return path
}

func cleanUpRemote(_ paths: [String]) {
    for path in paths {
        try? FileManager.default.removeItem(atPath: path)
    }
}
