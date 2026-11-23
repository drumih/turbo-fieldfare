import Foundation
import Synchronization
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore

/// Serves a local directory the way Hugging Face serves a pinned repository —
/// `HEAD` with the commit, linked size and range support, `GET` ranges as 206 —
/// read from disk one range at a time, so a 20 GB source streams through the
/// installer without being held in memory.
final class FileBackedHFURLProtocol: URLProtocol, @unchecked Sendable {
    // unchecked-invariant: `root` and `commit` are set once before a test starts
    // a session and only read while it runs.
    nonisolated(unsafe) static var root = ""
    nonisolated(unsafe) static var commit = String(repeating: "a", count: 40)

    override static func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "hf-mirror.test"
    }

    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let name = FakeHFURLProtocol.filename(from: url),
            !name.contains(".."),
            let handle = FileHandle(
                forReadingAtPath: (Self.root as NSString)
                    .appendingPathComponent(name)),
            let size = try? handle.seekToEnd()
        else {
            respond(status: 404)
            return
        }
        defer { try? handle.close() }
        var headers = [
            "X-Repo-Commit": Self.commit, "X-Linked-Size": "\(size)",
            "X-Linked-ETag": "\"\(name)\"", "Accept-Ranges": "bytes",
            "Content-Encoding": "identity",
        ]
        guard request.httpMethod == "GET" else {
            headers["Content-Length"] = "\(size)"
            respond(status: 200, headers: headers)
            return
        }
        let range = request.value(forHTTPHeaderField: "Range") ?? "bytes=0-\(max(size, 1) - 1)"
        let bounds = range.dropFirst("bytes=".count).split(separator: "-").compactMap {
            UInt64($0)
        }
        guard bounds.count == 2, bounds[0] <= bounds[1], bounds[1] < size,
            (try? handle.seek(toOffset: bounds[0])) != nil,
            let body = try? handle.read(upToCount: Int(bounds[1] - bounds[0] + 1))
        else {
            respond(status: 416)
            return
        }
        headers["Content-Length"] = "\(body.count)"
        headers["Content-Range"] = "bytes \(bounds[0])-\(bounds[1])/\(size)"
        respond(status: 206, headers: headers, body: body)
    }

    override func stopLoading() {}

    private func respond(status: Int, headers: [String: String]? = nil, body: Data? = nil) {
        guard let url = request.url,
            let response = HTTPURLResponse(
                url: url, statusCode: status, httpVersion: nil, headerFields: headers)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if let body { client?.urlProtocol(self, didLoad: body) }
        client?.urlProtocolDidFinishLoading(self)
    }
}

/// The Inkling 2 remote install path (TT-121 phase 1). Two opt-in checks:
///
/// - `TINYTITAN_INKLING2_HF=1`: a dry run against the real Hugging Face
///   repositories fetches only headers and metadata, and its plan must equal
///   the plan made from the local copies of the same pinned files.
/// - `TINYTITAN_INKLING2_MIRROR_INSTALL=gguf|checkpoint`: a full install
///   streamed from the local source through `FileBackedHFURLProtocol` into
///   `TINYTITAN_INKLING2_MIRROR_OUTPUT`, whose files must hash exactly as the
///   local install's (`TINYTITAN_INKLING2_MIRROR_REFERENCE`) do.
///
/// Both need the local sources: `TINYTITAN_INKLING2_GGUF` (the GGUF file) and
/// `TINYTITAN_INKLING2_CHECKPOINT` (the checkpoint directory).
@Suite(.serialized) struct Inkling2RemoteInstallTests {
    private static let environment = ProcessInfo.processInfo.environment
    private static var gguf: String? { environment["TINYTITAN_INKLING2_GGUF"] }
    private static var checkpoint: String? { environment["TINYTITAN_INKLING2_CHECKPOINT"] }
    private static var realHF: Bool {
        environment["TINYTITAN_INKLING2_HF"] == "1" && gguf != nil && checkpoint != nil
    }
    private static var mirror: String? {
        guard gguf != nil, checkpoint != nil,
            environment["TINYTITAN_INKLING2_MIRROR_OUTPUT"] != nil,
            environment["TINYTITAN_INKLING2_MIRROR_REFERENCE"] != nil
        else { return nil }
        return environment["TINYTITAN_INKLING2_MIRROR_INSTALL"]
    }

    /// What a plan copies, from where to where: every resident entry and every
    /// expert sub-tensor, without the output directory, which differs by run.
    static func summary(_ plan: RepackPlan) -> [String] {
        plan.resident.entries.map {
            "\($0.name) \($0.dtype) \($0.fileOffset) \($0.sizeBytes) "
                + "\($0.sourceWeight.shardPath)@\($0.sourceWeight.absoluteOffset)"
        }
            + plan.layers.flatMap { layer in
                ["layer \(layer.layerIndex) stride \(layer.expertStride)"]
                    + layer.subTensors.map {
                        "\($0.role).\($0.component) \($0.dtype) \($0.logicalShape) "
                            + "\($0.offsetInExpertBlob)+\($0.sizeInExpertBlob) "
                            + "\($0.sourceTensor.shardPath)@\($0.sourceTensor.absoluteOffset)"
                    }
            } + plan.passthroughFiles.map { "\($0.sourceName)->\($0.destinationName) \($0.size)" }
            + ["model \(plan.matchedModelID ?? "none")"]
    }

    static func localPlan(_ source: SupportedModelSource) throws -> RepackPlan {
        var plan: RepackPlan
        if case .gguf(let fileName) = source.inkling2 {
            let (header, _, size) = try RemoteStreamingRepacker.readGGUFHeader(
                at: try #require(gguf))
            plan = try Inkling2GGUFPlanner.plan(
                source: .init(fileName: fileName, header: header, fileSize: size),
                outputDir: "/plan")
        } else {
            plan = try Inkling2CheckpointPlanner.plan(
                source: try RemoteStreamingRepacker.loadInkling2Checkpoint(
                    directory: try #require(checkpoint)),
                outputDir: "/plan")
        }
        plan.matchedModelID = source.modelID
        return plan
    }

    @Test(.enabled(if: realHF)) func aDryRunAgainstHuggingFacePlansTheLocalInstall() async throws {
        for source in [SupportedModelSource.inkling2GGUFQ4, .inkling2NVFP4] {
            let output = NSTemporaryDirectory() + "inkling-hf-\(UUID().uuidString)"
            defer { try? FileManager.default.removeItem(atPath: output) }
            let options = source.installOptions(
                outputDirectory: URL(fileURLWithPath: output), overwrite: false, token: nil)
            let dryRun = RemoteStreamingRepackOptions(
                repoID: options.repoID, revision: options.revision, outputDir: output,
                requireKnownSource: true, minFreeReserveBytes: 0, dryRunSpaceCheck: true,
                inkling2: options.inkling2)
            let result = try await RemoteStreamingRepacker(options: dryRun).run()
            #expect(result.dryRun)
            #expect(result.resolvedCommit == source.revision)
            let remote = Self.summary(result.plan).map {
                $0.replacingOccurrences(of: output + ".partial", with: "/plan")
            }
            #expect(remote == Self.summary(try Self.localPlan(source)), "\(source.name)")
        }
    }

    @Test(.enabled(if: mirror != nil)) func aMirroredInstallMatchesTheLocalOne() async throws {
        let kind = try #require(Self.mirror)
        let source: SupportedModelSource = kind == "gguf" ? .inkling2GGUFQ4 : .inkling2NVFP4
        FileBackedHFURLProtocol.root =
            kind == "gguf"
            ? ((try #require(Self.gguf)) as NSString).deletingLastPathComponent
            : try #require(Self.checkpoint)
        FileBackedHFURLProtocol.commit = source.revision
        let output = try #require(Self.environment["TINYTITAN_INKLING2_MIRROR_OUTPUT"])
        let options = RemoteStreamingRepackOptions(
            repoID: source.repoID, revision: source.revision, outputDir: output,
            requireKnownSource: true, minFreeReserveBytes: 0,
            downloadSession: RemoteDownloadSession(
                protocolClasses: [FileBackedHFURLProtocol.self]),
            baseURL: try #require(URL(string: "https://hf-mirror.test")),
            inkling2: source.inkling2)
        _ = try await RemoteStreamingRepacker(options: options).run()
        func files(_ install: String) throws -> [String: String] {
            let manifest = try #require(
                try JSONSerialization.jsonObject(
                    with: try Data(contentsOf: URL(fileURLWithPath: install + "/manifest.json")))
                    as? [String: Any])
            let entries = try #require(manifest["files"] as? [String: [String: Any]])
            return entries.mapValues { "\($0["size"] ?? "") \($0["sha256"] ?? "")" }
        }
        let reference = try #require(Self.environment["TINYTITAN_INKLING2_MIRROR_REFERENCE"])
        #expect(try files(output) == files(reference))
    }
}
