import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

// The fake HF URL protocol is process-global state (`resetFakeHF`);
// tests that share it cannot run concurrently. Nesting under the serialized
// RemotePayloadCopyTests (like RemoteRangeTransferTests) serializes this
// suite with every other user of the fake; a separate top-level suite,
// serialized only within itself, still raced them and left a payload copy
// mid-flight with an emptied file table (remote HTTP 404).
extension RemotePayloadCopyTests {
    @Suite
    struct RemoteDownloadSessionTests {
        @Test func typedSessionUsesStallTolerantSerialDefaults() {
            let session = RemoteDownloadSession()
            let configuration = session.configurationSnapshot

            #expect(session.policy.requestTimeoutSeconds == 300)
            #expect(session.policy.resourceTimeoutSeconds == 7 * 24 * 60 * 60)
            #expect(session.policy.waitsForConnectivity)
            #expect(session.policy.maximumConnectionsPerHost == 1)
            #expect(session.policy.maximumRedirects == 5)
            #expect(configuration.requestTimeoutSeconds == 300)
            #expect(configuration.resourceTimeoutSeconds == 7 * 24 * 60 * 60)
            #expect(configuration.waitsForConnectivity)
            #expect(configuration.maximumConnectionsPerHost == 1)
            #expect(configuration.requestCachePolicy == .reloadIgnoringLocalCacheData)
            #expect(!configuration.hasURLCache)
        }

        @Test func injectedPolicyIsAppliedToOwnedConfiguration() {
            let policy = RemoteDownloadSessionPolicy(
                requestTimeoutSeconds: 12,
                resourceTimeoutSeconds: 34,
                waitsForConnectivity: false,
                maximumConnectionsPerHost: 1,
                maximumRedirects: 2)
            let session = RemoteDownloadSession(policy: policy)
            let configuration = session.configurationSnapshot

            #expect(configuration.requestTimeoutSeconds == 12)
            #expect(configuration.resourceTimeoutSeconds == 34)
            #expect(!configuration.waitsForConnectivity)
            #expect(configuration.maximumConnectionsPerHost == 1)
            #expect(session.policy.maximumRedirects == 2)
        }

        @Test func metadataRedirectsFollowOnlyBoundedSameHostHTTPS() {
            var original = URLRequest(url: URL(string: "https://hf.test/model/file")!)
            original.httpMethod = "HEAD"
            original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
            var policy = RemoteMetadataRedirectPolicy(
                originalRequest: original,
                maximumRedirects: 2)

            let sameHost = policy.request(
                proposedRequest: URLRequest(
                    url: URL(string: "https://hf.test/api/cache/file?etag=value")!))
            #expect(sameHost?.httpMethod == "HEAD")
            #expect(sameHost?.value(forHTTPHeaderField: "Accept-Encoding") == "identity")
            #expect(sameHost?.value(forHTTPHeaderField: "Authorization") == "Bearer secret")

            #expect(
                policy.request(
                    proposedRequest: URLRequest(
                        url: URL(string: "https://us.aws.cdn.hf.co/xet-bridge-us/object")!)) == nil)
            #expect(
                policy.request(
                    proposedRequest: URLRequest(
                        url: URL(string: "https://cdn-lfs.huggingface.co/object")!)) == nil)
            #expect(
                policy.request(
                    proposedRequest: URLRequest(
                        url: URL(string: "https://storage.test/signed?token=private")!)) == nil)

            var hopLimited = RemoteMetadataRedirectPolicy(
                originalRequest: original,
                maximumRedirects: 1)
            #expect(
                hopLimited.request(
                    proposedRequest: URLRequest(
                        url: URL(string: "https://hf.test/one")!)) != nil)
            #expect(
                hopLimited.request(
                    proposedRequest: URLRequest(
                        url: URL(string: "https://hf.test/too-many")!)) == nil)
        }

        @Test func resolveFileInfoAcceptsHubRedirectWithLinkedMetadata() async throws {
            resetFakeHF()
            FakeHFURLProtocol.files["model.bin"] = Data([1, 2, 3, 4])
            FakeHFURLProtocol.xetHashOverrides["model.bin"] = String(repeating: "ab", count: 32)
            FakeHFURLProtocol.failures["HEAD:model.bin"] = [
                .response(
                    status: 302,
                    headers: [
                        "Location": "https://us.aws.cdn.hf.co/xet-bridge-us/object",
                        "X-Repo-Commit": FakeHFURLProtocol.commit,
                        "X-Linked-Size": "4",
                        "X-Linked-ETag": "\"model-etag\"",
                        "X-Xet-Hash": String(repeating: "ab", count: 32),
                        "Accept-Ranges": "bytes",
                        "Content-Length": "64",
                    ],
                    body: Data())
            ]

            let remote = HuggingFaceRemoteSource(
                repoID: "owner/model",
                requestedRevision: "main",
                downloadSession: fakeHFSession(),
                baseURL: URL(string: "https://hf.test")!)
            let info = try await remote.resolveFileInfo(filename: "model.bin")

            #expect(info.resolvedCommit == FakeHFURLProtocol.commit)
            #expect(info.size == 4)
            #expect(info.etag == "\"model-etag\"")
            #expect(info.xetHash == String(repeating: "ab", count: 32))
            #expect(info.acceptsRanges)
        }

        @Test func resolveFileInfoRejectsBareRedirectWithoutLinkedSize() async throws {
            resetFakeHF()
            FakeHFURLProtocol.files["model.bin"] = Data([1])
            FakeHFURLProtocol.failures["HEAD:model.bin"] = [
                .response(
                    status: 302,
                    headers: [
                        "Location": "https://us.aws.cdn.hf.co/xet-bridge-us/object",
                        "X-Repo-Commit": FakeHFURLProtocol.commit,
                        "Content-Length": "64",
                    ],
                    body: Data())
            ]

            let remote = HuggingFaceRemoteSource(
                repoID: "owner/model",
                requestedRevision: "main",
                downloadSession: fakeHFSession(),
                baseURL: URL(string: "https://hf.test")!)
            await #expect(throws: RepackError.self) {
                _ = try await remote.resolveFileInfo(filename: "model.bin")
            }
        }
    }
}
