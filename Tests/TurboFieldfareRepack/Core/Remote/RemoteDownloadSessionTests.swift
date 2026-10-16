import Foundation
import Testing
@testable import TurboFieldfareRepackCore

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

        let sameHost = policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://hf.test/api/cache/file?etag=value")!))
        #expect(sameHost?.httpMethod == "HEAD")
        #expect(sameHost?.value(forHTTPHeaderField: "Accept-Encoding") == "identity")
        #expect(sameHost?.value(forHTTPHeaderField: "Authorization") == "Bearer secret")

        #expect(policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://storage.test/signed?token=private")!)) == nil)
        #expect(policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://hf.test/too-many")!)) == nil)
    }

    @Test func metadataRedirectForwardsAuthorizationToAllowedCDNHost() {
        var original = URLRequest(url: URL(string: "https://hf.test/model/file")!)
        original.httpMethod = "HEAD"
        original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        var policy = RemoteMetadataRedirectPolicy(
            originalRequest: original,
            maximumRedirects: 2)

        let cdn = policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://cdn-lfs.huggingface.co/repo/file.bin")!))
        #expect(cdn?.httpMethod == "HEAD")
        #expect(cdn?.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        #expect(cdn?.value(forHTTPHeaderField: "Accept-Encoding") == "identity")

        let unknown = policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://evil.test/stolen")!))
        #expect(unknown == nil)
    }
}
