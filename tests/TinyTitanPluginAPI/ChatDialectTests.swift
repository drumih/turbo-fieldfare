import Foundation
import Testing
import TinyTitanCore
import TinyTitanPluginAPI

@Suite("ChatDialect (TPP/1.1)")
struct ChatDialectTests {
    private enum ChatMLStubPlugin: EnginePlugin {
        static let pluginID = "ai.tinytitan.test.chatml-dialect"
        static let protocolVersion = ProtocolVersion.current
        static let displayName = "ChatML Stub"

        static func probe(_ install: InstallProbe) -> MatchStrength? { .exact }
        static func parameters(for install: InstallProbe) -> ParameterSupport {
            ParameterSupport(
                kvCachePrecision: .supported, ropeScaling: .supported,
                hostKVWindow: .refused(reason: "stub"), concurrentSequences: .supported,
                multiTokenPrediction: .refused(reason: "stub"), expertStreaming: .supported)
        }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
            .binaryThinking
        }
        static func chatDialect(for install: InstallProbe) -> (any ChatDialect)? {
            ChatMLDialect()
        }
        static func engineFactory(host: HostServices) -> EngineFactory {
            fatalError("unused in dialect test")
        }
    }

    @Test func protocolVersionIsOnePointOne() {
        #expect(ProtocolVersion.current == ProtocolVersion(major: 1, minor: 1))
    }

    @Test func referenceDialectPropertiesMatchContracts() {
        let chatML = ChatMLDialect()
        let k2 = K2Dialect()
        let inkling = InklingDialect()

        #expect(chatML.identifier == "chatml")
        #expect(chatML.supportsChatMLContinuationBridge)
        #expect(!chatML.closesThoughtOnMarkerText)
        #expect(!chatML.usesStructuredInklingParser)

        #expect(k2.identifier == "k2")
        #expect(k2.supportsChatMLContinuationBridge)
        #expect(k2.closesThoughtOnMarkerText)
        #expect(!k2.usesStructuredInklingParser)

        #expect(inkling.identifier == "inkling")
        #expect(!inkling.supportsChatMLContinuationBridge)
        #expect(!inkling.closesThoughtOnMarkerText)
        #expect(inkling.usesStructuredInklingParser)
    }

    @Test func resolvedPluginExposesDeclaredChatDialect() throws {
        let registry = PluginRegistry()
        try registry.register(ChatMLStubPlugin.self)
        let resolved = try registry.resolve(URL(fileURLWithPath: "/tmp/nonexistent-dialect-probe"))
        #expect(resolved.chatDialect?.identifier == "chatml")
    }
}
