import Foundation
import Testing

@testable import TinyTitanCore

@Test func sessionBuildOptionsPreserveTheFactoryInputs() {
    let options = SessionBuildOptions(
        context: nil,
        maxContext: 4096,
        slots: 2,
        promptCacheMaximumEntries: 8,
        promptCacheMemoryLimitBytes: 1024,
        promptCacheDiskDirectory: URL(fileURLWithPath: "/tmp/cache"),
        promptCacheDiskLimitBytes: 2048,
        prefillChunkTokens: 512,
        kvCachePrecision: .fp16,
        hostKVWindowTokens: 256,
        ropeScalingMode: .yarn,
        thinkingMode: .on,
        reasoningEffort: .high,
        expertCacheSlots: 16,
        expertCacheBudgetBytes: 4096,
        mtpModelDirectory: URL(fileURLWithPath: "/tmp/draft"),
        mtpMemoryMiB: 128)

    #expect(options.maxContext == 4096)
    #expect(options.slots == 2)
    #expect(options.promptCacheMaximumEntries == 8)
    #expect(options.promptCacheDiskLimitBytes == 2048)
    #expect(options.prefillChunkTokens == 512)
    #expect(options.kvCachePrecision == .fp16)
    #expect(options.hostKVWindowTokens == 256)
    #expect(options.ropeScalingMode == .yarn)
    #expect(options.thinkingMode == .on)
    #expect(options.reasoningEffort == .high)
    #expect(options.expertCacheSlots == 16)
    #expect(options.mtpMemoryMiB == 128)
}
