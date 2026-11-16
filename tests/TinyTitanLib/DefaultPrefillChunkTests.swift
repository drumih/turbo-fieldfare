import Foundation
import Testing

import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanCore

/// The prefill chunk a family gets when neither the caller nor the model's
/// profile row names one.
///
/// This exists because the rule was lost once: the CLI had its own switch with a
/// dense Qwen 3.5 case, the library took the path over without the case, and
/// dense installs silently dropped from 4,096 to the 128 default — which also
/// takes them off the ANE, since the sidecar only accepts exactly 4,096. The
/// assertion is cheap; the regression was invisible in every output.
@Suite struct DefaultPrefillChunkTests {
    private let fallback = 128

    @Test func theLongChunkFamiliesAskForTheLongChunk() {
        let long = RuntimeConfiguration.qwenLongPrefillChunkTokens
        for family in [ModelFamily.qwen36, .qwen38flash, .qwen35Dense] {
            #expect(
                defaultPrefillChunkTokens(family: family, fallback: fallback) == long,
                "\(family.rawValue) should prefill in \(long)-token chunks")
        }
    }

    @Test func everyOtherFamilyKeepsTheConfiguredFallback() {
        for family in [ModelFamily.qwen36MTP, .qwen38flashMTP] {
            #expect(
                defaultPrefillChunkTokens(family: family, fallback: fallback) == fallback,
                "\(family.rawValue) has no long-chunk case of its own")
        }
    }
}
