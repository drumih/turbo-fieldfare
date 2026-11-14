import Testing

@testable import TinyTitan
import TinyTitanCore

/// T25/T26: the page-cache decision is a pure function of physical memory,
/// install size and one explicit override. A 24 GB host keeps the bounded
/// reader for every install that does not fit; a host with twice the install's
/// RAM may use the page cache; an unreadable input fails closed.
@Suite struct PageCachePolicyTests {
    private let gib: UInt64 = 1 << 30

    private func resolve(
        ram: UInt64, install: UInt64?, environment: [String: String] = [:]
    ) -> PageCacheDecision {
        PageCachePolicyResolver.resolve(
            physicalMemory: ram, installBytes: install, environment: environment)
    }

    @Test func aTwentyFourGigabyteHostStaysBounded() {
        // The three installs under models/ are 19, 34 and 96 GB. None is
        // below half of 24 GB, so the automatic rule changes nothing here.
        for install in [19 * gib, 34 * gib, 96 * gib] {
            let decision = resolve(ram: 24 * gib, install: install)
            #expect(decision.policy == .bounded)
            #expect(decision.reason.contains("2x"))
        }
    }

    @Test func twiceTheInstallEnablesThePageCache() {
        #expect(resolve(ram: 128 * gib, install: 19 * gib).policy == .pageCache)
        // Exactly 2x is "comfortably exceeds" by the stated rule.
        #expect(resolve(ram: 64 * gib, install: 32 * gib).policy == .pageCache)
        // One byte below it is not.
        #expect(resolve(ram: 64 * gib, install: 32 * gib + 1).policy == .bounded)
    }

    @Test func unreadableInputsFailClosed() {
        let noMemory = resolve(ram: 0, install: 1 * gib)
        #expect(noMemory.policy == .bounded)
        #expect(noMemory.reason.contains("memory"))
        let noInstall = resolve(ram: 128 * gib, install: nil)
        #expect(noInstall.policy == .bounded)
        #expect(noInstall.reason.contains("install size"))
        let zeroInstall = resolve(ram: 128 * gib, install: 0)
        #expect(zeroInstall.policy == .bounded)
        let overflow = resolve(ram: UInt64.max, install: UInt64.max)
        #expect(overflow.policy == .bounded)
        #expect(overflow.reason.contains("overflow"))
    }

    @Test func anExplicitOverrideWinsInBothDirections() {
        let enabled = resolve(
            ram: 8 * gib, install: 19 * gib,
            environment: ["TINYTITAN_BOUNDED_IO": "0"])
        #expect(enabled.policy == .pageCache)
        #expect(enabled.reason.contains("explicit"))
        let disabled = resolve(
            ram: 512 * gib, install: 1 * gib,
            environment: ["TINYTITAN_BOUNDED_IO": "1"])
        #expect(disabled.policy == .bounded)
        #expect(disabled.reason.contains("explicit"))
    }

    @Test func anInvalidOverrideIsRefusedRatherThanReadAsEnable() {
        for value in ["true", "on", "0x0", ""] {
            let decision = resolve(
                ram: 512 * gib, install: 1 * gib,
                environment: ["TINYTITAN_BOUNDED_IO": value])
            #expect(decision.policy == .bounded)
            #expect(decision.reason.contains("not 0 or 1"))
        }
    }

    @Test func bypassesCacheFollowsThePolicy() {
        #expect(PageCachePolicy.bounded.bypassesCache)
        #expect(!PageCachePolicy.pageCache.bypassesCache)
        #expect(PageCachePolicy.pageCache.rawValue == "page-cache")
    }

    @Test func installSizeSumsManifestFilesAndRefusesToWrap() {
        #expect(PageCachePolicyResolver.installSize(fileSizes: []) == 0)
        #expect(
            PageCachePolicyResolver.installSize(fileSizes: [3, 4, 5]) == 12)
        #expect(
            PageCachePolicyResolver.installSize(
                fileSizes: [UInt64.max, 1]) == UInt64.max)
        #expect(
            PageCachePolicyResolver.installSize(
                fileSizes: [UInt64.max - 1, 1]) == UInt64.max - 1 + 1)
    }
}
