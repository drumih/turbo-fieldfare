import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppRuntimeOptionsTests {
    @Test func defaultsMatchProduction() throws {
        let options = AppRuntimeOptions()
        #expect(options.expertCacheSlots == 16)
        #expect(options.expertCachePolicy == .lfu)
        #expect(options.prefillEnabled)
        #expect(options.prefillChunkTokens == 128)
        #expect(options.rdadvisePolicy == .off)
        #expect(options.modelVerification == .fullSha256)

        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime == .production)
        #expect(options.resultSummary ==
            "Cache 16 LFU, prefill 128, FP16 KV, RDADVISE off, full SHA-256")
    }

    @Test func everyPublicChoiceMapsToRuntime() throws {
        for slots in AppRuntimeOptions.allowedSlotCounts {
            let runtime = try AppRuntimeOptions(expertCacheSlots: slots)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.expertCacheSlots == slots)
        }
        for chunk in AppRuntimeOptions.allowedPrefillChunkTokens {
            let runtime = try AppRuntimeOptions(prefillChunkTokens: chunk)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.prefillConfig.chunkTokens == chunk)
        }
        for policy in AppRDAdvicePolicy.allCases {
            let runtime = try AppRuntimeOptions(rdadvisePolicy: policy)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.rdadvisePolicy == policy.runtimeValue)
        }
    }

    @Test func runtimeAndTrustChoicesAreExplicit() throws {
        let options = AppRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: .adaptive,
            modelVerification: .trustedInstall)
        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: true)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.rdadvisePolicy == .adaptive)
        #expect(runtime.headPath == .logits)
        #expect(options.modelVerification.runtimeValue == .sizeCheckTrustedReceipt)
    }

    @Test func validationRejectsValuesOutsideClosedSets() {
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(expertCacheSlots: 12).validate()
        }
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(prefillChunkTokens: 96).validate()
        }
    }

    @Test func loadedRuntimeKeyTracksOnlyLoadTimeChoices() {
        let directory = URL(fileURLWithPath: "/tmp/model.gturbo")
        let base = AppRuntimeOptions()
        let baseline = AppLoadedRuntimeKey(
            modelDirectory: directory, maxContextTokens: 4096, options: base)

        var variants: [AppRuntimeOptions] = []
        var value = base
        value.expertCacheSlots = 24; variants.append(value)
        value = base; value.expertCachePolicy = .lru; variants.append(value)
        value = base; value.rdadvisePolicy = .bounded; variants.append(value)
        value = base; value.modelVerification = .trustedInstall; variants.append(value)

        for variant in variants {
            #expect(AppLoadedRuntimeKey(
                modelDirectory: directory,
                maxContextTokens: 4096,
                options: variant) != baseline)
        }
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: base,
            forceLogitsHead: true) != baseline)

        value = base; value.prefillEnabled = false
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: value) == baseline)
        value = base; value.prefillChunkTokens = 64
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: value) == baseline)
    }
}
