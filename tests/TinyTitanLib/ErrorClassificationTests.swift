import Foundation
import Testing
import TinyTitan
@testable import TinyTitanCore

@testable import TinyTitanLib

/// What a load failure means to an embedder.
///
/// The loader's `ModelError` used to escape the facade unclassified, so a caller
/// could not tell "this is not an install / not a format I read" (fix your path)
/// from "this install is corrupt" (download it again) from "this build cannot run
/// this model" (ask). These pin the three-way split — and the fourth outcome,
/// where the facade deliberately declines to classify.
@Suite struct ErrorClassificationTests {
    private let directory = URL(fileURLWithPath: "/tmp/some-install")
    private let family = ModelFamily.qwen35Dense

    private func classify(_ error: ModelError) -> TinyTitanError? {
        Engine.classify(error, directory: directory, family: family)
    }

    @Test func aFormatThisBuildDoesNotReadIsTheCallersToFix() {
        let cases: [ModelError] = [
            .notASSDAIDirectory,
            .unsupportedVersion(major: 2, minor: 0),
            .unknownFlag(name: "quant"),
        ]
        for error in cases {
            guard case .unsupportedFormat? = classify(error) else {
                Issue.record("\(error) should classify as .unsupportedFormat")
                continue
            }
        }
    }

    @Test func aCorruptInstallNamesTheFileItWasReading() {
        guard
            case .integrityFailure(let path, let detail)? = classify(
                .checksumMismatch(file: "model_weights.bin"))
        else {
            Issue.record("a checksum mismatch is an integrity failure")
            return
        }
        #expect(path == "model_weights.bin")
        #expect(detail.contains("model_weights.bin"))

        // The receipt and index failures know the install rather than a file, so
        // the path is the install's.
        guard
            case .integrityFailure(let installPath, _)? = classify(
                .trustedReceiptInvalid(detail: "receipt digest mismatch"))
        else {
            Issue.record("an invalid receipt is an integrity failure")
            return
        }
        #expect(installPath == directory.path)
    }

    @Test func anArchitectureThisBuildCannotRunIsASupportQuestion() {
        let cases: [ModelError] = [
            .unsupportedArchitecture(detail: "no such family"),
            .archMismatch(field: "numLayers", expected: "40", actual: "48"),
        ]
        for error in cases {
            guard case .unsupportedFamily(let named)? = classify(error) else {
                Issue.record("\(error) should classify as .unsupportedFamily")
                continue
            }
            #expect(named == family.rawValue)
        }
    }

    @Test func anInstallMissingItsPiecesIsNotAnInstall() {
        let cases: [ModelError] = [
            .partialInstall(path: "/tmp/some-install"),
            .missingFile(name: "manifest.json"),
            .tensorNotFound(name: "layers.0.attn"),
            .tensorSizeMismatch(name: "layers.0.attn", expected: 16, actual: 8),
            .expertStrideNotPageAligned(stride: 100, pageSize: 16_384),
        ]
        for error in cases {
            guard case .notAnInstall? = classify(error) else {
                Issue.record("\(error) should classify as .notAnInstall")
                continue
            }
        }
    }

    @Test func theRuntimeAndTheMachineAreNotTheInstall() {
        // These are about the machine or the runtime, not about the install, so
        // the facade declines to classify them: `nil` means "rethrown as it is"
        // rather than flattened into a case that would claim more than it knows.
        let cases: [ModelError] = [
            .commandBufferFailed(detail: "GPU error"),
            .internalInconsistency(detail: "arch/kernel mismatch"),
            .residentBufferWrapFailed,
            .expertCacheUnplaceable(detail: "budget too small"),
            .posixFailed(call: "pread", errno: 5),
        ]
        for error in cases {
            #expect(classify(error) == nil, "\(error) should be rethrown, not classified")
        }
    }
}
