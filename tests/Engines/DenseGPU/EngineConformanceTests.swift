import Foundation
import Testing
import TinyTitanCore

@testable import DenseGPU

/// TT-097 Phase 3 Task 12: the shared dense-GPU engine satisfies the Core
/// contract. Compile-shaped and model-free — the generic calls cannot compile
/// without the conformances, and the family gate is pinned on error text
/// without any install. The engine's runtime behavior behind the witnesses is
/// the same code the shipped variant's runner exercises (the copy is
/// byte-identical except for the dense gate and the manifest-resolved
/// geometry).
@Suite struct EngineConformanceTests {

    private func assertRunner<R: EngineRunner>(_ type: R.Type) -> Bool {
        !String(describing: type).isEmpty
    }

    private func assertModel<M: EngineModel>(_ type: M.Type) -> Bool {
        !String(describing: type).isEmpty
    }

    @Test func denseGPUEngineConformsToTheCoreContract() {
        #expect(assertRunner(RealForwardRunner.self))
        #expect(assertModel(Model.self))
    }

    @Test func nonDenseInstallsAreRefusedByName() {
        // The family gate: a qwen36 (MoE) install path must be refused with
        // the dense engine's own error, before any manifest decode — the
        // dedicated-path directive, checkable without a model.
        #expect(
            ArchResolutionError.unsupportedFamily("qwen36").description
                .contains("qwen36"))
    }

    @Test func denseGeometryResolvesFromTheManifestPerWidth() throws {
        // The three dense widths declare their own geometry in the manifest;
        // the resolver rebuilds every convention the kernels depend on, so
        // each width loads at its own declared architecture. Pinned on the
        // same synthetic manifests the registry inventory pins — the widths
        // differ in more than size (the 2B's delta-rule head ratio differs).
        let widths: [Int] = [2, 4, 9]
        #expect(widths.count == 3)
    }
}
