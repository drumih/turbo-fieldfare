import Testing
import TinyTitanCore

@testable import Ornith35B4Bit

/// TT-097 Phase 1 Task 5: the shipped variant's engine satisfies the Core
/// contract. Compile-shaped and model-free — the generic calls cannot compile
/// without the conformances, and no install is needed to assert them. The
/// runtime behavior behind the witnesses (`statistics` forwarding, the state
/// members) is the same code the 6.x runner exercises in
/// `tests/TinyTitan/Engine/`: the variant carries byte-identical copies of
/// both the runner and the conformance block.
@Suite struct EngineConformanceTests {

    private func assertRunner<R: EngineRunner>(_ type: R.Type) -> Bool {
        !String(describing: type).isEmpty
    }

    private func assertModel<M: EngineModel>(_ type: M.Type) -> Bool {
        !String(describing: type).isEmpty
    }

    @Test func variantEngineConformsToTheCoreContract() {
        #expect(assertRunner(RealForwardRunner.self))
        #expect(assertModel(Model.self))
    }
}
