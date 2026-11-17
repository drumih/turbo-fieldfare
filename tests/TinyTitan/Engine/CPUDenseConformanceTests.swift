import Testing
import TinyTitanCore

import Qwen35Dense2B4Bit
import Qwen35Dense2B8Bit
import Qwen35Dense4B4Bit
import Qwen35Dense4B8Bit
import Qwen35Dense9B4Bit
import Qwen35Dense9B8Bit

/// TT-097 Phase 1 Task 5: every dense variant's CPU engine conforms to the
/// Core `CPUEngineRunner` contract. Compile-shaped on purpose — a generic
/// call that cannot compile without the conformance — because the engines'
/// member surface is byte-identical to the runtime's `CPUQwen35`, whose
/// contract surface is driven for real in `CPUEngineTests`. The six variant
/// modules are imported here, and nowhere else, so the duplicated engine
/// family's identical type names never share one scope unqualified.
@Suite struct CPUDenseConformanceTests {

    private func serve<R: CPUEngineRunner>(_ type: R.Type) -> Bool {
        !String(describing: type).isEmpty
    }

    @Test func everyDenseCPEngineConformsToTheCoreContract() {
        #expect(serve(Qwen35Dense2B4Bit.CPUQwen35.self))
        #expect(serve(Qwen35Dense2B8Bit.CPUQwen35.self))
        #expect(serve(Qwen35Dense4B4Bit.CPUQwen35.self))
        #expect(serve(Qwen35Dense4B8Bit.CPUQwen35.self))
        #expect(serve(Qwen35Dense9B4Bit.CPUQwen35.self))
        #expect(serve(Qwen35Dense9B8Bit.CPUQwen35.self))
    }
}
