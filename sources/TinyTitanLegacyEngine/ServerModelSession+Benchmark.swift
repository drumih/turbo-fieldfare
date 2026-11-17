import TinyTitan
import TinyTitanCore
import TinyTitanLib

// Benchmark access to the loaded runner.
//
// The pass-cost bench (`TinyTitanBench passcost`, gate 2 of
// docs/plan-wide-verify.md) has to time the runner production builds -- the
// same expert-cache slots, read-ahead and head -- so it loads through
// `ServerModelSession.load` and takes the parts from here rather than
// re-deriving a RuntimeConfiguration that could drift from the server's.
// `package`, not `public`: this is not part of the library's promise.
extension ServerModelSession {
    package var benchmarkRunner: RealForwardRunner {
        guard let runner = runner as? RealForwardRunner else {
            fatalError("benchmark probes require the 6.x runner")
        }
        return runner
    }
    package var benchmarkTokenizer: GFTokenizer { tokenizer }
}
