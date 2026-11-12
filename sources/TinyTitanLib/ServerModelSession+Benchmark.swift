import TinyTitan

// Benchmark access to the loaded runner.
//
// The pass-cost bench (`TinyTitanBench passcost`, gate 2 of
// docs/plan-wide-verify.md) has to time the runner production builds -- the
// same expert-cache slots, read-ahead and head -- so it loads through
// `ServerModelSession.load` and takes the parts from here rather than
// re-deriving a RuntimeConfiguration that could drift from the server's.
// `package`, not `public`: this is not part of the library's promise.
extension ServerModelSession {
    package nonisolated var benchmarkRunner: RealForwardRunner { runner }
    package nonisolated var benchmarkTokenizer: GFTokenizer { tokenizer }
}
