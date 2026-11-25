import TinyTitanCore

// Compatibility shims for types moved from 6.x `TinyTitan` into
// `TinyTitanCore` (TT-134). Kept inside `sources/TinyTitan/` so legacy files
// keep compiling without touching every 6.x call site.
public typealias Quantization = TinyTitanCore.Quantization
package typealias TensorOpsMode = TinyTitanCore.TensorOpsMode
package typealias TensorOpsPlan = TinyTitanCore.TensorOpsPlan
package typealias TensorOpsPolicy = TinyTitanCore.TensorOpsPolicy
package typealias TensorOpsRuntimePath = TinyTitanCore.TensorOpsRuntimePath
package typealias TensorOpsDecision = TinyTitanCore.TensorOpsDecision
