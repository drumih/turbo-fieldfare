import TinyTitanCPUEngine
import TinyTitanLib

/// What the CLI needs from either engine: the GPU `Engine` and the dense CPU
/// `CPUEngine` publish the same descriptor, sampling row, `Session` and
/// unload/warm-up shape. The protocol exists only in the CLI; the facade
/// targets do not depend on it.
package protocol EngineFacade: Sendable {
    var descriptor: ModelDescriptor { get }
    var samplingDefaults: SamplingDefaults { get }
    func session(system: String?) async -> Session
    func awaitExpertWarmup() async
    func unload() async
}

extension Engine: EngineFacade {}
extension CPUEngine: EngineFacade {}
