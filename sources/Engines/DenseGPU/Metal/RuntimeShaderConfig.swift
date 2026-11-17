import Foundation
import Metal
import TinyTitanCore

/// The runtime serves every shipped family, including the model-specific
/// Nemotron shader module that the generic Core library deliberately does not
/// name (spec §1.3: no model knowledge in Core). The full shader configuration
/// lives here, in the runtime, and is what the 6.x engine compiles. A 7.0
/// variant that needs only the generic shaders uses `MetalContext`'s Core
/// default directly.
extension MetalContext {
    public static func runtimeDefault(device: MTLDevice? = nil) throws -> MetalContext {
        let modules = MetalContext.shaderModules + ["nemotron"]
        let subdirectories = MetalContext.shaderSubdirectories
            .merging(["nemotron": "Metal/Nemotron"]) { _, new in new }
        let dev: MTLDevice
        if let device {
            dev = device
        } else {
            guard let created = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
            dev = created
        }
        return try MetalContext(
            device: dev, shaderModules: modules, shaderSubdirectories: subdirectories)
    }
}
