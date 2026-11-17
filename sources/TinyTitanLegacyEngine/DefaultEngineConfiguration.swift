import TinyTitan
import TinyTitanCore

/// The 6.x loader's value inputs, passed opaquely through Core to its factory.
package struct DefaultEngineConfiguration: Sendable {
    package let loadRuntime: RuntimeConfiguration
    package let integrityPolicy: ModelIntegrityPolicy
    package let requestedIntegrityPolicy: ModelIntegrityPolicy?
    package let loadSlots: Int
    package let runtime: RuntimeConfiguration?

    package init(
        loadRuntime: RuntimeConfiguration,
        integrityPolicy: ModelIntegrityPolicy,
        requestedIntegrityPolicy: ModelIntegrityPolicy?,
        loadSlots: Int,
        runtime: RuntimeConfiguration? = nil
    ) {
        self.loadRuntime = loadRuntime
        self.integrityPolicy = integrityPolicy
        self.requestedIntegrityPolicy = requestedIntegrityPolicy
        self.loadSlots = loadSlots
        self.runtime = runtime
    }
}
