import Metal
import Testing

@testable import TinyTitan
import TinyTitanCore

/// The context is where a caller's device becomes the engine's device.
///
/// `Engine` used to refuse anything but the system default. Now it hands the
/// caller's device to the loader, and this is the link in that chain: the
/// context — and so every queue, pipeline and buffer built from it — belongs to
/// the device it was given, not to whichever one happens to be default.
@Suite struct MetalContextTests {
    @Test func theContextUsesTheDeviceItWasGiven() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let context = try MetalContext(device: device)
        #expect(context.device === device)
        #expect(context.queue.device === device)
    }

    /// The argument-free initializer keeps its old meaning, which is what the
    /// server and the CLI still rely on.
    @Test func theDefaultInitializerStillTakesTheSystemDevice() throws {
        let context = try MetalContext()
        let system = try #require(MTLCreateSystemDefaultDevice())
        #expect(context.device.registryID == system.registryID)
    }
}
