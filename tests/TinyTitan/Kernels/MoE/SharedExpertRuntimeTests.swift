import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The shared-expert dispatcher's own contract: which widths it builds an
/// implementation for, and what it does with the ones it does not.
///
/// A 6-bit arm used to sit in this switch and compose `AffineQuantGEMV`s -- and
/// that GEMV's initializer asserts `[4, 8].contains(weightBits)`, so the arm
/// could only ever abort the process. It was never a supported width: 6-bit was
/// withdrawn as a format, is refused by `ManifestReader.supportedWeightBits`
/// when a slot declares it, and is refused by `SSDAIManifestQuantV1` when an
/// override does. The only reachable behaviour left for it was the trap, so the
/// arm is gone and 6 falls through to the error every other unsupported width
/// already gets.
@Suite struct SharedExpertRuntimeTests {

    @Test(arguments: [4, 8])
    func aSupportedWidthBuildsAnImplementation(bits: Int) throws {
        let context = try MetalContext()
        let runtime = try SharedExpertRuntime(
            context: context, weightBits: bits, siluActivation: true)
        #expect(runtime.weightBits == bits)
    }

    /// The point of this suite: an unsupported width is *reported*. A test that
    /// caught a trap would have to survive it first, which is why the 6-bit arm
    /// being deleted is the fix and not the test's job.
    @Test(arguments: [6, 3, 16, 0, 32])
    func anUnsupportedWidthThrowsNamingTheWidth(bits: Int) throws {
        let context = try MetalContext()
        do {
            _ = try SharedExpertRuntime(
                context: context, weightBits: bits, siluActivation: true)
            Issue.record("a \(bits)-bit shared expert must not be constructed")
        } catch let error as SharedExpertError {
            guard case .unsupportedWeightBits(let reported) = error else {
                Issue.record("expected unsupportedWeightBits, got \(error)")
                return
            }
            #expect(reported == bits)
            #expect(error.description.contains("\(bits)"))
        }
    }
}
