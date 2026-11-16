import Darwin
import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
@testable import TinyTitanCore

@testable import TinyTitan

/// The per-layer ring-depth seam §5al's layer-selective arm needs: a profile
/// naming which links get more than the profile's one read, the cap applied
/// after the residency filter, and every other link left exactly as it ships.
@Suite struct ExpertPrefetchDepthTests {

    // MARK: - the profile document

    @Test func noEnvLeavesEveryLinkAtTheProfileDepth() throws {
        let table = try PrefetchDepthProfile.parse(raw: nil, numLayers: 48, topKExperts: 10)
        #expect(table.isEmpty)
        let empty = try PrefetchDepthProfile.parse(raw: "", numLayers: 48, topKExperts: 10)
        #expect(empty.isEmpty, "an explicitly blank switch is off, not a parse error")
    }

    @Test func aProfileKeyNamesTheSourceLayerOfTheLinkItDeepens() throws {
        // A link is layer L -> layer L+1, keyed by L exactly as the trace screen
        // keys it, so the layer list from §5al transcribes without a shift.
        // The ring is told the target layer, so the table comes out keyed L+1.
        let table = try PrefetchDepthProfile.parse(raw: "3=2", numLayers: 48, topKExperts: 10)
        #expect(table == [4: 2])
        let many = try PrefetchDepthProfile.parse(
            raw: "0=2,7=3,46=2", numLayers: 48, topKExperts: 10)
        #expect(many == [1: 2, 8: 3, 47: 2])
    }

    @Test func aProfileItemItCannotApplyIsRefused() {
        // Each of these would silently mis-stage reads if it were ignored, and
        // an arm's layer list is transcribed from a document, so a typo has to
        // stop the run rather than quietly run a different arm.
        let bad = [
            "3", "3=", "=2", "x=2", "3=y", "3=2,", "3=2,,4=2", "3 =2", " 3=2",
            "3=2,3=4", "3=0", "3=-1", "3=11", "47=2", "48=2",
        ]
        for item in bad {
            #expect(
                throws: ModelError.self, "profile \(item) must be refused"
            ) {
                _ = try PrefetchDepthProfile.parse(raw: item, numLayers: 48, topKExperts: 10)
            }
        }
    }

    // MARK: - how a depth becomes reads

    @Test func theDepthCapAppliesAfterTheResidencyFilter() {
        // The withdrawn v4.3 ring capped the ranking *first*: here that would
        // spend both slots on 10 and 11, drop them as resident, and issue
        // nothing. Filter-then-cap issues the next two names instead.
        let staged = ExpertPrefetchRing.staged(
            experts: [10, 11, 12, 13], resident: [10, 11], active: [],
            freeSlots: 4, depth: 2)
        #expect(staged == [12, 13])
        let shallower = ExpertPrefetchRing.staged(
            experts: [10, 11, 12, 13], resident: [10, 11], active: [],
            freeSlots: 4, depth: 1)
        #expect(shallower == [12])
    }

    @Test func theCapDropsNamesAlreadyInFlightAndRepeats() {
        let staged = ExpertPrefetchRing.staged(
            experts: [12, 12, 13, 14], resident: [], active: [13], freeSlots: 4, depth: 3)
        #expect(staged == [12, 14])
    }

    @Test func freeSlotsRemainTheHardCap() {
        let staged = ExpertPrefetchRing.staged(
            experts: [12, 13, 14, 15], resident: [], active: [], freeSlots: 2, depth: 4)
        #expect(staged == [12, 13])
    }

    @Test func aCapAtTheRingCapacityStagesWhatNoCapWould() {
        // The shipped behaviour: every link's depth equals the slot count, so
        // adding the parameter must not change one read. Free slots never
        // exceed the capacity, so a cap at the capacity is a no-op.
        let capped = ExpertPrefetchRing.staged(
            experts: [4, 5, 6], resident: [5], active: [], freeSlots: 2, depth: 2)
        #expect(capped == [4, 6])
        let full = ExpertPrefetchRing.staged(
            experts: [4, 5, 6], resident: [], active: [], freeSlots: 3, depth: 3)
        #expect(full == [4, 5, 6])
    }

    // MARK: - the seam in a constructed runner

    /// The env is read at runner construction, so it is set before each load.
    private func makeRunner(
        from dir: URL, ctx: MetalContext, layerDepth: String?
    ) throws -> RealForwardRunner {
        setenv("TINYTITAN_PREDICTIVE_PREFETCH", "1", 1)
        if let layerDepth {
            setenv("TINYTITAN_PREFETCH_LAYER_DEPTH", layerDepth, 1)
        } else {
            unsetenv("TINYTITAN_PREFETCH_LAYER_DEPTH")
        }
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(forceLogitsHead: true)
        return try RealForwardRunner(
            model: model, context: ctx, maxContext: 64, slots: 1,
            runtimeConfiguration: runtime)
    }

    private func decode(_ runner: RealForwardRunner, ctx: MetalContext) async throws -> [Float] {
        guard
            let logits = ctx.device.makeBuffer(
                length: 1024 * MemoryLayout<Float16>.stride, options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        return Fp16Buffer.read(logits, count: 1024)
    }

    /// The toy is 4 layers, so its last link is layer 2 -> layer 3.
    @Test func theRunnerRefusesAProfileThatNamesANonexistentLink() throws {
        defer {
            unsetenv("TINYTITAN_PREFETCH_LAYER_DEPTH")
            unsetenv("TINYTITAN_PREDICTIVE_PREFETCH")
        }
        let dir = try QwenToySynthetic.write(weightBits: 8)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        var message = ""
        do {
            _ = try makeRunner(from: dir, ctx: ctx, layerDepth: "9=2")
            Issue.record("a profile naming layer 9 of a 4-layer toy must not build")
        } catch let error as ModelError {
            if case .internalInconsistency(let detail) = error { message = detail }
        }
        #expect(message.contains("9"), Comment(rawValue: message))
    }

    @Test func aProfileWidensTheRingAndMovesNoLogit() async throws {
        defer {
            unsetenv("TINYTITAN_PREFETCH_LAYER_DEPTH")
            unsetenv("TINYTITAN_PREDICTIVE_PREFETCH")
        }
        let dir = try QwenToySynthetic.write(weightBits: 8)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        // Both arms read the same install, so a difference would be the seam's.
        let profiled = try makeRunner(from: dir, ctx: ctx, layerDepth: "0=2,2=3")
        let uniform = try makeRunner(from: dir, ctx: ctx, layerDepth: nil)
        #expect(profiled.prefetchRingCapacity == 3)
        #expect(uniform.prefetchRingCapacity == 1)
        let profiledLogits = try await decode(profiled, ctx: ctx)
        let uniformLogits = try await decode(uniform, ctx: ctx)
        #expect(profiledLogits.allSatisfy { $0.isFinite })
        // A speculative read only moves bytes into a staging slot; the exact
        // route still decides what is used, so the depth cannot move a logit.
        #expect(profiledLogits == uniformLogits)
    }
}
