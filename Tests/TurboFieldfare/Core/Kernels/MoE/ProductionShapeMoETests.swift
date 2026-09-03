import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The routed-width contract at the shape the specialised pipelines demand.
///
/// `MoE` compiles a second set of pipelines with the checkpoint's shape baked
/// into function constants, and selects them only when the runtime shape agrees
/// on every axis, K included. The K term carries real weight: `FC_ROUTER_TOP_K`
/// and `FC_MOE_TOP_K` override the runtime `top_k` the kernels read, so an
/// `--experts-per-token 4` or `6` run that reached a specialised pipeline would
/// select and reduce eight experts through a K-slot buffer.
///
/// No toy shape can hold that gate. The specialised pipelines are keyed on
/// D=2816, F=704 and 128 experts; `MoEFusedFFNTests` and `RouterTopKTests` run
/// at D=128 / F=64, so they take the generic pipelines at every width and would
/// stay green with the `topK == realDecodeTopK` term deleted. Only a suite at
/// the production shape puts the specialised pipelines within reach of a narrow
/// width, which is what these tests do.
///
/// Real decode shapes mean real allocations, so this suite is opt-in behind
/// `GTURBO_HEAVY=1` and must run alone: one `--filter` per invocation, with
/// `--no-parallel`, and never alongside another production-shape or real-model
/// process.
@Suite(.enabled(if: HeavyTestGate.enabled), .serialized)
struct ProductionShapeMoETests {
    private static let hiddenD = 2816
    private static let routedF = 704
    private static let numExperts = 128
    /// The routed argument buffer's hard cap, and the width the specialised
    /// pipelines bake. Output buffers are allocated this wide so a kernel that
    /// ignored the narrower runtime K overruns a sentinel instead of the heap.
    private static let maxRoutedSlots = 8
    /// Enough repeats that a threadgroup barrier missing from the phase-2
    /// reduce shows up as a changed byte rather than as a lucky pass.
    private static let synchronizationRepetitions = 32

    private static let indexSentinel = UInt32.max
    private static let weightSentinel: UInt16 = 0xDEAD

    // MARK: - Router

    private struct RouterFixture {
        let rows: [Quantization.Int8AffineRow]
        let hidden: [Float]
        /// Held as the exact BF16 value the kernel reads, so the reference and
        /// the kernel scale the hidden state by the same number.
        let effectiveScale: [Float]
        let expertScale: [Float]
    }

    private struct RouterResult {
        let indices: [UInt32]
        let weights: [Float]
        let weightBits: [UInt16]
    }

    /// 128 experts at the production D, with logits that fall monotonically in
    /// the expert index and a deliberate exact tie between experts 3 and 4. The
    /// tie straddles the narrowest width's boundary: at K=4 expert 3 is the last
    /// selected and expert 4 the first rejected, so membership rests on the
    /// kernel's `e < top_idx[i]` tie-break rather than only on order.
    ///
    /// The gains are spaced so the top eight logits span roughly one nat. A
    /// wider spacing would drive the softmax to one-hot, where renormalising
    /// over eight and over four are numerically indistinguishable and the
    /// renormalisation invariant below would stop discriminating.
    private static func routerFixture() -> RouterFixture {
        var rng = SeedTree(0x0166_2816).key("prod-router")
        let pattern = (0..<hiddenD).map { _ in rng.uniform(0.2, 1.0) }
        var gains = (0..<numExperts).map { 1.0 - Float($0) * 0.005 }
        gains[4] = gains[3]
        let scale = Quantization.bf16ToFloat(
            Quantization.bf16Bits(1.0 / Float(hiddenD).squareRoot()))
        return RouterFixture(
            rows: gains.map { gain in
                Quantization.quantizeInt8Affine(pattern.map { $0 * gain })
            },
            hidden: (0..<hiddenD).map { _ in rng.uniform(0.5, 1.5) },
            effectiveScale: [Float](repeating: scale, count: hiddenD),
            // Exactly representable in BF16 and in FP32, so dividing the output
            // weights by them recovers the softmax without adding error.
            expertScale: (0..<numExperts).map { 0.5 + Float($0 % 5) * 0.25 })
    }

    /// The gate the specialised router select protects: at a narrow width the
    /// kernel must select K experts, write exactly K slots, and softmax over
    /// those K. A pipeline with `FC_ROUTER_TOP_K = 8` would instead select eight
    /// and renormalise over eight, scaling the routed branch down by the mass
    /// the slots past K carry.
    @Test(arguments: RuntimeConfiguration.allowedExpertsPerToken)
    func routerSelectsExactlyKExpertsAndRenormalisesOverThoseK(_ topK: Int) throws {
        let fixture = Self.routerFixture()
        let expected = Self.routerReference(fixture, topK: topK)
        let actual = try Self.runRouter(fixture, topK: topK)

        let indexTail = Array(actual.indices[topK...])
        #expect(indexTail == [UInt32](repeating: Self.indexSentinel,
                                      count: Self.maxRoutedSlots - topK),
                "router select wrote past slot \(topK) at K=\(topK): \(indexTail)")
        let weightTail = Array(actual.weightBits[topK...])
        #expect(weightTail == [UInt16](repeating: Self.weightSentinel,
                                       count: Self.maxRoutedSlots - topK),
                "router select wrote weights past slot \(topK) at K=\(topK)")

        let indices = Array(actual.indices[..<topK])
        #expect(indices == expected.indices,
                "router indices at K=\(topK) differ from the CPU top-K")
        // The fixture's logits fall with the expert index, so the top-K is the
        // first K experts at every width; the tie at 3/4 is what makes K=4 a
        // real boundary rather than a truncation of the K=8 answer.
        #expect(indices == (0..<topK).map(UInt32.init),
                "router selected \(indices) at K=\(topK)")

        let softmax = zip(indices, actual.weights[..<topK]).map { index, weight in
            weight / fixture.expertScale[Int(index)]
        }
        // Derived, not chosen. Each stored weight is `round_fp16(s_i * g_i)`;
        // FP16 round-to-nearest has unit roundoff 2^-11, so a stored value sits
        // within `2^-11 * s_i * g_i` of the exact product. The gains are exact
        // in BF16 and FP32, so dividing by one leaves each term within
        // `2^-11 * s_i`, and the exact terms sum to one, so their errors sum to
        // at most 2^-11. The FP32 divides and the accumulation add a few 2^-24,
        // which 2^-20 covers.
        let sumTolerance: Float = 0x1p-11 + 0x1p-20
        let sum = softmax.reduce(0, +)
        #expect(abs(sum - 1.0) < sumTolerance,
                "routed weights at K=\(topK) sum to \(sum), not renormalised over \(topK)")

        let maxError = zip(actual.weights[..<topK], expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        // The kernel takes `fast::exp` where the reference takes `exp`; that
        // difference dominates the FP16 storage error bounded above. Same
        // allowance `RouterTopKTests` uses against the same reference.
        #expect(maxError < 5e-3,
                "router weights at K=\(topK) differ from the reference by \(maxError)")
    }

    private static func routerReference(_ fixture: RouterFixture,
                                        topK: Int) -> (indices: [UInt32], weights: [Float]) {
        let scaled = zip(fixture.hidden, fixture.effectiveScale).map { $0 * $1 }
        let logits = DequantInt8GemvRef.apply(weightRows: fixture.rows,
                                              x: scaled,
                                              n: hiddenD)
        var paired = (0..<numExperts).map { (logits[$0], UInt32($0)) }
        paired.sort { lhs, rhs in
            lhs.0 == rhs.0 ? lhs.1 < rhs.1 : lhs.0 > rhs.0
        }
        let selected = Array(paired.prefix(topK))
        let maximum = selected[0].0
        let exponents = selected.map { exp($0.0 - maximum) }
        let sum = exponents.reduce(0, +)
        return (selected.map { $0.1 },
                zip(selected, exponents).map { item, value in
                    value / sum * fixture.expertScale[Int(item.1)]
                })
    }

    private static func runRouter(_ fixture: RouterFixture,
                                  topK: Int) throws -> RouterResult {
        let packed = fixture.rows.flatMap(\.packed)
        let scales = fixture.rows.flatMap(\.scales)
        let biases = fixture.rows.flatMap(\.biases)
        let context = try MetalContext()
        let kernel = try MoE(context: context)

        let routerWeightBuffer = try #require(context.device.makeBuffer(
            bytes: packed, length: packed.count, options: .storageModeShared))
        let scaleBuffer = try #require(bf16Buffer(context.device, scales))
        let biasBuffer = try #require(bf16Buffer(context.device, biases))
        let hiddenBuffer = try #require(Fp16Buffer.make(context.device,
                                                        values: fixture.hidden))
        let effectiveBuffer = try #require(
            bf16Buffer(context.device, fixture.effectiveScale.map(Quantization.bf16Bits)))
        let expertScaleBuffer = try #require(
            bf16Buffer(context.device, fixture.expertScale.map(Quantization.bf16Bits)))
        // Both outputs are one slot wider than the run's K and prefilled with a
        // sentinel, so an eight-wide select overwrites the sentinel instead of
        // running off the end of a K-slot allocation.
        let sentinelIndices = [UInt32](repeating: indexSentinel, count: maxRoutedSlots)
        let indexBuffer = try #require(context.device.makeBuffer(
            bytes: sentinelIndices,
            length: maxRoutedSlots * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let sentinelWeights = [UInt16](repeating: weightSentinel, count: maxRoutedSlots)
        let outWeightBuffer = try #require(bf16Buffer(context.device, sentinelWeights))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeRouterGemma4(commandBuffer: commandBuffer,
                                  weights: routerWeightBuffer,
                                  scales: scaleBuffer,
                                  biases: biasBuffer,
                                  hidden: hiddenBuffer,
                                  effectiveScale: effectiveBuffer,
                                  perExpertScale: expertScaleBuffer,
                                  outIndices: indexBuffer,
                                  outWeights: outWeightBuffer,
                                  numExperts: UInt32(numExperts),
                                  d: UInt32(hiddenD),
                                  topK: UInt32(topK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: maxRoutedSlots)
        let bitPointer = outWeightBuffer.contents().bindMemory(
            to: UInt16.self, capacity: maxRoutedSlots)
        return RouterResult(
            indices: (0..<maxRoutedSlots).map { indexPointer[$0] },
            weights: Fp16Buffer.read(outWeightBuffer, count: maxRoutedSlots),
            weightBits: (0..<maxRoutedSlots).map { bitPointer[$0] })
    }

    // MARK: - Routed FFN

    private struct RoutedExpert {
        let gate: [Quantization.Int4AffineRow]
        let up: [Quantization.Int4AffineRow]
        let down: [Quantization.Int4AffineRow]
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    /// The gate the specialised phase-1 and phase-2 pipelines protect. Phase 1
    /// writes `K * F` activation rows and phase 2 launches exactly `32 * K`
    /// threads so every SIMD group owns a slot and reaches the barrier; a
    /// pipeline with `FC_MOE_TOP_K = 8` would sum eight partials out of a
    /// K-group threadgroup, reading slots no group ever wrote.
    @Test(arguments: RuntimeConfiguration.allowedExpertsPerToken)
    func routedFFNAtProductionShapeReducesExactlyKSlots(_ topK: Int) throws {
        let d = Self.hiddenD
        let f = Self.routedF
        let context = try MetalContext()
        let kernel = try MoE(context: context)

        // Seeded per slot, so slot s holds the same expert at every width and
        // the narrow runs are a prefix of the wide one rather than a fresh draw.
        // The packed blob is uploaded and dropped inside the loop; only the
        // quantised rows the CPU reference needs stay resident.
        var gates: [[Quantization.Int4AffineRow]] = []
        var ups: [[Quantization.Int4AffineRow]] = []
        var downs: [[Quantization.Int4AffineRow]] = []
        var routedBlobs: [MTLBuffer] = []
        var slotOffsets: MoEExpertOffsets?
        for slot in 0..<topK {
            let expert = Self.routedExpert(slot: slot)
            routedBlobs.append(try #require(context.device.makeBuffer(
                bytes: expert.bytes,
                length: expert.bytes.count,
                options: .storageModeShared)))
            gates.append(expert.gate)
            ups.append(expert.up)
            downs.append(expert.down)
            slotOffsets = slotOffsets ?? expert.offsets
        }
        let offsets = try #require(slotOffsets)

        var rng = SeedTree(0x0167_2816).key("prod-moe-activations")
        let x = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let descendingWeights: [Float] = [0.20, 0.17, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05]
        let routingWeights = Array(
            descendingWeights.map { Float(Float16($0)) }.prefix(topK))

        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let residualBuffer = try #require(Fp16Buffer.make(context.device, values: residual))
        let routingBuffer = try #require(Fp16Buffer.make(context.device,
                                                         values: routingWeights))
        let fusedActs = try #require(Fp16Buffer.make(context.device, count: topK * f))
        let splitActs = try #require(Fp16Buffer.make(context.device, count: topK * f))
        let fusedOutput = try #require(Fp16Buffer.make(context.device, count: d))
        let splitOutput = try #require(Fp16Buffer.make(context.device, count: d))
        let argumentBuffer = try #require(
            kernel.makeRoutedArgumentBuffer(routedBlobs: routedBlobs, topK: UInt32(topK)))

        var fusedBaseline: [UInt16]?
        for repetition in 0..<Self.synchronizationRepetitions {
            let command = try #require(context.queue.makeCommandBuffer())
            kernel.encodeRoutedPersistentPhase1U16Load(commandBuffer: command,
                                                       routedArgBuffer: argumentBuffer,
                                                       routedBlobs: routedBlobs,
                                                       routedOffsets: offsets,
                                                       x: xBuffer,
                                                       acts: fusedActs,
                                                       d: UInt32(d),
                                                       f: UInt32(f),
                                                       topK: UInt32(topK))
            kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: command,
                                                      routedArgBuffer: argumentBuffer,
                                                      routedBlobs: routedBlobs,
                                                      routedOffsets: offsets,
                                                      acts: fusedActs,
                                                      routingWeights: routingBuffer,
                                                      residual: residualBuffer,
                                                      y: fusedOutput,
                                                      d: UInt32(d),
                                                      f: UInt32(f),
                                                      topK: UInt32(topK))
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw error }
            let bits = Self.halfBits(fusedOutput, count: d)
            if let fusedBaseline {
                #expect(bits == fusedBaseline,
                        "routed MoE output at K=\(topK) changed on repeat \(repetition)")
            } else {
                fusedBaseline = bits
            }
        }

        // The runtime splits phase 1 into a cache-hit encode and a cache-miss
        // encode, so the subset pipeline has to reach the same activations the
        // one-shot pipeline does before phase 2 reduces them.
        let hitSlots = (0..<(topK / 2)).map(UInt32.init)
        let missSlots = ((topK / 2)..<topK).map(UInt32.init)
        let hitSlotBuffer = try #require(context.device.makeBuffer(
            bytes: hitSlots,
            length: hitSlots.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let missSlotBuffer = try #require(context.device.makeBuffer(
            bytes: missSlots,
            length: missSlots.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        var splitBaseline: [UInt16]?
        for repetition in 0..<Self.synchronizationRepetitions {
            let command = try #require(context.queue.makeCommandBuffer())
            for (indices, buffer) in [(hitSlots, hitSlotBuffer), (missSlots, missSlotBuffer)] {
                kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: command,
                    routedArgBuffer: argumentBuffer,
                    routedBlobs: routedBlobs,
                    routedOffsets: offsets,
                    x: xBuffer,
                    acts: splitActs,
                    activeSlots: buffer,
                    activeSlotIndices: indices,
                    activeCount: UInt32(indices.count),
                    d: UInt32(d),
                    f: UInt32(f),
                    topK: UInt32(topK))
            }
            kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: command,
                                                      routedArgBuffer: argumentBuffer,
                                                      routedBlobs: routedBlobs,
                                                      routedOffsets: offsets,
                                                      acts: splitActs,
                                                      routingWeights: routingBuffer,
                                                      residual: residualBuffer,
                                                      y: splitOutput,
                                                      d: UInt32(d),
                                                      f: UInt32(f),
                                                      topK: UInt32(topK))
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw error }
            let bits = Self.halfBits(splitOutput, count: d)
            if let splitBaseline {
                #expect(bits == splitBaseline,
                        "split routed MoE output at K=\(topK) changed on repeat \(repetition)")
            } else {
                splitBaseline = bits
            }
        }

        let expected = MoeRef.applyStreamedRouted(x: x,
                                                  residual: residual,
                                                  routedGate: gates,
                                                  routedUp: ups,
                                                  routedDown: downs,
                                                  indices: Array(0..<topK),
                                                  routingWeights: routingWeights,
                                                  d: d,
                                                  f: f)
        let fused = Fp16Buffer.read(fusedOutput, count: d)
        let split = Fp16Buffer.read(splitOutput, count: d)
        #expect(fused == split,
                "subset phase 1 diverged from the one-shot pipeline at K=\(topK)")

        // `fp16ChainedReduction` (1e-2 relative) is the bar for a multi-stage
        // FP16 chain, and this is one: gate and up each accumulate 2816 affine
        // terms, the product lands back in FP16 in `acts`, the down projection
        // accumulates 704 of those, and the K partials are summed once more
        // before `y` is stored as FP16. The reference dequantises in bulk and
        // dots with vDSP, so the summation order differs at every stage; the gap
        // is FP16 rounding at the two storage points, not a difference in math.
        let relative = RelError.compute(actual: fused, reference: expected)
        let maxAbs = RelError.maxAbsDiff(fused, expected)
        #expect(relative < Tolerance.fp16ChainedReduction,
                "routed MoE D=\(d) F=\(f) K=\(topK) rel=\(relative) maxAbs=\(maxAbs)")
        let splitRelative = RelError.compute(actual: split, reference: expected)
        let splitMaxAbs = RelError.maxAbsDiff(split, expected)
        #expect(splitRelative < Tolerance.fp16ChainedReduction,
                "split routed MoE D=\(d) F=\(f) K=\(topK) rel=\(splitRelative) maxAbs=\(splitMaxAbs)")
    }

    /// Quantises row by row rather than materialising the FP32 matrix: one
    /// production-shape expert is 2816 x 704 x 2 plus 2816 x 704, and keeping
    /// the raw floats for eight of them would cost hundreds of megabytes.
    private static func int4Rows(m: Int, n: Int, seed: UInt64, label: String)
        -> [Quantization.Int4AffineRow] {
        var rng = SeedTree(seed).key(label)
        return (0..<m).map { _ in
            Quantization.quantizeInt4Affine((0..<n).map { _ in rng.uniform(-0.4, 0.4) })
        }
    }

    private static func routedExpert(slot: Int) -> RoutedExpert {
        let gate = int4Rows(m: routedF, n: hiddenD,
                            seed: 0xE100 + UInt64(slot), label: "prod-moe-gate")
        let up = int4Rows(m: routedF, n: hiddenD,
                          seed: 0xE200 + UInt64(slot), label: "prod-moe-up")
        let down = int4Rows(m: hiddenD, n: routedF,
                            seed: 0xE300 + UInt64(slot), label: "prod-moe-down")

        let groups = hiddenD / Quantization.groupSize
        var bytes = [UInt8]()
        bytes.reserveCapacity(3 * routedF * hiddenD / 2 + 6 * routedF * groups * 2)
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateW = UInt32(bytes.count); append(gate.flatMap(\.packed))
        let gateS = UInt32(bytes.count); append(gate.flatMap(\.scales))
        let gateB = UInt32(bytes.count); append(gate.flatMap(\.biases))
        let upW = UInt32(bytes.count); append(up.flatMap(\.packed))
        let upS = UInt32(bytes.count); append(up.flatMap(\.scales))
        let upB = UInt32(bytes.count); append(up.flatMap(\.biases))
        let downW = UInt32(bytes.count); append(down.flatMap(\.packed))
        let downS = UInt32(bytes.count); append(down.flatMap(\.scales))
        let downB = UInt32(bytes.count); append(down.flatMap(\.biases))

        return RoutedExpert(
            gate: gate,
            up: up,
            down: down,
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }

    // MARK: - Buffers

    private static func bf16Buffer(_ device: MTLDevice, _ bits: [UInt16]) -> MTLBuffer? {
        device.makeBuffer(bytes: bits,
                          length: bits.count * MemoryLayout<UInt16>.stride,
                          options: .storageModeShared)
    }

    /// Compares the FP16 bits the kernel wrote, not the decoded floats: a
    /// synchronisation fault that moves one output by a single ulp has to fail
    /// the repeat check, and a comparison through `Float` could round it away.
    private static func halfBits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }
}
