import Foundation
import TinyTitanCore
import TinyTitanFormat

/// How an Inkling 2 Q4 install is loaded on this device: the context window
/// and the expert slots per layer, from the request, `--ram-budget` and the
/// GPU working set (TT-121 phase 2).
///
/// What the process holds is the resident weights (mapped `model_weights.bin`),
/// the expert slots, the per-visit KV caches for the window, and a fixed
/// reserve for everything that does not scale (sliding rings, hidden pool,
/// chunk scratch, logits). A budget is split in that order: weights and
/// reserve first, then the window's KV, then as many slots as the rest holds.
package struct InklingQ4LoadPlan: Equatable, Sendable {
    /// Bytes kept for the buffers that do not scale with the window: measured
    /// on the GGUF route at about 420 MB fixed plus headroom.
    package static let reserveBytes = 512 << 20
    /// The smallest window worth serving.
    package static let minimumContext = 4_096
    /// Expert memory when no `--ram-budget` is given: the same default the
    /// shipped streaming variant uses for its expert cache.
    package static let defaultExpertBytes = 8 << 30

    package let context: Int
    package let slots: Int
    /// Bytes the plan expects the process to hold (weights, slots, KV, reserve).
    package let plannedBytes: Int

    /// Geometry the arithmetic needs, read from the install without loading it.
    package struct Geometry: Equatable, Sendable {
        package let residentBytes: Int
        /// One slot in every layer: the sum of the layers' expert strides.
        package let bytesPerSlot: Int
        package let bytesPerPosition: Int
        package let experts: Int
        package let expertsPerToken: Int
        package let modelContext: Int

        package init(
            residentBytes: Int, bytesPerSlot: Int, bytesPerPosition: Int, experts: Int,
            expertsPerToken: Int, modelContext: Int
        ) {
            self.residentBytes = residentBytes
            self.bytesPerSlot = bytesPerSlot
            self.bytesPerPosition = bytesPerPosition
            self.experts = experts
            self.expertsPerToken = expertsPerToken
            self.modelContext = modelContext
        }

        package init(directory: URL) throws {
            let install = try InklingQ4Install(directory: directory)
            let manifest = try InklingQ4Manifest(install: install)
            var fullVisits = 0
            for visit in 0..<manifest.unrolledLayers
            where !manifest.layers[manifest.storedLayer(forUnrolled: visit)].isSliding {
                fullVisits += 1
            }
            let kvWidth = manifest.headDimension * manifest.layers[0].kvHeads
            // A slot holds the repacked expert (TT-125), which can be larger than
            // the file's stride (the MXFP8 layers).
            let strides = install.slotPlans.map(\.slotStride)
            self.init(
                residentBytes: Int(
                    install.manifest.files[InklingQ4Install.residentFile]?.size ?? 0),
                bytesPerSlot: strides.reduce(0, +),
                bytesPerPosition: fullVisits * kvWidth * 2 * 2,
                experts: manifest.expertCount, expertsPerToken: manifest.expertsPerToken,
                modelContext: manifest.contextLength)
        }
    }

    package enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case budgetTooSmall(budget: Int, minimum: Int)

        package var description: String {
            switch self {
            case .budgetTooSmall(let budget, let minimum):
                "--ram-budget \(budget >> 20) MiB cannot hold the Inkling 2 Q4 weights, a "
                    + "\(InklingQ4LoadPlan.minimumContext)-token window and the minimum expert "
                    + "slots; it needs at least \(minimum >> 20) MiB"
            }
        }
    }

    /// Default context cap when `--ram-budget` is active and the caller did not
    /// pass an explicit `--max-context` (TT-126).
    package static let defaultBudgetedContext = 32_768

    /// The plan for `requested` context tokens. `ramBudget` caps the whole
    /// process; `explicitSlots` (`--expert-cache-slots`) wins over the budget's
    /// slot count but never exceeds the expert count; `workingSet` is the
    /// GPU's recommended maximum, which nothing may exceed.
    package static func make(
        geometry g: Geometry, requested: Int, ramBudget: Int?, explicitSlots: Int?,
        workingSet: Int, maxContextExplicit: Bool = true
    ) throws -> InklingQ4LoadPlan {
        let fixed = g.residentBytes + reserveBytes
        let minimumSlots = g.expertsPerToken
        let ceiling = min(ramBudget ?? Int.max, workingSet)
        let minimum = fixed + minimumSlots * g.bytesPerSlot + minimumContext * g.bytesPerPosition
        guard ceiling >= minimum else {
            throw Error.budgetTooSmall(budget: ceiling, minimum: minimum)
        }
        let targetContext =
            (!maxContextExplicit && ramBudget != nil)
            ? min(requested, defaultBudgetedContext) : requested
        // The window takes what is left after the weights and the minimum
        // slots, up to the request and the model's own window.
        let forWindow = ceiling - fixed - minimumSlots * g.bytesPerSlot
        let fits = forWindow / max(g.bytesPerPosition, 1)
        let context = max(
            min(targetContext, g.modelContext, fits) / 1_024 * 1_024, minimumContext)
        let left = ceiling - fixed - context * g.bytesPerPosition
        var slots = left / max(g.bytesPerSlot, 1)
        if ramBudget == nil { slots = min(slots, defaultExpertBytes / max(g.bytesPerSlot, 1)) }
        if let explicitSlots { slots = explicitSlots }
        slots = max(minimumSlots, min(slots, g.experts))
        return InklingQ4LoadPlan(
            context: context, slots: slots,
            plannedBytes: fixed + slots * g.bytesPerSlot + context * g.bytesPerPosition)
    }

    package init(context: Int, slots: Int, plannedBytes: Int) {
        self.context = context
        self.slots = slots
        self.plannedBytes = plannedBytes
    }
}
