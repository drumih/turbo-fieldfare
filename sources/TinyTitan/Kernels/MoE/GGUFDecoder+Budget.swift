import Foundation

/// How the as-is decoder sizes what it holds: the expert cache from the RAM
/// budget, the read-ahead depth, and the opt-in routing log. Split from
/// `GGUFDecoder.swift` as pure code motion.
extension GGUFDecoder {
    /// Expert slots per layer: the RAM budget (`TINYTITAN_GGUF_EXPERT_CACHE_GB`,
    /// default a twelfth of physical memory) divided by what one slot costs
    /// across all layers, at least two top-k sets (the pinned layer and the one
    /// being fetched) and at most every expert.
    ///
    /// The default is measured, not guessed. The pool is wired Metal memory, so
    /// every GiB it takes is a GiB the mmap'd attention and GDN weights lose to
    /// the page cache. On the 24 GB M-series Mac (essay prompt, 80 tokens, two
    /// rounds, 2026-11-11): 2 GiB 2.07/2.46 tok/s, 3 GiB 1.07/2.46, 4 GiB
    /// 0.75/2.34 with wait(attn) doubling to 199 ms; the hit rate rose 52% -> 64%
    /// and bought nothing. 6 GiB (the old quarter) drove free memory to 28% and
    /// the GPU into page-fault stalls. A twelfth is 2 GiB there.
    ///
    /// With a process target (`ramBudgetBytes`, the CLI's `--ram-budget`, the
    /// launcher's `--ram`) the cache gets what the target leaves after the rest
    /// of the model: every non-expert tensor as stored (the mapped matrices are
    /// paged in on use and count against the process), the bf16 arena built
    /// from some of them, and a working margin for KV, scratch and the
    /// runtime. A target that leaves less than the two-set floor is refused
    /// with the floor named, rather than silently exceeded.
    static func expertSlotsPerLayer(
        model: GGUFModel, arch: ArchConfig, ramBudgetBytes: Int?, arenaBytes: Int
    ) throws -> Int {
        let perSlot = try GGUFExpertBufferCache.bytesPerSlotAcrossLayers(model: model)
        let floor = 2 * arch.topKExperts
        let budget: Double
        if let ramBudgetBytes {
            let resident = residentBytesOutsideTheExpertCache(
                model: model, arenaBytes: arenaBytes)
            let required = resident + floor * perSlot
            guard ramBudgetBytes >= required else {
                throw MetalError.invalidState(
                    "--ram-budget \(ramBudgetBytes / 1_048_576) MiB is below this model's floor of "
                        + "\(required / 1_048_576) MiB (\(resident / 1_048_576) MiB outside the "
                        + "expert cache plus \(floor) expert slots per layer)")
            }
            budget = Double(ramBudgetBytes - resident)
        } else if let value = ProcessInfo.processInfo.environment["TINYTITAN_GGUF_EXPERT_CACHE_GB"]
            .flatMap(Double.init), value > 0
        {
            budget = value * 1_073_741_824
        } else {
            budget = Double(ProcessInfo.processInfo.physicalMemory) / 12
        }
        let slots = Int(budget / Double(perSlot))
        return min(arch.numExperts, max(floor, slots))
    }

    /// What the process holds besides the expert pools: the non-expert
    /// tensors as stored, the bf16 arena, and a 768 MiB margin (KV cache,
    /// scratch, Metal and the runtime itself).
    ///
    /// The PLE n-gram table is left out: it is 26.8 GiB of this checkpoint's
    /// 77.2, and the stage reads a handful of its rows per token from the
    /// file, so what it costs the process is those pages, not the table.
    /// Counting it would put every RAM target under 33 GiB below the floor.
    /// The token embedding (0.49 GB) is left out for the same reason: the
    /// decoder reads one row a token from the file.
    static func residentBytesOutsideTheExpertCache(model: GGUFModel, arenaBytes: Int) -> Int {
        let notResident = model.tensors.filter { name, _ in
            name.contains("_exps.") || name.hasSuffix("per_layer_token_embd.weight")
                || name == "token_embd.weight"
        }
        .reduce(0) { $0 + Int($1.value.tensor.byteCount) }
        let resident = Int(model.totalTensorBytes) - notResident
        return resident + arenaBytes + 768 * 1_048_576
    }

    static func openRoutingLog(_ path: String?) -> FileHandle? {
        guard let path, !path.isEmpty else { return nil }
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let handle = FileHandle(forWritingAtPath: path) else { return nil }
        handle.seekToEndOfFile()
        return handle
    }
}
