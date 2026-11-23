import Foundation

// Output-file bookkeeping for the remote repacker: completed-range recovery,
// per-file recording and the manifest/receipt writers.
//
// Split out of `RemoteStreamingRepacker.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension RemoteStreamingRepacker {

    static func validatedCompletedRanges(
        _ completed: [RemoteCompletedRange],
        copies: [CoalescedRangeCopy],
        partialDirectory: String
    ) throws -> [RemoteCompletedRange] {
        let copiesByID = Dictionary(uniqueKeysWithValues: copies.map { ($0.id, $0) })
        var valid: [RemoteCompletedRange] = []
        for range in completed {
            guard let copy = copiesByID[range.id],
                range.sourceBytes == copy.size,
                range.destinationBytes
                    == copy.destinations.reduce(UInt64(0), { $0 + $1.size })
            else {
                throw RepackError.installStateCorrupt(
                    path: partialDirectory,
                    detail: "checkpoint contains an unknown range")
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy,
                partialDirectory: partialDirectory)
            if digest == range.destinationDigest {
                valid.append(range)
            }
        }
        return valid.sorted { $0.id < $1.id }
    }

    func recordOutputFile(
        relativePath: String,
        path: String,
        progress: @Sendable (ModelInstallProgress) -> Void
    ) throws {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        // O_NOFOLLOW: hashing must never follow a symlink planted inside the
        // partial directory.
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
        let sha = try WriterCore.hashEntireFile(
            path: path,
            size: size,
            audit: audit,
            cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(relativePath: relativePath, size: size, sha256: sha))
    }

    func writeSmall(path: String, data: Data) throws {
        let directory = (path as NSString).deletingLastPathComponent
        try Posix.mkdirP(directory)
        try Posix.atomicWrite(data, to: path, durableIn: directory)
        audit.recordWrite(bytes: data.count)
    }

    func copyRemoteMetadataSidecars(
        snapshot: RemoteSnapshot,
        remote: HuggingFaceRemoteSource,
        partialDir: String,
        progress: @Sendable (ModelInstallProgress) -> Void
    ) async throws {
        let tokenizerDir = (partialDir as NSString).appendingPathComponent("tokenizer")
        let pinned = remote.pinned(commit: snapshot.resolvedCommit)
        try Posix.mkdirP(tokenizerDir)

        // config.json is a required sidecar: the source snapshot always
        // declares it (metadata loading fails without it). Prefer the copy
        // fetched this run into .remote-metadata; if a resume-mismatch wiped
        // the partial directory (and with it the fresh metadata), re-fetch it
        // from the remote instead of silently skipping it. The read doubles as
        // the existence check, and atomicWrite fsyncs both the file and its
        // directory after the rename.
        let localConfig = (snapshot.metadataDirectory as NSString)
            .appendingPathComponent("config.json")
        let dstConfig = (tokenizerDir as NSString).appendingPathComponent("config.json")
        do {
            let configData = try Posix.readBoundedData(
                localConfig,
                maximumBytes: 1024 * 1024)
            try Posix.atomicWrite(configData, to: dstConfig, durableIn: tokenizerDir)
        } catch {
            if (try? Posix.entryKind(localConfig)) != .regular {
                let info = try await pinned.resolveFileInfo(
                    filename: "config.json",
                    audit: audit)
                guard info.size <= 1024 * 1024 else {
                    throw RepackError.remoteFileTooLarge(
                        path: "config.json",
                        size: info.size,
                        cap: 1024 * 1024)
                }
                try await pinned.fetchSmallFile(
                    filename: "config.json",
                    info: info,
                    capBytes: 1024 * 1024,
                    outputPath: dstConfig,
                    audit: audit)
            } else {
                throw error
            }
        }
        try recordOutputFile(
            relativePath: "tokenizer/config.json",
            path: dstConfig,
            progress: progress)

        let tokenizerFiles: [(name: String, cap: UInt64, required: Bool)] = [
            ("tokenizer.json", 64 * 1024 * 1024, true),
            ("tokenizer_config.json", 4 * 1024 * 1024, true),
            ("special_tokens_map.json", 1 * 1024 * 1024, false),
            ("chat_template.jinja", 4 * 1024 * 1024, false),
            ("chat_template.json", 4 * 1024 * 1024, false),
        ]
        for file in tokenizerFiles {
            try Task.checkCancellation()
            let info: RemoteFileInfo
            do {
                info = try await pinned.resolveFileInfo(filename: file.name, audit: audit)
            } catch {
                if file.required || !isRemoteNotFound(error) {
                    throw error
                }
                continue
            }
            let dst = (tokenizerDir as NSString).appendingPathComponent(file.name)
            try await pinned.fetchSmallFile(
                filename: file.name,
                info: info,
                capBytes: file.cap,
                outputPath: dst,
                audit: audit)
            try recordOutputFile(
                relativePath: "tokenizer/\(file.name)",
                path: dst,
                progress: progress)
        }
        // The Inkling checkpoint's tiktoken model and its quantization and
        // generation configs, the same optional files a local import copies.
        guard snapshot.arch.family == .inkling2 else { return }
        for extra in Inkling2CheckpointPlanner.tokenizerExtras {
            try Task.checkCancellation()
            let info: RemoteFileInfo
            do {
                info = try await pinned.resolveFileInfo(filename: extra.source, audit: audit)
            } catch  where isRemoteNotFound(error) {
                continue
            }
            let dst = (tokenizerDir as NSString).appendingPathComponent(extra.name)
            try await pinned.fetchSmallFile(
                filename: extra.source, info: info, capBytes: 64 * 1024 * 1024,
                outputPath: dst, audit: audit)
            try recordOutputFile(
                relativePath: "tokenizer/\(extra.name)", path: dst, progress: progress)
        }
    }

    func isRemoteNotFound(_ error: Error) -> Bool {
        if case RepackError.remoteHTTPStatus(_, 404) = error {
            return true
        }
        if case RepackError.remoteHTTPResponse(_, 404, _) = error {
            return true
        }
        return false
    }

    func writeManifest(
        plan: RepackPlan,
        partialDir: String,
        metadata: IndexLoader.SourceMetadata,
        expertStride: UInt64,
        resolvedCommit: String,
        modelIDOverride: String?
    ) throws {
        // Determine quantization bits from actual tensor data, not hardcoded.
        //
        // `routedExpert` starts at the source's *base* affine width rather than
        // a literal 4, and that is load-bearing for a model with no routed
        // experts at all. The dense Qwen 3.5 installs are exactly that: no
        // `.mlp.switch_mlp.*`, so the loop below never assigns the slot and the
        // layer-derived override at the end never fires, leaving the width at
        // whatever it started as. At a literal 4 every dense install claimed to
        // be 4-bit -- and the routed-expert width is what `ManifestIdentity`
        // reads and `apiModelID` turns into the `_<bits>-Bit` suffix, so the
        // 8-bit 2B/4B/9B came back as `qwen3.5-2b_4-Bit` and the catalog
        // skipped them as duplicates of the 4-bit ones. A MoE install is
        // unaffected: the layer-derived width still overrides this below.
        var bits = SSDAIJSON.QuantBitWidths(
            embedding: metadata.baseBits,
            attention: metadata.baseBits,
            router: 8,
            sharedExpert: 8,
            routedExpert: metadata.baseBits)
        for e in plan.resident.entries {
            guard let quantSpec = e.quantSpec else { continue }
            if e.name.hasSuffix(".embed_tokens.weight") {
                bits.embedding = quantSpec.bits
            }
            if e.name.hasSuffix(".self_attn.q_proj.weight")
                || e.name.hasSuffix(".self_attn.k_proj.weight")
                || e.name.hasSuffix(".self_attn.v_proj.weight")
                || e.name.hasSuffix(".self_attn.o_proj.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_qkv.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_z.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_a.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_b.weight")
                || e.name.hasSuffix(".linear_attn.out_proj.weight")
            {
                bits.attention = quantSpec.bits
            }
            // Router slot: the Qwen router tensor is `.mlp.gate.weight`.
            if e.name.hasSuffix(".router.proj.weight")
                || e.name.hasSuffix(".mlp.gate.weight")
            {
                bits.router = quantSpec.bits
            }
            // Shared-expert slot: the sigmoid-gated shared expert MLP. Routed
            // experts (`.mlp.switch_mlp.*`) are deliberately excluded here —
            // their bits land in `bits.routedExpert` below from the layer
            // sub-tensors, so no tensor feeds more than one slot.
            if e.name.hasSuffix(".mlp.shared_expert.gate_proj.weight")
                || e.name.hasSuffix(".mlp.shared_expert.up_proj.weight")
                || e.name.hasSuffix(".mlp.shared_expert.down_proj.weight")
            {
                bits.sharedExpert = quantSpec.bits
            }
        }
        if let layer = plan.layers.first(where: { !$0.subTensors.isEmpty }),
            let routedBits = layer.subTensors.first?.bitsForWeights
        {
            bits.routedExpert = routedBits
        }
        let files = audit.outputFiles.map {
            ($0.relativePath, SSDAIJSON.FileEntry(size: $0.size, sha256: $0.sha256))
        }
        let data = try SSDAIJSON.encodeManifest(
            plan: plan,
            modelID: modelIDOverride ?? plan.matchedModelID ?? "unknown/snapshot",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer
                ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits)
        let tmp = (partialDir as NSString).appendingPathComponent("manifest.json.tmp")
        let final = (partialDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: tmp, data: data)
        try Posix.rename(from: tmp, to: final)
        try Posix.fsyncDirectory(partialDir)
        let manifestSha = try Sha256Stream.hashFile(path: final)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: options.outputDir,
            manifestSha256: manifestSha,
            manifestSize: UInt64(data.count),
            sourceRepoID: options.repoID,
            sourceRevision: resolvedCommit,
            files: audit.outputFiles)
        let receiptPath = (partialDir as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        let tmpReceiptPath = receiptPath + ".tmp"
        try writeSmall(path: tmpReceiptPath, data: receipt)
        try Posix.rename(from: tmpReceiptPath, to: receiptPath)
        try Posix.fsyncDirectory(partialDir)
    }
}
