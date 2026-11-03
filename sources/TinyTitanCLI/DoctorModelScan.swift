import Foundation
import TinyTitan
import TinyTitanLib

/// Reads a `models/` tree without loading a model or touching its weights.
///
/// The scan is deliberately the same shape as `ModelCatalog.scan` (one
/// directory per install, skipped directories carry a reason) but stays in the
/// CLI target: pulling `TinyTitanServerCore` in would link NIO into a command
/// whose whole job is to print facts. Everything the facts are derived from is
/// still the shared reader and configuration code.
enum DoctorModelScanner {
    struct ScanResult {
        let exists: Bool
        let models: [DoctorReport.Model]
        let skipped: [DoctorReport.Skipped]
    }

    static func scan(
        directory: URL,
        physicalMemory: UInt64,
        contextTokens: Int,
        environment: [String: String]
    ) -> ScanResult {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard
            fileManager.fileExists(
                atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            return ScanResult(exists: false, models: [], skipped: [])
        }
        let children: [URL]
        do {
            children = try fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        } catch {
            return ScanResult(
                exists: true, models: [],
                skipped: [
                    DoctorReport.Skipped(
                        name: directory.lastPathComponent,
                        reason: "cannot list: \(error)")
                ])
        }
        var models: [DoctorReport.Model] = []
        var skipped: [DoctorReport.Skipped] = []
        for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let path = child.resolvingSymlinksInPath()
            guard
                (try? path.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory
                    == true
            else { continue }
            if fileManager.fileExists(
                atPath: path.appendingPathComponent("manifest.json").path)
            {
                switch probeInstall(
                    path, physicalMemory: physicalMemory,
                    contextTokens: contextTokens, environment: environment)
                {
                case .success(let model):
                    models.append(model)
                case .failure(let failure):
                    skipped.append(
                        DoctorReport.Skipped(
                            name: child.lastPathComponent, reason: failure.reason))
                }
            } else if fileManager.fileExists(
                atPath: path.appendingPathComponent("config.json").path)
            {
                switch probeSnapshot(path) {
                case .success(let model):
                    models.append(model)
                case .failure(let failure):
                    skipped.append(
                        DoctorReport.Skipped(
                            name: child.lastPathComponent, reason: failure.reason))
                }
            }
        }
        models.sort { $0.id < $1.id }
        return ScanResult(exists: true, models: models, skipped: skipped)
    }

    private struct ProbeFailure: Error {
        let reason: String
    }

    /// Why an installed family cannot be run on this build, or nil when it
    /// can. Shared with `ModelCatalog`'s reasons in spirit, not in code: the
    /// server target links NIO and doctor deliberately does not.
    static func unsupportedInstallReason(_ family: ModelFamily) -> String? {
        switch family {
        case .qwen36MTP, .qwen38flashMTP, .qwen4flashMTP:
            return "MTP draft head, served only beside its target"
        case .k2Horizon:
            return "K2-Horizon is not served by this CLI: the server's probe "
                + "backend serves it, and `TinyTitanBench k2gpu` runs it "
                + "directly (docs/plan-k2-horizon.md)"
        case .laguna:
            return "Laguna-S-2.1's decode probe is implemented; the serving "
                + "graph is not, so the install validates but is not served "
                + "(docs/plan-laguna-s-2.1.md)"
        case .qwen36, .qwen38flash, .qwen4flash, .qwen35Dense:
            return nil
        }
    }

    private static func probeInstall(
        _ directory: URL,
        physicalMemory: UInt64,
        contextTokens: Int,
        environment: [String: String]
    ) -> Result<DoctorReport.Model, ProbeFailure> {
        let identity: ManifestIdentity
        let manifest: Manifest
        do {
            identity = try ManifestReader.peekIdentity(directoryURL: directory)
            manifest = try ManifestReader.read(directoryURL: directory)
        } catch {
            return .failure(
                ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        // A family the runtime cannot serve is not a model this machine can
        // run, whatever its memory fit says. The catalog skips the same set;
        // doctor reporting K2 as a fit would send a person to launch a model
        // that stops at Model.load's graph refusal.
        if let reason = Self.unsupportedInstallReason(identity.family) {
            return .failure(ProbeFailure(reason: reason))
        }
        let profile = ModelProfile.resolve(identity: identity)
        let routedExperts = manifest.expertsPerLayer > 0 && manifest.expertStride > 0
        let expertSlots: Int?
        let cacheFloorBytes: UInt64
        let expertCacheBytes: UInt64
        if routedExperts {
            let budget = RuntimeConfiguration.affordableExpertCacheBudget(
                profile.expertCacheBudgetBytes, physicalMemory: physicalMemory)
            expertSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride,
                layers: manifest.arch.numLayers,
                budgetBytes: budget)
            cacheFloorBytes = clampedProduct(
                UInt64(RuntimeConfiguration.minimumExpertCacheSlots),
                manifest.expertStride,
                UInt64(manifest.arch.numLayers))
            expertCacheBytes = clampedProduct(
                UInt64(expertSlots ?? 0),
                manifest.expertStride,
                UInt64(manifest.arch.numLayers))
        } else {
            expertSlots = nil
            cacheFloorBytes = 0
            expertCacheBytes = 0
        }
        let weights = manifest.files["model_weights.bin"]?.size ?? 0
        let kvPerToken = kvBytesPerToken(manifest.arch, precision: .int8)
        let kvBytes = clampedProduct(
            UInt64(kvPerToken), UInt64(contextTokens))
        let total = clampedSum(weights, kvBytes, expertCacheBytes)
        let fit = DoctorReport.Model.Fit(
            weightsBytes: weights,
            kvBytes: kvBytes,
            kvPrecision: KVCachePrecision.int8.label,
            contextTokens: contextTokens,
            cacheFloorBytes: cacheFloorBytes,
            expertCacheBytes: expertCacheBytes,
            totalBytes: total,
            ramBytes: physicalMemory,
            fits: total <= physicalMemory,
            utilization: physicalMemory == 0
                ? 0 : Double(total) / Double(physicalMemory))
        let installBytes = PageCachePolicyResolver.installSize(
            fileSizes: manifest.files.values.map(\.size))
        let pageCache = PageCachePolicyResolver.resolve(
            physicalMemory: physicalMemory,
            installBytes: installBytes,
            environment: environment)
        let chunk =
            profile.prefillChunkTokens
            ?? defaultPrefillChunkTokens(
                family: identity.family,
                fallback: RuntimeConfiguration.production.prefillChunkTokens)
        return .success(
            DoctorReport.Model(
                id: identity.modelID,
                directory: directory.lastPathComponent,
                family: identity.family.rawValue,
                weightBits: identity.weightBits,
                sizeBytes: totalFileSize(directory),
                fit: fit,
                expertSlots: expertSlots,
                prefillChunkTokens: chunk,
                pageCache: DoctorReport.Model.PageCache(
                    policy: pageCache.policy.rawValue,
                    reason: pageCache.reason),
                note: nil))
    }

    private static func probeSnapshot(
        _ directory: URL
    ) -> Result<DoctorReport.Model, ProbeFailure> {
        let data: Data
        do {
            data = try BoundedMetadataRead.read(
                fileAt: directory.appendingPathComponent("config.json"),
                maxBytes: ManifestReader.defaultMaxBytes)
        } catch {
            return .failure(
                ProbeFailure(reason: "unreadable config.json: \(error)"))
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            return .failure(
                ProbeFailure(reason: "config.json is not a JSON object"))
        }
        let declared =
            (object["model_id"] as? String)
            ?? (object["display_name"] as? String)
        let id =
            declared.flatMap { $0.isEmpty ? nil : $0 }
            ?? directory.lastPathComponent
        let family = (object["model_type"] as? String) ?? "cpu-snapshot"
        let bits =
            (object["quantization"] as? [String: Any])?["bits"] as? Int ?? 0
        return .success(
            DoctorReport.Model(
                id: id,
                directory: directory.lastPathComponent,
                family: family,
                weightBits: bits,
                sizeBytes: totalFileSize(directory),
                fit: nil,
                expertSlots: nil,
                prefillChunkTokens: nil,
                pageCache: nil,
                note: "CPU snapshot: fit is not estimated"))
    }

    /// K/V bytes a token occupies across every row-bearing layer, mirroring
    /// `KVCacheManager`'s storage layout (the same `rowLayout` the allocator
    /// uses) at the server's default 8-bit KV precision.
    private static func kvBytesPerToken(
        _ arch: ManifestArch,
        precision: KVCachePrecision
    ) -> Int {
        var total = 0
        for mask in arch.fullAttentionLayerMask {
            if mask == 2 { continue }
            let elements =
                mask == 0
                ? arch.numKVHeads * arch.headDim
                : arch.numFullKVHeads * arch.fullHeadDim
            let layout = KVCacheManager.rowLayout(
                elements: elements, precision: precision)
            // One K row and one V row per token.
            total += layout.stride * 2
        }
        return total
    }

    private static func clampedSum(_ values: UInt64...) -> UInt64 {
        values.reduce(UInt64(0)) { total, value in
            let (sum, overflow) = total.addingReportingOverflow(value)
            return overflow ? UInt64.max : sum
        }
    }

    private static func clampedProduct(_ values: UInt64...) -> UInt64 {
        values.reduce(UInt64(1)) { total, value in
            let (product, overflow) = total.multipliedReportingOverflow(by: value)
            return overflow ? UInt64.max : product
        }
    }

    /// Logical file sizes, not allocated blocks.
    ///
    /// `ModelCatalog` reports allocated bytes because it describes what an
    /// install occupies; `doctor` reports logical bytes because its output is
    /// compared against a golden and APFS allocation is not stable across
    /// filesystems. Nothing here reads a byte of a weight file.
    static func totalFileSize(_ directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.fileSizeKey, .isRegularFileKey]
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: keys)
        else { return 0 }
        var total: Int64 = 0
        for case let file as URL in walker {
            guard
                let values = try? file.resourceValues(forKeys: Set(keys)),
                values.isRegularFile == true
            else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }
}
