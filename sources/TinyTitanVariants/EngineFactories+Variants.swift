import Foundation
import DenseGPU
import Ornith35B4Bit
import Qwen35Dense2B4Bit
import Qwen35Dense2B8Bit
import Qwen35Dense4B4Bit
import Qwen35Dense4B8Bit
import Qwen35Dense9B4Bit
import Qwen35Dense9B8Bit
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib
import TinyTitanPluginAPI

/// The engine a caller asks for explicitly.
public enum EngineChoice: String, Sendable, CaseIterable, Equatable {
    case cpu
    case gpu
}

extension EngineFactories {

    // MARK: - TT-097 Phase 3 Task 10: the shipped variants as server engines

    /// Which GPU engine a catalog identity selects. The table is explicit and
    /// small: a shipped variant joins it when it qualifies (registry
    /// `shipped: true`, `catalog --check` green), and every identity that
    /// misses falls back to the 6.x engine — which stays the serving engine
    /// for every model without a shipped variant (TT-087). `Equatable` so a
    /// test can pin the selection without running a model.
    public enum GPUVariantSelection: Equatable {
        case legacy
        case ornith35B4Bit
        case qwen35DenseGPU
    }

    /// The selection for a GPU catalog identity, decided from the manifest
    /// identity the entry's directory carries: the shipped Ornith 1.5 35B-A3B
    /// install at 4 bits routes to its variant, every dense Qwen 3.5 install
    /// routes to the shared dense-GPU engine, and everything else stays on
    /// the 6.x engine.
    public static func gpuSelection(
        family: String,
        manifestModelID: String,
        routedExpertWeightBits: Int
    ) -> GPUVariantSelection {
        if family == "qwen36",
            manifestModelID.hasPrefix("ornith-1.5-35b"),
            routedExpertWeightBits == 4
        {
            return .ornith35B4Bit
        }
        if family == "qwen3_5_dense" {
            return .qwen35DenseGPU
        }
        return .legacy
    }

    /// The GPU factory a catalog identity selects. The router reads the
    /// install manifest identity (the same bounded read its plan performs)
    /// and dispatches through the selection table: the shipped variant
    /// engines for their installs, the 6.x engine for everything else —
    /// which keeps serving every model without a shipped variant (TT-087).
    ///
    /// Resolved through the plugin registry (TPP/1, TT-130): the strongest
    /// plugin probe wins. An install no plugin claims specifically and that has
    /// no readable manifest is refused here, as before the registry.
    public static func gpuFactory(directoryURL: URL) throws -> EngineFactory {
        let resolved = try pluginRegistry.resolve(directoryURL)
        if resolved.strength == .fallback {
            _ = try SSDAIManifestReader.read(directoryURL: directoryURL)
        }
        return resolved.engineFactory()
    }

    /// The CPU factory for a dense Qwen 3.5 install: the per-width variant
    /// engine the server serves, built from the install's own hidden size and
    /// quant width. A family the CPU engine does not implement is refused by
    /// name, before the model loads.
    public static func routingCPUFactory(directoryURL: URL) throws -> CPUEngineFactory {
        // Read through `DenseCPUInstall`, not the 6.x manifest reader (TT-134).
        guard let install = try? DenseCPUInstall(directory: directoryURL), install.isSSDAI
        else {
            throw EngineFactoryError.unsupported(
                "--engine cpu needs an .ssdai install with a readable manifest; "
                    + "\(directoryURL.path) has none")
        }
        guard install.family == Self.denseCPUFamily else {
            throw EngineFactoryError.unsupported(
                "\(install.modelID ?? directoryURL.lastPathComponent) declares family "
                    + "\(install.family ?? "none"); "
                    + "the CPU engine implements the dense Qwen 3.5 family only, so "
                    + "this install is GPU-only")
        }
        return try variantCPUEngine(
            hiddenSize: install.hiddenSize,
            weightBits: install.weightBits,
            snapshotDirectory: directoryURL,
            isSSDAI: true)
    }

    /// The family a dense `.ssdai` install declares.
    static let denseCPUFamily = "qwen3_5_dense"

    /// The GPU factory a single-model front end (the CLI) and an embedder
    /// use: the same selection table the server's router uses, keyed on the
    /// install's manifest identity. A directory without a readable manifest
    /// (an affine snapshot, a GGUF directory) falls back to the 6.x factory,
    /// which is what those paths used before this surface existed.
    public static func routingFactory(directoryURL: URL) -> EngineFactory {
        (try? pluginRegistry.resolve(directoryURL))?.engineFactory() ?? sixXGPU()
    }

    public static func gpuFactory(
        family: String,
        manifestModelID: String,
        routedExpertWeightBits: Int
    ) -> EngineFactory {
        switch gpuSelection(
            family: family,
            manifestModelID: manifestModelID,
            routedExpertWeightBits: routedExpertWeightBits)
        {
        case .ornith35B4Bit: return Ornith35B4BitPlugin.engineFactory(host: .standard)
        case .qwen35DenseGPU: return DenseQwen35Plugin.engineFactory(host: .standard)
        case .legacy: return sixXGPU()
        }
    }

    // MARK: - the dense CPU variants

    /// Which dense variant target a CPU install's width selects. Each dense
    /// model owns its dedicated code path (the 7.0 directive), so the width —
    /// not a shared engine — decides the target. 6 bits is the withdrawn
    /// legacy format and is refused by name, like the registry does.
    public enum DenseCPUVariantSelection: Equatable {
        case qwen35Dense2B4Bit
        case qwen35Dense2B8Bit
        case qwen35Dense4B4Bit
        case qwen35Dense4B8Bit
        case qwen35Dense9B4Bit
        case qwen35Dense9B8Bit
    }

    /// The dense CPU target a served snapshot selects: hidden size names the
    /// model, the width its quant twin. An unknown combination is refused by
    /// name rather than run through a sibling's statics.
    public static func variantCPUEngineSelection(
        hiddenSize: Int,
        weightBits: Int
    ) throws -> DenseCPUVariantSelection {
        switch (hiddenSize, weightBits) {
        case (2_048, 4): return .qwen35Dense2B4Bit
        case (2_048, 8): return .qwen35Dense2B8Bit
        case (2_560, 4): return .qwen35Dense4B4Bit
        case (2_560, 8): return .qwen35Dense4B8Bit
        case (4_096, 4): return .qwen35Dense9B4Bit
        case (4_096, 8): return .qwen35Dense9B8Bit
        default:
            throw ServerInferenceError.unsupportedModel(
                "no dense CPU variant ships for hiddenSize \(hiddenSize) at "
                    + "\(weightBits) bits; supported sizes: 2048, 2560, 4096 at 4 or 8 bits")
        }
    }

    /// The dense CPU variant for a snapshot's size and width. Each dense
    /// model owns its dedicated code path (the 7.0 directive): the hidden
    /// size decides the model (2B 2048 / 4B 2560 / 9B 4096) and the width the
    /// quant twin. The variant's engine consumes its own module's snapshot
    /// type, so the closure builds it from the install directory — the 6.x
    /// snapshot the backend holds is a different module's class and cannot
    /// cross. An unknown size/width combination is refused by name.
    public static func variantCPUEngine(
        hiddenSize: Int,
        weightBits: Int,
        snapshotDirectory: URL,
        isSSDAI: Bool,
        threads: Int? = nil
    ) throws -> CPUEngineFactory {
        func engine(_ build: @Sendable @escaping () throws -> any CPUEngineRunner)
            -> CPUEngineFactory
        {
            CPUEngineFactory(build: build)
        }
        switch try variantCPUEngineSelection(hiddenSize: hiddenSize, weightBits: weightBits) {
        case .qwen35Dense2B4Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense2B4Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense2B4Bit.AffineSnapshot(directory: snapshotDirectory)
                guard snapshot.family != nil else {
                    throw ServerInferenceError.unsupportedModel(
                        Qwen35Dense2B4Bit.CPUModelFamily.refusal(modelType: snapshot.modelType))
                }
                return try Qwen35Dense2B4Bit.CPUQwen35(snapshot: snapshot, threads: threads)
            }
        case .qwen35Dense2B8Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense2B8Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense2B8Bit.AffineSnapshot(directory: snapshotDirectory)
                guard snapshot.family != nil else {
                    throw ServerInferenceError.unsupportedModel(
                        Qwen35Dense2B8Bit.CPUModelFamily.refusal(modelType: snapshot.modelType))
                }
                return try Qwen35Dense2B8Bit.CPUQwen35(snapshot: snapshot, threads: threads)
            }
        case .qwen35Dense4B4Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense4B4Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense4B4Bit.AffineSnapshot(directory: snapshotDirectory)
                guard snapshot.family != nil else {
                    throw ServerInferenceError.unsupportedModel(
                        Qwen35Dense4B4Bit.CPUModelFamily.refusal(modelType: snapshot.modelType))
                }
                return try Qwen35Dense4B4Bit.CPUQwen35(snapshot: snapshot, threads: threads)
            }
        case .qwen35Dense4B8Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense4B8Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense4B8Bit.AffineSnapshot(directory: snapshotDirectory)
                guard snapshot.family != nil else {
                    throw ServerInferenceError.unsupportedModel(
                        Qwen35Dense4B8Bit.CPUModelFamily.refusal(modelType: snapshot.modelType))
                }
                return try Qwen35Dense4B8Bit.CPUQwen35(snapshot: snapshot, threads: threads)
            }
        case .qwen35Dense9B4Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense9B4Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense9B4Bit.AffineSnapshot(directory: snapshotDirectory)
                guard snapshot.family != nil else {
                    throw ServerInferenceError.unsupportedModel(
                        Qwen35Dense9B4Bit.CPUModelFamily.refusal(modelType: snapshot.modelType))
                }
                return try Qwen35Dense9B4Bit.CPUQwen35(snapshot: snapshot, threads: threads)
            }
        case .qwen35Dense9B8Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense9B8Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense9B8Bit.AffineSnapshot(directory: snapshotDirectory)
                guard snapshot.family != nil else {
                    throw ServerInferenceError.unsupportedModel(
                        Qwen35Dense9B8Bit.CPUModelFamily.refusal(modelType: snapshot.modelType))
                }
                return try Qwen35Dense9B8Bit.CPUQwen35(snapshot: snapshot, threads: threads)
            }
        }
    }
}

/// CPU int8 affine GEMV entry point exposed for benchmarks without importing
/// the 6.x runtime (TT-134).
public enum VariantCPUInt8GEMV {
    public static var preferredThreads: Int {
        Qwen35Dense4B4Bit.Int8AffineGEMV.preferredThreads
    }

    public static func threaded(
        weights: UnsafePointer<UInt8>,
        scales: UnsafePointer<UInt16>,
        biases: UnsafePointer<UInt16>,
        x: UnsafePointer<Float>,
        rows: Int,
        n: Int,
        out: UnsafeMutablePointer<Float>,
        threads: Int
    ) {
        Qwen35Dense4B4Bit.Int8AffineGEMV.threaded(
            weights: weights, scales: scales, biases: biases,
            x: x, rows: rows, n: n, out: out, threads: threads)
    }
}

/// Load the Ornith variant model with the resolved slot count.
///
/// A nil `slots` is the legacy `buildModel` path: the env override, then the
/// variant's own 32-slot default. The session's resolved count (profile,
/// `--ram-budget`, `--expert-cache-slots`) arrives through
/// `buildModelWithSlots`. The env stays the operator's last word.
