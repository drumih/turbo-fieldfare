import Foundation
import TinyTitan
import TinyTitanCore
import TinyTitanLegacyEngine

extension CPUEngineFactory {
    static func legacy(snapshot: AffineSnapshot) -> CPUEngineFactory {
        CPUEngineFactory { try CPUQwen35(snapshot: snapshot) }
    }

    /// The 6.x engine built from the install directory — the fallback form
    /// the width-selected default uses, so the caller does not have to hold a
    /// snapshot to ask for an engine.
    static func legacy(snapshotDirectory: URL, isSSDAI: Bool) -> CPUEngineFactory {
        CPUEngineFactory {
            let snapshot =
                isSSDAI
                ? try AffineSnapshot(ssdai: snapshotDirectory)
                : try AffineSnapshot(directory: snapshotDirectory)
            return try CPUQwen35(snapshot: snapshot)
        }
    }
}

extension CPUEngineFactory {
    /// The width-selected variant engine for a dense install (TT-097 Task
    /// 13): hidden size names the model, the width its quant twin. Falls back
    /// to the 6.x engine for a combination no variant ships (matching the GPU
    /// table's legacy-miss rule) — the CPUModelFamily refusal still guards
    /// the families that were never dense at all.
    static func widthSelected(
        snapshotDirectory: URL,
        isSSDAI: Bool,
        hiddenSize: Int,
        weightBits: Int
    ) -> CPUEngineFactory {
        (try? EngineFactories.variantCPUEngine(
            hiddenSize: hiddenSize,
            weightBits: weightBits,
            snapshotDirectory: snapshotDirectory,
            isSSDAI: isSSDAI))
            ?? CPUEngineFactory.legacy(snapshotDirectory: snapshotDirectory, isSSDAI: isSSDAI)
    }
}
