import Foundation
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanVariants

extension CPUEngineFactory {
    /// The variant engine for a dense install (TT-097 Task 13, TT-134): the
    /// hidden size names the model, the width its quant twin -- the caller's
    /// width when it names one, otherwise the install's own. A combination no
    /// variant ships, and an architecture the selected variant does not
    /// implement, are refused by name; the 6.x engine is no longer a fallback.
    static func widthSelected(install: DenseCPUInstall, weightBits: Int?) throws
        -> CPUEngineFactory
    {
        try EngineFactories.variantCPUEngine(
            hiddenSize: install.hiddenSize,
            weightBits: weightBits ?? install.weightBits,
            snapshotDirectory: install.directory,
            isSSDAI: install.isSSDAI)
    }
}
