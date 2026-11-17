import Foundation
import TinyTitan
import TinyTitanCore
import TinyTitanLib

extension EngineFactories {
    func makeWarmup(_ engineModel: any EngineModel, requested: Int?) -> EngineExpertWarmup? {
        guard let slotCount = engineModel.routedExpertCacheSlotCount(),
            let model = engineModel as? Model
        else { return nil }
        let requested = requested ?? ExpertWarmup.requestedPerLayer(
            environment: ProcessInfo.processInfo.environment)
        let perLayer = ExpertWarmup.perLayer(
            requested: requested, slotCount: slotCount,
            expertsPerLayer: model.config.numExperts)
        guard perLayer > 0 else { return nil }
        return EngineExpertWarmup {
            let report = await ExpertWarmupJob(model: model).run(perLayer: perLayer)
            ServerLog.diagnostic(
                report?.logLine
                    ?? "TinyTitan warm experts: \(perLayer)/layer, none ranked "
                    + "(no expert profile yet)")
            return report
        }
    }
}
