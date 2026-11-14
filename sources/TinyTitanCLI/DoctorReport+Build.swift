import Foundation
import TinyTitan
import TinyTitanCore

extension DoctorReport {
    /// Assemble a report from injected facts. This is the whole of `doctor`'s
    /// decision-making: the live command only fills an environment.
    static func build(_ environment: DoctorEnvironment) -> DoctorReport {
        let scan = DoctorModelScanner.scan(
            directory: environment.modelsDirectory,
            physicalMemory: environment.capabilities.physicalMemoryBytes,
            contextTokens: environment.maxContextTokens,
            environment: environment.environment)
        let resolution = TensorOpsMode.resolve(environment: environment.environment)
        let plan = TensorOpsPolicy.plan(
            capabilities: environment.capabilities,
            mode: resolution.mode,
            note: resolution.note)
        let tensorOps = TensorOps(
            mode: resolution.mode.rawValue,
            path: plan.attemptsPipeline ? "tensor-ops" : "fallback",
            reason: plan.reason,
            capabilityPresent: environment.capabilities.supportsApple10TensorOps)
        var messages: [String] = []
        if !scan.exists {
            messages.append(
                "models directory not found: \(environment.modelsDirectory.path)")
        }
        if scan.models.isEmpty {
            messages.append("no installed models found")
        }
        let machine = Machine(
            chip: environment.capabilities.chipModel,
            gpuName: environment.capabilities.gpuName,
            gpuFamily: environment.capabilities.gpuFamily?.rawValue,
            apple10TensorOps: environment.capabilities.supportsApple10TensorOps,
            metal4: environment.capabilities.metal4Available,
            ane: environment.capabilities.anePresent,
            ramBytes: environment.capabilities.physicalMemoryBytes,
            osVersion: environment.capabilities.osVersion)
        return DoctorReport(
            schema: DoctorReport.schemaVersion,
            machine: machine,
            modelsDir: environment.modelsDirectory.path,
            modelsDirExists: scan.exists,
            disk: environment.disk,
            tensorOps: tensorOps,
            models: scan.models,
            skipped: scan.skipped,
            calibration: environment.calibration,
            otherModelProcesses: environment.otherProcesses,
            messages: messages)
    }

    func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }
}
