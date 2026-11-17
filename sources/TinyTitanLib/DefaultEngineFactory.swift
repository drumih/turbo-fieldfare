import Foundation
import TinyTitan
import TinyTitanCore

/// unchecked-invariant: the lock protects the runtime and MTP decoder while the
/// sequential session loader moves from model construction to runner construction.
package final class DefaultEngineFactory: @unchecked Sendable {
    private let loadRuntime: RuntimeConfiguration
    private let integrityPolicy: ModelIntegrityPolicy
    private let requestedIntegrityPolicy: ModelIntegrityPolicy?
    private let loadSlots: Int
    private let lock = NSLock()
    private var runtime: RuntimeConfiguration?
    private var decoder: StreamingMTPDecoder?

    package init(
        loadRuntime: RuntimeConfiguration,
        integrityPolicy: ModelIntegrityPolicy,
        requestedIntegrityPolicy: ModelIntegrityPolicy?,
        loadSlots: Int
    ) {
        self.loadRuntime = loadRuntime
        self.integrityPolicy = integrityPolicy
        self.requestedIntegrityPolicy = requestedIntegrityPolicy
        self.loadSlots = loadSlots
    }

    package var factory: EngineFactory {
        EngineFactory(
            buildModel: { directory, arch, context in
                guard let context else {
                    throw ServerInferenceError.unsupportedModel("6.x model load requires Metal")
                }
                return try Model.load(
                    directoryURL: directory,
                    device: context.device,
                    expecting: arch,
                    streamingMode: .pread(slotCount: self.loadSlots),
                    expertCachePolicy: self.loadRuntime.modelExpertCachePolicy,
                    expertIOTimeline: self.loadRuntime.expertIOTimeline,
                    integrityPolicy: self.integrityPolicy)
            },
            buildRunner: { model, options in
                try self.buildRunner(model: model, options: options)
            })
    }

    package func setRuntime(_ runtime: RuntimeConfiguration) {
        lock.withLock { self.runtime = runtime }
    }

    package var mtpDecoder: StreamingMTPDecoder? {
        lock.withLock { decoder }
    }

    private func buildRunner(
        model: any EngineModel, options: SessionBuildOptions
    ) throws -> any EngineRunner {
        guard let model = model as? Model, let context = options.context else {
            throw ServerInferenceError.unsupportedModel(
                "6.x runner requires its model and Metal context")
        }
        let runtime = try lock.withLock {
            guard let runtime else {
                throw ServerInferenceError.unsupportedModel("6.x runner configuration is missing")
            }
            return runtime
        }
        if let mtpModelDirectory = options.mtpModelDirectory {
            let sidecarFamily = try ManifestReader.peekFamily(
                directoryURL: mtpModelDirectory)
            let sidecarArch: ArchConfig
            do {
                sidecarArch = try ArchConfig.resolved(
                    forFamily: sidecarFamily, directoryURL: mtpModelDirectory)
            } catch {
                throw ServerInferenceError.unsupportedModel(
                    "MTP sidecar declares family \(sidecarFamily.rawValue), "
                        + "which this runtime does not implement")
            }
            let sidecar = try Model.load(
                directoryURL: mtpModelDirectory,
                device: context.device,
                expecting: sidecarArch,
                streamingMode: .pread(slotCount: StreamingMTPMemoryPlan.expertSlots),
                expertCachePolicy: runtime.modelExpertCachePolicy,
                expertIOTimeline: runtime.expertIOTimeline,
                integrityPolicy: requestedIntegrityPolicy
                    ?? .resolved(directoryURL: mtpModelDirectory))
            let decoder = try StreamingMTPDecoder(
                targetModel: model,
                mtpSidecar: sidecar,
                context: context,
                maxContext: options.maxContext,
                memoryBudgetMiB: options.mtpMemoryMiB,
                runtimeConfiguration: runtime)
            lock.withLock { self.decoder = decoder }
            return decoder.target
        }
        lock.withLock { decoder = nil }
        return try RealForwardRunner(
            model: model,
            context: context,
            maxContext: options.maxContext,
            slots: options.slots,
            runtimeConfiguration: runtime)
    }
}
