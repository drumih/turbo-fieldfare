import Foundation

struct VisionSidecarPlan: Sendable {
    let resident: ResidentFilePlan
    let sourceTensorCount: Int

    var entryCount: Int { resident.entries.count }
}

enum VisionSidecarPlanner {
    static let visionPrefixes = ["vision_tower.", "embed_vision."]

    static func plan(
        meta: IndexLoader.SourceMetadata,
        shardHeaders: [Safetensors.Header],
        outputDirectory: String
    ) throws -> VisionSidecarPlan {
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for header in shardHeaders {
            for tensor in header.tensors {
                registry[tensor.name] = tensor
            }
        }

        let sourceNames = registry.keys.filter(isVisionTensor).sorted()
        guard !sourceNames.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: "source snapshot contains no Gemma vision tensors")
        }

        // Quantized scale and bias companions are represented by offsets on
        // their base `.weight` entry in the resident index, just as they are in
        // the language-model resident file.
        let baseNames = sourceNames.filter {
            !$0.hasSuffix(".scales") && !$0.hasSuffix(".biases")
        }
        let path = (outputDirectory as NSString).appendingPathComponent("weights.bin")
        let resident = try RepackPlanner.planResidentFile(
            path: path,
            baseNames: baseNames,
            registry: registry,
            meta: meta)
        return VisionSidecarPlan(
            resident: resident,
            sourceTensorCount: sourceNames.count)
    }

    static func isVisionTensor(_ name: String) -> Bool {
        visionPrefixes.contains { name.hasPrefix($0) }
    }
}
