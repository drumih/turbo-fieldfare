import Foundation
import TinyTitanFormat
import TinyTitanCore

extension Model {
    static func ransLayoutAndPhases(
        forLayer layer: Int, layout: PackedExpertsLayout, url: URL
    ) throws -> (ExpertRANSIndex.Layout?, [UInt8]) {
        guard layout.compression == .rans else { return (nil, []) }
        let rans = try ransLayout(
            beside: url, expertsPerLayer: layout.expertsPerLayer,
            expertStride: layout.expertStride)
        var phases: [UInt8] = []
        let experts = layout.layers[layer].experts
        for (index, expert) in experts.enumerated() {
            // Schema checks widths; inference reuses expert 0's byte geometry.
            let canonical = experts[0].subTensors
            guard expert.subTensors.count == canonical.count,
                expert.subTensors.allSatisfy({ name, tensor in
                    guard let reference = canonical[name] else { return false }
                    return tensor.offset == reference.offset && tensor.size == reference.size
                        && tensor.dtype == reference.dtype && tensor.shape == reference.shape
                })
            else {
                throw ModelError.indexCorrupt(
                    detail: "rANS expert \(index) has different tensor geometry from expert 0")
            }
            phases = try validateRANSBlocks(rans.blocks[index], expert: expert)
        }
        return (rans, phases)
    }

    private static func validateRANSBlocks(
        _ blocks: [ExpertRANSIndex.Block], expert: ExpertEntry
    ) throws -> [UInt8] {
        var phases: [UInt8] = []
        for name in expert.subTensors.keys.sorted() {
            guard let tensor = expert.subTensors[name] else { continue }
            let stride = tensor.dtype == "BF16" ? 2 : 1
            guard tensor.size % UInt64(stride) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "rANS tensor \(name) has an incomplete byte plane")
            }
            for plane in 0..<stride {
                let index = phases.count
                guard blocks.indices.contains(index) else {
                    throw ModelError.indexCorrupt(
                        detail: "rANS expert \(expert.expert) is missing blocks")
                }
                let block = blocks[index]
                guard UInt64(block.rawOffset) == tensor.offset + UInt64(plane),
                    UInt64(block.symbolCount) == tensor.size / UInt64(stride),
                    Int(block.stride) == stride, Int(block.role) == index
                else {
                    throw ModelError.indexCorrupt(
                        detail:
                            "rANS expert \(expert.expert) block \(index) does not match tensor \(name)"
                    )
                }
                phases.append(name.hasPrefix("down") ? 1 : 0)
            }
        }
        guard phases.count == blocks.count else {
            throw ModelError.indexCorrupt(detail: "rANS expert \(expert.expert) has extra blocks")
        }
        return phases
    }

    static func ransLayout(
        beside url: URL,
        expertsPerLayer: Int,
        expertStride: UInt64
    ) throws -> ExpertRANSIndex.Layout {
        let name = ExpertRANSIndex.sidecarName(forLayerFile: url.lastPathComponent)
        let indexURL = url.deletingLastPathComponent().appendingPathComponent(name)
        let cap = UInt64(
            ExpertRANSIndex.headerBytes
                + Int(ExpertRANSIndex.maximumBlockCount) * 512
                + expertsPerLayer * Int(ExpertRANSIndex.maximumBlockCount)
                * ExpertRANSIndex.blockBytes)
        let data = try BoundedMetadataRead.read(fileAt: indexURL, maxBytes: cap)
        let payloadSize =
            try FileManager.default
            .attributesOfItem(atPath: url.path)[.size] as? UInt64 ?? 0
        return try ExpertRANSIndex.decode(
            data, expectedExperts: expertsPerLayer,
            expertStride: expertStride, payloadSize: payloadSize)
    }
}
