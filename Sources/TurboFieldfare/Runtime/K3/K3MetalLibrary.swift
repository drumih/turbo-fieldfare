import Foundation
import Metal

/// Runtime-compiled library for the K3 decode shaders (`Metal/K3/*.metal`),
/// mirroring `MetalContext`'s compile-and-cache pattern but self-contained:
/// the K3 port adds files without touching the Gemma shader manifest.
///
/// The three MoE/router modules share one namespace (mxfp4.metal defines the
/// common MXFP4/SiTU helpers; moe_k3.metal's include-guarded copy is skipped
/// when concatenated), so all K3 modules are compiled together, in this
/// fixed order. kda/mla/attnres_k3 are self-contained.
///
/// Safe math mode: the dequant kernel builds its products bit-by-bit (see
/// mxfp4.metal) because Apple GPUs flush subnormals in FP32 arithmetic, and
/// the remaining float ops (SiTU `tanh`/`exp`, the router sigmoid, the
/// signed-zero multiplies) get precise transcendentals and no reassociation
/// here. Decode is SSD-bound, so the precise-math cost is immaterial.
/// `@unchecked Sendable`: the compile products are immutable and Metal
/// objects are thread-safe for encoding; the entry/pipeline caches are the
/// only mutable state and are lock-guarded (same pattern as MetalContext).
final class K3MetalLibrary: @unchecked Sendable {
    static let shared = K3MetalLibrary()

    private struct PipelineCacheKey: Hashable {
        var name: String
        var constants: [MetalFunctionConstant]
        var maxTotalThreadsPerThreadgroup: Int?
    }

    private struct DeviceEntry {
        var library: MTLLibrary
        var pipelines: [PipelineCacheKey: MTLComputePipelineState] = [:]
    }

    private var entries: [ObjectIdentifier: DeviceEntry] = [:]
    private let lock = NSLock()

    private init() {}

    private static let modules: [String] = [
        "mxfp4", "moe_k3", "router_k3", "kda", "mla", "attnres_k3",
        "embed_k3", "sampling_k3", "trunk_k3", "dequant_k3", "prefill_k3",
    ]

    func pipeline(device: MTLDevice,
                  name: String,
                  constants: [MetalFunctionConstant] = [],
                  maxTotalThreadsPerThreadgroup hint: Int? = nil) throws
        -> MTLComputePipelineState
    {
        if let hint {
            precondition(hint > 0, "maxTotalThreadsPerThreadgroup must be positive")
        }
        let sortedConstants = constants.sorted {
            if $0.index != $1.index { return $0.index < $1.index }
            return String(describing: $0.value) < String(describing: $1.value)
        }
        let key = PipelineCacheKey(
            name: name, constants: sortedConstants,
            maxTotalThreadsPerThreadgroup: hint)
        lock.lock()
        if let cached = entries[ObjectIdentifier(device)]?.pipelines[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let library = try self.library(device: device)
        guard library.functionNames.contains(name) else {
            throw MetalError.missingFunction(name)
        }
        let values = MTLFunctionConstantValues()
        for constant in sortedConstants {
            switch constant.value {
            case .bool(let value):
                var v = value
                values.setConstantValue(&v, type: .bool, index: constant.index)
            case .uint32(let value):
                var v = value
                values.setConstantValue(&v, type: .uint, index: constant.index)
            case .float(let value):
                var v = value
                values.setConstantValue(&v, type: .float, index: constant.index)
            }
        }
        let function = try library.makeFunction(name: name, constantValues: values)
        let pipeline: MTLComputePipelineState
        if let hint {
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.maxTotalThreadsPerThreadgroup = hint
            pipeline = try device.makeComputePipelineState(
                descriptor: descriptor, options: [], reflection: nil)
        } else {
            pipeline = try device.makeComputePipelineState(function: function)
        }
        lock.lock()
        entries[ObjectIdentifier(device)]?.pipelines[key] = pipeline
        lock.unlock()
        return pipeline
    }

    /// Compile a single K3 shader module on its own (NOT concatenated, NOT
    /// safe-math). Used for `tensorops_k3` — the house keeps its tensorops
    /// module out of the shared concatenated library too, and the MSL 4.0
    /// `mpp::tensor_ops` path does not need the K3 decode library's safe-math
    /// subnormal handling. Throws on compile failure; the caller maps that to
    /// its non-NAX fallback (pre-M5 silicon, where `__HAVE_TENSOR__` is
    /// undefined and the module compiles to an empty library).
    static func separateModuleLibrary(device: MTLDevice, module: String) throws
        -> MTLLibrary
    {
        guard let url = Bundle.module.url(
            forResource: module, withExtension: "metal",
            subdirectory: "Metal/K3") else {
            throw MetalError.missingShaderResource("K3/\(module)")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            throw MetalError.libraryCompileFailed(
                "K3 separate Metal module \(module) failed: \(error)")
        }
    }

    private func library(device: MTLDevice) throws -> MTLLibrary {
        let id = ObjectIdentifier(device)
        lock.lock()
        if let entry = entries[id] {
            lock.unlock()
            return entry.library
        }
        lock.unlock()

        var combined = ""
        for name in Self.modules {
            guard let url = Bundle.module.url(
                forResource: name, withExtension: "metal",
                subdirectory: "Metal/K3") else {
                throw MetalError.missingShaderResource("K3/\(name)")
            }
            let source = try String(contentsOf: url, encoding: .utf8)
            combined += "\n// ==== K3/\(name).metal ====\n" + source + "\n"
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        // See the type docstring: bit-exact MXFP4 dequant needs IEEE subnormal
        // and signed-zero behavior.
        options.mathMode = .safe
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: combined, options: options)
        } catch {
            throw MetalError.libraryCompileFailed(
                "K3 combined decode Metal library failed: \(error)")
        }
        lock.lock()
        if let existing = entries[id] {
            lock.unlock()
            return existing.library
        }
        entries[id] = DeviceEntry(library: library)
        lock.unlock()
        return library
    }
}
