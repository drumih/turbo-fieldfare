import Foundation
import Metal

package enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case noQueue
    case missingShaderResource(String)
    case missingFunction(String)
    case libraryCompileFailed(String)
    case commandEncoderFailed
    case bufferAllocationFailed(String)
    case invalidState(String)

    public var description: String {
        switch self {
        case .noDevice: return "No Metal device"
        case .noQueue: return "Failed to create Metal command queue"
        case .missingShaderResource(let n): return "Shader resource missing: \(n)"
        case .missingFunction(let n): return "Metal function missing in library: \(n)"
        case .libraryCompileFailed(let s): return "Metal library compile failed: \(s)"
        case .commandEncoderFailed: return "Failed to create Metal compute command encoder"
        case .bufferAllocationFailed(let label):
            return "Failed to allocate Metal buffer: \(label)"
        case .invalidState(let detail): return "Invalid Metal encoder state: \(detail)"
        }
    }
}

public struct MetalFunctionConstant: Hashable, Sendable {
    public enum Value: Hashable, Sendable {
        case bool(Bool)
        case uint32(UInt32)
        case float(Float)
    }

    public let index: Int
    public let value: Value

    public init(index: Int, value: Value) {
        self.index = index
        self.value = value
    }
}

/// Single owner of the `MTLDevice`, queue, and the runtime-compiled shader library.
/// On Mac and iOS we ship `.metal` source files as bundle resources and compile
/// them into one combined `MTLLibrary` at startup. This keeps the dev loop
/// fast — edit a shader, rebuild the Swift target, no Xcode metallib step.
/// `@unchecked Sendable`: device/queue/library are immutable and Metal objects
/// are thread-safe for encoding; the pipeline cache is the only mutable state
/// and is lock-guarded.
/// unchecked-invariant: device, queue and library are set once in `init` and
/// never reassigned; MTLDevice and MTLCommandQueue are documented as safe to
/// use from multiple threads. The pipeline cache behind them is the only
/// mutable state and is guarded by its own lock.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary

    private struct PipelineCacheKey: Hashable {
        var name: String
        var constants: [MetalFunctionConstant]
        var maxTotalThreadsPerThreadgroup: Int?
    }

    private var pipelineCache: [PipelineCacheKey: MTLComputePipelineState] = [:]
    private let pipelineCacheLock = NSLock()

    /// The context for the system default device, which is what a command-line
    /// front end wants.
    public convenience init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        try self.init(device: dev)
    }

    /// The context for a device the caller owns.
    ///
    /// Every queue, pipeline and buffer this context hands out belongs to that
    /// device, which is the point: a process with more than one GPU, or holding
    /// a device it took from somewhere other than the system default, runs on
    /// the one it named rather than on whichever device happens to be default.
    public init(device dev: MTLDevice) throws {
        guard let q = dev.makeCommandQueue() else { throw MetalError.noQueue }
        self.device = dev
        self.queue = q
        self.library = try Self.compileShaderLibrary(device: dev)
    }

    deinit {
        // NOTE: Metal MTLCommandQueue does not provide deinit-safe cleanup.
        // All pending command buffers should be completed via waitForCompletion()
        // before the app quits. The Metal device/queue will be released when the
        // process terminates.
    }

    /// Production shader modules compiled into the shared runtime library.
    private static let shaderModules: [String] = [
        "dequant_int4",
        "dequant_int8",
        "dequant_affine",
        "kv_cache_quantize",
        "rmsnorm",
        "rope",
        "attention",
        "moe",
        "logit",
        "utility",
        "fused",
        "prefill",
        "gdn",
        "qsa",
        "bf16_gemv",
        "dequant_iq",
        "gguf_moe",
        "gguf_embed",
        "ple_iq",
        "nemotron",
    ]

    /// Bundle locations for runtime shader modules.
    private static let shaderSubdirectories: [String: String] = [
        "attention": "Metal/Attention",
        "qsa": "Metal/Attention",
        "dequant_int4": "Metal/Quant",
        "dequant_int8": "Metal/Quant",
        "dequant_affine": "Metal/Quant",
        "dequant_iq": "Metal/Quant",
        "gguf_moe": "Metal/MoE",
        "gguf_embed": "Metal/Quant",
        "ple_iq": "Metal/Prefill",
        "bf16_gemv": "Metal/Quant",
        "kv_cache_quantize": "Metal/Quant",
        "fused": "Metal/Fusions",
        "gdn": "Metal/GDN",
        "logit": "Metal/Sampling",
        "moe": "Metal/MoE",
        "prefill": "Metal/Prefill",
        "rmsnorm": "Metal/Primitives",
        "rope": "Metal/Primitives",
        "tensorops": "Metal/TensorCore",
        "utility": "Metal/Primitives",
        "nemotron": "Metal/Nemotron",
    ]

    private static func shaderURL(module: String) -> URL? {
        guard let subdirectory = shaderSubdirectories[module] else { return nil }
        return Bundle.module.url(
            forResource: module, withExtension: "metal",
            subdirectory: subdirectory)
    }

    private static func compileShaderLibrary(device: MTLDevice) throws -> MTLLibrary {
        var combined = ""
        for name in shaderModules {
            guard let url = shaderURL(module: name) else {
                throw MetalError.missingShaderResource(name)
            }
            // lint:allow-unbounded-read Bundle.module resolves this to a .metal
            // source this package itself ships and compiles in full; no path a
            // caller can hand the engine reaches this read.
            let src = try String(contentsOf: url, encoding: .utf8)
            combined += "\n// ==== \(name).metal ====\n" + src + "\n"
        }
        do {
            let opts = MTLCompileOptions()
            // The MPP prefill path requires MSL 4.0 tensor operations.
            opts.languageVersion = .version4_0
            return try device.makeLibrary(source: combined, options: opts)
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    /// Compile a shader module separately from the shared runtime library.
    public static func moduleLibrary(device: MTLDevice, module: String) throws -> MTLLibrary {
        guard let url = shaderURL(module: module) else {
            throw MetalError.missingShaderResource(module)
        }
        // lint:allow-unbounded-read Same resource as `compileShaderLibrary`: a
        // .metal file this package ships, named by `module`, not by a caller.
        let src = try String(contentsOf: url, encoding: .utf8)
        let opts = MTLCompileOptions()
        opts.languageVersion = .version4_0
        do {
            return try device.makeLibrary(source: src, options: opts)
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        try pipeline(name, constants: [])
    }

    public func pipeline(
        _ name: String,
        constants: [MetalFunctionConstant]
    ) throws -> MTLComputePipelineState {
        try pipeline(name, constants: constants, maxTotalThreadsPerThreadgroup: nil)
    }

    public func pipeline(
        _ name: String,
        constants: [MetalFunctionConstant],
        maxTotalThreadsPerThreadgroup hint: Int?
    ) throws -> MTLComputePipelineState {
        if let hint {
            precondition(hint > 0, "maxTotalThreadsPerThreadgroup must be positive")
        }
        let sortedConstants = constants.sorted {
            if $0.index != $1.index { return $0.index < $1.index }
            return Self.constantSortKey($0.value) < Self.constantSortKey($1.value)
        }
        let key = PipelineCacheKey(
            name: name,
            constants: sortedConstants,
            maxTotalThreadsPerThreadgroup: hint)
        // K2: compile+insert happen inside the lock. The naive double-checked
        // locking compiled the PSO *outside* the lock, so a concurrent miss on
        // the same key compiled the pipeline multiple times. Compiling under
        // the lock serializes only the (rare, startup-only) cold-miss path;
        // reads after initialization are served from the fast path above.
        pipelineCacheLock.lock()
        defer { pipelineCacheLock.unlock() }
        if let existing = pipelineCache[key] { return existing }

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

        let fn = try library.makeFunction(name: name, constantValues: values)
        let p: MTLComputePipelineState
        if let hint {
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = fn
            descriptor.maxTotalThreadsPerThreadgroup = hint
            var reflection: MTLAutoreleasedComputePipelineReflection?
            p = try device.makeComputePipelineState(
                descriptor: descriptor,
                options: [],
                reflection: &reflection)
        } else {
            p = try device.makeComputePipelineState(function: fn)
        }
        pipelineCache[key] = p
        return p
    }

    private static func constantSortKey(_ value: MetalFunctionConstant.Value) -> String {
        switch value {
        case .bool(let v): return "b:\(v ? 1 : 0)"
        case .uint32(let v): return "u:\(v)"
        case .float(let v): return "f:\(v.bitPattern)"
        }
    }
}
