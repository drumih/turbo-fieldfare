import Foundation
import Metal

enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case noQueue
    case missingShaderResource(String)
    case missingFunction(String)
    case libraryCompileFailed(String)
    case commandBufferFailed(String)

    public var description: String {
        switch self {
        case .noDevice:                   return "No Metal device"
        case .noQueue:                    return "Failed to create Metal command queue"
        case .missingShaderResource(let n): return "Shader resource missing: \(n)"
        case .missingFunction(let n):     return "Metal function missing in library: \(n)"
        case .libraryCompileFailed(let s):return "Metal library compile failed: \(s)"
        case .commandBufferFailed(let s): return "Metal command buffer failed: \(s)"
        }
    }
}

func metalCommandBufferStatusName(_ status: MTLCommandBufferStatus) -> String {
    switch status {
    case .notEnqueued: return "notEnqueued"
    case .enqueued:    return "enqueued"
    case .committed:   return "committed"
    case .scheduled:   return "scheduled"
    case .completed:   return "completed"
    case .error:       return "error"
    @unknown default:  return "unknown(\(status.rawValue))"
    }
}

/// Builds the diagnostic for a command buffer that did not complete, or nil
/// when it did. Status is checked separately because a failed buffer is not
/// guaranteed to carry an error object.
func metalCommandBufferFailureDetail(label: String?,
                                     status: MTLCommandBufferStatus,
                                     error: (any Error)?) -> String? {
    if status == .completed && error == nil { return nil }

    var parts = ["label=\(label.map { $0.isEmpty ? "<empty>" : $0 } ?? "<none>")"]
    parts.append("status=\(metalCommandBufferStatusName(status))")
    if let error {
        let nsError = error as NSError
        parts.append("domain=\(nsError.domain)")
        parts.append("code=\(nsError.code)")
        parts.append("description=\(nsError.localizedDescription)")
        if !nsError.userInfo.isEmpty {
            parts.append("userInfoKeys=\(nsError.userInfo.keys.sorted().joined(separator: ","))")
        }
    } else {
        parts.append("error=<none>")
    }
    return parts.joined(separator: " ")
}

func checkCommandBufferError(_ commandBuffer: MTLCommandBuffer) throws {
    guard let detail = metalCommandBufferFailureDetail(label: commandBuffer.label,
                                                       status: commandBuffer.status,
                                                       error: commandBuffer.error) else {
        return
    }
    throw MetalError.commandBufferFailed(detail)
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
public final class MetalContext: @unchecked Sendable {
    public let device:  MTLDevice
    public let queue:   MTLCommandQueue
    public let library: MTLLibrary

    private struct PipelineCacheKey: Hashable {
        var name: String
        var constants: [MetalFunctionConstant]
        var maxTotalThreadsPerThreadgroup: Int?
    }

    private var pipelineCache: [PipelineCacheKey: MTLComputePipelineState] = [:]
    private let pipelineCacheLock = NSLock()

    private static func relaxInteractivityWatchdog() {
        #if os(macOS)
        // The AGX driver reads this once at first device creation. Long prefill
        // dispatches can otherwise be killed as compositor-impacting on macOS
        // 26. Overwrite 0 preserves an operator's explicit stock-behaviour
        // override. This relaxes the deadline; it does not guarantee survival.
        setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 0)
        #endif
    }

    /// Routes every production device creation through the watchdog mitigation
    /// before the AGX driver's process-wide one-time environment read.
    public static func makeSystemDefaultDevice() -> MTLDevice? {
        relaxInteractivityWatchdog()
        return MTLCreateSystemDefaultDevice()
    }

    public init() throws {
        guard let dev = Self.makeSystemDefaultDevice() else { throw MetalError.noDevice }
        guard let q   = dev.makeCommandQueue()           else { throw MetalError.noQueue }
        self.device  = dev
        self.queue   = q
        self.library = try Self.compileShaderLibrary(device: dev)
    }

    /// Production shader modules compiled into the shared runtime library.
    private static let shaderModules: [String] = [
        "dequant_int4",
        "dequant_int8",
        "rmsnorm",
        "rope",
        "attention",
        "moe",
        "logit",
        "utility",
        "fused",
        "prefill",
        "vision",
    ]

    /// Bundle locations for runtime shader modules.
    private static let shaderSubdirectories: [String: String] = [
        "attention": "Metal/Attention",
        "dequant_int4": "Metal/Quant",
        "dequant_int8": "Metal/Quant",
        "fused": "Metal/Fusions",
        "logit": "Metal/Sampling",
        "moe": "Metal/MoE",
        "prefill": "Metal/Prefill",
        "rmsnorm": "Metal/Primitives",
        "rope": "Metal/Primitives",
        "tensorops": "Metal/TensorCore",
        "utility": "Metal/Primitives",
        "vision": "Metal/Vision",
        "vision_register_gemm": "Metal/Vision",
        "vision_resize": "Metal/Vision",
    ]

    private static func shaderURL(module: String) -> URL? {
        guard let subdirectory = shaderSubdirectories[module] else { return nil }
        return Bundle.module.url(forResource: module, withExtension: "metal",
                                 subdirectory: subdirectory)
    }

    private static func compileShaderLibrary(device: MTLDevice) throws -> MTLLibrary {
        var combined = ""
        for name in shaderModules {
            guard let url = shaderURL(module: name) else {
                throw MetalError.missingShaderResource(name)
            }
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

    /// Compile one shader module into its own library, leaving the shared
    /// runtime library untouched.
    ///
    /// Cached per device, module, math mode and source variant: libraries are
    /// immutable, and
    /// one `VisionRuntime` init otherwise compiles the identical tensorops
    /// source three times (linear, attention, projector) on every load.
    public static func privateLibrary(device: MTLDevice, module: String,
                                      mathMode: MTLMathMode? = nil,
                                      includeVisionTensorOps: Bool = false) throws
        -> MTLLibrary {
        let key = "\(ObjectIdentifier(device).hashValue)#\(module)"
            + "#\(mathMode?.rawValue ?? -1)#\(includeVisionTensorOps)"
        privateLibraryLock.lock()
        defer { privateLibraryLock.unlock() }
        if let cached = privateLibraryCache[key] {
            return cached
        }
        guard let url = shaderURL(module: module) else {
            throw MetalError.missingShaderResource(module)
        }
        let src = try String(contentsOf: url, encoding: .utf8)
        let opts = MTLCompileOptions()
        opts.languageVersion = .version4_0
        if let mathMode {
            opts.mathMode = mathMode
        }
        if includeVisionTensorOps {
            opts.preprocessorMacros = [
                "TURBO_FIELDFARE_VISION_TENSOROPS": NSNumber(value: true)
            ]
        }
        do {
            let library = try device.makeLibrary(source: src, options: opts)
            privateLibraryCache[key] = library
            return library
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    private static let privateLibraryLock = NSLock()
    private static nonisolated(unsafe) var privateLibraryCache: [String: MTLLibrary] = [:]

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        try pipeline(name, constants: [])
    }

    public func pipeline(_ name: String,
                         constants: [MetalFunctionConstant]) throws -> MTLComputePipelineState {
        try pipeline(name, constants: constants, maxTotalThreadsPerThreadgroup: nil)
    }

    public func pipeline(_ name: String,
                         constants: [MetalFunctionConstant],
                         maxTotalThreadsPerThreadgroup hint: Int?) throws -> MTLComputePipelineState {
        if let hint {
            precondition(hint > 0, "maxTotalThreadsPerThreadgroup must be positive")
        }
        let sortedConstants = constants.sorted {
            if $0.index != $1.index { return $0.index < $1.index }
            return Self.constantSortKey($0.value) < Self.constantSortKey($1.value)
        }
        let key = PipelineCacheKey(name: name,
                                   constants: sortedConstants,
                                   maxTotalThreadsPerThreadgroup: hint)
        pipelineCacheLock.lock()
        let cached = pipelineCache[key]
        pipelineCacheLock.unlock()
        if let cached { return cached }

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
            p = try device.makeComputePipelineState(descriptor: descriptor,
                                                    options: [],
                                                    reflection: &reflection)
        } else {
            p = try device.makeComputePipelineState(function: fn)
        }
        pipelineCacheLock.lock()
        pipelineCache[key] = p
        pipelineCacheLock.unlock()
        return p
    }

    private static func constantSortKey(_ value: MetalFunctionConstant.Value) -> String {
        switch value {
        case .bool(let v):   return "b:\(v ? 1 : 0)"
        case .uint32(let v): return "u:\(v)"
        case .float(let v):  return "f:\(v.bitPattern)"
        }
    }
}
