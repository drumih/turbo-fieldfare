import Foundation
import Metal
import TinyTitanCore

extension NemotronMetal {
    /// The engine's entry point: the shared `MetalContext` has already compiled
    /// the two shaders, so only the generic (non-constant-folded) pipelines are
    /// built here.
    public convenience init(
        install: NemotronInstall, context: MetalContext
    ) throws {
        let int4 = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [
                MetalFunctionConstant(index: 22, value: .bool(false)),
                MetalFunctionConstant(index: 26, value: .bool(false)),
            ])
        let int8 = try context.pipeline(
            "dequant_int8_gemv_simd",
            constants: [MetalFunctionConstant(index: 72, value: .bool(false))])
        try self.init(
            install: install, device: context.device, int4: int4, int8: int8)
    }
}
