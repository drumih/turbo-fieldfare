import Foundation

public enum ForwardRunnerFactory {
    public static func make(model: Model,
                            context: MetalContext,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production) throws -> any ForwardRunner {
        switch model.config.modelFamily {
        case .gemma4:
            return try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration)
        case .qwen36MoeText:
            return try QwenForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration)
        }
    }
}