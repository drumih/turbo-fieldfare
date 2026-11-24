import Foundation
import TinyTitanCore
import TinyTitanPluginAPI

/// The server's half of TPP/1 §4: every parameter the operator set is checked
/// against the resolved plugin's `ParameterSupport` before the model loads.
/// An explicit value the engine does not accept is refused with the plugin's
/// reason; a default it does not accept is replaced by the engine's own and
/// announced. Nothing the operator typed is accepted and silently ignored.
public enum PluginParameterGate {
    /// The refusal for the first explicit unsupported parameter, if any, and
    /// the notices for defaults the engine replaces.
    public static func check(_ arguments: ServerArguments, plugin: ResolvedPlugin)
        -> (refusal: String?, notices: [String])
    {
        let table = plugin.parameters
        let name = plugin.plugin.displayName
        var notices: [String] = []
        if arguments.kvCachePrecisionExplicit {
            if let refusal = table.kvCachePrecision.refusal(
                for: arguments.kvCachePrecision, flag: "--kv-bits", plugin: name,
                spelled: { "\($0.rawValue)" })
            {
                return (refusal, [])
            }
        } else if let notice = table.kvCachePrecision.resolveDefault(
            arguments.kvCachePrecision, flag: "--kv-bits", plugin: name,
            spelled: { "\($0.rawValue)" }
        ).notice {
            notices.append(notice)
        }
        if arguments.ropeScalingMode != .none,
            let refusal = table.ropeScaling.refusal(
                for: arguments.ropeScalingMode, flag: "--rope-scaling", plugin: name,
                spelled: { $0.rawValue })
        {
            return (refusal, [])
        }
        if let window = arguments.hostKVWindowTokens,
            let refusal = table.hostKVWindow.refusal(
                for: window, flag: "--host-kv-window", plugin: name)
        {
            return (refusal, [])
        }
        if arguments.maxConcurrentSequences != 1,
            let refusal = table.concurrentSequences.refusal(
                for: arguments.maxConcurrentSequences, flag: "--max-concurrent-sequences",
                plugin: name)
        {
            return (refusal, [])
        }
        if arguments.mtpModel != nil,
            let refusal = table.multiTokenPrediction.refusal(
                for: true, flag: "--mtp-model", plugin: name)
        {
            return (refusal, [])
        }
        return (nil, notices)
    }
}
