import Foundation

public enum RuntimeGPUKeepWarm {
    public static let minimumIntervalUs = 100
    public static let maximumIntervalUs = 20_000

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Int? {
        guard let raw = environment["TINYTITAN_GPU_KEEPWARM_US"] else { return nil }
        guard let value = Int(raw) else {
            throw GPUKeepWarmError.invalidInterval(raw)
        }
        try validate(value)
        return value
    }

    public static func validate(_ intervalUs: Int) throws {
        guard intervalUs >= minimumIntervalUs, intervalUs <= maximumIntervalUs else {
            throw GPUKeepWarmError.invalidInterval("\(intervalUs)")
        }
    }
}

public enum GPUKeepWarmError: Error, CustomStringConvertible, Equatable {
    case invalidInterval(String)

    public var description: String {
        switch self {
        case .invalidInterval(let value):
            "unsupported GPU keep-warm interval \(value) us; allowed: "
                + "\(RuntimeGPUKeepWarm.minimumIntervalUs)...\(RuntimeGPUKeepWarm.maximumIntervalUs)"
        }
    }
}
