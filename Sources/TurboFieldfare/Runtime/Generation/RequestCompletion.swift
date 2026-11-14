import Foundation

/// The initiating failure remains authoritative; cleanup context is retained
/// even when a caller classifies cancellation or a typed request error.
public struct RequestCompletionFailure: Error, CustomStringConvertible {
    public let primary: any Error
    public let drain: any Error

    public init(primary: any Error, drain: any Error) {
        self.primary = primary
        self.drain = drain
    }

    public var description: String { "\(primary) (request drain also failed: \(drain))" }
}

public func requestPrimaryError(_ error: any Error) -> any Error {
    var primary = error
    while let combined = primary as? RequestCompletionFailure {
        primary = combined.primary
    }
    return primary
}

/// Optional internal capability: scripted producers need no cleanup hook.
/// The caller already owns the producer exclusively across the operation and
/// finish; workers are joined synchronously before recovery can regain control.
protocol RequestFinishingProducer: LogitProducer {
    func finishRequest() throws
}

func withFinishedInferenceRequest<Value>(
    producer: any LogitProducer,
    operation: () async throws -> Value
) async throws -> Value {
    let result: Result<Value, any Error>
    do { result = .success(try await operation()) }
    catch { result = .failure(error) }
    do {
        try (producer as? any RequestFinishingProducer)?.finishRequest()
    } catch {
        if case .failure(let primary) = result {
            throw RequestCompletionFailure(primary: primary, drain: error)
        }
        throw error
    }
    return try result.get()
}
