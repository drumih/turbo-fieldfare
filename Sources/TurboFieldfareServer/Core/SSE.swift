import Foundation

/// Server-Sent Events framing. One OpenAI JSON object per `data:` frame,
/// terminated by the literal `[DONE]` sentinel.
public enum SSEFramer {
    /// Shared encoder; JSONEncoder is documented as thread-safe for reuse.
    private static let encoder = JSONEncoder()

    public static func frame<T: Encodable>(_ value: T) throws -> Data {
        var data = Data("data: ".utf8)
        data.append(try encoder.encode(value))
        data.append(Data("\n\n".utf8))
        return data
    }

    public static var doneFrame: Data {
        Data("data: [DONE]\n\n".utf8)
    }
}
