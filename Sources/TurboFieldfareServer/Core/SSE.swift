import Foundation

/// Server-Sent Events framing. The OpenAI surface emits one JSON object per
/// `data:` frame terminated by the literal `[DONE]` sentinel; the Anthropic
/// surface adds an `event:` line per frame and has no sentinel.
public enum SSEFramer {
    /// Shared encoder; JSONEncoder is documented as thread-safe for reuse.
    private static let encoder = JSONEncoder()

    public static func frame<T: Encodable>(_ value: T) throws -> Data {
        var data = Data("data: ".utf8)
        data.append(try encoder.encode(value))
        data.append(Data("\n\n".utf8))
        return data
    }

    /// Named-event variant for the Anthropic Messages API, whose SSE streams
    /// tag every frame with an `event:` line.
    public static func frame<T: Encodable>(event: String, _ value: T) throws -> Data {
        var data = Data("event: \(event)\ndata: ".utf8)
        data.append(try encoder.encode(value))
        data.append(Data("\n\n".utf8))
        return data
    }

    public static var doneFrame: Data {
        Data("data: [DONE]\n\n".utf8)
    }
}
