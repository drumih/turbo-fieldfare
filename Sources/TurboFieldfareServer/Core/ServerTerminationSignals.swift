import Darwin
import Dispatch

public actor ServerTerminationSignals {
    private let stream: AsyncStream<Int32>
    private let continuation: AsyncStream<Int32>.Continuation
    private let sources: [any DispatchSourceSignal]

    public init(_ signals: [Int32] = [SIGINT, SIGTERM]) {
        var capturedContinuation: AsyncStream<Int32>.Continuation?
        let stream = AsyncStream<Int32>(bufferingPolicy: .bufferingOldest(1)) {
            capturedContinuation = $0
        }
        let continuation = capturedContinuation!

        self.stream = stream
        self.continuation = continuation
        self.sources = signals.map {
            Darwin.signal($0, SIG_IGN)
            return Self.makeSource(signal: $0, continuation: continuation)
        }
        for source in sources {
            source.resume()
        }
    }

    public func wait() async -> Int32 {
        for await signal in stream {
            return signal
        }
        preconditionFailure("termination signal stream ended without a signal")
    }

    public func cancel() {
        for source in sources {
            source.cancel()
        }
        continuation.finish()
    }

    private nonisolated static func makeSource(
        signal: Int32,
        continuation: AsyncStream<Int32>.Continuation
    ) -> any DispatchSourceSignal {
        let source = DispatchSource.makeSignalSource(signal: signal, queue: .global())
        source.setEventHandler { @Sendable [continuation] in
            continuation.yield(signal)
        }
        return source
    }
}
