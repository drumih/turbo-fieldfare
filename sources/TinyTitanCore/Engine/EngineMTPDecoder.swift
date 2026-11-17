/// The session-facing capability of an engine's speculative decoder. The
/// ordinary generation loop already consumes `LogitProducer`; MTP adds only
/// its independent draft-window bound.
public protocol EngineMTPDecoder: LogitProducer, ContextWindowReporting {
    var draftMaxContext: Int { get }
    func diagnosticLines(result: RawDecodeResult, includePhases: Bool) -> [String]
}

extension EngineMTPDecoder {
    public func diagnosticLines(result: RawDecodeResult, includePhases: Bool) -> [String] { [] }
}
