/// The dense CPU engine's TT-097 contract: exactly the surface
/// `CPUModelBackend` drives — the geometry the grammar sizes its mask from,
/// the thread width, and the three-step generate loop (`reset`, `prefill`,
/// `step`). Conformed today by the legacy runtime and by every dense
/// variant's byte-identical copy, so the Phase 2 backend can hold `any
/// CPUEngineRunner` and a factory can choose which engine builds it.
///
/// Deliberately not `Sendable`: a CPU engine owns mutable KV and recurrent
/// state and is confined to the actor that serves it. The GPU contract
/// (`EngineRunner`) is Sendable because its session crosses concurrency
/// domains; this one has no such caller.
///
/// The plan sketched a model/runner split; the split does not exist on this
/// path — CPU construction is single-stage and the backend consumes one
/// object — so one protocol carries the whole surface.
public protocol CPUEngineRunner: AnyObject {
    /// The geometry the engine was built from (hidden sizes, head counts,
    /// and the `vocabulary` the structured-output mask is sized from).
    var configuration: CPUEngineConfiguration { get }
    /// The GEMV row-split width in force. A scheduling decision, not a
    /// property of the model.
    var threads: Int { get }
    /// Forget the conversation: every carried state goes, and the next token
    /// is position zero again.
    func reset()
    /// Prefill a whole prompt; returns the next-token logits.
    func prefill(tokens: [Int]) throws -> [Float]
    /// One decode step from `token`; returns the next-token logits.
    func step(token: Int) throws -> [Float]
    /// The sampling row the model's card specifies, used wherever a request
    /// leaves a field unset. The engine owns it, not the host (TT-134): a
    /// host that switched on the model family to find it is model knowledge
    /// outside the plugin.
    var samplingDefaults: SamplingProfile { get }
    /// Fault the weights in and keep them there; returns the bytes made
    /// resident. The engine owns its storage, so only it can pin it.
    func makeResident() -> Int
}
