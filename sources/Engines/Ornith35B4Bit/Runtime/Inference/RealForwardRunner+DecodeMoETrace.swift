import Foundation

extension RealForwardRunner {
    /// The per-layer decode MoE trace, split out of `encodeDecodeRoutedMoE` to
    /// keep that file under the production line cap. Pure code motion: the
    /// wording and the stderr destination are unchanged.
    ///
    /// stderr, not stdout: this target is inside `TinyTitanLib`'s closure,
    /// and stdout belongs to the program that embeds the library.
    func emitDecodeMoELayerTrace(
        position: Int, layer: Int,
        bodyUs: UInt64, waitUs: UInt64, ioUs: UInt64, cb1Us: UInt64, cb2Us: UInt64,
        attnUs: Int, tailUs: Int, routedUs: Int
    ) {
        let trace: String =
            "TinyTitan layer pos=\(position) L=\(layer) "
            + "body_us=\(bodyUs) "
            + "wait_us=\(waitUs) io_us=\(ioUs) "
            + "cb1_us=\(cb1Us) "
            + "cb2_us=\(cb2Us) "
            + "gpu_attn_us=\(attnUs) gpu_tail_us=\(tailUs) "
            + "gpu_routed_us=\(routedUs)\n"
        FileHandle.standardError.write(Data(trace.utf8))
    }
}
