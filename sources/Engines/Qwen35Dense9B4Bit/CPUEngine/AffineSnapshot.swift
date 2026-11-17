import Foundation
import TinyTitanCore

/// The side-engine's view of a snapshot written by `tools/prepare_qwen35.py`.
///
/// One directory: a `config.json` carrying the architecture and the
/// quantization block, an index naming which shard holds what, and the
/// shards themselves. Everything the engine needs to run is here, and
/// nothing it does not.
public struct AffineSnapshot: Sendable {

    /// A quantized matrix, as the kernels want it: unsigned lanes packed
    /// low-first into `UInt32`, one BF16 scale and bias per group.
    ///
    /// The pointers are into the mapping and stay valid for the snapshot's
    /// life. Nothing is copied, which is what makes loading a two-gigabyte
    /// model instant and its resident cost the page cache's problem rather
    /// than the process's.
    ///
    /// unchecked-invariant: every pointer addresses a read-only `MAP_PRIVATE`
    /// mapping owned by the `SafeTensorsFile` this came from, which outlives
    /// the snapshot that vends it. The pages are never written, by anyone, so
    /// concurrent readers cannot race and threading the GEMV over row ranges
    /// is safe by construction.
    public struct Matrix: @unchecked Sendable {
        public let weights: UnsafeRawBufferPointer
        public let scales: UnsafePointer<UInt16>
        public let biases: UnsafePointer<UInt16>
        public let rows: Int
        public let columns: Int
        public let bits: Int
        public let groupSize: Int
    }

    // TT-097: the configuration value type was promoted to TinyTitanCore
    // (as CPUEngineConfiguration) because the CPUEngineRunner contract
    // exposes it; this nested alias keeps every `AffineSnapshot.Configuration`
    // spelling resolving.
    public typealias Configuration = TinyTitanCore.CPUEngineConfiguration

    /// Where the tensors actually live.
    ///
    /// Two shapes carry the same architecture: a plain affine safetensors
    /// snapshot, which the converter writes and the CPU engine has always
    /// read, and a `.ssdai` install, which is what every other model here
    /// is. A `.ssdai` is a byte copy of the same quantized tensors (both
    /// quantizers are group-64 affine), so this is a storage difference and
    /// not a semantic one -- which `tools/ssdai_diff_snapshot.py` checks
    /// rather than assumes.
    enum Storage {
        case safetensors(
            shards: [String: SafeTensorsFile],
            placement: [String: String])
        case ssdai(index: ResidentIndex, weights: ResidentWeights)
    }

    /// One read-only mapping of a `.ssdai`'s resident payload, held for the
    /// life of the snapshot.
    ///
    /// Mapping once matters: `matrix(_:)` is called per tensor (several
    /// hundred times a token), and mapping the 1.3 GB file on each call paged
    /// the whole payload in repeatedly -- generation went from seconds to
    /// never finishing.
    ///
    /// unchecked-invariant: the file is opened read-only and mapped with
    /// `.alwaysMapped`, no operation in this project writes it, and the
    /// mapping outlives every pointer vended from it because the `Data` is
    /// held for the snapshot's lifetime. Readers therefore only ever read
    /// immutable pages, so handing the same base address to the CPU engine's
    /// row-range reads from several threads cannot race. The same reasoning
    /// the safetensors case documents for its shard mappings.
    struct ResidentWeights: @unchecked Sendable {
        let data: Data
        var base: UnsafeRawPointer? {
            data.withUnsafeBytes { $0.baseAddress }
        }
    }

    public let directory: URL
    public let configuration: Configuration
    let storage: Storage
    let baseBits: Int
    let groupSize: Int
    /// Per-tensor width overrides, keyed by stem. The 4-bit build keeps the
    /// tied embedding and the attention K/V at 8 bits, because measuring
    /// said that is where the error actually is.
    let widths: [String: Int]

    /// The safetensors shards, for the paths that are only reachable from the
    /// snapshot initializer. Nil for a `.ssdai` install.
    var shards: [String: SafeTensorsFile] {
        if case .safetensors(let shards, _) = storage { return shards }
        return [:]
    }
    var placement: [String: String] {
        if case .safetensors(_, let placement) = storage { return placement }
        return [:]
    }

    public func bits(forStem stem: String) -> Int { widths[stem] ?? baseBits }

    /// Refuse a quantized width the CPU kernels do not implement.
    ///
    /// The `.ssdai` reader gets this for free -- `SSDAIManifestQuantV1.init(from:)`
    /// refuses an override outside `supportedWeightBits` while decoding -- but the
    /// snapshot reader reads `config.json`'s `quantization` block with plain JSON
    /// casts, so nothing on that route looked at the number at all.
    ///
    /// Nothing downstream does either. `dequantize` derives its lane count as
    /// `32 / bits`, so a 6-bit tensor does not fail: it unpacks five lanes per
    /// word, shifts past the mask, and returns plausible nonsense. The GEMV at
    /// least ends in `preconditionFailure`, but that is an abort on model data.
    /// Either way the engine stops being what it exists to be -- an independent
    /// reference for the GPU path -- so the width is checked where it is read.
    /// 4 and 8 are what `tinytitan_int4_affine_gemv` and
    /// `tinytitan_int8_affine_gemv` implement.
    static func validateWidths(baseBits: Int, widths: [String: Int]) throws {
        let declared: [(String, Int)] =
            [("<quantization>.bits", baseBits)]
            + widths.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
        for (source, bits) in declared {
            guard [4, 8].contains(bits) else {
                throw SafeTensorsFile.Failure.malformed(
                    "\(source) declares \(bits)-bit weights. The CPU kernels "
                        + "implement 4 and 8 only, so this snapshot cannot be "
                        + "read correctly -- not as a reference and not as output.")
            }
        }
    }

    /// The stem of a `.weight` name, which is how widths and the index are
    /// keyed.
    func stem(of name: String) -> String {
        name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
    }

    /// One affine matrix out of a `.ssdai` resident payload.
    ///
    /// The index already carries the packed shape, the weight extent and the
    /// scale/bias extents, so this is a mapping rather than a parse. The
    /// resident file is opened read-only and never written, so the pointers
    /// handed out live as long as the mapping and concurrent row-range reads
    /// cannot race -- the same invariant the snapshot case documents.
    static func matrix(
        _ name: String,
        index: ResidentIndex,
        weights: ResidentWeights,
        groupSize: Int,
        bits: Int
    ) throws -> Matrix {
        let stem =
            name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count)) : name
        guard let entry = index.entries[name] else {
            throw SafeTensorsFile.Failure.missing(name)
        }
        let rows = Int(entry.shape.0)
        // The resident index stores the *logical* (unpacked) width -- the
        // repacker derives it from the scales, `lastScale * groupSize` -- where
        // a safetensors snapshot stores the packed word count and needs
        // `* lanes`. Reading it the snapshot's way made every matrix 2-4x too
        // wide, which showed up as a generation that never finished rather
        // than as an error.
        let columns = Int(entry.shape.1)
        // The companions are offsets on the weight's own entry, not entries of
        // their own: a repacked `.ssdai` carries one record per tensor and
        // points at its scale and bias spans. (A safetensors snapshot names
        // them as separate tensors, which is why the two readers differ here.)
        guard entry.sizeBytes > 0, entry.scaleSize > 0, entry.biasSize > 0 else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): empty weight, scales or biases in the resident index")
        }
        // The same extent check the safetensors branch below makes, on the same
        // arithmetic, so a corrupt or hand-edited index is refused at load
        // instead of being read as neighbouring bytes of the same mapping. The
        // resident file is one mapping, so a wrong span does not fault -- it
        // silently dequantizes whatever is next to it. BF16 companions are two
        // bytes per group per row, and the scales and biases must match.
        // `columns / groupSize` is integer division, so a width that is not a
        // whole number of groups truncates the group count and the guard below
        // then accepts scales sized for the wrong number of groups: the
        // dequantize reads fewer groups per row than the weights hold and returns
        // plausible nonsense. Producers derive the width from the scale count, so
        // this is defence in depth -- the class C34 guarded in the GEMVs -- and it
        // is checked here because this is where the division is.
        guard columns % groupSize == 0 else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): width \(columns) is not a whole number of "
                    + "\(groupSize)-element groups")
        }
        let perRow = columns / groupSize
        guard entry.scaleSize == UInt64(rows * perRow * 2),
            entry.biasSize == entry.scaleSize
        else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): scales/biases are \(entry.scaleSize)/\(entry.biasSize) bytes "
                    + "but \(rows)x\(columns) at group \(groupSize) needs "
                    + "\(rows * perRow * 2) each")
        }
        guard let base = weights.base else {
            throw SafeTensorsFile.Failure.malformed("model_weights.bin could not be mapped")
        }
        return Matrix(
            weights: UnsafeRawBufferPointer(
                start: base.advanced(by: Int(entry.fileOffset)),
                count: Int(entry.sizeBytes)),
            scales: base.advanced(by: Int(entry.scaleOffset))
                .assumingMemoryBound(to: UInt16.self),
            biases: base.advanced(by: Int(entry.biasOffset))
                .assumingMemoryBound(to: UInt16.self),
            rows: rows, columns: columns, bits: bits, groupSize: groupSize)
    }

    /// Fault every shard in, so the model is in memory rather than in the
    /// page cache's good graces. Returns the bytes made resident.
    @discardableResult
    public func makeResident() -> Int {
        switch storage {
        case .safetensors:
            return shards.values.reduce(0) { $0 + $1.makeResident() }
        case .ssdai(_, let weights):
            // A `.ssdai`'s resident payload is one file, mapped once at load.
            // Faulting it in is the same intent as faulting every snapshot
            // shard in: touch a byte per page so the pages are resident rather
            // than at the page cache's mercy.
            return Self.faultIn(weights)
        }
    }

    /// Touch a byte per page so the mapping is resident. Returns the bytes.
    private static func faultIn(_ weights: ResidentWeights) -> Int {
        var touched = 0
        weights.data.withUnsafeBytes { raw in
            // `madvise(WILLNEED)` is not exposed here and this is a load-time
            // cost paid once.
            var offset = 0
            while offset < raw.count {
                touched &+= Int(raw[offset])
                offset += 4096
            }
        }
        return weights.data.count
    }

    /// The family this snapshot's layer shape belongs to, or nil when the
    /// CPU engine does not implement it.
    public var family: CPUModelFamily? {
        CPUModelFamily.resolve(modelType: modelType)
    }

    public let modelType: String?

}
