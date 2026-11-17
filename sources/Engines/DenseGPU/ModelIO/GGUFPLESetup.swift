import Foundation

package enum GGUFPLESetupError: Error, Equatable, CustomStringConvertible {
    case missingKey(String)
    case inconsistent(String)

    package var description: String {
        switch self {
        case .missingKey(let key): return "the checkpoint declares no \(key)"
        case .inconsistent(let detail):
            return "the checkpoint's PLE constants are inconsistent: \(detail)"
        }
    }
}

/// The per-layer n-gram embedding constants, read from the checkpoint.
///
/// The multipliers, per-head offsets and per-head vocab sizes are data, not
/// something to re-derive: they are what decides which row of the 320-million
/// row table a token's n-gram hashes to. The GGUF carries all of them, plus the
/// table geometry, so the hash can be built and checked against the table it
/// indexes.
package struct GGUFPLESetup {
    package let hash: PLEHash
    package let layerIndices: [Int]
    package let ngramSize: Int
    package let headsPerNgram: Int
    package let convKernelSize: Int
    package let headCount: Int
    package let embedDim: Int
    package let tableRows: UInt64

    package static func read(from model: GGUFModel) throws -> GGUFPLESetup {
        func integers(_ key: String) throws -> [Int64] {
            guard let value = model.metadata["qwen4exp.\(key)"] else {
                throw GGUFPLESetupError.missingKey("qwen4exp.\(key)")
            }
            switch value {
            case .array(let items):
                return try items.map {
                    guard let integer = $0.integer else {
                        throw GGUFPLESetupError.inconsistent("\(key) holds a non-integer")
                    }
                    return integer
                }
            default:
                guard let integer = value.integer else {
                    throw GGUFPLESetupError.inconsistent("\(key) is not an integer")
                }
                return [integer]
            }
        }
        func integer(_ key: String) throws -> Int64 {
            guard let value = model.metadata["qwen4exp.\(key)"] else {
                throw GGUFPLESetupError.missingKey("qwen4exp.\(key)")
            }
            guard let integer = value.integer else {
                throw GGUFPLESetupError.inconsistent("\(key) is not an integer")
            }
            return integer
        }

        let layers = try integers("ple.layers").map(Int.init)
        let ngramSize = Int(try integer("ple.ngram_size"))
        let headsPerNgram = Int(try integer("ple.heads_per_ngram"))
        let convKernel = Int(try integer("ple.conv_kernel"))
        let eos = Int32(truncatingIfNeeded: try integer("ple.eos_token_id"))
        let multipliers = try integers("ple.layer_multipliers")
        let offsets = try integers("ple.head_offsets")
        let vocabSizes = try integers("ple.head_vocab_sizes")

        guard let table = model.tensor(named: "per_layer_token_embd.weight") else {
            throw GGUFPLESetupError.missingKey("per_layer_token_embd.weight")
        }
        let rowWidth = Int(table.tensor.dimensions[0])
        let tableRows = UInt64(table.tensor.dimensions[1])
        let headCount = headsPerNgram * (ngramSize - 1)
        guard multipliers.count == ngramSize else {
            throw GGUFPLESetupError.inconsistent(
                "\(multipliers.count) multipliers for \(ngramSize) n-gram slots")
        }
        guard offsets.count == headCount, vocabSizes.count == headCount else {
            throw GGUFPLESetupError.inconsistent(
                "\(offsets.count) offsets and \(vocabSizes.count) vocab sizes for \(headCount) heads"
            )
        }
        // Every head's range must be addressable in the table. It need not
        // end exactly at the last row: this checkpoint's ranges cover
        // 320,001,446 of the table's 320,001,536 rows, so the tail is simply
        // unused and asserting equality would refuse a valid file.
        let covered = zip(offsets, vocabSizes).map { UInt64($0 + $1) }.max() ?? 0
        guard covered <= tableRows else {
            throw GGUFPLESetupError.inconsistent(
                "the head ranges end at \(covered), past the \(tableRows)-row table")
        }
        let hash = PLEHash(
            multipliers: multipliers.map { UInt64(bitPattern: $0) },
            offsets: offsets.map { UInt64(bitPattern: $0) },
            vocabSizes: vocabSizes.map { UInt64(bitPattern: $0) },
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            eosTokenID: eos)
        return GGUFPLESetup(
            hash: hash,
            layerIndices: layers,
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            convKernelSize: convKernel,
            headCount: headCount,
            embedDim: headCount * rowWidth,
            tableRows: tableRows)
    }

    /// The row ids one token gathers, checked against the table they index.
    package func rows(context: [Int32]) -> [UInt32] {
        hash.rows(context: context)
    }
}
