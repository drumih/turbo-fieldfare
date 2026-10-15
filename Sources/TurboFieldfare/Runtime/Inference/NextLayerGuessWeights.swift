import CryptoKit
import Darwin
import Foundation
import Metal

/// The fitted next-layer guess for `NextLayerExpertPrefetcher`, mapped
/// read-only. For each target layer 1..numLayers-1 it holds one router-shaped
/// matrix: that layer's router plus a ridge correction fitted offline on decode
/// routing traces, already multiplied by the effective scale. It is stored in
/// the router's own 8-bit affine format, so `router_gemv_gemma4_r4` runs it
/// unchanged with `unitScale` as the effective scale.
///
/// Layout, little-endian: a 64-byte header (8-byte magic `TFNLG1\0\0`, then
/// UInt32 numLayers, numExperts, hiddenSize, groupSize, firstTargetLayer,
/// targetLayerCount), then per target layer UInt8 weights [E][D], BF16
/// scales [E][D/64] and BF16 biases [E][D/64].
///
/// The runtime ships fitted files in its `NextLayerGuess/` resources, listed
/// in `index.json` by the model's manifest `sourceSnapshotHash` with the
/// file's SHA-256. A model with no entry gets no fitted guess; the early read
/// then scores with the next layer's own router.
final class NextLayerGuessWeights {
    static let magic: [UInt8] = Array("TFNLG1".utf8) + [0, 0]
    static let headerBytes = 64

    struct Matrix {
        let weightsOffset: Int
        let scalesOffset: Int
        let biasesOffset: Int
    }

    /// The whole mapped file; `Matrix` offsets index into it.
    let buffer: MTLBuffer
    /// `hiddenSize` BF16 ones, the effective scale the guess runs with.
    let unitScale: MTLBuffer
    /// File name and SHA-256 of the mapped bytes, for logs.
    let name: String
    let sha256: String
    private let numLayers: Int
    private let weightBytes: Int
    private let groupBytes: Int

    init(fileURL: URL, numLayers: Int, numExperts: Int, hiddenSize: Int,
         device: MTLDevice) throws {
        let groups = hiddenSize / Quantization.groupSize
        weightBytes = numExperts * hiddenSize
        groupBytes = numExperts * groups * MemoryLayout<UInt16>.size
        let expectedSize = Self.headerBytes
            + (numLayers - 1) * (weightBytes + 2 * groupBytes)

        var info = stat()
        guard stat(fileURL.path, &info) == 0 else {
            throw ModelError.posixFailed(call: "stat(\(fileURL.path))", errno: errno)
        }
        guard Int(info.st_size) == expectedSize else {
            throw ModelError.indexCorrupt(detail: "next-layer guess file is \(info.st_size) "
                + "bytes, expected \(expectedSize)")
        }
        // The buffer's deallocator unmaps the file.
        buffer = try ResidentBuffer(fileURL: fileURL, fileOffset: 0,
                                    residentSize: UInt64(expectedSize), device: device).buffer

        let header = UnsafeRawBufferPointer(start: buffer.contents(), count: Self.headerBytes)
        let fields = (0..<6).map {
            UInt32(littleEndian: header.loadUnaligned(fromByteOffset: 8 + 4 * $0,
                                                      as: UInt32.self))
        }
        let expected = [numLayers, numExperts, hiddenSize, Quantization.groupSize,
                        1, numLayers - 1].map(UInt32.init)
        guard Array(header.prefix(8)) == Self.magic, fields == expected else {
            throw ModelError.indexCorrupt(detail: "next-layer guess header \(fields) does not "
                + "match this model \(expected)")
        }
        self.numLayers = numLayers
        name = fileURL.lastPathComponent
        sha256 = SHA256.hash(data: UnsafeRawBufferPointer(start: buffer.contents(),
                                                          count: expectedSize))
            .map { String(format: "%02x", $0) }.joined()

        guard let ones = device.makeBuffer(length: hiddenSize * MemoryLayout<UInt16>.size,
                                           options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        let one = Quantization.bf16Bits(1)
        ones.contents().initializeMemory(as: UInt16.self, repeating: one, count: hiddenSize)
        unitScale = ones
    }

    private struct IndexEntry: Decodable {
        let sourceSnapshotHash: String
        let file: String
        let sha256: String
    }

    /// The fitted guess the runtime ships for the model whose manifest has
    /// `sourceSnapshotHash`, or nil when it ships none. Throws when the
    /// shipped file is damaged: its hash must match the index.
    static func bundled(sourceSnapshotHash: String?, numLayers: Int, numExperts: Int,
                        hiddenSize: Int, device: MTLDevice,
                        bundle: Bundle = .module) throws -> NextLayerGuessWeights? {
        guard let sourceSnapshotHash,
              let indexURL = bundle.url(forResource: "index", withExtension: "json",
                                        subdirectory: "NextLayerGuess") else { return nil }
        let entries = try JSONDecoder().decode([IndexEntry].self,
                                               from: Data(contentsOf: indexURL))
        guard let entry = entries.first(where: { $0.sourceSnapshotHash == sourceSnapshotHash })
        else { return nil }
        let url = indexURL.deletingLastPathComponent().appendingPathComponent(entry.file)
        let weights = try NextLayerGuessWeights(fileURL: url, numLayers: numLayers,
                                                numExperts: numExperts,
                                                hiddenSize: hiddenSize, device: device)
        guard weights.sha256 == entry.sha256 else {
            throw ModelError.indexCorrupt(detail: "shipped next-layer guess \(entry.file) has "
                + "SHA-256 \(weights.sha256), expected \(entry.sha256)")
        }
        return weights
    }

    /// The guess matrix that scores `targetLayer`'s experts, 1..numLayers-1.
    func matrix(targetLayer: Int) -> Matrix {
        precondition(targetLayer >= 1 && targetLayer < numLayers)
        let start = Self.headerBytes + (targetLayer - 1) * (weightBytes + 2 * groupBytes)
        return Matrix(weightsOffset: start,
                      scalesOffset: start + weightBytes,
                      biasesOffset: start + weightBytes + groupBytes)
    }
}
