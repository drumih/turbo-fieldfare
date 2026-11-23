import Foundation
import Testing

@testable import TinyTitanFormat

/// The installer's GGUF header reader (TT-121 phase 1).
@Suite struct GGUFHeaderV3Tests {
    /// A minimal v3 file: two metadata keys, one 2 x 64 F32 tensor at data offset 0.
    static func tinyFile() -> [UInt8] {
        var out: [UInt8] = []
        func u32(_ v: UInt32) { out += withUnsafeBytes(of: v.littleEndian, Array.init) }
        func u64(_ v: UInt64) { out += withUnsafeBytes(of: v.littleEndian, Array.init) }
        func str(_ s: String) {
            u64(UInt64(s.utf8.count))
            out += Array(s.utf8)
        }
        out += Array("GGUF".utf8)
        u32(3)
        u64(1)  // tensors
        u64(3)  // metadata keys
        str("general.architecture")
        u32(8)
        str("inkling2")
        str("inkling2.block_count")
        u32(4)
        u32(24)
        str("inkling2.layers")
        u32(9)
        u32(5)  // int32 array
        u64(3)
        for v: UInt32 in [0, 1, 2] { u32(v) }
        str("w")
        u32(2)
        u64(64)
        u64(2)
        u32(0)  // F32
        u64(0)
        while out.count % 32 != 0 { out.append(0) }
        out += [UInt8](repeating: 0, count: 2 * 64 * 4)
        return out
    }

    @Test func aSmallFileParses() throws {
        let bytes = Self.tinyFile()
        let header = try GGUFHeaderV3(bytes: bytes)
        #expect(header.version == 3)
        #expect(header.string("general.architecture") == "inkling2")
        #expect(header.number("inkling2.block_count") == 24)
        #expect(header.numbers("inkling2.layers") == [0, 1, 2])
        #expect(header.alignment == 32)
        #expect(header.dataStart % 32 == 0)
        #expect(
            header.tensors == [
                .init(
                    name: "w", dimensions: [64, 2], ggmlType: 0, absoluteOffset: header.dataStart)
            ])
    }

    @Test func aTruncatedHeaderAsksForMoreBytes() throws {
        let bytes = Array(Self.tinyFile().prefix(60))
        #expect {
            _ = try GGUFHeaderV3(bytes: bytes)
        } throws: { error in
            guard
                case GGUFHeaderV3.ReadError.needMoreBytes(let count)? = error
                    as? GGUFHeaderV3.ReadError
            else { return false }
            return count > 60
        }
    }

    @Test func aWrongMagicIsRefused() {
        var bytes = Self.tinyFile()
        bytes[0] = 0x58
        #expect(throws: GGUFHeaderV3.ReadError.self) { _ = try GGUFHeaderV3(bytes: bytes) }
    }

    private static var inklingGGUF: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_GGUF"]
    }

    /// The real Q4_K_XL header: 734 tensors, and the first expert tensor at the
    /// absolute offset llama.cpp's gguf-py reports for it.
    @Test(.enabled(if: inklingGGUF != nil)) func theInklingHeaderMatchesGGUFPy() throws {
        let path = try #require(Self.inklingGGUF)
        let handle = try #require(FileHandle(forReadingAtPath: path))
        defer { try? handle.close() }
        let prefix = [UInt8](try handle.read(upToCount: 64 << 20) ?? Data())
        let header = try GGUFHeaderV3(bytes: prefix)
        #expect(header.tensors.count == 734)
        #expect(header.string("general.architecture") == "inkling2")
        let gateUp = try #require(
            header.tensors.first { $0.name == "blk.0.ffn_gate_up_exps.weight" })
        #expect(gateUp.absoluteOffset == 1_217_729_504)
        #expect(gateUp.dimensions == [2_304, 2_304, 128])
        #expect(gateUp.ggmlType == 40)  // NVFP4
    }
}
