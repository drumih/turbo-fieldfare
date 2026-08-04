import Foundation
import TurboFieldfareFormat

/// Little-endian binary encoders for the resident `IndexHeader` and 72-byte
/// `IndexEntry` records. Used by the resident writer and the
/// matching loader-side parsers; kept in one place so the on-disk layout
/// changes only here.
enum GTurboBinary {

    static let indexHeaderBytes = GTurboFormatV1.residentHeaderBytes
    static let indexEntryBytes = GTurboFormatV1.residentEntryBytes

    /// Write `IndexHeader { indexSize, residentSize, entryCount }` (24 bytes, LE).
    static func writeIndexHeader(into buf: UnsafeMutableRawPointer,
                                        indexSize: UInt64,
                                        residentSize: UInt64,
                                        entryCount: UInt64) {
        GTurboResidentIndexCodec.writeHeader(
            into: buf,
            header: GTurboResidentIndexHeaderV1(
                indexSize: indexSize,
                residentSize: residentSize,
                entryCount: entryCount))
    }

    /// Write one `IndexEntry` (72 bytes, LE) at `dst`. See gturbo-format.md.
    static func writeIndexEntry(into dst: UnsafeMutableRawPointer,
                                       entry: ResidentEntry,
                                       nameOffset: UInt32) {
        GTurboResidentIndexCodec.writeEntry(
            into: dst,
            entry: GTurboResidentIndexEntryV1(
                name: entry.name,
                dtype: entry.dtype,
                fileOffset: entry.fileOffset,
                sizeBytes: entry.sizeBytes,
                shape: entry.logicalShape4,
                scaleOffset: entry.scaleOffset,
                scaleSize: entry.scaleSize,
                biasOffset: entry.biasOffset,
                biasSize: entry.biasSize),
            nameOffset: nameOffset)
    }
}
