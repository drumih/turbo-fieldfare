import Metal

/// A QSA chunk's key selection in both forms.
///
/// `mask` is one byte per visible key per query: the form the attention kernel
/// originally consumed, and still what the activation dump compares. `indices`
/// is the same selection compacted to ascending key ids with `counts[query]`
/// valid entries -- what the kernel loops over now.
///
/// The two are kept together deliberately. The mask is how the selection is
/// built and how it is checked; the indices are how it is executed. Deriving
/// one from the other at the point of use is what made attention O(context)
/// when the selection is O(budget).
struct QSASelection {
    let mask: MTLBuffer
    let maskStride: Int
    let indices: MTLBuffer
    let indexStride: Int
    let counts: MTLBuffer
}

/// One decode step's key selection, in the forms the two attention paths need.
///
/// `mask` is the historical form the contiguous full-attention kernel reads.
/// `indices` is the same selection compacted to ascending logical keys with
/// `count` valid entries; the host-KV window path needs that list on the host
/// to decide which rows to copy in, and the kernel iterates it directly.
struct DecodeKeySelection {
    let mask: MTLBuffer?
    let indices: MTLBuffer?
    let count: Int
}
