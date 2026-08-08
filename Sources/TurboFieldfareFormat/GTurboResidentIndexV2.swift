import Foundation

// v2 reuses the v1 resident-index wire format verbatim: 24-byte header,
// 72-byte entries, the same dtype byte set { u32, bf16, fp16, fp32 }, and the
// same scale/bias companion ranges. Every v2 resident trunk tensor is
// affine4-g64 / affine8-g64 (dtype u32 payload + BF16 scale/bias ranges),
// BF16, or FP32, all of which the v1 entry already expresses. MXFP4 never
// appears in model_weights.bin — routed experts live in
// packed_experts/layer_XX.bin and are addressed exclusively through
// layout.json — so no v2 dtype byte is needed and the v1 byte layout stays
// untouched. These aliases exist so v2 call sites read as v2 without
// duplicating the codec.
package typealias GTurboResidentIndexHeaderV2 = GTurboResidentIndexHeaderV1
package typealias GTurboResidentIndexEntryV2 = GTurboResidentIndexEntryV1
package typealias GTurboResidentIndexCodecV2 = GTurboResidentIndexCodec
