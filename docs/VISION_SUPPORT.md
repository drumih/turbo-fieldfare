# Vision support assessment

## Finding

Gemma 4 26B-A4B supports image-to-text inference, but this checkout is a
text-only runtime by design. Its installer excludes every `vision_tower.*` and
`embed_vision.*` tensor, so the installed `.gturbo` package cannot turn pixels
into the image embeddings expected by the language model.

The pinned source configuration describes a 27-layer, 1,152-wide vision tower,
followed by a projection into the language model's 2,816-wide hidden space. An
image is represented by up to 280 soft tokens at the default budget. See the
[pinned source configuration](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/blob/0d77464eeb233a2da68ebf9d7dc4edaac7db956d/config.json)
and Google's [Gemma image guide](https://ai.google.dev/gemma/docs/capabilities/vision/image).

## Memory answer

Image support cannot have *zero* memory impact while an image is being
processed: the vision weights, vision activations, and projected image tokens
must exist at some point. It can, however, preserve the current text-only
steady-state footprint:

| Mode | Resident effect |
| --- | --- |
| Text-only chat | No vision weights or image buffers are allocated. |
| Image encoding | An optional vision sidecar and bounded image workspace are loaded temporarily. |
| Language generation after encoding | The sidecar is released; only the projected soft-token embeddings remain for the prompt. At 280 × 2,816 FP16 values, that is about 1.5 MiB per image, before normal prompt/KV-cache costs. |

The image token budget also consumes context capacity. Higher image resolution
increases visual tokens and prefill time; it should be exposed as a deliberate
quality-versus-memory control rather than an invisible default.

## Required implementation

1. Add an opt-in vision payload to the installer rather than changing the
   existing text-only install. The payload needs its own manifest and checksum
   verification, so existing `.gturbo` installs remain valid.
2. Add a bounded image decoder/preprocessor for local files and enforce file,
   pixel-count, image-count, and visual-token limits before allocating Metal
   buffers.
3. Implement the Gemma 4 vision tower in Metal: patch embedding, position
   handling, 27 encoder blocks, pooling, and the `embed_vision` projector.
4. Extend prefill to inject projected embeddings in place of the model's image
   placeholder tokens. This needs a mixed token/embedding input path and
   multimodal prompt accounting; a text-token-only prefill loop is not enough.
5. Keep the sidecar load/encode/release lifecycle inside the existing single
   decode service. It must never load a second language model and must release
   the vision buffers on cancellation and terminal events.
6. Add reference-image tests and memory measurements on a machine that meets
   the project's real-model-run checks. The feature should not claim support
   until captioning and multi-turn image follow-ups match a reference runtime.

## Recommended first release

Support local PNG, JPEG, HEIC, and WebP images; one image per user turn; a
default 280-token visual budget; and automatic sidecar unload immediately after
projecting features. Keep images attached to the in-memory chat so follow-up
questions can refer to them, but store only the source URL and compact feature
cache policy—not full-resolution image copies—in a future persistent-history
feature.

Video and audio should remain out of scope for this first vision release.
