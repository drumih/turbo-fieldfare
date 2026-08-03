# Gemma 4 image input

The Mac chat app supports local PNG, JPEG, HEIC, and WebP images. Attach an
image with the photo button, enter a question, and send it normally. Images are
stored as managed local files and persist with their chat; image bytes are not
embedded in chat-history JSON or decode-service messages.

Image support is opt-in because the existing `.gturbo` package intentionally
contains only the language model. Install the optional vision tensors into an
existing model directory:

```bash
swift run -c release TurboFieldfareRepack \
  --install-vision \
  --output scratch/gemma4.gturbo
```

See [Optional vision sidecar](VISION_SIDECAR.md) for resume, discard, and
verification commands.

## Runtime behavior

Gemma 4's native image pipeline is used end to end: bounded ImageIO decoding,
aspect-preserving resize and patchification, the 27-layer vision encoder,
3×3 pooling, projection into the language-model hidden size, and replacement
of the prompt's image-placeholder embeddings during prefill.

The language model remains the only resident model process. Text-only chats do
not open the sidecar or allocate vision activations. For an image request,
vision tensors are streamed one layer at a time from `vision/weights.bin` into
a reusable buffer; tower weights and intermediate activations are released
before language-model prefill. Only projected FP16 features may be retained in
a bounded LRU cache for multi-turn follow-ups. A maximum-size projection is
`280 × 2,816 × 2` bytes, about 1.5 MiB per image; the app permits one image per
user turn and at most eight images in a request.

Input is bounded to 25 MiB and 64 megapixels per image. Managed attachment
references are relative paths, regular files are required, and the stored
SHA-256 and dimensions are checked before projected features are used.

## Current limitations

- A text question is required; image-only turns are not accepted.
- Image upload is available in the Mac app, not the CLI or OpenAI-compatible
  server API.
- Tool-calling conversations do not accept image inputs.
- Multimodal prompts are re-prefilled on each turn. Cached image projections
  avoid rerunning the vision tower, but language-model KV continuation is not
  yet enabled for multimodal chats.
- Real-model visual-quality and peak-memory measurements still require a
  completed vision sidecar on a supported macOS 26 / Apple Silicon machine.

The pinned checkpoint configuration specifies a 1,152-wide vision tower and a
2,816-wide language projection with up to 280 soft tokens per image. See the
[pinned source configuration](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/blob/0d77464eeb233a2da68ebf9d7dc4edaac7db956d/config.json)
and Google's [Gemma image guide](https://ai.google.dev/gemma/docs/capabilities/vision/image).
