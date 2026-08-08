These Base64 files contain the exact v2 bytes emitted by the
GTurboManifestCodecV2 / GTurboPackedExpertsLayoutCodecV2 writers at commit
3249be4. The compatibility suite decodes these frozen bytes and requires the
shared writers to reproduce them exactly. v2 reuses the v1 resident-index wire
format verbatim, so there is no resident-index.bin fixture here.
