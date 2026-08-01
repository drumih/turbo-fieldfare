# Model Picker and Model Switcher

Date: 2026-08-02
Status: Approved design, not yet implemented

## Summary

Replace the single pinned model with a catalog the user can browse, install,
switch between, and extend with custom Hugging Face repositories. Scope is
limited to Gemma 4 26B-A4B and finetunes of it. No new Metal kernels.

## Motivation

The app installs exactly one model. `SupportedModelSource` pins one repository,
one revision, and one index hash, and `RepackModelInstallerClient` passes
`requireKnownSource: true`. A user who wants a finetune, even one with an
identical architecture, has no path.

Three gates enforce the single-model assumption:

1. `SourceFingerprint.knownFingerprints` is an allowlist with one entry.
   `requireKnownSource: true` rejects any unlisted repository before repacking.
2. `ManifestReader.validateArch` compares field-by-field against the single
   constant `ArchConfig.gemma4_26B_A4B`.
3. `ManifestReader.validateQuant` accepts only affine quantization with bf16
   scales and biases at group size 64.

Gate 1 is the only one that blocks a Gemma finetune. Finetuning does not change
architecture, so a finetune of Gemma 4 26B-A4B produces a byte-identical
`ArchConfig` and passes gates 2 and 3 unchanged.

## Scope

In scope:

- Model catalog with curated and user-added entries
- Per-model install, resume, cancel, and delete
- Switching the loaded model without restarting the decode service
- Custom Hugging Face repositories with a trust-on-first-use policy
- Per-architecture context and memory budgeting, ceiling raised to 4 GB
- Conversations that survive a model switch

Out of scope:

- Any architecture other than Gemma 4 26B-A4B (notably Qwen, which needs Gated
  DeltaNet kernels that do not exist in this build)
- New Metal kernels of any kind
- A remotely-hosted catalog

## Design decisions

### Architecture validation stays exact-match, over a list

`ManifestReader` validates against a list of known `ArchConfig` values rather
than a single constant. The list has one entry today. Rejection names the
mismatched field.

This is deliberately less than a capability-predicate registry. Every supported
model shares one `ArchConfig`, so predicates would buy nothing now. The list
keeps the extension point open for the eventual Qwen3.6 work without paying for
it today.

### Trust is two-tier, with trust-on-first-use for custom entries

Two mechanisms exist and are separable:

- `verified-install.json` receipts are written by the repacker after repacking
  and validated at load. This is a local integrity check that files have not
  been corrupted or swapped since install. It is generated per install, works
  for any model, and stays on unconditionally for both tiers.
- `SourceFingerprint` / `requireKnownSource` is a supply-chain check asserting
  that this exact upload was validated by the project. Only this relaxes.

Curated entries keep the pinned fingerprint and show a "Verified" badge. Custom
entries install after explicit one-time consent, show "Unverified", and record
the observed index SHA-256 on first install. A later change to that repository
is detected and blocked pending re-consent.

If a user adds a custom entry whose repository ID matches a curated one, the
curated entry wins and the custom entry is not created. Otherwise a user could
downgrade a pinned model to the unverified tier by re-adding it.

### The catalog is compiled in, not fetched

A remote catalog could only list models whose kernels already ship in the
binary, so it could never introduce a new architecture without a new build. It
would add a trust problem and an offline-fallback problem for almost no gain.

### Conversations move to a global store

`ConversationFileStore` currently stores conversations inside the model install
directory. That conflicts with switching: it swaps the chat list on every
switch, prevents a conversation from spanning two models, and destroys chat
history when a model is deleted.

Conversations move to a single global store. `ChatTurn` gains an optional
`modelID` recording which model produced the turn. History replays as structured
`[{role, content}]` messages, re-templated by whichever model is loaded, so
switching mid-conversation works without flattening to text.

Migration runs once on first launch after upgrade: every per-model
`conversations.json` found under an install directory is merged into the global
store with its turns tagged, then the source file is renamed to
`conversations.migrated.json`. The rename is what makes the migration
idempotent; a second launch finds nothing to migrate.

This modifies work in flight on `feat/mac-app-chat-history`.

## Components

| Component | Responsibility | Status |
|---|---|---|
| `ModelCatalogEntry` | Identity, display name, repo, revision, trust tier, byte estimates | Generalizes `AppModelInstallDescriptor` |
| `ModelCatalog` | Curated entries plus persisted custom entries | New |
| `ModelTrustPolicy` | Fingerprint pinning for curated, TOFU for custom | New |
| `ModelInstallCoordinator` | Per-model install, resume, cancel, delete; serialized queue | New, wraps `RepackModelInstallerClient` |
| `ModelSwitcher` | Unload then load over the existing decode protocol | New, in `AppModel` |
| `ModelPickerView` | Catalog list with per-entry state; custom-repo add sheet | New |
| `AppModelLocation` | Per-model install directories keyed by slug | Modify |
| `AppContextLengthOption` | Context options computed from `ArchConfig` and resident bytes | Modify |
| `ManifestReader` | Validate against a list of known architectures | Modify |
| `SourceFingerprint` | Expose curated fingerprints; allow unlisted with consent | Modify |
| `ConversationFileStore` | Global store plus migration from per-model stores | Modify |
| `ChatTurn` | Optional `modelID` | Modify |

Runtime changes are confined to `ManifestReader` and `SourceFingerprint`.
Everything else is in `TurboFieldfareApp`.

## Storage layout

```
~/Library/Application Support/TurboFieldfare/
  models/<slug>/model.gturbo/     manifest.json, verified-install.json, packed_experts/
  catalog.json                    custom entries and recorded TOFU fingerprints
  conversations.json              global conversation store
```

`<slug>` derives deterministically from the repository ID with `/` replaced by
`--`, so re-adding a repository finds its existing install. The dev-build
convenience in `AppModelLocation.resolve`, which installs into the package root,
is preserved and re-pointed at `scratch/models/<slug>/`.

Slug derivation must reject path escapes. The repository ID is user input that
becomes a directory name, so `../`, absolute paths, and any component that
normalizes outside the models directory are rejected before use.

## Install flow

1. User picks a curated entry, or pastes a repository ID into the custom sheet.
2. Metadata fetch resolves the revision and reads the source `config.json`.
3. **Architecture pre-flight.** The source architecture is checked against the
   known list before any payload transfer. An unsupported model fails in
   seconds, naming the architecture, with nothing downloaded. This is the
   expected outcome for Qwen repositories and is the most load-bearing UX
   detail in the feature.
4. For custom entries, a consent sheet shows repository, revision, index
   SHA-256, and download size. On confirm the fingerprint is recorded.
5. Disk readiness is checked using per-entry `requiredFreeBytes`.
6. Install proceeds through the existing `AppModelInstallState` machine,
   including `recoverable` for resume.
7. The repacker writes `verified-install.json` as it does today.

Install state becomes a map keyed by model ID. Installs are serialized; only
one runs at a time and the rest queue. Concurrent multi-gigabyte repacks would
thrash the SSD, and `requiredFreeBytes` assumes exclusive use of the reserve.

On reinstall of a custom entry, an observed index SHA differing from the
recorded one blocks the install and surfaces both values.

## Switch flow

1. Refuse to switch while generating; offer to cancel first.
2. Cancel any generation, then `unloadModel()`.
3. Invalidate the prompt cache explicitly. `AppPromptCacheDomain` already keys
   on `modelID` and `sourceSnapshotHash`, so a stale entry cannot match, but
   dropping it on unload is cheaper than carrying dead KV.
4. Point at the new directory and `loadModel()`.
5. On failure, surface the error and offer to restore the previous model.

The decode protocol already carries `load(DecodeLoadRequest)` with `modelPath`
and `unload`, so no process restart is needed.

The last successfully loaded model is persisted, so a relaunch never boots into
a broken selection.

## Context and memory budget

`AppContextLengthOption` is currently a fixed enum with hardcoded labels and
`fp16KVBytes` hardcoding `ArchConfig.gemma4_26B_A4B`. It becomes computed:
options and their KV costs derive from the loaded `ArchConfig` plus measured
resident weight bytes.

Resident weight bytes are read from the manifest at load time, not sampled at
runtime, so the option list is correct before the first token is generated.

The ceiling is:

```
ceiling = min(4 GB, max(1.5 GB, 0.25 × installedRAM))
```

which yields 2 GB on an 8 GB Mac, 4 GB on 16 GB and above. Options above the
ceiling render disabled with the reason shown rather than disappearing.

Raising the ceiling to 4 GB is not free on small machines. Resident state
competes with the page cache that makes expert streaming fast, so the adaptive
share matters more than the absolute cap.

For Gemma this unlocks context beyond today's 64K limit on larger Macs; the
model supports 256K. The same code path serves a future architecture without
rework.

## Error handling

| Failure | Detected at | Behavior |
|---|---|---|
| Unsupported architecture | pre-flight `config.json` | Fails in seconds naming the architecture; nothing downloaded |
| Unsupported quantization | `validateQuant` | Reports the offending group size or scheme |
| Repository missing or gated | metadata fetch | "Not found or requires a token", with a token field |
| Source changed since added | TOFU SHA compare | Blocked; shows recorded and observed SHA; requires re-consent |
| Insufficient disk | `AppModelInstallRequirement` | Existing `insufficientSpace` path with per-model estimates |
| Interrupted download | existing `recoverable` | Resume |
| Receipt invalid at load | `trustedReceiptInvalid` | Offers repair by reinstall |
| Load fails after switch | `AppModelLoadState.failed` | Restores previous model, keeps error visible |
| Delete while loaded | guard | Refused; offers to unload first |
| Corrupt `catalog.json` | catalog load | Backs up the bad file, falls back to curated-only, never crashes |

## Testing

Swift Testing, matching the existing `@Suite` / `@Test` style.

- `ModelCatalogTests` — slug derivation is stable and collision-free; custom
  entries round-trip; corrupt catalog degrades to curated-only
- `ModelTrustPolicyTests` — curated requires a pinned fingerprint; custom
  records SHA on first install; changed SHA blocks; receipts enforced in both
  tiers
- `AppModelLocationTests` (extend) — per-slug paths; dev package-root branch
  preserved; path escape via `../` or absolute repository IDs rejected
- `ArchPreflightTests` — a real Qwen `config.json` fixture is rejected naming
  the architecture; a Gemma finetune config is accepted
- `ModelSwitcherTests` — switching blocked mid-generation; prompt cache
  invalidated on unload; failed load restores the previous model; last-good
  selection persists
- `ContextBudgetTests` — options computed from `ArchConfig`; ceiling adapts to
  installed RAM; over-ceiling options disabled rather than hidden
- `ConversationMigrationTests` — per-model stores merge into the global store;
  turns tagged with the correct `modelID`; re-running is idempotent
- `AppModelInstallTests` (extend) — per-model state map; serialized queue

## Consequences

The curated catalog ships with one entry. "Custom model" in practice means a
Gemma 4 26B-A4B finetune. Users will paste Qwen repositories and receive a clear
rejection; the architecture pre-flight exists to make that rejection fast and
legible rather than a wasted multi-gigabyte download.

What this delivers beyond finetune support is the scaffolding every future
architecture plugs into: per-model storage, install queueing, trust tiers,
switching, and per-architecture budgeting. Adding Qwen3.6-35B-A3B later becomes
kernels plus a catalog entry, not an app rewrite.

## Follow-on work, not in this spec

Supporting Qwen3.6-35B-A3B requires, in rough order:

1. Gated DeltaNet kernels — chunked parallel scan for prefill, recurrent state
   update for decode, across 30 of 40 layers. The dominant piece of work.
2. Constant-state cache manager — `KVCacheManager` assumes every layer holds a
   growing KV ring.
3. Gated Attention layers at 16 Q heads / 2 KV heads / head dim 256. Mostly
   reuse of existing specialized pipelines.
4. MoE and repack changes — 256 experts, top-8 plus shared, expert intermediate
   512, and stripping vision towers from the `mlx-vlm` checkpoint.
5. Chat template and thinking-mode toggle — `enable_thinking`, `<think>` block
   handling in the UI.
6. Multi-token-prediction speculative decoding. Optional, and well matched to
   an I/O-bound decode path.
