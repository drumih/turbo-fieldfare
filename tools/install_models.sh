#!/usr/bin/env bash
# Install any supported model at any supported width.
#
#   tools/install_models.sh                 # what is installed, what is missing
#   tools/install_models.sh ornith15-8bit   # install one
#   tools/install_models.sh --all-4bit      # every 4-bit model
#   tools/install_models.sh --all-8bit      # every 8-bit model
#
# Most models install straight from a pinned Hugging Face release through
# TinyTitanRepack, which streams and verifies in one pass. Qwen3.8-Flash-Next is
# the exception and is documented below, because the difference matters when
# choosing what to trust.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Same override as the launcher: a checkout builds into `.build/release`, an
# install from the release tarball keeps its binaries in `~/.tinytitan/bin`.
BIN="${TINYTITAN_BIN_DIR:-$ROOT/.build/release}/TinyTitanRepack"
MODELS="${TINYTITAN_MODELS_DIR:-$ROOT/models}"
# Where a download and its conversion are staged, and where the converted
# snapshot is kept until the install exists. It must be absolute and independent
# of the caller's working directory: every staging path below used to be a bare
# `.build/...`, so a factory-new install launched from the user's home staged
# tens to hundreds of GB in `~/.build` — outside the install root, and outside
# the two directories the installer says removing uninstalls everything. Found
# and fixed 2026-09-24; `TINYTITAN_WORK_DIR` moves it to another volume.
WORK="${TINYTITAN_WORK_DIR:-$ROOT/.build}"

# The install menu's list of models, labels and sizes comes from the shared
# catalogue so this file and the installer cannot disagree about what exists.
# shellcheck source=tools/tinytitan_models.sh
source "$ROOT/tools/tinytitan_models.sh"

# The converters need a Python with numpy/ml_dtypes/safetensors at 3.10 or
# newer. Resolved by capability rather than by name: `python3` is 3.9 on a
# stock macOS and lacks the packages, while a pinned `python3.13` fails on a
# machine whose stack lives under another version. See tools/lib/python.sh.
# shellcheck source=tools/lib/python.sh
source "$ROOT/tools/lib/python.sh"

# --- disk space and staging hygiene -----------------------------------------
#
# Two rules make an install predictable on a machine that is somebody's only
# disk. A download is refused before it starts if it cannot finish, and staging
# is removed as soon as it can no longer save work — a converted snapshot exists
# so the *second* width of a model costs no fetch, so it is kept exactly until
# that width is installed, and draft-head staging is never reusable at all.

# Free GB on the volume holding <path>, or empty when df cannot say.
free_gb() {
  local path="$1" available
  mkdir -p "$path" 2>/dev/null || true
  available="$(df -g "$path" 2>/dev/null | awk 'NR==2 {print $4}')"
  [[ "$available" =~ ^[0-9]+$ ]] && printf '%s' "$available"
}

# The catalogue's installed size for a row, in GB. May be fractional.
choice_gb() {
  local want="$1" row key _label _bits gb
  for row in "${TINYTITAN_MODEL_CHOICES[@]+"${TINYTITAN_MODEL_CHOICES[@]}"}"; do
    IFS='|' read -r key _label _bits gb _ <<<"$row"
    [[ "$key" == "$want" ]] && { printf '%s' "$gb"; return 0; }
  done
}

# Refuse when `path` cannot hold `need` GB. Empty `need` means "do not check",
# and an unreadable df means the same: this never invents a limit it cannot
# justify.
require_gb() {
  local label="$1" path="$2" need="$3" available
  [[ -n "$need" ]] || return 0
  available="$(free_gb "$path")"
  [[ -n "$available" ]] || return 0
  if awk -v a="$available" -v n="$need" 'BEGIN { exit !(a < n) }'; then
    printf '  %s holds %s GB but about %s GB is needed there.\n' "$label" "$available" "$need" >&2
    return 1
  fi
}

# The paths under $WORK that this tool owns. Deliberately a narrow pattern rather
# than "everything in $WORK": a checkout keeps its release build, benchmark
# results and scratch checkouts there too, and measuring those as "staging" made
# `clean` claim 12 GB it had nothing to do with.
staging_paths() {
  local dir
  for dir in "$WORK"/*-affine* "$WORK"/*-shards "$WORK"/ornith-mtp-src; do
    [[ -e "$dir" ]] && printf '%s\n' "$dir"
  done
}

# A KB count as a person reads it: MB below a gigabyte, GB above. Storage here
# spans three orders of magnitude, and "0 GB" or "1 GB" for a few megabytes is a
# number that teaches the reader to distrust the rest.
human_kb() {
  local kb="$1"
  if (( kb < 1048576 )); then
    printf '%s MB' "$(( (kb + 1023) / 1024 ))"
  else
    printf '%s GB' "$(( (kb + 1048575) / 1048576 ))"
  fi
}

# Total KB of staging, printed as a number only (empty when there is none), so
# callers can both compare and display it.
staging_kb() {
  local dir kb total=0
  while IFS= read -r dir; do
    kb="$(du -sk "$dir" 2>/dev/null | awk 'NR==1 {print $1}')"
    [[ "$kb" =~ ^[0-9]+$ ]] && total=$(( total + kb ))
  done < <(staging_paths)
  (( total > 0 )) || return 0
  printf '%s' "$total"
}

# The staging size as text ("12 GB", "850 MB"), or empty when there is none.
staging_size() {
  local kb
  kb="$(staging_kb)"
  [[ -n "$kb" ]] && human_kb "$kb"
}

# The staging paths, comma-separated, for a hint that says what is kept.
staging_names() {
  local dir names=()
  while IFS= read -r dir; do names+=("${dir#"$WORK"/}"); done < <(staging_paths)
  (( ${#names[@]} > 0 )) && printf '%s' "$(IFS=', '; printf '%s' "${names[*]+"${names[*]}"}")"
}

# The install directory of the row that shares this row's download, if any.
# Paired MoE widths share one conversion, the two Qwen3.8 widths share their
# shard directory, and the dense Qwen 3.5 sizes share theirs. A draft head
# shares nothing. This decides whether staging is still worth keeping.
row_sibling_dir() {
  local name="$1" row _n dir _w source preset sibling other base row2 n2 d2
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r _n dir _w source preset sibling <<<"$row"
    [[ "$_n" == "$name" ]] || continue
    case "$source" in
      convert_qwen35moe) other="$sibling" ;;
      convert|convert_qwen35)
        base="${name%-8bit}"
        if [[ "$name" == *-8bit ]]; then other="$base"; else other="${base}-8bit"; fi ;;
      *) return 0 ;;
    esac
    [[ -n "$other" ]] || return 0
    for row2 in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
      IFS='|' read -r n2 d2 _ _ _ _ <<<"$row2"
      [[ "$n2" == "$other" ]] && { printf '%s' "$d2"; return 0; }
    done
    return 0
  done
}

# The staging one catalogue row owns, as `reuse:<path>` (a sibling width can
# still use it) or `drop:<path>` (nobody can).
row_staging_dirs() {
  local name="$1" row _n _dir _w source preset
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r _n _dir _w source preset _ <<<"$row"
    [[ "$_n" == "$name" ]] || continue
    case "$source" in
      convert_qwen36_mtp)  printf 'drop:%s\n' "$WORK/qwen36-mtp-affine" ;;
      prepare_ornith_mtp)  printf 'drop:%s\n' "$WORK/ornith-mtp-affine" "$WORK/ornith-mtp-src" ;;
      convert)             printf 'drop:%s\n' "$WORK/qwen38-affine-${_w}bit"
                           printf 'reuse:%s\n' "$WORK/qwen38-shards" ;;
      # Both MoE widths are quantized into the same pair of snapshots by one
      # run, so every one of these is what the *other* width would consume.
      convert_qwen35moe)   printf 'reuse:%s\n' "$WORK/${preset}-affine-4bit" "$WORK/${preset}-affine-8bit" \
                                              "$WORK/${preset}-affine" "$WORK/${preset}-shards" ;;
      # The bf16-dense build is a third conversion of the same checkpoint, not
      # a width of either snapshot above: its dense blocks are copied through
      # at bf16, so neither `<preset>-affine-*` can stand in for it. Nothing
      # else can reuse it either, hence `drop` for the snapshot and the shards.
      convert_qwen35moe_dense16)
        printf 'drop:%s\n' "$WORK/${preset}-bf16dense-affine" "$WORK/${preset}-bf16dense-shards" ;;
      # A 122B snapshot is 69.5 GB (4-bit) or 129.8 GB (8-bit). Nobody can reuse
      # it -- the sibling width is a separate fetch by design -- so it goes as
      # soon as its own install exists. The shard directory holds only partial
      # shards by then (the converter deletes each converted one) and is dropped
      # on the same terms.
      convert_qwen35moe_solo) printf 'drop:%s\n' "$WORK/${preset}-affine-${_w}bit" "$WORK/${preset}-shards" ;;
      convert_qwen35)      printf 'drop:%s\n' "$WORK/${_n%-8bit}-affine-${_w}bit"
                           printf 'reuse:%s\n' "$WORK/${_n%-8bit}-shards" ;;
    esac
    return 0
  done
}

# The install directory one catalogue row would fill.
row_install_dir() {
  local name="$1" row _n dir
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r _n dir _ <<<"$row"
    [[ "$_n" == "$name" ]] && { printf '%s' "$dir"; return 0; }
  done
}

# Remove staging that cannot save a future download, and say what happened.
#
# `mode` "keep" is the normal install path: staging is left alone while any row
# that would consume it is still missing, and the hint says what is kept and how
# to reclaim it. `mode` "reap" is the explicit `clean` command: the same rules,
# silently, for every row. Nothing is removed on a guess — a shared snapshot goes
# only when every width that reuses it is installed, and a per-width one goes
# once its own width is.
cleanup_staging() {
  local name="$1" mode="${2:-keep}" dir kind freed=0 removed=()
  local -a victims=() kept=()
  local sibling own_dir
  sibling="$(row_sibling_dir "$name")"
  own_dir="$(row_install_dir "$name")"
  while IFS=: read -r kind dir; do
    [[ -n "$dir" && -e "$dir" ]] || continue
    if [[ "$kind" == reuse ]]; then
      # Kept while either consumer is still uninstalled.
      if [[ -n "$own_dir" && ! -d "$MODELS/$own_dir" ]] \
         || { [[ -n "$sibling" ]] && [[ ! -d "$MODELS/$sibling" ]]; }; then
        kept+=("$dir")
        continue
      fi
    elif [[ -n "$own_dir" && ! -d "$MODELS/$own_dir" ]]; then
      # Per-width staging (a snapshot, a draft head's source): worth keeping
      # until this row itself is installed, then it is only wasted disk.
      kept+=("$dir")
      continue
    fi
    victims+=("$dir")
  done < <(row_staging_dirs "$name")

  # Remove each one, counting what it held before it went. The list needs no
  # de-duplication: `row_staging_dirs` names each path once per row.
  for dir in "${victims[@]+"${victims[@]}"}"; do
    local before
    before="$(du -sk "$dir" 2>/dev/null | awk 'NR==1 {print $1}')"
    rm -rf "$dir"
    freed=$(( freed + ${before:-0} ))
    removed+=("${dir#"$WORK"/}")
  done
  if (( ${#removed[@]} > 0 )); then
    printf '  staging cleaned (%s freed): %s\n' \
      "$(human_kb "$freed")" "$(IFS=', '; printf '%s' "${removed[*]+"${removed[*]}"}")"
  fi

  # Only the install path explains itself; `clean` has its own summary.
  if [[ "$mode" == "keep" && ${#kept[@]} -gt 0 ]]; then
    local kept_kb=0 kept_paths=()
    for dir in "${kept[@]+"${kept[@]}"}"; do
      kept_paths+=("${dir#"$WORK"/}")
      local size
      size="$(du -sk "$dir" 2>/dev/null | awk 'NR==1 {print $1}')"
      kept_kb=$(( kept_kb + ${size:-0} ))
    done
    printf '  kept %s of staging (%s) so the other width needs no download;\n' \
      "$(human_kb "$kept_kb")" "$(IFS=', '; printf '%s' "${kept_paths[*]+"${kept_paths[*]}"}")"
    printf '  reclaim it when you have the widths you want:  %s clean\n' "$0"
  fi
  return 0
}

# The explicit `clean`: the rules above for every model, plus empty shard
# directories, then a summary of what is still staged and why. Safe at any time:
# it only touches this script's own staging.
cmd_clean() {
  echo "Cleaning install staging under $WORK"
  local row name dir
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name _ <<<"$row"
    cleanup_staging "$name" reap
  done
  for dir in "$WORK"/*-shards; do
    [[ -d "$dir" && -z "$(ls -A "$dir" 2>/dev/null)" ]] && rmdir "$dir" 2>/dev/null
  done
  local staged
  staged="$(staging_size)"
  if [[ -n "$staged" ]]; then
    echo "  still staged: $staged — $(staging_names)"
    echo "  kept for a width you have not installed yet; it goes automatically once"
    echo "  both widths of that model are installed"
  else
    echo "  nothing left staged"
  fi
}

# name|install directory|width|source[|preset[|both-widths-directory]]
#
# The preset field names the converter preset; it appears on every row a
# converter branch reads (`convert_qwen35moe`, `convert_qwen35moe_dense16`,
# `convert_qwen35moe_solo`). The sibling field appears only on the paired
# `convert_qwen35moe` rows: their converter can write *both* widths from one
# 70 GB download, so a row names the sibling width it pairs with. That is what
# makes `tools/install_models.sh katcoder both` cost one download instead of
# two, and it also lets a second width reuse a snapshot a previous run already
# converted. The bf16-dense row is not paired: it is a third conversion, so it
# has a preset and no sibling.
CATALOGUE=(
  "ornith15|ornith-1.5_35B_A3B_4Bit|4|convert_qwen35moe|ornith15|ornith15-8bit"
  "ornith15-8bit|ornith-1.5_35B_A3B_8Bit|8|convert_qwen35moe|ornith15|ornith15"
  # One width only, and a different dense policy: see convert_qwen35moe_dense16.
  "ornith15-bf16dense|ornith-1.5_35B_A3B-bf16dense_4Bit|4|convert_qwen35moe_dense16|ornith15"
  "ornith15-mtp|ornith-1.5_35B_A3B_MTP_4Bit|4|prepare_ornith_mtp"
  # Ornith-1.5-122B. Deliberately *not* a paired `convert_qwen35moe` row: one
  # `--bits 4 8` pass would stage 69.5 GB and 129.8 GB of snapshots at once,
  # and then a 120 GB install beside them, which no 24 GB/1 TB Mac can hold.
  # The solo source converts the requested width only and drops its snapshot
  # once the install exists, so each width costs one fetch (the source shards
  # are streamed and deleted, never kept). An interrupted width resumes its
  # own partial snapshot; the other width starts its own fetch.
  "ornith122|ornith-1.5_122B_4Bit|4|convert_qwen35moe_solo|ornith122"
  "ornith122-8bit|ornith-1.5_122B_8Bit|8|convert_qwen35moe_solo|ornith122"
  "qwen36|qwen3.6_35B_A3B_4Bit|4|convert_qwen35moe|qwen36|qwen36-8bit"
  "qwen36-8bit|qwen3.6_35B_A3B_8Bit|8|convert_qwen35moe|qwen36|qwen36"
  "qwen36-mtp|qwen3.6_35B_A3B_MTP_4Bit|4|convert_qwen36_mtp"
  # Qwen3.8-Flash-Next ships two install types and the quant recipe is
  # hardcoded per type, not a --bits dial: the "4-bit" type routes its experts
  # at 4 bits and keeps every always-resident tensor (attention, GDN,
  # hyper-connection, PLE, indexer, shared expert, embedding, LM head, router)
  # at 8 bits, the n-gram table at its fp16 -- the 4/8/16 recipe. The
  # "8-bit" type is uniform 8-bit with the same fp16 table. The directory keeps
  # the canonical `_4Bit` spelling; the recipe lives in the manifest, not the
  # name. The MTP draft head is no longer an install type.
  "qwen38flash|qwen3.8-flash-next_125B_A6B_4Bit|4|convert"
  "qwen38flash-8bit|qwen3.8-flash-next_125B_A6B_8Bit|8|convert"
  "katcoder|kat-coder-v2.5_35B_A3B_4Bit|4|convert_qwen35moe|katcoder|katcoder-8bit"
  "katcoder-8bit|kat-coder-v2.5_35B_A3B_8Bit|8|convert_qwen35moe|katcoder|katcoder"
  "agentworld|qwen-agentworld_35B_A3B_4Bit|4|convert_qwen35moe|agentworld|agentworld-8bit"
  "agentworld-8bit|qwen-agentworld_35B_A3B_8Bit|8|convert_qwen35moe|agentworld|agentworld"
  # The dense Qwen 3.5 models. Small enough to run on the CPU, and the only
  # installs that do: the 2B beside a big GPU model, the 9B on its own.
  "qwen35-2b|qwen3.5_2B_4Bit|4|convert_qwen35"
  "qwen35-2b-8bit|qwen3.5_2B_8Bit|8|convert_qwen35"
  "qwen35-4b|qwen3.5_4B_4Bit|4|convert_qwen35"
  "qwen35-4b-8bit|qwen3.5_4B_8Bit|8|convert_qwen35"
  "qwen35-9b|qwen3.5_9B_4Bit|4|convert_qwen35"
  "qwen35-9b-8bit|qwen3.5_9B_8Bit|8|convert_qwen35"
)

usage() {
  cat <<'USAGE'
Commands

  tools/install_models.sh                 what is installed, and what staging is kept
  tools/install_models.sh --choose        the install menu (the one real choice)
  tools/install_models.sh <name>          download, convert and install one width
  tools/install_models.sh <name> both     4-bit and 8-bit from one download
  tools/install_models.sh --all-4bit      every 4-bit model
  tools/install_models.sh --all-8bit      every 8-bit model
  tools/install_models.sh clean           drop staging that cannot be reused

Where things go

  Models install to $TINYTITAN_MODELS_DIR (the installer sets
  ~/.tinytitan/models; a checkout uses <repo>/models), one directory per row
  with its verified-install.json receipt. A download and its conversion are
  staged under $TINYTITAN_WORK_DIR (default <tools>/../.build) — for an
  installed copy that is ~/.tinytitan/src/.build.

  Staging is kept only while it can save a download: the converted snapshot is
  what makes the *other* width of the same model free, so it lives until that
  width is installed and is removed automatically at that point. Draft-head
  staging has no second consumer and goes as soon as the install exists.
  `clean` applies those rules to everything at once and says what is left.

  A download is refused before it starts when the staging volume or the models
  volume cannot hold it, naming both numbers and both ways out (free space, or
  TINYTITAN_WORK_DIR on another volume). TINYTITAN_SKIP_DISK_CHECK=1 starts
  anyway. A resumed conversion does not need the room again.

What a run costs

  The sizes in the menu are the space an install occupies. Installing also
  downloads the model's own bf16 release and stages it under
  $TINYTITAN_WORK_DIR: about 70 GB for a 35B-A3B, which serves both widths,
  and 360 GB per width for Qwen3.8-Flash-Next. The bf16-dense variant is a
  third conversion of the same 70 GB, not a width of either snapshot, so it
  fetches again when installed on its own. A draft head fetches only its
  own shards. The disk guard above names both volumes and both numbers before
  anything starts.

Coverage

  Ornith 1.5 35B-A3B        4-bit, 8-bit, bf16 dense (4-bit experts), MTP draft
  Ornith 1.5 122B           4-bit, 8-bit (one width per run)
  Qwen 3.6 35B-A3B          4-bit, 8-bit, MTP draft
  Qwen3.8-Flash-Next        4-bit (experts 4-bit, the rest 8-bit), 8-bit
  Qwen-AgentWorld 35B-A3B   4-bit, 8-bit
  KAT-Coder-V2.5-Dev 35B-A3B 4-bit, 8-bit
  Qwen 3.5 2B / 4B / 9B     4-bit, 8-bit (CPU models)

Sources

  Every install is built from the model's own bf16 release, quantized here
  (group-64 affine) by the tools in tools/ and imported by TinyTitanRepack.
  Third-party quantizations are deliberately not used: their group sizes,
  widths and norm conventions are theirs, and here the router, the
  shared-expert gate, the DeltaNet gating projections and every norm stay
  at bf16 in both widths.

  convert_qwen35moe   tools/prepare_agentworld.py --model {ornith15,qwen36,agentworld,katcoder}
                      One ~70 GB download yields both widths, so
                      `<name> both` installs 4-bit and 8-bit for the cost of
                      one fetch; one width alone already converts both and
                      keeps the other snapshot for a later run.
  convert_qwen35moe_dense16
                      the same converter for Ornith-1.5-35B, with the dense
                      projections kept at the checkpoint's bf16 and only the
                      routed experts at the build width (--bits 4
                      --dense-bits 16). One width only: the install is a
                      different model, not a second width of the 4-bit one.
  convert_qwen35moe_solo
                      the same converter for Ornith-1.5-122B, one width per
                      run. A 250 GB source cannot stage both snapshots (69.5
                      and 129.8 GB) beside a 120 GB install, so the requested
                      width converts alone, is repacked, and its snapshot is
                      dropped. The source shards are streamed and deleted as
                      they convert, so the fetch is repeated for the second
                      width -- the disk is the constraint, not the link.
  convert_qwen35      tools/prepare_qwen35.py --size {2b,4b,9b}, then
                      TinyTitanRepack --input-snapshot. One fetch yields both
                      widths. These are the dense models, and the only ones
                      the CPU engine runs; the 9B is the vision-language
                      build, converted text-only like the others.
                      tools/repack_dense.sh re-runs the repack and the
                      equivalence check against a retained snapshot.
  convert             tools/prepare_qwen38.py, one 360 GB fetch per width.
                      The quant recipe is hardcoded per width: 4-bit keeps the
                      routed experts at 4 and moves the rest to 8 (--rest-8bit,
                      the 4E-8R-16N hybrid); 8-bit quantizes everything at 8.
                      The n-gram table stays fp16 and the norms and conv1d stay
                      bf16 in both. Qwen's own FP8 build is not used either: it
                      quantizes only the routed experts, in [128, 128] blocks
                      that do not map onto affine group-64.
  convert_qwen36_mtp  the same converter's --draft-head mode (two shards).
  prepare_ornith_mtp  tools/prepare_ornith_mtp.py (shard 16 of the original).

USAGE
}

# What a catalogue `source` means, in words a person reads. The branch names are
# this script's own, and `convert` alone reads as "converts a file you already
# fetched" — which is exactly how people read it. Every row downloads first.
source_label() {
  case "$1" in
    convert|convert_qwen35|convert_qwen35moe|convert_qwen35moe_solo|convert_qwen35moe_dense16)
      printf 'download + convert' ;;
    convert_qwen36_mtp|prepare_ornith_mtp)
      printf 'download + convert (draft head)' ;;
    repack)
      printf 'download + install' ;;
    *)
      printf '%s' "$1" ;;
  esac
}

status() {
  printf '%-20s %-8s %-13s %s\n' MODEL WIDTH STATE 'INSTALLS BY'
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name dir width source _ _ <<<"$row"
    if [[ -d "$MODELS/$dir" ]]; then
      state="installed"
    else
      state="not installed"
    fi
    printf '%-20s %-8s %-13s %s\n' "$name" "${width}-bit" "$state" "$(source_label "$source")"
  done
  echo
  echo "Nothing needs fetching by hand: installing a row downloads that model's"
  echo "pinned bf16 release and converts it here."
  echo
  local staged
  staged="$(staging_size)"
  if [[ -n "$staged" ]]; then
    echo "staging: $staged under $WORK — $(staging_names)"
    echo "  a converted snapshot is kept until the other width is installed, so the"
    echo "  second width needs no download; it is removed automatically at that point"
  fi
  echo "tools/install_models.sh --choose        the install menu: pick a model"
  echo "tools/install_models.sh <name>          download, convert and install one width"
  echo "tools/install_models.sh <name> both     4-bit and 8-bit from one download"
  echo "tools/install_models.sh clean           drop staging that cannot be reused"
  echo "tools/install_models.sh --help          sources, disk sizes and env vars"
}

# Does `TinyTitanRepack --model <key>` stream this one itself?
#
# These are the checkpoints the repacker fetches from Hugging Face and packages
# in one pass, with no converter involved — which means no Python. The keys are
# the same strings as this script's own, and the list is exactly what
# `TinyTitanRepack --help` accepts.
tinytitan_repack_streams() {
  case "$1" in
    qwen36|qwen36-8bit|qwen36-mtp|ornith15|ornith15-8bit) return 0 ;;
    *) return 1 ;;
  esac
}

# A Mac that has only Xcode's command-line tools ships Python 3.9, below the
# converters' 3.10 floor, and there is no Homebrew to install a newer one. Some
# of these models do not need a converter at all, so say which command works
# instead of leaving the person at "no usable Python interpreter found".
tinytitan_no_python_fallback() {
  local want="$1" row name dir
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name dir _ <<<"$row"
    [[ "$name" == "$want" ]] || continue
    echo >&2
    if tinytitan_repack_streams "$want"; then
      echo "  This one does not need Python: the repacker streams it itself." >&2
      echo "      swift run -c release TinyTitanRepack --model $want --output models/$dir" >&2
      echo "  or, with the release build already made," >&2
      echo "      .build/release/TinyTitanRepack --model $want --output models/$dir" >&2
    else
      echo "  This one is quantized here from its bf16 release by a Python" >&2
      echo "  converter, so it does need Python 3.10+. The repacker can stream" >&2
      echo "  qwen36 and ornith15 without any Python at all." >&2
    fi
    return 0
  done
  return 0
}

install_one() {
  local want="$1" found=0 preset_field=""
  # Every conversion path runs a Python converter, so resolve the interpreter
  # once, here, rather than emitting a raw "command not found" per call.
  local python
  python="$(tinytitan_resolve_python)" || { tinytitan_no_python_fallback "$want"; return 1; }
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    # Up to six fields on the MoE rows, four elsewhere. The preset is read here
    # because the MoE branches address staging by it (`$WORK/<preset>-affine-*`);
    # it used to be left unset, which worked only while every key equalled its
    # preset by coincidence.
    IFS='|' read -r name dir width source preset_field _ <<<"$row"
    [[ "$name" == "$want" ]] || continue
    found=1
    if [[ -d "$MODELS/$dir" ]]; then
      # A dense install from before this project repacked them is an affine
      # snapshot: the directory exists but carries no manifest and no receipt.
      # Returning here would leave it that way forever, because "the directory
      # exists" is what this check means. Fall through and let the branch
      # repack it, which is the only way it becomes verifiable in place.
      if [[ "$source" == convert_qwen35 && ! -f "$MODELS/$dir/manifest.json" ]]; then
        echo "$name is an affine snapshot; repacking it as .ssdai"
      else
        echo "$name is already installed at models/$dir"
        cleanup_staging "$name" keep
        return 0
      fi
    fi

    # --- can this download finish? -----------------------------------------
    # The converted snapshot and the install exist at the same time, so each
    # destination has to hold a copy: the catalogue's size for the install, a
    # quarter more on the staging volume for the quantized snapshot's overhead,
    # and the shards being streamed. A Qwen3.5-MoE conversion keeps twenty
    # fetchers busy (~90 GB in flight for a 35B-A3B) and a few finished shards
    # waiting for the converter; with both widths' snapshots also on the
    # volume, 104 GB is the upper bound on everything the source shards can
    # occupy. Every other source streams one shard at a time (12 GB). A
    # snapshot that is already on disk is not fetched again, so it is not
    # counted twice. An unreadable df is not a limit: require_gb says nothing
    # then, which is the same rule the runtime's own guard follows.
    if [[ "${TINYTITAN_SKIP_DISK_CHECK:-0}" != "1" ]]; then
      local size_gb="" snap_exists=0 install_exists=0 probe work_ok=1 models_ok=1
      local stream_gb=12 stage_need=""
      [[ "$source" == convert_qwen35moe || "$source" == convert_qwen35moe_dense16 ]] \
        && stream_gb=104
      size_gb="$(choice_gb "$name")"
      if [[ -n "$size_gb" ]]; then
        while IFS=: read -r _kind probe; do
          [[ -n "$probe" && -e "$probe" ]] && snap_exists=1
        done < <(row_staging_dirs "$name")
        [[ -d "$MODELS/$dir" ]] && install_exists=1
        if [[ "$source" == convert_qwen35moe_solo ]]; then
          # The solo path consumes its source shards as the install is written
          # (--consume-snapshot), so the peak is the snapshot plus the
          # bytes written before the first unlink -- not the snapshot plus the
          # whole install. 24 GB is the pre-unlink headroom (resident body,
          # four layer files and the reserve) with two fetchers in flight.
          stage_need="$(awk -v s="$size_gb" -v have="$snap_exists" \
               'BEGIN { printf "%d", have ? 24 : s + 24 }')"
        else
          stage_need="$(awk -v s="$size_gb" -v have="$snap_exists" -v stream="$stream_gb" \
               'BEGIN { printf "%d", have ? 2 : s * 1.25 + stream }')"
        fi
        require_gb "the staging volume ($WORK)" "$WORK" "$stage_need" \
          || work_ok=0
        local model_need
        if [[ "$source" == convert_qwen35moe_solo ]]; then
          # The solo import consumes its snapshot as the install is written, so
          # the install is not a second full copy: the repacker charges the
          # output less the shard bytes that import will release. 24 GB is the
          # pre-unlink headroom plus the reserve; requiring s + 3 here refused
          # a build that had 123 GB free for a 123 GB install and a 130 GB
          # snapshot.
          if (( install_exists )); then model_need=2; else model_need=24; fi
        else
          model_need="$(awk -v s="$size_gb" -v have="$install_exists" \
               'BEGIN { printf "%d", have ? 2 : s + 3 }')"
        fi
        require_gb "the models volume ($MODELS)" "$MODELS" "$model_need" \
          || models_ok=0
        if (( ! work_ok || ! models_ok )); then
          echo "  Not starting $name: there is not enough room to finish it." >&2
          echo "  Free space, or stage on another volume:" >&2
          echo "      TINYTITAN_WORK_DIR=/Volumes/scratch/tt $0 $name" >&2
          echo "  (TINYTITAN_SKIP_DISK_CHECK=1 starts anyway, at your own risk.)" >&2
          return 1
        fi
      fi
    fi

    case "$source" in
      repack)
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        echo "installing $name -> models/$dir"
        # --resume is refused when there is nothing to resume; pass it only
        # when a previous attempt left its state behind.
        if [[ -f "$MODELS/$dir.resume.json" ]]; then
          "$BIN" --model "$name" --output "$MODELS/$dir" --resume
        else
          "$BIN" --model "$name" --output "$MODELS/$dir"
        fi
        ;;
      convert)
        # Qwen's own bf16 release, quantized one shard at a time by
        # tools/prepare_qwen38.py (a 360 GB fetch per width), then repacked.
        # The quant recipe is hardcoded per width (see the CATALOGUE comment):
        # at 4 bits the routed experts keep the build width and the rest moves
        # to 8, so the snapshot is the 4E-8R-16N hybrid and the install is
        # named for it; at 8 bits everything quantised is 8.
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        local rest8_arg="" model_id="qwen3.8-flash-next"
        if [[ "$width" == 4 ]]; then
          rest8_arg="--rest-8bit"
          model_id="qwen3.8-flash-next-8bitrest"
        fi
        # A mirror: HF_ENDPOINT is the Hub's own variable, and the converter
        # fetches the checkpoint and its small JSON files from it when set.
        local endpoint_arg=""
        if [[ -n "${HF_ENDPOINT:-}" ]]; then endpoint_arg="--endpoint=${HF_ENDPOINT}"; fi
        if [[ ! -f "$WORK/qwen38-affine-${width}bit/model.safetensors.index.json" ]]; then
          echo "downloading and converting $name (360 GB fetch) -> $WORK/qwen38-affine-${width}bit"
          # A run killed mid-conversion resumes: finished output shards are kept
          # and the checkpoint shards they came from are not fetched again.
          "$python" "$ROOT/tools/prepare_qwen38.py" --bits "$width" ${rest8_arg:+"$rest8_arg"} \
              --output "$WORK/qwen38-affine-${width}bit" \
              --work "$WORK/qwen38-shards" ${endpoint_arg:+"$endpoint_arg"} || return 1
        fi
        # `--share-ngram-table` hardlinks the 102 GB table from the staging
        # snapshot into the install, which only works on one filesystem. With
        # TINYTITAN_MODELS_DIR on another volume, leave the flag off so the
        # repack copies it and its own disk-space check is the one that applies.
        local share_arg=""
        mkdir -p "$MODELS" || return 1
        local staging_device models_device
        staging_device="$(stat -f '%d' "$WORK/qwen38-affine-${width}bit" 2>/dev/null || echo)"
        models_device="$(stat -f '%d' "$MODELS" 2>/dev/null || echo)"
        if [[ -n "$staging_device" && "$staging_device" == "$models_device" ]]; then
          share_arg="--share-ngram-table"
        else
          echo "note: staging and $MODELS are on different filesystems;"
          echo "      ngram_table.bin will be copied rather than hardlinked"
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot "$WORK/qwen38-affine-${width}bit" \
            --model-id "$model_id" ${share_arg:+"$share_arg"} --output "$MODELS/$dir"
        ;;
      convert_qwen36_mtp)
        # Qwen3.6's draft head: 19 tensors of the `mtp.*` namespace in two
        # shards of Qwen's original, converted as a qwen3_5_mtp sidecar.
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        if [[ ! -f "$WORK/qwen36-mtp-affine/model.safetensors.index.json" ]]; then
          echo "downloading the draft head (two shards) and converting $name -> $WORK/qwen36-mtp-affine"
          "$python" "$ROOT/tools/prepare_agentworld.py" --model qwen36 --draft-head --bits "$width" \
              --output "$WORK/qwen36-mtp-affine" --work "$WORK/qwen36-mtp-shards" || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot "$WORK/qwen36-mtp-affine" \
            --model-id qwen3.6-35b-a3b-mtp-4bit --output "$MODELS/$dir"
        ;;
      convert_qwen35moe)
        # Qwen's own bf16 release, quantized one shard at a time by
        # tools/prepare_agentworld.py (about 70 GB fetched, at most two
        # shards on disk), then repacked. `--bits 4 8` writes *both* widths
        # from that one download, to `$WORK/<preset>-affine-{4,8}bit`, so
        # neither width may be thrown away: the snapshot is kept until both
        # installs exist, and the other width then costs no fetch at all.
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        local preset="${preset_field:-${name%-8bit}}" model_id
        case "$preset" in
          agentworld) model_id="qwen-agentworld" ;;
          katcoder)   model_id="kat-coder-v2.5" ;;
          qwen36)     model_id="qwen3.6-35b-a3b" ;;
          ornith15)   model_id="ornith-1.5-35b-a3b" ;;
        esac
        # Reuse whichever snapshot already exists, in either spelling: the
        # paired `-8bit` form this script writes, or the plain directory an
        # earlier manual `--bits 8` run leaves behind. Without this second
        # check a model already converted by hand re-downloads 70 GB.
        local snap="$WORK/${preset}-affine-${width}bit"
        if [[ ! -f "$snap/model.safetensors.index.json" \
              && "$width" == 8 && -f "$WORK/${preset}-affine/model.safetensors.index.json" ]]; then
          snap="$WORK/${preset}-affine"
        fi
        if [[ ! -f "$snap/model.safetensors.index.json" ]]; then
          echo "downloading and converting $preset (~70 GB fetch, both widths) -> $WORK/${preset}-affine-{4,8}bit"
          "$python" "$ROOT/tools/prepare_agentworld.py" --model "$preset" --bits 4 8 \
              --output "$WORK/${preset}-affine" \
              --work "$WORK/${preset}-shards" || return 1
          snap="$WORK/${preset}-affine-${width}bit"
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot "$snap" \
            --model-id "$model_id" --output "$MODELS/$dir"
        ;;
      convert_qwen35moe_dense16)
        # Ornith 1.5 35B with the dense blocks at the checkpoint's bf16.
        # `--dense-bits 16` copies attention, GDN and shared-expert weights
        # through unquantized, so the runtime reads them by dtype through
        # BF16GEMV, while the routed experts keep the 4-bit width the catalogue
        # names. The repacker records the promoted tensors in the manifest's
        # `promoted` list, which is what stops a reader from dequantizing them.
        # The model id carries the variant: the catalog id is `<id>_4-Bit`, and
        # without the suffix this install would advertise the same one as the
        # plain 4-bit row.
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        local preset="${preset_field:-${name}}" model_id
        case "$preset" in
          ornith15) model_id="ornith-1.5-35b-a3b-bf16dense" ;;
          *) echo "unknown bf16-dense preset: $preset" >&2; return 2 ;;
        esac
        local snap="$WORK/${preset}-bf16dense-affine"
        if [[ ! -f "$snap/model.safetensors.index.json" ]]; then
          echo "downloading and converting $preset (~70 GB fetch, bf16 dense) -> $snap"
          "$python" "$ROOT/tools/prepare_agentworld.py" --model "$preset" \
              --bits "$width" --dense-bits 16 \
              --output "$snap" --work "$WORK/${preset}-bf16dense-shards" || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot "$snap" --model-id "$model_id" --output "$MODELS/$dir" \
          || return 1
        "$BIN" --verify-install --input-ssdai "$MODELS/$dir" || return 1
        ;;
      convert_qwen35moe_solo)
        # The 122B path. One width per run, because the two snapshots plus an
        # install cannot coexist on a 1 TB Mac (69.5 + 129.8 + 120 GB). The
        # converter streams the 250 GB source four shards at a time and deletes
        # each as it converts, so the fetch is the only thing that repeats.
        # A width that was already converted by hand is reused rather than
        # fetched again; a partial snapshot is resumed by the converter.
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        local preset="${preset_field:-${name%-8bit}}" model_id
        case "$preset" in
          ornith122) model_id="ornith-1.5-122b" ;;
          *) echo "unknown large MoE preset: $preset" >&2; return 2 ;;
        esac
        local snap="$WORK/${preset}-affine-${width}bit"
        if [[ ! -f "$snap/model.safetensors.index.json" ]]; then
          echo "downloading and converting $preset at ${width}-bit (~250 GB fetch) -> $snap"
          TINYTITAN_FETCHERS="${TINYTITAN_FETCHERS:-4}" \
          TINYTITAN_PREFETCH_DEPTH="${TINYTITAN_PREFETCH_DEPTH:-2}" \
            "$python" "$ROOT/tools/prepare_agentworld.py" --model "$preset" --bits "$width" \
              --output "$snap" --work "$WORK/${preset}-shards" || return 1
        fi
        echo "installing $name -> models/$dir"
        # The source snapshot is consumed as the install is written: 130 GB of
        # snapshot + a ~125 GB install cannot coexist on this class of Mac.
        "$BIN" --input-snapshot "$snap" --consume-snapshot \
            --model-id "$model_id" --output "$MODELS/$dir"
        ;;
      prepare_ornith_mtp)
        # Ornith's draft head lives in shard 16 of its original checkpoint;
        # tools/prepare_ornith_mtp.py verifies that shard against its pinned
        # revision and converts it.
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        local src="$WORK/ornith-mtp-src" rev=e4dfb35a93d4b6822a811a7676f3488514abe7e2
        local base="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/$rev"
        mkdir -p "$src"
        # A file is trusted only at the size the server reports; a partial
        # left by an interrupted run is resumed, not skipped.
        for f in config.json model.safetensors.index.json model-00016-of-00016.safetensors; do
          local want have
          want=$(curl -sIL --retry 5 "$base/$f" | grep -i '^content-length:' | tail -1 | tr -dc '0-9')
          have=$(stat -f %z "$src/$f" 2>/dev/null || echo 0)
          if [[ -z "$want" || "$have" != "$want" ]]; then
            curl -fL --retry 20 --retry-delay 15 --retry-all-errors -C - -o "$src/$f" "$base/$f" || return 1
          fi
        done
        if [[ ! -f "$WORK/ornith-mtp-affine/model.safetensors.index.json" ]]; then
          echo "downloading shard 16 and converting $name -> $WORK/ornith-mtp-affine"
          "$python" "$ROOT/tools/prepare_ornith_mtp.py" --bits "$width" \
              --source-shard "$src/model-00016-of-00016.safetensors" \
              --source-config "$src/config.json" --source-index "$src/model.safetensors.index.json" \
              --output "$WORK/ornith-mtp-affine" || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot "$WORK/ornith-mtp-affine" \
            --model-id ornith-1.5-35b-a3b-mtp-4bit --output "$MODELS/$dir"
        ;;
      convert_qwen35)
        # The dense Qwen 3.5 models, from Qwen's own bf16 release. One
        # download yields both widths, so the other width installs without a
        # second fetch: only the converted *staging* directory is per-width,
        # the source shards in $WORK/<preset>-shards are shared.
        #
        # Convert, then repack, then drop the staging directory. The snapshot
        # the converter writes is an intermediate, not the install: every
        # model this project serves is a .ssdai directory with a manifest and
        # a path-bound receipt, and a snapshot has neither. Keeping the
        # intermediate would double the disk for a 9B and buy nothing, since
        # it is reproducible from the cached shards.
        #
        # The receipt is bound to the absolute output path, so the repack must
        # write straight into models/. Nothing here may move the directory
        # afterwards.
        local preset="${name%-8bit}" size_key model_id
        case "$preset" in
          qwen35-2b) size_key=2b; model_id="qwen3.5-2b" ;;
          qwen35-4b) size_key=4b; model_id="qwen3.5-4b" ;;
          qwen35-9b) size_key=9b; model_id="qwen3.5-9b" ;;
          *) echo "unknown Qwen 3.5 size: $preset" >&2; return 2 ;;
        esac
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        # The 9B checkpoint is the vision-language build; the converter drops
        # the model.visual.* tower and writes the text model, so the install
        # is text-only like every other model here.
        local stage="$WORK/qwen35-${size_key}-affine-${width}bit"
        if [[ ! -f "$stage/config.json" ]]; then
          if [[ -f "$MODELS/$dir/config.json" ]]; then
            # A legacy snapshot: move it into the converter's staging area
            # rather than converting it again. It is the same bytes the
            # converter would fetch and quantize, and it is already here.
            echo "staging the existing snapshot -> $stage"
            rm -rf "$stage"
            mv "$MODELS/$dir" "$stage"
          else
            echo "downloading and converting Qwen 3.5 ${size_key} ${width}-bit -> $stage"
            "$python" "$ROOT/tools/prepare_qwen35.py" --size "$size_key" --bits "$width" \
                --output "$stage" \
                --work "$WORK/${preset}-shards" || return 1
          fi
        fi
        # The receipt is bound to the absolute output path below, and it is
        # written by the repack, so the destination must be empty first and
        # must never be moved afterwards.
        echo "repacking $stage -> models/$dir"
        # `${var:?}` because an unset MODELS or dir would make this `rm -rf /`
        # or `rm -rf <models>`; the staging path is checked before it is used.
        rm -rf "${MODELS:?}/${dir:?}"
        "$BIN" --input-snapshot "$stage" --model-id "$model_id" \
            --output "$MODELS/$dir" || return 1
        "$BIN" --verify-install --input-ssdai "$MODELS/$dir" || return 1
        # The staging snapshot is an intermediate and is reproducible from the
        # cached shards, so it does not outlive the install. Use
        # tools/repack_dense.sh instead if you want it kept for the
        # equivalence gate.
        rm -rf "$stage"
        echo "installed $name -> models/$dir (.ssdai)"
        ;;
      unsupported)
        # No catalogue row uses this today, and the message it used to carry was
        # false: it said 8-bit Qwen3.8-Flash-Next "cannot execute -- the runtime
        # refuses it at load", which predates `SlotGEMV` giving the
        # hyper-connection, PLE and QSA-indexer projections both a 4- and an
        # 8-bit path. `validateFamilyQuantSupport` now refuses only a width
        # neither GEMV implements, and the indexer's bf16 prefill branch exists.
        # Kept as a generic refusal rather than deleted, so wiring a genuinely
        # unsupported row here later fails loudly instead of falling through this
        # switch and reporting a successful install it never performed.
        cat <<EOF
$name cannot be run by this installer.

A row reaches this branch only when it is known to build an install the runtime
cannot execute. Check --help for what that model would require; if nothing
explains it, this message is stale and the row should be fixed rather than
shipped.
EOF
        return 1
        ;;
    esac
    # Installed. Drop whatever staging can no longer save a download, and say
    # what is kept and why — a snapshot a second width will reuse, or nothing.
    cleanup_staging "$name" keep
    return 0
  done
  [[ "$found" == 1 ]] || { echo "unknown model: $want" >&2; status >&2; return 2; }
}

# Install one model's 4-bit and 8-bit builds from a single download.
#
# Only the Qwen3.5-MoE rows have a sibling width; for anything else there is
# nothing to pair with, so say so rather than silently installing one width.
install_both() {
  local want="$1" row name dir width source preset sibling
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name dir width source preset sibling <<<"$row"
    [[ "$name" == "$want" ]] || continue
    if [[ "$source" != convert_qwen35moe || -z "$sibling" ]]; then
      echo "$want has no second width to pair with; install it by name instead" >&2
      return 2
    fi
    # The 4-bit row carries the sibling; install it first so the one download
    # converts into both snapshots, then the 8-bit install reuses what is on
    # disk. `install_one` finds it because it checks both snapshot spellings.
    echo "installing $want at both widths from one download"
    install_one "$name" || return 1
    install_one "$sibling" || return 1
    return 0
  done
  echo "unknown model: $want" >&2
  status >&2
  return 2
}

# Install every model at one width, and report on all of them.
#
# The batch used to be the dispatcher's own loop, whose body ended in
# `[[ "$w" == 4 ]] && install_one "$n"`. A loop's status is its last body
# statement's, so the verdict was the *final catalogue row's width test*: with an
# 8-bit row last, `--all-4bit` exited 1 after installing everything and
# `--all-8bit` exited 0 whenever the last 8-bit model happened to work, however
# many before it had been refused (AUD-253). One refused model is enough to make
# the run incomplete, so every row's status is collected and the count of what
# was missed is what leaves the script.
install_width() {
  local want="$1" row name _ width
  local ran=() missed=()
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name _ width _ <<<"$row"
    [[ "$width" == "$want" ]] || continue
    if install_one "$name"; then
      ran+=("$name")
    else
      missed+=("$name")
    fi
  done
  echo
  echo "installed ${#ran[@]} of $(( ${#ran[@]} + ${#missed[@]} )) ${want}-bit model(s)"
  if [[ "${#missed[@]}" -gt 0 ]]; then
    printf 'NOT INSTALLED: %s\n' "${missed[@]+"${missed[@]}"}" >&2
    return 1
  fi
  return 0
}

# The one question an install has to ask.
#
# The model is the only real choice a person makes — everything else about
# setting TinyTitan up is automatic — so it is shown as a list with the disk each
# one costs and what it is for, with the verified default first. Enter takes the
# default. The entries and their sizes come from `TINYTITAN_MODEL_CHOICES` in
# tools/tinytitan_models.sh, so the installer cannot offer a model that does not
# exist or quote a size no one re-measured.
choose_model() {
  local count=${#TINYTITAN_MODEL_CHOICES[@]}
  local index key label bits gb note reply default_label
  if (( count == 0 )); then
    echo "no models are listed in TINYTITAN_MODEL_CHOICES" >&2
    return 2
  fi
  IFS='|' read -r _ default_label _ _ _ <<<"${TINYTITAN_MODEL_CHOICES[0]}"

  echo
  echo "Which model? This is the only choice; the rest is automatic."
  echo
  printf '  %-4s %-32s %-6s %-10s %s\n' "#" "model" "bits" "installed" "what it is for"
  for (( index = 0; index < count; index++ )); do
    IFS='|' read -r key label bits gb note <<<"${TINYTITAN_MODEL_CHOICES[$index]}"
    printf '  %2d)  %-32s %-6s %7s GB  %s\n' \
      "$((index + 1))" "$label" "${bits}-bit" "$gb" "$note"
  done
  echo
  echo "  Sizes are installed sizes: installing also downloads the model's bf16"
  echo "  release first and stages it separately — --help has the numbers."
  echo
  printf 'Choice [1-%d] (Enter for %s): ' "$count" "$default_label"
  # An EOF is not an answer. Treating it as "Enter" started the recommended
  # 36.9 GB download for a caller that never chose anything (found 2026-09-24 by
  # closing stdin on `--choose`); only a real empty line takes the default.
  if ! read -r reply; then
    echo
    echo "No answer given; nothing was installed." >&2
    return 2
  fi
  reply="${reply:-1}"
  # Both reads are forced to decimal. `(( ))` and `$(( ))` take a leading zero as
  # octal, and this guard asks the range question in the negative, so an
  # unreadable reply answers "no verdict" and means ACCEPTED: `010` passed as 8
  # and installed row 8 for a person who pointed at row 10, while `08` passed the
  # test and then died at the index below. The row a reply names is the row it
  # installs. Forcing the base is what the launcher does with a number it consumes
  # on the spot (`ram_tier` and `valid_concurrency` in server_launcher.sh); a port
  # is refused instead because it flows outward to readers that script cannot fix.
  if [[ ! "$reply" =~ ^[0-9]+$ ]] || (( 10#$reply < 1 || 10#$reply > count )); then
    echo "not a choice: $reply" >&2
    return 2
  fi
  key="${TINYTITAN_MODEL_CHOICES[$((10#$reply - 1))]%%|*}"
  echo
  install_one "$key"
}

case "${1:-}" in
  "")            status ;;
  --help|-h)     usage ;;
  clean)         cmd_clean ;;
  # The install menu: one numbered list, default first, then install the pick.
  --choose|--menu)
                 if [[ ! -t 0 ]]; then
                   echo "--choose needs a terminal to ask in; pass a model name instead" >&2
                   exit 2
                 fi
                 choose_model ;;
  --all-4bit)    install_width 4 ;;
  --all-8bit)    install_width 8 ;;
  # `both` installs a model's 4-bit and 8-bit builds from ONE download. The
  # MoE checkpoints convert both widths in a single pass, so asking for them
  # one at a time would fetch the same ~70 GB twice; this is the cheap way and
  # the one the help text points at.
  both)          [[ -n "${2:-}" ]] || { echo "usage: tools/install_models.sh <model> both" >&2; exit 2; }
                 install_both "$2" ;;
  *)             if [[ "${2:-}" == "both" ]]; then install_both "$1"; else install_one "$1"; fi ;;
esac
