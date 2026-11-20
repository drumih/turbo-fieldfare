#!/usr/bin/env bash
# Is CI green on one commit? Used by tools/release.sh before it tags a release,
# and runnable on its own against any commit.
#
#   tools/ci-green.sh Pummelchen/TinyTitan "$(git rev-parse v5.18^{commit})"
#
# The shape of the question matters. A release tag is a claim that the commit it
# names passes, and every other gate in this repository is one machine's view of
# a working tree: only CI runs against a clean clone, on the pinned toolchains,
# across the install matrix. v5.18 was published an hour after its own tag commit
# came back `CI | completed | failure`, because nothing asked.
#
# Three answers are refusals, not warnings:
#   * a run that has not finished -- releasing over an in-flight run is guessing;
#   * no run at all for that sha -- a short sha matches nothing, and a check that
#     queries nothing and reports green is how a vacuous pass gets made;
#   * a conclusion that is not success or skipped, including cancelled, timed
#     out and action-required, each of which means CI did not pass.
#
# **CodeQL is advisory, not a release gate.** The release rule is that the
# compile+test workflow is green; the security analysis may still be running (it
# can take hours) or report a finding, and neither holds a patch back. Its state
# is printed so a failure is still visible and handled as a follow-up. This is
# the 2026-11-20 rule: shipping a patch waits for the tests, not for CodeQL.
#
# `--allow-red "<reason>"` lets a deliberate release continue. It is the same
# bargain as the skipped golden baselines: the reason is mandatory, printed, and
# the caller requires the failing run's URL in the release notes so the choice
# stays on the record.
#
# Output: human-readable lines, then a tab-separated result line for the caller:
#   result        green|overridden       <first red run URL, or ->
# Exit 0 on green or overridden, 1 on any refusal, 2 on a usage or transport
# error (the check could not be made at all).
set -uo pipefail

REPO="${1:-}"
SHA="${2:-}"
ALLOW_REASON="${4:-}"

if [ -z "$REPO" ] || [ -z "$SHA" ]; then
  echo "usage: tools/ci-green.sh <owner/repo> <full-commit-sha> [--allow-red \"<reason>\"]" >&2
  exit 2
fi
case "${3:-}" in
  --allow-red) ;;
  '') ;;
  *)
    echo "usage: third argument must be --allow-red \"<reason>\" (got $3)" >&2
    exit 2
    ;;
esac
if [ "$#" -gt 3 ] && [ -z "$ALLOW_REASON" ]; then
  echo "error: --allow-red needs a non-empty reason; a release over a red CI must say why" >&2
  exit 2
fi
# 40 hex digits is a full object name. Anything shorter is the mistake the header
# names: `head_sha` with a short sha matches no run, and "no runs" would otherwise
# arrive as a question rather than as an answer.
case "${#SHA}" in
  40) ;;
  *)
    echo "error: $SHA is not a full commit sha (need 40 hex characters)" >&2
    exit 2
    ;;
esac

if ! RUNS="$(gh api "repos/$REPO/actions/runs?head_sha=$SHA&per_page=100" \
  --jq '.workflow_runs[] | [.name, .status, (.conclusion // "-"), .html_url] | @tsv' 2>&1)"; then
  echo "error: cannot read GitHub Actions runs for $SHA on $REPO: $RUNS" >&2
  exit 2
fi

TOTAL=0
PENDING=""
RED=""
RED_URL=""
ADVISORY=""
while IFS=$'\t' read -r name status conclusion url; do
  [ -n "$name" ] || continue
  # CodeQL is deliberately outside the gate: the release waits for compile+test,
  # not for the security scan. A pending or red scan is reported, never fatal.
  if [ "$name" = "CodeQL" ]; then
    case "$status/$conclusion" in
      completed/success|completed/skipped) ;;
      *) ADVISORY="$ADVISORY CodeQL ($status/$conclusion)" ;;
    esac
    continue
  fi
  TOTAL=$((TOTAL + 1))
  if [ "$status" != "completed" ]; then
    PENDING="$PENDING $name ($status)"
    continue
  fi
  case "$conclusion" in
    success | skipped) ;;
    *)
      RED="$RED $name ($conclusion)"
      [ -n "$RED_URL" ] || RED_URL="$url"
      ;;
  esac
done <<< "$RUNS"

if [ "$TOTAL" -eq 0 ]; then
  echo "error: no workflow run on $REPO for $SHA -- CI never saw this commit" >&2
  echo "       Push the branch that carries it and let CI finish; a check that did" >&2
  echo "       not run is not a pass." >&2
  exit 1
fi
if [ -n "$PENDING" ]; then
  echo "error: CI has not finished on $SHA:$PENDING" >&2
  echo "       Wait for it. An in-flight run is not what --allow-red is for." >&2
  exit 1
fi
if [ -n "$ADVISORY" ]; then
  echo "  advisory (does not gate a release):$ADVISORY"
fi
if [ -n "$RED" ]; then
  if [ -z "$ALLOW_REASON" ]; then
    echo "error: CI is red on $SHA:$RED" >&2
    echo "       $RED_URL" >&2
    echo "       Fix main and re-tag, or go over it deliberately with" >&2
    echo "       --allow-red \"<reason>\" and put that reason in the release notes." >&2
    exit 1
  fi
  echo "  !! CI is red on $SHA:$RED ($RED_URL)"
  echo "  !! releasing over it: $ALLOW_REASON"
  printf 'result\toverridden\t%s\n' "$RED_URL"
  exit 0
fi
echo "  CI green on $SHA ($TOTAL gating run(s))"
printf 'result\tgreen\t-\n'
