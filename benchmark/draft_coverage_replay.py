#!/usr/bin/env python3
"""Gate 1 of the wide-verify plan: how much of an agent's output a prompt-lookup
drafter could have copied, measured on real transcripts. Model-free.

For every assistant turn, the drafter sees the whole conversation before it
(system, user, tool results, earlier assistant turns) plus the turn's own
tokens so far. At each step it looks for the longest recent n-gram match of the
current suffix in that history and proposes the tokens that followed it, up to
`--width - 1` of them. The replay scores the proposal against what the model
actually wrote: a pass emits the accepted prefix plus one token (the row the
target model always produces), exactly as a greedy verify would.

Reported per run:

- `copy_share`: share of output tokens emitted inside accepted runs of at least
  `--min-run` tokens. The plan's stop line is 0.40 at min-run 8.
- `projected_tok_s`: a projection, not a measurement, from a linear pass-cost
  model (`--pass-base-ms` + `--pass-row-ms` per row, against `--scalar-ms` for a
  plain token). The defaults are the fit from the Qwen3.8 4-bit two-row MTP pass
  (480 ms) and the 200-token prefill (~55 ms/row), against the v5.20 run
  record's 211 ms token. A controller drafts only when the match proposes at
  least `--min-draft` tokens; otherwise it takes a scalar step.

Inputs are JSONL transcripts in any of these shapes, auto-detected per line:
Claude Code session logs (`{"type": "user"|"assistant", "message": {...}}`),
Codex rollouts (`{"type": "response_item", "payload": {...}}`), OpenAI-style
`{"role": ..., "content": ...}` lines, or one `{"messages": [...]}` per line.
Tool calls count as assistant output; tool results count as context.

Tokenization: with `--tokenizer path/to/tokenizer.json` and the `tokenizers`
package, the model's own BPE is used. Otherwise a BPE-shaped regex split
(words with a leading space, digit runs, punctuation, whitespace runs) stands
in; it tracks copy structure but not exact token counts, and the report says
which one ran.

    python3 benchmark/draft_coverage_replay.py ~/.codex/sessions/2026/11/*.jsonl \
        --width 8 --width 16 --out .build/draft-coverage.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

_REGEX_TOKEN = re.compile(r" ?[A-Za-z_]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9_]+|\s+")


class TranscriptError(ValueError):
    """A transcript line that cannot be read as a conversation turn."""


@dataclass(frozen=True)
class Turn:
    role: str  # "assistant" emits tokens; every other role is context
    text: str


@dataclass
class ReplayStats:
    output_tokens: int = 0
    copied_tokens: int = 0  # emitted inside accepted runs >= min_run
    passes: int = 0
    scalar_steps: int = 0
    accepted_total: int = 0
    rows_total: int = 0
    time_ms: float = 0.0
    turns: int = 0
    files: list[str] = field(default_factory=list)


def _text_of(content: object) -> str:
    """Flatten a message content field (string or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        raise TranscriptError(f"unreadable content of type {type(content).__name__}")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in {"text", "input_text", "output_text"}:
            parts.append(str(block.get("text", "")))
        elif kind == "tool_use":
            parts.append(json.dumps(block.get("input", {}), ensure_ascii=False))
        elif kind == "tool_result":
            parts.append(_text_of(block.get("content")))
    return "\n".join(part for part in parts if part)


def _turns_of_record(record: object) -> list[Turn]:
    if not isinstance(record, dict):
        return []
    if "messages" in record and isinstance(record["messages"], list):
        return [turn for message in record["messages"] for turn in _turns_of_record(message)]
    if record.get("type") in {"user", "assistant"} and isinstance(record.get("message"), dict):
        message = record["message"]
        role = str(message.get("role", record["type"]))
        content = message.get("content")
        # A Claude Code "user" record that carries tool results is context either way.
        return [Turn(role, _text_of(content))]
    if record.get("type") == "response_item" and isinstance(record.get("payload"), dict):
        payload = record["payload"]
        kind = payload.get("type")
        if kind == "message":
            return [Turn(str(payload.get("role", "user")), _text_of(payload.get("content")))]
        if kind in {"function_call", "custom_tool_call"}:
            args = payload.get("arguments", payload.get("input", ""))
            return [Turn("assistant", args if isinstance(args, str) else json.dumps(args))]
        if kind in {"function_call_output", "custom_tool_call_output"}:
            out = payload.get("output", "")
            return [Turn("tool", out if isinstance(out, str) else json.dumps(out))]
        return []
    if "role" in record:
        role = str(record["role"])
        turns = [Turn(role, _text_of(record.get("content")))]
        for call in record.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            turns.append(Turn("assistant", str(function.get("arguments", ""))))
        return turns
    return []


def read_transcript(path: pathlib.Path) -> list[Turn]:
    turns: list[Turn] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TranscriptError(f"{path}:{number}: not JSON ({error.msg})") from error
            turns.extend(turn for turn in _turns_of_record(record) if turn.text)
    return turns


def regex_tokenize(text: str) -> list[str]:
    return _REGEX_TOKEN.findall(text)


def load_tokenizer(path: pathlib.Path | None):
    """Return (tokenize, name). Refuses a named tokenizer it cannot load."""
    if path is None:
        return regex_tokenize, "regex-approximation"
    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
    except ImportError as error:
        raise SystemExit("--tokenizer needs the 'tokenizers' package") from error
    model = Tokenizer.from_file(str(path))
    return (lambda text: model.encode(text, add_special_tokens=False).ids), f"bpe:{path.name}"


class NgramIndex:
    """Most-recent-occurrence index over a growing token history."""

    def __init__(self, n_max: int, n_min: int) -> None:
        self.n_max, self.n_min = n_max, n_min
        self.history: list = []
        self.last: dict[tuple, int] = {}  # n-gram -> index of the token that followed it

    def append(self, token) -> None:
        # Register the n-grams that end just before the new token: their continuation
        # (this token) now exists. The current suffix stays unregistered, so a lookup
        # of it finds an earlier occurrence rather than itself.
        self.history.append(token)
        end = len(self.history) - 1
        for n in range(self.n_min, self.n_max + 1):
            if end - n >= 0:
                self.last[tuple(self.history[end - n : end])] = end

    def propose(self, limit: int) -> list:
        h = self.history
        for n in range(self.n_max, self.n_min - 1, -1):
            if len(h) < n:
                continue
            start = self.last.get(tuple(h[-n:]))
            if start is not None and start < len(h):
                return h[start : start + limit]
        return []


def replay(
    turns: list[Turn],
    tokenize,
    width: int,
    min_draft: int,
    min_run: int,
    scalar_ms: float,
    pass_base_ms: float,
    pass_row_ms: float,
    stats: ReplayStats,
    n_max: int = 8,
    n_min: int = 2,
) -> None:
    index = NgramIndex(n_max, n_min)

    push = index.append

    for turn in turns:
        tokens = tokenize(turn.text)
        if turn.role != "assistant":
            for token in tokens:
                push(token)
            continue
        stats.turns += 1
        i = 0
        while i < len(tokens):
            draft = index.propose(width - 1) if len(index.history) >= n_min else []
            if len(draft) < min_draft:
                stats.scalar_steps += 1
                stats.time_ms += scalar_ms
                push(tokens[i])
                i += 1
                stats.output_tokens += 1
                continue
            accepted = 0
            while (
                accepted < len(draft)
                and i + accepted < len(tokens)
                and draft[accepted] == tokens[i + accepted]
            ):
                accepted += 1
            emitted = min(accepted + 1, len(tokens) - i)
            rows = len(draft) + 1
            stats.passes += 1
            stats.accepted_total += accepted
            stats.rows_total += rows
            stats.time_ms += pass_base_ms + pass_row_ms * rows
            if emitted >= min_run:
                stats.copied_tokens += emitted
            for token in tokens[i : i + emitted]:
                push(token)
            i += emitted
            stats.output_tokens += emitted


def summarize(stats: ReplayStats, scalar_ms: float) -> dict:
    if stats.output_tokens == 0:
        raise SystemExit("no assistant output found in the given transcripts; refusing to report")
    baseline_ms = stats.output_tokens * scalar_ms
    return {
        "output_tokens": stats.output_tokens,
        "assistant_turns": stats.turns,
        "copy_share": round(stats.copied_tokens / stats.output_tokens, 4),
        "passes": stats.passes,
        "scalar_steps": stats.scalar_steps,
        "mean_accepted_per_pass": round(stats.accepted_total / stats.passes, 3)
        if stats.passes
        else 0.0,
        "mean_rows_per_pass": round(stats.rows_total / stats.passes, 3) if stats.passes else 0.0,
        "projected_tok_s": round(1000.0 * stats.output_tokens / stats.time_ms, 3),
        "baseline_tok_s": round(1000.0 * stats.output_tokens / baseline_ms, 3),
        "projected_speedup": round(baseline_ms / stats.time_ms, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("transcripts", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--width", type=int, action="append", help="rows per pass (repeatable); default 8 and 16"
    )
    parser.add_argument(
        "--min-draft",
        type=int,
        action="append",
        help="controller threshold (repeatable); default 3 and 7",
    )
    parser.add_argument("--min-run", type=int, default=8)
    parser.add_argument("--scalar-ms", type=float, default=211.0)
    parser.add_argument("--pass-base-ms", type=float, default=370.0)
    parser.add_argument("--pass-row-ms", type=float, default=55.0)
    parser.add_argument("--tokenizer", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)
    widths = args.width or [8, 16]
    thresholds = args.min_draft or [3, 7]
    if any(w < 2 for w in widths) or any(m < 1 for m in thresholds) or args.min_run < 1:
        parser.error("--width must be >= 2, --min-draft and --min-run >= 1")
    if min(args.scalar_ms, args.pass_base_ms, args.pass_row_ms) <= 0:
        parser.error("cost parameters must be positive")

    tokenize, tokenizer_name = load_tokenizer(args.tokenizer)
    sessions = []
    for path in args.transcripts:
        try:
            sessions.append((path, read_transcript(path)))
        except (OSError, TranscriptError) as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2

    rows = []
    for width in widths:
        for threshold in thresholds:
            if threshold > width - 1:
                continue
            stats = ReplayStats()
            for _path, turns in sessions:
                replay(
                    turns,
                    tokenize,
                    width,
                    threshold,
                    args.min_run,
                    args.scalar_ms,
                    args.pass_base_ms,
                    args.pass_row_ms,
                    stats,
                )
            rows.append(
                {"width": width, "min_draft": threshold, **summarize(stats, args.scalar_ms)}
            )

    report = {
        "tokenizer": tokenizer_name,
        "files": len(sessions),
        "min_run": args.min_run,
        "cost_model_ms": {
            "scalar": args.scalar_ms,
            "pass_base": args.pass_base_ms,
            "pass_row": args.pass_row_ms,
        },
        "note": "projected_tok_s is a projection from the cost model, not a measurement",
        "rows": rows,
    }
    print(f"tokenizer={tokenizer_name} files={len(sessions)} min_run={args.min_run}")
    print(
        f"{'W':>3} {'min_draft':>9} {'out_tok':>8} {'copy_share':>10} {'acc/pass':>8} {'proj_tok_s':>10} {'speedup':>7}"
    )
    for row in rows:
        print(
            f"{row['width']:>3} {row['min_draft']:>9} {row['output_tokens']:>8} {row['copy_share']:>10.3f} "
            f"{row['mean_accepted_per_pass']:>8.2f} {row['projected_tok_s']:>10.2f} {row['projected_speedup']:>7.2f}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
