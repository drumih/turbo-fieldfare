#!/usr/bin/env python3
"""Tests for the draft-coverage replay: transcript parsing, the n-gram drafter,
and the replay's accounting on conversations whose answer is known by hand.

    cd benchmark && python3 -m unittest test_draft_coverage_replay -v
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

BENCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import draft_coverage_replay as dcr  # noqa: E402


def chars(text: str) -> list[str]:
    return list(text)


def run(turns, width=8, min_draft=1, min_run=8, tokenize=chars):
    stats = dcr.ReplayStats()
    dcr.replay(turns, tokenize, width, min_draft, min_run, 211.0, 370.0, 55.0, stats)
    return stats


class ParsingTests(unittest.TestCase):
    def write(self, lines) -> pathlib.Path:
        directory = tempfile.mkdtemp()
        path = pathlib.Path(directory) / "t.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
        return path

    def test_claude_code_records_with_tool_use_and_result(self) -> None:
        path = self.write(
            [
                {"type": "user", "message": {"role": "user", "content": "fix it"}},
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {"type": "tool_use", "name": "Edit", "input": {"old": "a"}},
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "content": "done"}],
                    },
                },
            ]
        )
        turns = dcr.read_transcript(path)
        self.assertEqual([t.role for t in turns], ["user", "assistant", "user"])
        self.assertIn('"old": "a"', turns[1].text)
        self.assertEqual(turns[2].text, "done")

    def test_codex_rollout_calls_are_output_and_results_context(self) -> None:
        path = self.write(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "function_call", "arguments": '{"cmd":1}'},
                },
                {
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "output": "x"},
                },
                {"type": "session_meta", "payload": {}},
            ]
        )
        self.assertEqual([t.role for t in dcr.read_transcript(path)], ["user", "assistant", "tool"])

    def test_openai_messages_wrapper(self) -> None:
        path = self.write(
            [
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        self.assertEqual([t.role for t in dcr.read_transcript(path)], ["system", "assistant"])

    def test_refuses_a_line_that_is_not_json(self) -> None:
        directory = tempfile.mkdtemp()
        path = pathlib.Path(directory) / "bad.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        with self.assertRaises(dcr.TranscriptError):
            dcr.read_transcript(path)


class DrafterTests(unittest.TestCase):
    def test_proposes_the_continuation_of_the_latest_earlier_match(self) -> None:
        index = dcr.NgramIndex(n_max=4, n_min=2)
        for token in "abcdXabcd" + "Yab":
            index.append(token)
        self.assertEqual(index.propose(3), ["c", "d", "Y"])

    def test_current_suffix_never_matches_itself(self) -> None:
        index = dcr.NgramIndex(n_max=4, n_min=2)
        for token in "abcd":
            index.append(token)
        self.assertEqual(index.propose(3), [])


class ReplayTests(unittest.TestCase):
    def test_novel_output_is_all_scalar_and_projects_baseline(self) -> None:
        stats = run([dcr.Turn("user", "wxyz"), dcr.Turn("assistant", "abcdefgh")])
        self.assertEqual((stats.output_tokens, stats.passes, stats.copied_tokens), (8, 0, 0))
        summary = dcr.summarize(stats, 211.0)
        self.assertAlmostEqual(summary["projected_speedup"], 1.0)

    def test_verbatim_copy_is_counted_and_charged_by_rows(self) -> None:
        source = "0123456789abcdefghij"
        # The first two tokens are scalar (no suffix of length 2 yet in output);
        # from then on every W=8 pass proposes 7 correct tokens and emits 8.
        stats = run([dcr.Turn("user", source), dcr.Turn("assistant", source)], width=8, min_run=8)
        self.assertEqual(stats.output_tokens, 20)
        self.assertEqual(stats.scalar_steps, 2)
        self.assertEqual(stats.passes, 3)  # 8 + 8 + final 2
        self.assertEqual(stats.copied_tokens, 16)  # the final 2-token pass is below min_run
        self.assertEqual(stats.rows_total, 3 * 8)  # each pass verifies a full 7-token draft
        self.assertAlmostEqual(stats.time_ms, 2 * 211.0 + 3 * 370.0 + 55.0 * 24)

    def test_controller_threshold_skips_short_drafts(self) -> None:
        turns = [dcr.Turn("user", "abcXabcY"), dcr.Turn("assistant", "abcZ")]
        self.assertGreater(run(turns, min_draft=1).passes, 0)
        self.assertEqual(run(turns, min_draft=6).passes, 0)

    def test_summary_refuses_an_empty_run(self) -> None:
        with self.assertRaises(SystemExit):
            dcr.summarize(dcr.ReplayStats(), 211.0)


if __name__ == "__main__":
    unittest.main()
