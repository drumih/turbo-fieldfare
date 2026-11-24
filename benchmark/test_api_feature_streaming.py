"""TT-128: the API feature check fails a route that buffers its answer.

`tools/api-feature-check.py` talks to a live server at import time, so the
streaming predicate is lifted out of its source with `ast` and exercised on
synthetic event timings: a burst of deltas (decoded first, sent at the end)
fails, deltas spaced like decoded tokens pass, and a slow prefill before the
first delta does not count against a route that streams.
"""

import ast
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "tools" / "api-feature-check.py"


def load_predicate():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "streamed_incrementally"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace["streamed_incrementally"]


def events(times, text_from=1):
    out = [{"_arrived": 0.0, "choices": [{"delta": {"role": "assistant"}}]}]
    for i, t in enumerate(times):
        out.append({"_arrived": t, "choices": [{"delta": {"content": f"t{i}"}}]})
    out.append({"_arrived": times[-1] + 0.001, "choices": [], "usage": {}})
    return out


def has_text(event):
    return any(c.get("delta", {}).get("content") for c in event.get("choices", []))


class StreamingPredicate(unittest.TestCase):
    def setUp(self) -> None:
        self.streamed = load_predicate()

    def test_burst_after_decode_fails(self) -> None:
        burst = [5.0 + i * 0.00001 for i in range(12)]
        self.assertFalse(self.streamed(events(burst), has_text))

    def test_tokens_spaced_by_decode_pass(self) -> None:
        paced = [0.5 + i * 0.08 for i in range(12)]
        self.assertTrue(self.streamed(events(paced), has_text))

    def test_slow_prefill_does_not_fail_a_streaming_route(self) -> None:
        paced = [9.0 + i * 0.05 for i in range(12)]
        self.assertTrue(self.streamed(events(paced), has_text))

    def test_single_delta_is_not_streaming(self) -> None:
        self.assertFalse(self.streamed(events([1.0]), has_text))


if __name__ == "__main__":
    unittest.main()
