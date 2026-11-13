#!/usr/bin/env python3
"""Expert-cache hit-rate bounds from real routing traces (task 18).

Reads `TINYTITAN_ROUTE_TRACE` files (one line per layer and token:
`position layer e0 e1 ...`) and, per layer, replays the decode tokens through
a fixed-size slot cache under three policies:

  LRU   least recently used
  LFU   least frequently used (counts since the start of the trace)
  OPT   Belady's optimal policy: evict the expert whose next use is furthest
        away. No online policy can beat it, so it is the physical ceiling for
        that slot count on that token stream.

It also reports the decode working set per layer (distinct experts) and, for
several traces, how much of one trace's hottest experts the others reuse --
the quantity a warm-start preload or a session-sticky cache relies on.

  python3 benchmark/route_cache_bound.py --slots 48 --decode-from 400 a.trace b.trace
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys


def load(path: pathlib.Path, decode_from: int) -> dict[int, list[list[int]]]:
    """layer -> per-token expert lists, decode positions only, in order."""
    layers: dict[int, list[tuple[int, list[int]]]] = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        position, layer = int(parts[0]), int(parts[1])
        if position < decode_from:
            continue
        layers[layer].append((position, [int(x) for x in parts[2:]]))
    return {
        layer: [e for _, e in sorted(rows, key=lambda r: r[0])] for layer, rows in layers.items()
    }


def simulate(
    tokens: list[list[int]], slots: int, policy: str, warm: list[int] | None = None
) -> tuple[int, int]:
    """Hits and accesses for one layer's token stream."""
    cache: set[int] = set(warm[:slots]) if warm else set()
    last_use: dict[int, int] = {}
    count: collections.Counter[int] = collections.Counter()
    flat = [(t, e) for t, experts in enumerate(tokens) for e in experts]
    hits = 0
    for i, (_, e) in enumerate(flat):
        count[e] += 1
        if e in cache:
            hits += 1
        else:
            if len(cache) >= slots:
                if policy == "LRU":
                    victim = min(cache, key=lambda x: last_use.get(x, -1))
                elif policy == "LFU":
                    victim = min(cache, key=lambda x: (count[x], last_use.get(x, -1)))
                else:  # OPT
                    nxt = {}
                    for x in cache:
                        nxt[x] = 1 << 60
                    for j in range(i + 1, len(flat)):
                        x = flat[j][1]
                        if x in nxt and nxt[x] == 1 << 60:
                            nxt[x] = j
                            if all(v != 1 << 60 for v in nxt.values()):
                                break
                    victim = max(cache, key=lambda x: nxt[x])
                cache.discard(victim)
            cache.add(e)
        last_use[e] = i
    return hits, len(flat)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=pathlib.Path)
    parser.add_argument("--slots", type=int, default=48)
    parser.add_argument(
        "--decode-from",
        type=int,
        nargs="*",
        default=[0],
        help="first decode position per trace (prompt length); one value or one per trace",
    )
    args = parser.parse_args()
    starts = (
        args.decode_from
        if len(args.decode_from) == len(args.traces)
        else [args.decode_from[0]] * len(args.traces)
    )
    data = [load(p, s) for p, s in zip(args.traces, starts, strict=True)]

    print(f"slots per layer: {args.slots}")
    hot: list[dict[int, list[int]]] = []
    for path, layers in zip(args.traces, data, strict=True):
        totals = {p: [0, 0] for p in ("LRU", "LFU", "OPT")}
        distinct = []
        hot_layers = {}
        for layer, tokens in sorted(layers.items()):
            counter = collections.Counter(e for experts in tokens for e in experts)
            distinct.append(len(counter))
            hot_layers[layer] = [e for e, _ in counter.most_common()]
            for policy in totals:
                h, n = simulate(tokens, args.slots, policy)
                totals[policy][0] += h
                totals[policy][1] += n
        hot.append(hot_layers)
        tokens_n = len(next(iter(layers.values()))) if layers else 0
        print(f"\n{path.name}: {tokens_n} decode tokens, {len(layers)} layers")
        print(
            f"  distinct experts per layer: mean {sum(distinct) / max(len(distinct), 1):.0f}, max {max(distinct, default=0)}"
        )
        for policy, (h, n) in totals.items():
            print(f"  {policy}: hit rate {h / max(n, 1):.3f}")

    if len(data) > 1:
        print(f"\nwarm start from the other traces' top-{args.slots} per layer (LRU, then OPT):")
        for i, layers in enumerate(data):
            pooled: dict[int, collections.Counter[int]] = collections.defaultdict(
                collections.Counter
            )
            for j, other in enumerate(data):
                if j == i:
                    continue
                for layer, tokens in other.items():
                    pooled[layer].update(e for experts in tokens for e in experts)
            sums = {"LRU": [0, 0], "OPT": [0, 0]}
            for layer, tokens in layers.items():
                warm = [e for e, _ in pooled[layer].most_common(args.slots)]
                for policy in sums:
                    h, n = simulate(tokens, args.slots, policy, warm)
                    sums[policy][0] += h
                    sums[policy][1] += n
            print(
                f"  {args.traces[i].name}: "
                + ", ".join(f"{p} {h / max(n, 1):.3f}" for p, (h, n) in sums.items())
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
