#!/usr/bin/env python3
"""Render self-hosted flat-square SVG badges under .github/badges/.

Avoids broken README stats bars when img.shields.io or GitHub's camo proxy
times out or rate-limits external badge fetches.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone

REPO = "Pummelchen/TinyTitan"
ROOT = pathlib.Path(__file__).resolve().parent.parent
BADGES_DIR = ROOT / ".github" / "badges"
TRAFFIC_JSON = ROOT / ".github" / "traffic.json"


def _text_width(text: str) -> int:
    # Approximate Verdana 11px width in SVG user units.
    return max(24, int(round(len(text) * 6.5)) + 12)


def render_badge(label: str, message: str, color: str) -> str:
    lw = _text_width(label)
    mw = _text_width(message)
    total = lw + mw
    lx = lw / 2
    mx = lw + mw / 2
    esc_label = html.escape(label)
    esc_msg = html.escape(message)
    esc_title = html.escape(f"{label}: {message}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" '
        f'role="img" aria-label="{esc_title}">'
        f"<title>{esc_title}</title>"
        f'<g shape-rendering="crispEdges">'
        f'<rect width="{lw}" height="20" fill="#555"/>'
        f'<rect x="{lw}" width="{mw}" height="20" fill="{color}"/>'
        f"</g>"
        f'<g fill="#fff" text-anchor="middle" '
        f'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        f'text-rendering="geometricPrecision" font-size="11">'
        f'<text x="{lx:.1f}" y="14">{esc_label}</text>'
        f'<text x="{mx:.1f}" y="14">{esc_msg}</text>'
        f"</g></svg>\n"
    )


def fetch_stars(token: str | None) -> str:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "TinyTitan-badges"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(int(data["stargazers_count"]))


def read_views() -> str:
    if not TRAFFIC_JSON.exists():
        return "0"
    data = json.loads(TRAFFIC_JSON.read_text(encoding="utf-8"))
    return str(data.get("message", "0"))


def read_last_commit() -> str:
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%cI"],
        text=True,
    ).strip()
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render self-hosted SVG badges.")
    parser.add_argument("--stars", help="Override star count instead of querying GitHub API.")
    parser.add_argument("--views", help="Override 14-day views count.")
    parser.add_argument("--last-commit", help="Override last-commit date string.")
    parser.add_argument("--token", help="Optional GitHub API bearer token.")
    args = parser.parse_args()

    stars = args.stars or fetch_stars(args.token)
    views = args.views or read_views()
    last_commit = args.last_commit or read_last_commit()

    BADGES_DIR.mkdir(parents=True, exist_ok=True)
    (BADGES_DIR / "stars.svg").write_text(render_badge("Stars", stars, "#e3b341"), encoding="utf-8")
    (BADGES_DIR / "views.svg").write_text(
        render_badge("Views (14d)", views, "#8a2be2"), encoding="utf-8"
    )
    (BADGES_DIR / "last-commit.svg").write_text(
        render_badge("Last Commit", last_commit, "#2ea44f"), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
