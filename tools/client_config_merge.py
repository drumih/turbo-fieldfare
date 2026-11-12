#!/usr/bin/env python3
"""Merge the TinyTitan provider into a coding client's JSONC settings file.

`tools/server_launcher.sh` calls this for the clients that have no environment
override for a provider (OpenCode, Zed), so the provider has to be written into
the user's own config. Everything else in that file is kept. Comments and
trailing commas are tolerated on read; the file is rewritten as plain JSON, which
both clients accept, and only when its content actually changes. Before any
write the current file is copied beside it: `<file>.tinytitan-backup` the first
time, `<file>.tinytitan-backup-<timestamp>` after that, so every edit has the
version it replaced on disk.

    opencode  the `tinytitan` provider block comes from
              plugins/opencode-tinytitan/opencode.json, the checked-in source of
              truth (`--source`); only its `options.baseURL` is set to the port
              this server listens on, and the launched model's `limit.context`
              to the server's --max-context. A launched model the file does not
              list gets a plain entry. A file that does not exist yet becomes a
              copy of the source. `tinytitan` is removed from
              `disabled_providers` -- in the target and in every `--also` file --
              and added to an existing `enabled_providers` allow-list, because
              either one hides the provider no matter what the block says.
    zed       language_models.openai_compatible.<provider> = {api_url, available_models}

Stays runnable by a stock macOS /usr/bin/python3 (3.9): the launcher calls plain
`python3`, and none of this needs the analysis stack.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time

PROVIDER = "tinytitan"


def strip_comments(text):
    """JSONC -> JSON: drop // and /* */ comments outside strings."""
    out, i, n, in_str, esc = [], 0, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i)
            i = n if j == -1 else j + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse(stripped):
    """Comment-free JSONC text -> value; trailing commas are tolerated."""
    text = re.sub(r",(\s*[}\]])", r"\1", stripped)
    return json.loads(text) if text.strip() else {}


def load_jsonc(path):
    """Return (data, raw, had_comments); data is None when the file is unusable.

    A missing file is an empty object. A file that does not parse, or is not a
    JSON object, is reported and left alone: rewriting it would lose the user's
    settings.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return {}, None, False
    stripped = strip_comments(raw)
    try:
        data = parse(stripped)
    except json.JSONDecodeError as error:
        print(f"  WARNING: {path} is not parseable ({error}); left untouched.", file=sys.stderr)
        return None, raw, False
    if not isinstance(data, dict):
        print(f"  WARNING: {path} is not a JSON object; left untouched.", file=sys.stderr)
        return None, raw, False
    return data, raw, stripped != raw


def backup(path):
    """Copy `path` beside itself before it is rewritten; return the copy's path."""
    target = f"{path}.tinytitan-backup"
    if os.path.exists(target):
        target = f"{path}.tinytitan-backup-{time.strftime('%Y%m%d-%H%M%S')}"
        suffix = 1
        while os.path.exists(target):
            suffix += 1
            target = f"{path}.tinytitan-backup-{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
    shutil.copy2(path, target)
    return target


def write_if_changed(path, data, raw, had_comments):
    """Write `data` as JSON when its content differs from what is on disk.

    The comparison is on the parsed values, so a file whose settings already say
    what is wanted is not reformatted, stripped of its comments, or backed up.
    """
    if raw is not None and parse(strip_comments(raw)) == data:
        return False
    rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if raw is not None:
        saved = backup(path)
        print(f"  Backed up {path} -> {saved}")
        if had_comments:
            print(f"  NOTE: {path} had comments; they are kept only in the backup.")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tinytitan-tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    os.replace(temporary, path)
    return True


def table(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def unhide_provider(data, path, provider):
    """Take `provider` out of disabled_providers; add it to an enabled_providers list."""
    disabled = data.get("disabled_providers")
    if isinstance(disabled, list) and provider in disabled:
        data["disabled_providers"] = [entry for entry in disabled if entry != provider]
        print(f"  Removed '{provider}' from disabled_providers in {path}")
    enabled = data.get("enabled_providers")
    if isinstance(enabled, list) and provider not in enabled:
        enabled.append(provider)
        print(f"  Added '{provider}' to enabled_providers in {path}")


def plain_model_entry(model, context, output):
    """An entry for a launched model the checked-in file does not list."""
    return {
        "name": model,
        "tool_call": True,
        "reasoning": True,
        "temperature": False,
        "attachment": False,
        "modalities": {"input": ["text"], "output": ["text"]},
        "limit": {"context": context, "output": output},
    }


def merge_opencode(args):
    with open(args.source, encoding="utf-8") as handle:
        source = json.load(handle)
    block = copy.deepcopy(source["provider"][PROVIDER])
    block.setdefault("options", {})["baseURL"] = args.base_url
    models = block.setdefault("models", {})
    if args.model in models:
        models[args.model].setdefault("limit", {})["context"] = args.context
    else:
        # The output cap is the one every checked-in model carries.
        output = next(iter(models.values()), {}).get("limit", {}).get("output", 32768)
        models[args.model] = plain_model_entry(args.model, args.context, output)

    data, raw, had_comments = load_jsonc(args.file)
    if data is None:
        return 1
    if raw is None:
        # No config yet: the checked-in file is the config, as its README says.
        data = copy.deepcopy(source)
        data["provider"][PROVIDER] = block
        data["model"] = f"{PROVIDER}/{args.model}"
    else:
        existing = table(table(data, "provider"), PROVIDER)
        # Models the user added under the provider survive; the checked-in ones win.
        kept = existing.get("models") if isinstance(existing.get("models"), dict) else {}
        merged_models = {key: value for key, value in kept.items() if key not in block["models"]}
        merged_models.update(block["models"])
        existing.clear()
        existing.update(block)
        existing["models"] = merged_models
    unhide_provider(data, args.file, PROVIDER)
    if write_if_changed(args.file, data, raw, had_comments):
        print(f"  OpenCode configured: {args.file}")
    else:
        print(f"  OpenCode already configured: {args.file}")

    for other in args.also:
        if os.path.abspath(other) == os.path.abspath(args.file) or not os.path.exists(other):
            continue
        other_data, other_raw, other_comments = load_jsonc(other)
        if other_data is None:
            continue
        unhide_provider(other_data, other, PROVIDER)
        write_if_changed(other, other_data, other_raw, other_comments)
    return 0


def merge_zed(args):
    data, raw, had_comments = load_jsonc(args.file)
    if data is None:
        return 1
    block = {
        "api_url": args.base_url,
        "available_models": [
            {
                "name": args.model,
                "display_name": f"TinyTitan — {args.model}",
                "max_tokens": args.context,
                "capabilities": {
                    "tools": True,
                    "images": False,
                    "parallel_tool_calls": False,
                    "prompt_cache_key": False,
                    "chat_completions": True,
                    "interleaved_reasoning": False,
                    "max_tokens_parameter": False,
                },
            }
        ],
    }
    table(table(data, "language_models"), "openai_compatible")[PROVIDER] = block
    write_if_changed(args.file, data, raw, had_comments)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schema", choices=("opencode", "zed"))
    parser.add_argument("--file", required=True, help="the client config to merge into")
    parser.add_argument("--model", required=True, help="the served model id")
    parser.add_argument("--base-url", required=True, help="http://127.0.0.1:<port>/v1")
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--source", help="opencode: plugins/opencode-tinytitan/opencode.json")
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        help="opencode: another config file OpenCode reads; only disabled_providers is fixed there",
    )
    args = parser.parse_args(argv)
    if args.schema == "opencode":
        if not args.source:
            parser.error("opencode needs --source")
        return merge_opencode(args)
    return merge_zed(args)


if __name__ == "__main__":
    sys.exit(main())
