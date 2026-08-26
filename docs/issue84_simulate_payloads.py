#!/usr/bin/env python3
"""Issue #84 payload simulation.

Faithful Python port of GemmaToolCallParser (as of acefaf1/main, bugs
included) + the real gemma4 tokenizer. Sweeps candidate payloads a drifting
model might emit between <|tool_call> and <tool_call|> and reports which
match the observed diagnostics: exactly 14 payload tokens and a `malformed`
classification (not unknown_tool / not ok).
"""
import json
import re
from tokenizers import Tokenizer

TOK = Tokenizer.from_file(
    "/Users/andreymikhaylov/development/turbo-fieldfare/scratch/gemma4.gturbo/tokenizer/tokenizer.json")

NUMBER_RE = re.compile(r'^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$')


class Malformed(Exception):
    pass


class UnknownTool(Exception):
    pass


class Parser:
    """Port of the private Parser struct in GemmaToolCallParser.swift.

    Deliberately reproduces both known bugs:
    - identifier() stops at '-' and '.' (letters/digits/_ only)
    - jsonString() calls take(), whose skipWhitespace eats ws inside strings
    """

    def __init__(self, text):
        self.c = list(text)
        self.i = 0

    def at_end(self):
        return self.i == len(self.c)

    def skip_ws(self):
        while self.i < len(self.c) and self.c[self.i].isspace():
            self.i += 1

    def consume(self, literal):
        self.skip_ws()
        n = len(literal)
        if self.c[self.i:self.i + n] != list(literal):
            raise Malformed(f"expected {literal!r} at {self.i}")
        self.i += n

    def identifier(self):
        self.skip_ws()
        start = self.i
        while self.i < len(self.c):
            ch = self.c[self.i]
            if not (ch.isalpha() or ch.isdigit() or ch == '_'):
                break
            self.i += 1
        if self.i == start:
            raise Malformed("empty identifier")
        return ''.join(self.c[start:self.i])

    def starts(self, literal):
        return self.c[self.i:self.i + len(literal)] == list(literal)

    def take(self, ch):
        self.skip_ws()
        if self.i < len(self.c) and self.c[self.i] == ch:
            self.i += 1
            return True
        return False

    def take_word(self, w):
        if self.starts(w):
            self.i += len(w)
            return True
        return False

    def object(self):
        self.consume('{')
        result = {}
        self.skip_ws()
        if self.take('}'):
            return result
        while True:
            key = self.object_key()
            self.consume(':')
            result[key] = self.value()
            self.skip_ws()
            if self.take('}'):
                return result
            self.consume(',')

    def object_key(self):
        self.skip_ws()
        start = self.i
        while self.i < len(self.c):
            ch = self.c[self.i]
            if not (ch.isalpha() or ch.isdigit() or ch in '_-.$'):
                break
            self.i += 1
        if self.i == start:
            raise Malformed("empty object key")
        return ''.join(self.c[start:self.i])

    def value(self):
        self.skip_ws()
        if self.starts('<|"|>'):
            return self.gemma_string()
        if self.starts('"'):
            return self.json_string()
        if self.starts('{'):
            return self.object()
        if self.starts('['):
            return self.array()
        if self.take_word('true'):
            return True
        if self.take_word('false'):
            return False
        if self.take_word('null'):
            return None
        return self.number()

    def array(self):
        self.consume('[')
        result = []
        self.skip_ws()
        if self.take(']'):
            return result
        while True:
            result.append(self.value())
            self.skip_ws()
            if self.take(']'):
                return result
            self.consume(',')

    def gemma_string(self):
        self.consume('<|"|>')
        out = ''
        while not self.at_end():
            if self.starts('<|"|>'):
                self.consume('<|"|>')
                return out
            if self.c[self.i] == '\\' and self.i + 1 < len(self.c):
                self.i += 1
                out += self.escaped_fragment()
            else:
                out += self.c[self.i]
                self.i += 1
        raise Malformed("unterminated gemma string")

    def json_string(self):
        # BUG-FAITHFUL: take() skips whitespace each iteration, so spaces,
        # tabs and newlines inside the string are silently dropped.
        self.consume('"')
        out = ''
        while not self.at_end():
            if self.take('"'):
                return out
            if self.take('\\'):
                out += self.escaped_fragment()
            else:
                out += self.c[self.i]
                self.i += 1
        raise Malformed("unterminated json string")

    def escaped_fragment(self):
        if self.i >= len(self.c):
            raise Malformed("dangling escape")
        e = self.c[self.i]
        self.i += 1
        simple = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f',
                  'n': '\n', 'r': '\r', 't': '\t'}
        if e in simple:
            return simple[e]
        if e == 'u':
            first = self.unicode_unit()
            if 0xD800 <= first <= 0xDBFF:
                if not (self.c[self.i:self.i + 2] == ['\\', 'u']):
                    raise Malformed("bad surrogate pair")
                self.i += 2
                second = self.unicode_unit()
                if not (0xDC00 <= second <= 0xDFFF):
                    raise Malformed("bad low surrogate")
                scalar = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
            else:
                if 0xDC00 <= first <= 0xDFFF:
                    raise Malformed("lone low surrogate")
                scalar = first
            return chr(scalar)
        raise Malformed(f"bad escape {e!r}")

    def unicode_unit(self):
        if self.i + 4 > len(self.c):
            raise Malformed("short \\u")
        s = ''.join(self.c[self.i:self.i + 4])
        try:
            v = int(s, 16)
        except ValueError:
            raise Malformed("bad hex")
        self.i += 4
        return v

    def number(self):
        start = self.i
        while self.i < len(self.c) and self.c[self.i] in '-+0123456789.eE':
            self.i += 1
        if self.i == start:
            raise Malformed("expected value")
        lit = ''.join(self.c[start:self.i])
        if not NUMBER_RE.match(lit):
            raise Malformed(f"bad number {lit!r}")
        return float(lit)


def parse(text, allowed):
    if len(text.encode()) > 256 * 1024:
        return "oversized", None
    p = Parser(text)
    try:
        p.consume('call:')
        name = p.identifier()
        if name not in allowed:
            raise UnknownTool(name)
        args = p.object()
        p.skip_ws()
        if not p.at_end():
            raise Malformed("trailing content")
        return "ok", (name, args)
    except UnknownTool as e:
        return "unknown_tool", str(e)
    except Malformed as e:
        return "malformed", str(e)
    except IndexError:
        return "malformed", "index"


def ntok(text):
    return len(TOK.encode(text, add_special_tokens=False).ids)


# --- Candidate sweep -------------------------------------------------------
# Generic agent-tool names of varying lengths; the real Hermes names are
# unknown, so treat counts as a band, not an exact match.
NAMES = ["search", "read_file", "web_search", "list_files", "get_weather",
         "execute_command", "todo-write", "browser.open", "final_answer"]
ALLOWED = set(NAMES)

candidates = []

for name in NAMES:
    # 1. Canonical OpenAI JSON drift
    candidates += [
        ('canonical json, empty args', f'{{"name": "{name}", "arguments": {{}}}}'),
        ('canonical json, compact', f'{{"name":"{name}","arguments":{{}}}}'),
        ('canonical json, str args', f'{{"name": "{name}", "arguments": "{{}}"}}'),
        ('canonical json, one arg', f'{{"name": "{name}", "arguments": {{"query": "x"}}}}'),
        ('canonical json, tool key', f'{{"tool": "{name}", "args": {{}}}}'),
        ('bare args object', '{"query": "weather today"}'),
        # 2. Native dialect, near misses
        ('native, ok empty', f'call:{name}{{}}'),
        ('native, ok gemma str', f'call:{name}{{query:<|"|>x<|"|>}}'),
        ('native, json-quoted key', f'call:{name}{{"query":"x"}}'),
        ('native, capital C', f'Call:{name}{{}}'),
        ('native, missing colon', f'call {name}{{}}'),
        ('native, paren args', f'call:{name}("x")'),
        ('native, single quotes', f"call:{name}{{query:'x'}}"),
        ('native, trailing text', f'call:{name}{{}} done'),
        ('native, unquoted value', f'call:{name}{{query:x}}'),
        ('native, leading-zero num', f'call:{name}{{n:01}}'),
        # 3. Other drift formats
        ('python style', f'{name}(query="x")'),
        ('fenced json', f'```json\n{{"name": "{name}"}}\n```'),
        ('tool_code style', f'print({name}(query="x"))'),
        ('name colon args', f'{name}: {{"query": "x"}}'),
    ]

print(f'{"verdict":13} {"tok":>3}  fits  {"kind":28} payload')
print('-' * 100)
rows = []
for kind, text in candidates:
    verdict, detail = parse(text, ALLOWED)
    n = ntok(text)
    fits = '  *' if 13 <= n <= 15 and verdict == 'malformed' else '   '
    rows.append((verdict, n, fits, kind, text))

# Matches first, then by verdict.
for verdict, n, fits, kind, text in sorted(rows, key=lambda r: (r[2] != '  *', r[0])):
    print(f'{verdict:13} {n:>3} {fits}  {kind:28} {text[:60]!r}')

# --- Bug demos on valid native calls --------------------------------------
print('\n--- silent-corruption / latent-bug checks (valid native dialect) ---')
demos = [
    ('space inside json string', 'call:read_file{path:"/tmp/a b c"}'),
    ('gemma string keeps space', 'call:read_file{path:<|"|>/tmp/a b c<|"|>}'),
    ('hyphenated tool', 'call:todo-write{}'),
    ('dotted tool', 'call:browser.open{}'),
    ('backslash-space in string', 'call:read_file{path:"a\\ b"}'),
]
for kind, text in demos:
    verdict, detail = parse(text, ALLOWED)
    print(f'{verdict:13} {ntok(text):>3}       {kind:28} {text!r} -> {detail!r}')
