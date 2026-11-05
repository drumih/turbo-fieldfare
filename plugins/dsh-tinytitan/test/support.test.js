/**
 * The supported harness release, the gate that enforces it, and the two
 * declarations that must agree with it.
 *
 * `package.json` cannot import a constant and a package version is not a shell
 * script, so the agreement is asserted here rather than derived. That is the
 * point of the file: widening the peer range, or moving the launcher's pin, has
 * to fail the suite instead of drifting quietly away from the release this
 * plugin was written against.
 *
 * The gate itself is tested for the three things that make it safe to ship:
 * it refuses, it refuses *without throwing*, and it refuses without touching
 * anything — no config resolution, no writes.
 */
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { resolveConfig } from "../src/config.js";
import { apply } from "../src/index.js";
import {
  SUPPORTED_DSH_VERSION,
  dshVersion,
  packageFrom,
  siblingPackage,
  supportDecision,
} from "../src/support.js";

const MANIFEST = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));

/** The harness packages this plugin declares. */
const DSH_PEERS = [
  "@deepseek-ai/dsh-agent-preset",
  "@deepseek-ai/dsh-compaction-basic",
  "@deepseek-ai/dsh-web-app",
];

/** A version we positively read as *not* the supported one. */
const OTHER_VERSIONS = [
  "0.1.5-rc.2",
  "0.1.6-alpha.1",
  "0.1.6-alpha.3",
  "0.1.6-rc.1",
  "0.1.6",
  "0.2.0-rc.2",
  "0.2.1-alpha.2",
  "0.2.2-alpha.3",
  "0.0.0-development",
];

/** Run `apply` under `version`, with every side effect off, capturing both sinks. */
function applyAs(version, config = {}) {
  const lines = [];
  const stderr = [];
  const realError = console.error;
  console.error = (message) => stderr.push(String(message));
  const previous = process.env.DSH_VERSION;
  if (version === undefined) delete process.env.DSH_VERSION;
  else process.env.DSH_VERSION = version;
  try {
    const result = apply(
      {},
      {
        registerRoute: false,
        writeCompactionPreset: false,
        watchModels: false,
        log: (message) => lines.push(String(message)),
        ...config,
      },
    );
    return { lines, result, stderr };
  } finally {
    console.error = realError;
    if (previous === undefined) delete process.env.DSH_VERSION;
    else process.env.DSH_VERSION = previous;
  }
}

test("the plugin declares exactly the one supported harness release", () => {
  for (const peer of DSH_PEERS) {
    const spec = MANIFEST.peerDependencies[peer];
    assert.equal(spec, SUPPORTED_DSH_VERSION, `${peer} must pin the supported release`);
    assert.doesNotMatch(spec, /[\^~*]|\|\||[<>]=?/, `${peer} must be exact, not a range`);
  }
});

test("the bundle's own version is the release it supports", () => {
  // The third place this release is declared: the package version an operator
  // reads off `dsh plugin list` and the profile pins. Tied here rather than left
  // to a release script, because bumping one without the other is the drift this
  // file's header is about.
  assert.equal(MANIFEST.version, SUPPORTED_DSH_VERSION);
});

test("an exact version is what npm treats as exact", () => {
  assert.match(SUPPORTED_DSH_VERSION, /^\d+\.\d+\.\d+/);
});

test("the supported release runs", () => {
  assert.deepEqual(supportDecision(SUPPORTED_DSH_VERSION), { run: true, refusal: null });
});

test("another release is refused, by name, in one line", () => {
  for (const version of OTHER_VERSIONS) {
    const { run, refusal } = supportDecision(version);
    assert.equal(run, false, `${version} must be refused`);
    assert.match(refusal, new RegExp(version.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.match(refusal, new RegExp(SUPPORTED_DSH_VERSION.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.doesNotMatch(refusal, /\n/, "a refusal is one line");
  }
});

test("an unreadable version is refused too, and says so", () => {
  for (const unknown of [null, undefined, ""]) {
    const { run, refusal } = supportDecision(unknown);
    assert.equal(run, false, `${String(unknown)} must be refused, not assumed`);
    assert.match(refusal, /could not be read/);
  }
});

test("every refusal promises the rest of DSH keeps working", () => {
  for (const version of [...OTHER_VERSIONS, null]) {
    assert.match(supportDecision(version).refusal, /DSH itself is unaffected and keeps working/);
    assert.match(supportDecision(version).refusal, /remove this plugin/);
  }
});

test("an unsupported harness is refused without throwing", () => {
  for (const version of [...OTHER_VERSIONS, undefined]) {
    const { lines, result } = applyAs(version);
    assert.equal(lines.length, 1, `${String(version)} must produce exactly one line`);
    assert.equal(result?.refused, true);
  }
});

test("a refusal reaches stderr and the host error channel, never info", () => {
  // Verified against a real harness in a throwaway home, and against
  // `dsh-app-boot`: its startup exporter is registered with
  // `levels: { default: 2 }` and its `startupLogs` are printed only when the boot
  // itself fails. So on a healthy boot *nothing* a plugin logs is printed — an
  // `info` refusal would leave the plugin silently absent, and even the host
  // error channel alone is not enough.
  const infos = [];
  const errors = [];
  const stderr = [];
  const realError = console.error;
  console.error = (m) => stderr.push(String(m));
  const ctx = {
    logger: { info: (m) => infos.push(String(m)), error: (m) => errors.push(String(m)) },
  };
  const previous = process.env.DSH_VERSION;
  process.env.DSH_VERSION = "0.1.6";
  try {
    apply(ctx, { registerRoute: false, writeCompactionPreset: false, watchModels: false });
  } finally {
    console.error = realError;
    if (previous === undefined) delete process.env.DSH_VERSION;
    else process.env.DSH_VERSION = previous;
  }
  assert.equal(stderr.length, 1, "stderr is the channel a healthy boot does not swallow");
  assert.equal(errors.length, 1, "the host logger still gets the record");
  assert.equal(infos.length, 0, "a refusal must not go where the harness filters it out");
  assert.match(stderr[0], /not supported/);
});

test("the refusal is on stderr for every rejected version", () => {
  for (const version of [...OTHER_VERSIONS, undefined]) {
    const { stderr } = applyAs(version);
    assert.equal(stderr.length, 1, `${String(version)} must be reported once on stderr`);
    assert.match(stderr[0], /not supported|could not be read/);
  }
});

test("a refusal does no work at all — not even resolving config", () => {
  const home = mkdtempSync(join(tmpdir(), "dsh-tinytitan-refused-"));
  // `port: 99999` makes `resolveConfig` throw, so reaching it would prove the
  // gate ran too late. Nothing may be created under `dshHome` either.
  const { lines, result } = applyAs("0.1.6", { dshHome: home, port: 99999 });
  assert.equal(result?.refused, true);
  assert.equal(lines.length, 1);
  assert.deepEqual(readdirSync(home), [], "a refusal must not write");
});

test("the supported harness is not refused", () => {
  assert.deepEqual(applyAs(SUPPORTED_DSH_VERSION).lines, []);
});

test("DSH_VERSION is the harness version when the launcher sets it", () => {
  assert.equal(dshVersion({ env: { DSH_VERSION: "9.9.9" }, argv: [], require: null }), "9.9.9");
});

test("the harness package is found by walking up from its own entry point", () => {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-support-"));
  const harness = join(root, "node_modules", "@deepseek-ai", "dsh");
  mkdirSync(join(harness, "lib"), { recursive: true });
  writeFileSync(
    join(harness, "package.json"),
    JSON.stringify({ name: "@deepseek-ai/dsh", version: "7.7.7" }),
  );
  assert.equal(packageFrom(join(harness, "lib", "bin.js"), "@deepseek-ai/dsh")?.version, "7.7.7");
  assert.equal(packageFrom(join(harness, "lib", "bin.js"), "some-other-package"), undefined);
});

test("the harness is found beside the peer this plugin declares", () => {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-support-"));
  const scope = join(root, "node_modules", "@deepseek-ai");
  mkdirSync(join(scope, "dsh"), { recursive: true });
  mkdirSync(join(scope, "dsh-compaction-basic", "lib"), { recursive: true });
  writeFileSync(
    join(scope, "dsh", "package.json"),
    JSON.stringify({ name: "@deepseek-ai/dsh", version: "8.8.8" }),
  );
  writeFileSync(
    join(scope, "dsh-compaction-basic", "package.json"),
    JSON.stringify({ name: "@deepseek-ai/dsh-compaction-basic" }),
  );
  const anchor = join(scope, "dsh-compaction-basic", "lib", "index.js");
  writeFileSync(anchor, "");
  assert.equal(siblingPackage(anchor, "dsh")?.version, "8.8.8");

  const load = {
    resolve: (specifier) => {
      if (specifier !== "@deepseek-ai/dsh-compaction-basic")
        throw new Error(`unexpected ${specifier}`);
      return anchor;
    },
  };
  assert.equal(dshVersion({ require: load, env: {}, argv: [] }), "8.8.8");
});

test("nothing is claimed when no anchor resolves", () => {
  const load = {
    resolve: () => {
      throw new Error("not resolvable");
    },
  };
  assert.equal(dshVersion({ require: load, env: {}, argv: [] }), null);
});

test("the plugin's config still resolves with the route and preset switches off", () => {
  const resolved = resolveConfig({
    registerRoute: false,
    writeCompactionPreset: false,
    watchModels: false,
  });
  assert.equal(resolved.registerRoute, false);
  assert.equal(resolved.writeCompactionPreset, false);
  assert.equal(resolved.watchModels, false);
});
