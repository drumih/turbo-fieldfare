/**
 * The one harness release this plugin supports, and how the running one is read.
 *
 * A harness upgrade is not cosmetic here. This plugin generates an agent preset
 * from the *shipped* `standard` composition, so its row ids and the way presets
 * are declared move under it — 0.2.1-alpha.2 is the release that dropped the
 * disabled `tool-ralph` row from the standard (ralph moved to the experimental
 * ralph bundle, which {@link module:dsh-tinytitan/preset} now appends when
 * autonomy asks for the loop) — and it mounts a compaction backend that
 * subclasses `dsh-compaction-basic`, so that engine's internals move under it
 * too.
 *
 * The supported range is therefore a single release. Older ones, later ones and
 * a harness built from `main` are out of support deliberately, not assumed to
 * work. `test/support.test.js` asserts {@link SUPPORTED_DSH_VERSION} against the
 * two places that cannot import it — this package's `peerDependencies` and its
 * own `version` — so widening one without the others fails the suite.
 *
 * Detection is best-effort and answers `null` when it cannot tell, which
 * {@link supportDecision} reads as a refusal rather than as permission: this
 * plugin writes into the harness home, so an unreadable version is not a reason
 * to proceed.
 *
 * @module dsh-tinytitan/support
 */
import { createRequire } from "node:module";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { dirname, join, parse } from "node:path";

/** The one DeepSeek Harness release this plugin is written and tested against. */
export const SUPPORTED_DSH_VERSION = "0.2.2-alpha.3";

/** The package whose version is the harness's. */
const HARNESS = "@deepseek-ai/dsh";

/**
 * A declared peer of this plugin, used as a second anchor.
 *
 * It is installed beside the harness in every layout this plugin runs in, so
 * walking out from a resolved file inside it reaches the harness package even
 * where `@deepseek-ai/dsh` itself does not resolve from the plugin's location —
 * which pnpm layouts routinely make true.
 */
const ANCHOR = "@deepseek-ai/dsh-compaction-basic";

/** Read and parse one `package.json`, or `undefined`. */
function readPackage(path) {
  if (!existsSync(path)) return undefined;
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"));
    return parsed?.name ? parsed : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Walk up from a file looking for the package that owns it.
 *
 * `require.resolve` on a `package.json` is refused whenever a package declares
 * `exports` — the harness does — so the walk is the reliable route.
 *
 * @param start - file to start from.
 * @param wanted - the package name to match, or `undefined` for any.
 * @returns the parsed `package.json`, or `undefined`.
 */
export function packageFrom(start, wanted) {
  let directory = dirname(start);
  for (let depth = 0; depth < 10; depth += 1) {
    const found = readPackage(join(directory, "package.json"));
    if (found && (!wanted || found.name === wanted)) return found;
    const parent = dirname(directory);
    if (parent === directory || parent === parse(directory).root) return undefined;
    directory = parent;
  }
  return undefined;
}

/**
 * Find `@deepseek-ai/<shortName>` beside an already-resolved sibling.
 *
 * A pnpm layout resolves `@deepseek-ai/dsh-compaction-basic` to a real path
 * inside `.pnpm`, where walking up never reaches `dsh`; the two packages do sit
 * side by side in the same scope directory, which is what this looks for.
 *
 * @param from - a resolved file inside the sibling.
 * @param shortName - the package to find, e.g. `dsh`.
 * @returns the parsed `package.json`, or `undefined`.
 */
export function siblingPackage(from, shortName) {
  let directory = dirname(from);
  for (let depth = 0; depth < 10; depth += 1) {
    const found = readPackage(join(directory, "@deepseek-ai", shortName, "package.json"));
    if (found) return found;
    const parent = dirname(directory);
    if (parent === directory || parent === parse(directory).root) return undefined;
    directory = parent;
  }
  return undefined;
}

/**
 * The harness version this plugin is running inside, or `null`.
 *
 * There is no host service carrying it, so several anchors are tried and the
 * first that answers wins. `DSH_VERSION` is checked first because a launcher
 * that pins a release knows the answer better than any directory walk.
 *
 * @param options - `{ require, argv, env }`, injectable for tests.
 * @returns the version string, or `null`.
 */
export function dshVersion({ require: load, argv = process.argv, env = process.env } = {}) {
  if (env?.DSH_VERSION) return String(env.DSH_VERSION);

  let resolver;
  try {
    resolver = load ?? createRequire(import.meta.url);
  } catch {
    return null;
  }

  // The surest anchor: a resolved harness entry point.
  try {
    const found = packageFrom(resolver.resolve(HARNESS), HARNESS);
    if (found?.version) return found.version;
  } catch {
    // Not resolvable from the plugin's own location.
  }

  // Then the sibling scope directory, which usually *is* resolvable.
  try {
    const found = siblingPackage(resolver.resolve(ANCHOR), "dsh");
    if (found?.version) return found.version;
  } catch {
    // No sibling either.
  }

  // Finally the running harness's own entry point, when it is a file we can walk.
  const entry = argv?.[1];
  if (entry) {
    try {
      const found = packageFrom(realpathSync(entry), HARNESS);
      if (found?.version) return found.version;
    } catch {
      // argv[1] can be a launcher symlink that does not resolve; fall through.
    }
  }
  return null;
}

/**
 * Whether this plugin runs on the harness it was handed, and why not when it does not.
 *
 * The plugin writes into the harness home — a route block, an agent preset — so
 * an unverified harness gets **nothing** from it rather than a best guess. Both
 * a version we can read as different and a version we cannot read at all are a
 * refusal: failing closed is the only answer that keeps "supports
 * `0.1.6-alpha.2`" a statement about the product rather than about our luck.
 *
 * A refusal is a **return, never a throw**. This plugin mounts into somebody
 * else's profile, so the harness must boot, every other plugin must load, and
 * removing this one must leave a working installation with nothing to undo. The
 * refusal is reported through the host logger, which is why it is a decision
 * object rather than an exception: the caller reports and returns.
 *
 * @param version - a version from {@link dshVersion}.
 * @param supported - the supported release, for tests.
 * @returns `{ run, refusal }` — `refusal` is one line, or `null` when it runs.
 */
export function supportDecision(version, supported = SUPPORTED_DSH_VERSION) {
  if (version === supported) return { run: true, refusal: null };
  const readable = typeof version === "string" && version.length > 0;
  const what = readable
    ? `DeepSeek Harness ${version} is not supported`
    : "the DeepSeek Harness version could not be read";
  return {
    run: false,
    refusal:
      `dsh-tinytitan: ${what}, and this plugin supports ${supported} exactly — ` +
      "not older, not newer, and not a build from main. It writes a route and an " +
      "agent preset into your harness home, so it is not running here. DSH itself " +
      `is unaffected and keeps working; pin the harness to ${supported}, or remove ` +
      "this plugin.",
  };
}
