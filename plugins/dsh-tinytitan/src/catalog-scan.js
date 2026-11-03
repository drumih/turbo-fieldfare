/**
 * The model catalogue read straight from `models/`, with no server binary.
 *
 * `TinyTitanServer --catalog` is the authority on what an install is: it derives
 * the served id from the manifest, decides which families may be listed at all,
 * and knows which thinking levels each template renders. The plugin normally
 * asks it. A profile that has installed models but has not built the server yet
 * has nothing to ask, though, and the picker then showed an empty route rather
 * than the installs sitting on disk.
 *
 * So this module walks the folder itself, and it is a *mirror*: every rule below
 * is the rule in `sources/TinyTitanServer/Core/ModelCatalog.swift` (with the
 * manifest identity in `ManifestReader.peekIdentity` and the id spelling in
 * `ServerModelIdentity`). `test/catalog-scan.test.js` compares its rows against
 * the binary's own `--catalog` output on the same folder, so a divergence fails
 * a test rather than quietly changing what the picker offers. When the Swift
 * side changes, this file and that test change with it.
 *
 * The output is the same shape `--catalog` prints, so `catalogRows` consumes
 * either source unchanged.
 *
 * @module dsh-tinytitan/catalog-scan
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { basename, join } from "node:path";

/** Human names for the installs this project ships; a manifest carries an id. */
const DISPLAY_NAMES = Object.freeze({
  "qwen3.6-35b-a3b": "Qwen 3.6 35B-A3B",
  "ornith-1.5-35b-a3b": "Ornith 1.5 35B-A3B",
  "qwen-agentworld": "Qwen AgentWorld 35B-A3B",
  "kat-coder-v2.5": "KAT-Coder-V2.5-Dev 35B-A3B",
  "qwen3.8-flash-next": "Qwen 3.8 Flash Next 125B-A6B",
  "qwen4-omni-120b-think": "Qwen 4.0 Flash (Qwen4-Omni-120B-Think)",
  "qwen3.5-2b": "Qwen 3.5 2B",
  "qwen3.5-4b": "Qwen 3.5 4B",
  "qwen3.5-9b": "Qwen 3.5 9B",
});

/** The levels each family's chat template honours, `.off` first. */
const FAMILY_LEVELS = Object.freeze({
  qwen36: ["off", "on"],
  qwen36_mtp: ["off", "on"],
  qwen38flash: ["off", "low", "medium", "xhigh"],
  qwen38flash_mtp: ["off", "low", "medium", "xhigh"],
  qwen4flash: ["off", "low", "medium", "xhigh"],
  qwen4flash_mtp: ["off", "low", "medium", "xhigh"],
  qwen3_5_dense: ["off", "on"],
});

/** A draft head has no tokenizer and no layers of its own to run. */
const MTP_FAMILIES = new Set(["qwen36_mtp", "qwen38flash_mtp", "qwen4flash_mtp"]);

/** Families the GPU engine lists (the MTP ones above are excluded). */
const GPU_FAMILIES = new Set(["qwen36", "qwen38flash", "qwen4flash", "qwen3_5_dense"]);

/** The one family both engines implement, so its rows advertise both. */
const DENSE_FAMILY = "qwen3_5_dense";

/** `config.json`'s `model_type` values that mean the dense Qwen 3.5 shape. */
const CPU_FAMILY_ALIASES = Object.freeze({
  qwen3_5_dense: DENSE_FAMILY,
  qwen3_5_text: DENSE_FAMILY,
  qwen3_5: DENSE_FAMILY,
});

/** Where an id-less install's name comes from, per family. */
const FAMILY_BASE = Object.freeze({
  qwen36: "qwen3.6-35b-a3b",
  qwen36_mtp: "qwen3.6-35b-a3b-mtp",
  qwen38flash: "qwen3.8-flash-next",
  qwen38flash_mtp: "qwen3.8-flash-next-mtp",
  qwen4flash: "qwen4-omni-120b-think",
  qwen4flash_mtp: "qwen4-omni-120b-think-mtp",
  qwen3_5_dense: "qwen3.5-dense",
});

/** A manifest may already spell the width; the suffix is added exactly once. */
const QUANT_SUFFIXES = ["-4bit", "-8bit", "-6bit"];

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

/** `ServerModelIdentity.base`: the id with any width the manifest spelled removed. */
function baseID(manifestModelID, family) {
  const lowered = manifestModelID.toLowerCase();
  for (const suffix of QUANT_SUFFIXES) {
    if (lowered.endsWith(suffix)) {
      return manifestModelID.slice(0, -suffix.length);
    }
  }
  if (manifestModelID !== "unknown/snapshot") return manifestModelID;
  return FAMILY_BASE[family] ?? manifestModelID;
}

/** `<tokenizer>/tokenizer.json`, or the override directory. */
function hasTokenizer(directory, env) {
  if (existsSync(join(directory, "tokenizer", "tokenizer.json"))) return true;
  const override = env?.TURBO_FIELDFARE_TOKENIZER_DIR;
  return Boolean(override) && existsSync(join(override, "tokenizer.json"));
}

/** A `.ssdai` install. Mirrors `ModelCatalog.probeInstall`. */
function probeInstall(directory, env) {
  let manifest;
  try {
    manifest = readJson(join(directory, "manifest.json"));
  } catch (error) {
    return { reason: `unreadable manifest.json: ${error.message}` };
  }
  if (typeof manifest?.modelID !== "string" || manifest.modelID === "") {
    return { reason: "manifest modelID is empty" };
  }
  const declared = manifest?.arch?.family;
  // A manifest that declares no known family was inferred from layer shape by
  // the server; the shape test needs the full arch, so an undeclared family is
  // read as the qwen3.6 default exactly as `peekIdentity` does.
  const family =
    typeof declared === "string" && FAMILY_LEVELS[declared] !== undefined ? declared : "qwen36";
  if (MTP_FAMILIES.has(family)) {
    return { reason: `an MTP draft head (${family}), served only beside its target` };
  }
  if (!GPU_FAMILIES.has(family)) return { reason: `unsupported family ${family}` };
  if (manifest?.arch?.hiddenActivation !== "silu") {
    return { reason: `hiddenActivation=${manifest?.arch?.hiddenActivation}` };
  }
  if (!hasTokenizer(directory, env)) {
    return { reason: "no tokenizer/tokenizer.json; the install is incomplete" };
  }
  const bits = Number(manifest?.quant?.routedExpert?.weightBits ?? 4);
  if (!Number.isFinite(bits) || bits <= 0) return { reason: "no routed-expert width" };
  const base = baseID(manifest.modelID, family);
  return {
    model: {
      id: `${base}_${bits}-Bit`,
      name: DISPLAY_NAMES[base] ?? base,
      family,
      quant: bits,
      backend: "gpu",
      engines: family === DENSE_FAMILY ? "gpu,cpu" : "gpu",
      path: directory,
      thinking: [...FAMILY_LEVELS[family]],
    },
  };
}

/** A converted safetensors snapshot. Mirrors `ModelCatalog.probeSnapshot`. */
function probeSnapshot(directory) {
  let config;
  try {
    config = readJson(join(directory, "config.json"));
  } catch (error) {
    return { reason: `unreadable config.json: ${error.message}` };
  }
  if (config === null || typeof config !== "object" || Array.isArray(config)) {
    return { reason: "config.json is not a JSON object" };
  }
  const modelType = typeof config.model_type === "string" ? config.model_type.toLowerCase() : null;
  const family = modelType === null ? undefined : CPU_FAMILY_ALIASES[modelType];
  if (family === undefined) {
    const named = modelType === null ? "an unnamed architecture" : `\`${config.model_type}\``;
    return { reason: `the CPU engine does not implement ${named}` };
  }
  const bits = config?.quantization?.bits;
  if (!Number.isInteger(bits)) {
    return {
      reason: "config.json has no quantization block; the CPU engine serves affine snapshots",
    };
  }
  for (const required of ["model.safetensors.index.json", "tokenizer.json"]) {
    if (!existsSync(join(directory, required))) {
      return { reason: `incomplete snapshot: no ${required}` };
    }
  }
  const declared =
    typeof config.model_id === "string" && config.model_id !== "" ? config.model_id : null;
  const id = declared ?? basename(directory);
  const display =
    typeof config.display_name === "string" && config.display_name !== ""
      ? config.display_name
      : null;
  return {
    model: {
      id,
      name: display ?? id,
      family,
      quant: bits,
      backend: "cpu",
      engines: "cpu",
      path: directory,
      thinking: [...FAMILY_LEVELS[family]],
    },
  };
}

/** Either shape the project ships, or the reason it is not one. */
function probe(directory, env) {
  if (existsSync(join(directory, "manifest.json"))) return probeInstall(directory, env);
  if (existsSync(join(directory, "config.json"))) return probeSnapshot(directory);
  return { reason: "neither manifest.json (a GPU install) nor config.json (a CPU snapshot)" };
}

/**
 * Every servable model under a `models/` folder, in the order the server lists
 * them (GPU first, then by id).
 *
 * @param directory - the folder to walk.
 * @param options - `env` for the tokenizer override; `readdir`/`stat` injectable
 *   for tests.
 * @returns `{ models, skipped }` — `models` is `--catalog`'s array, `skipped` is
 *   `{path, reason}` per directory that is not a model, for the caller to log.
 */
export function scanModelsFolder(
  directory,
  { env = process.env, readdir = readdirSync, stat = statSync } = {},
) {
  const models = [];
  const skipped = [];
  let children;
  try {
    children = readdir(directory);
  } catch (error) {
    return { models, skipped: [{ path: directory, reason: `cannot list: ${error.message}` }] };
  }
  for (const name of [...children].sort()) {
    // Hidden entries are the server's own caches and lock files, as in
    // `.skipsHiddenFiles`; a symlinked install is served from where it lives.
    if (name.startsWith(".")) continue;
    const child = join(directory, name);
    let childStat;
    try {
      childStat = stat(child);
    } catch {
      continue;
    }
    if (!childStat.isDirectory()) continue;
    const probed = probe(child, env);
    if (probed.model === undefined) {
      skipped.push({ path: child, reason: probed.reason });
      continue;
    }
    if (models.some((entry) => entry.id === probed.model.id)) {
      skipped.push({ path: child, reason: `duplicate id ${probed.model.id}` });
      continue;
    }
    models.push(probed.model);
  }
  models.sort((lhs, rhs) =>
    lhs.backend === rhs.backend
      ? lhs.id < rhs.id
        ? -1
        : lhs.id > rhs.id
          ? 1
          : 0
      : lhs.backend === "gpu"
        ? -1
        : 1,
  );
  return { models, skipped };
}
