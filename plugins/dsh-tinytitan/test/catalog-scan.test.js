/**
 * The folder scan must describe a models directory exactly as the server does.
 *
 * The scan exists so a profile that has installed models but has not built the
 * server still gets a picker. That only works if its rows are the server's rows,
 * so the first test below compares the block built from the scan with the block
 * built from `TinyTitanServer --catalog` on the same folder: the mirror is the
 * feature, and a divergence has to fail here rather than change what the picker
 * offers.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { scanModelsFolder } from "../src/catalog-scan.js";
import { catalogRows, generateBlock } from "../src/generate.js";

const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const MODELS_DIR = join(REPO_ROOT, "models");
const BINARY = join(REPO_ROOT, ".build", "release", "TinyTitanServer");

/** A `.ssdai` install the way the repacker writes one, minus the weights. */
function writeInstall(
  root,
  name,
  { modelID, family, bits, activation = "silu", tokenizer = true } = {},
) {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, "manifest.json"),
    JSON.stringify({
      magic: "SSDAI",
      modelID,
      quant: { routedExpert: { weightBits: bits } },
      arch: { family, hiddenActivation: activation },
    }),
  );
  if (tokenizer) {
    mkdirSync(join(directory, "tokenizer"), { recursive: true });
    writeFileSync(join(directory, "tokenizer", "tokenizer.json"), "{}");
  }
  return directory;
}

/** A converted safetensors snapshot. */
function writeSnapshot(
  root,
  name,
  { modelType = "qwen3_5", bits = 8, modelID, displayName, complete = true } = {},
) {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, "config.json"),
    JSON.stringify({
      model_type: modelType,
      quantization: bits === undefined ? undefined : { bits },
      model_id: modelID,
      display_name: displayName,
    }),
  );
  if (complete) {
    writeFileSync(join(directory, "model.safetensors.index.json"), "{}");
    writeFileSync(join(directory, "tokenizer.json"), "{}");
  }
  return directory;
}

test("the scan and the server catalog produce the same picker block", (t) => {
  if (!existsSync(BINARY) || !existsSync(MODELS_DIR)) {
    t.skip("no built TinyTitanServer or models/ here; skipping the parity check");
    return;
  }
  const printed = execFileSync(BINARY, ["--catalog", "--models-dir", MODELS_DIR], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const fromServer = JSON.parse(String(printed)).models;
  const fromFolder = scanModelsFolder(MODELS_DIR, { env: {} }).models;
  assert.equal(generateBlock(catalogRows(fromFolder)), generateBlock(catalogRows(fromServer)));
  assert.ok(fromFolder.length > 0, "the parity check needs at least one installed model");
});

test("a folder is read the way the server reads it", () => {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-scan-"));
  writeInstall(root, "qwen3.5_2B_4Bit", {
    modelID: "qwen3.5-2b",
    family: "qwen3_5_dense",
    bits: 4,
  });
  writeInstall(root, "qwen3.5_2B_8Bit", {
    modelID: "qwen3.5-2b",
    family: "qwen3_5_dense",
    bits: 8,
  });
  // An id the manifest already spelled a width into: the suffix is added once.
  writeInstall(root, "kat_4Bit", { modelID: "kat-coder-v2.5-4bit", family: "qwen36", bits: 4 });
  // An id outside the display-name table is listed by its id.
  writeInstall(root, "custom_4Bit", { modelID: "custom-model", family: "qwen36", bits: 4 });
  // The 4.0 Flash family lists like the 3.8 one, under its own name.
  writeInstall(root, "qwen4_4Bit", {
    modelID: "qwen4-omni-120b-think-4bit",
    family: "qwen4flash",
    bits: 4,
  });
  // Not models, for one reason each.
  writeInstall(root, "mtp_4Bit", {
    modelID: "qwen3.8-flash-next-mtp",
    family: "qwen38flash_mtp",
    bits: 4,
  });
  writeInstall(root, "incomplete_4Bit", {
    modelID: "half",
    family: "qwen36",
    bits: 4,
    tokenizer: false,
  });
  writeInstall(root, "gguf_4Bit", {
    modelID: "wrong-activation",
    family: "qwen36",
    bits: 4,
    activation: "gelu",
  });
  writeSnapshot(root, "qwen3.5_9B_8Bit", { modelID: "qwen3.5-9b", displayName: "Qwen 3.5 9B" });
  writeSnapshot(root, "llama_4Bit", { modelType: "llama" });
  writeSnapshot(root, "halfwritten_4Bit", { complete: false });
  mkdirSync(join(root, "loose-file-dir"), { recursive: true });
  mkdirSync(join(root, ".cache"), { recursive: true });
  writeFileSync(join(root, "qwen3.5_2B_4Bit.install.lock"), "");

  const { models, skipped } = scanModelsFolder(root, { env: {} });

  assert.deepEqual(
    models.map((model) => [
      model.id,
      model.backend,
      model.quant,
      model.engines,
      model.thinking.join(","),
    ]),
    [
      // GPU entries first, by id; then the CPU snapshot.
      ["custom-model_4-Bit", "gpu", 4, "gpu", "off,on"],
      ["kat-coder-v2.5_4-Bit", "gpu", 4, "gpu", "off,on"],
      ["qwen3.5-2b_4-Bit", "gpu", 4, "gpu,cpu", "off,on"],
      ["qwen3.5-2b_8-Bit", "gpu", 8, "gpu,cpu", "off,on"],
      ["qwen4-omni-120b-think_4-Bit", "gpu", 4, "gpu", "off,low,medium,xhigh"],
      ["qwen3.5-9b", "cpu", 8, "cpu", "off,on"],
    ],
  );
  assert.deepEqual(
    models.map((model) => model.name),
    [
      "custom-model",
      "KAT-Coder-V2.5-Dev 35B-A3B",
      "Qwen 3.5 2B",
      "Qwen 3.5 2B",
      "Qwen 4.0 Flash (Qwen4-Omni-120B-Think)",
      "Qwen 3.5 9B",
    ],
  );
  // Every directory that is not a model says why, so the caller can log it.
  assert.deepEqual(skipped.map((entry) => entry.path.split("/").pop()).sort(), [
    "gguf_4Bit",
    "halfwritten_4Bit",
    "incomplete_4Bit",
    "llama_4Bit",
    "loose-file-dir",
    "mtp_4Bit",
  ]);
  for (const entry of skipped) assert.ok(entry.reason.length > 0, entry.path);
  assert.ok(skipped.find((entry) => entry.path.endsWith("mtp_4Bit")).reason.includes("MTP"));
  assert.ok(
    skipped.find((entry) => entry.path.endsWith("incomplete_4Bit")).reason.includes("tokenizer"),
  );
  assert.ok(skipped.find((entry) => entry.path.endsWith("llama_4Bit")).reason.includes("llama"));
});

test("two directories claiming one id list the first and skip the second", () => {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-dupe-"));
  writeInstall(root, "a_4Bit", { modelID: "same-model", family: "qwen36", bits: 4 });
  writeInstall(root, "b_4Bit", { modelID: "same-model", family: "qwen36", bits: 4 });

  const { models, skipped } = scanModelsFolder(root, { env: {} });
  assert.equal(models.length, 1);
  assert.equal(models[0].id, "same-model_4-Bit");
  assert.equal(skipped.length, 1);
  assert.match(skipped[0].reason, /duplicate id same-model_4-Bit/);
});

test("an unreadable folder is reported rather than thrown", () => {
  const { models, skipped } = scanModelsFolder(join(tmpdir(), "dsh-tinytitan-absent-dir"));
  assert.deepEqual(models, []);
  assert.equal(skipped.length, 1);
  assert.match(skipped[0].reason, /cannot list/);
});
