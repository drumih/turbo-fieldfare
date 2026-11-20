/**
 * The keep-going switch: a manual prompt becomes a goal so the harness's own
 * round driver continues it, exactly as `/goal <prompt>` would.
 *
 * What this file pins is mostly what the switch must *not* do — feed on the
 * driver's own round, arm a goal for a subagent, replace an unfinished goal,
 * take a boot down when the harness has no goal service, or turn a prompt with
 * no text into an objective. Each of those would either loop a session or
 * silently change what a person asked for.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { resolveConfig } from "../src/config.js";
import { AUTONOMY_DEFAULT_ROUNDS } from "../src/preset.js";
import {
  DEFAULT_AUTO_GOAL_ROUNDS,
  GOALS_SERVICE,
  installAutoGoal,
  isDirectHuman,
  messageText,
} from "../src/keep-going.js";

/** A goal service stub that records creates and reports one current goal. */
function goalsService({ current } = {}) {
  const created = [];
  return {
    created,
    get: () => current,
    create: (agent, request) => {
      created.push({ agent, request });
      return { id: "goal-1", maxGoalRounds: request.maxGoalRounds };
    },
  };
}

/** A context whose events are recorded and whose `inject` resolves immediately. */
function context(services = {}, { inject = true } = {}) {
  const listeners = {};
  const ctx = {
    get: (name) => services[name],
    on: (name, handler) => {
      listeners[name] = handler;
    },
    listeners,
    logger: {},
  };
  if (inject) ctx.inject = (_names, run) => run(ctx);
  return ctx;
}

/** Fire the inbox event the switch listens to. */
function insert(ctx, agent, message) {
  ctx.listeners["agent/inbox/inserted"]({ agent, message });
}

/** A direct human message with text. */
function human(text, id = "m1") {
  return { id, content: [{ type: "text", text }], source: { kind: "user" } };
}

test("messageText reads a string, parts, and an object alike", () => {
  assert.equal(messageText("  ship it  "), "ship it");
  assert.equal(messageText([{ type: "text", text: "ship" }, { type: "image" }]), "ship");
  assert.equal(messageText([{ type: "text", text: "a" }, { text: "b" }]), "a\nb");
  assert.equal(messageText({ text: " plain " }), "plain");
  assert.equal(messageText([{ type: "image" }]), "");
  assert.equal(messageText(undefined), "");
});

test("isDirectHuman accepts a person's message and refuses everything else", () => {
  assert.equal(isDirectHuman({ source: { kind: "user" } }), true);
  assert.equal(isDirectHuman({ source: { kind: "goal", round: 1 } }), false);
  assert.equal(isDirectHuman({ source: { kind: "user", goalId: "goal-1" } }), false);
  assert.equal(isDirectHuman({ source: { kind: "agent" } }), false);
  assert.equal(isDirectHuman({}), false);
});

test("the switch is off unless the profile turns it on", () => {
  const ctx = context({ [GOALS_SERVICE]: goalsService() });
  const status = installAutoGoal({
    ctx,
    resolved: { autoGoal: false, autoGoalRounds: 12 },
    log: () => {},
  });
  assert.deepEqual(status, { installed: false, reason: "disabled" });
  assert.equal(ctx.listeners["agent/inbox/inserted"], undefined);
});

test("a harness with no goal service logs why and leaves boot alone", () => {
  const lines = [];
  const ctx = context({}, { inject: false });
  const status = installAutoGoal({
    ctx,
    resolved: { autoGoal: true, autoGoalRounds: 12 },
    log: (line) => lines.push(line),
  });
  assert.equal(status.installed, false);
  assert.equal(status.reason, "no-goals-service");
  assert.ok(lines.some((line) => line.includes("no goal service")));
});

test("a direct human prompt creates and arms a goal with the configured cap", () => {
  const goals = goalsService();
  const ctx = context({ [GOALS_SERVICE]: goals });
  const lines = [];
  const status = installAutoGoal({
    ctx,
    resolved: { autoGoal: true, autoGoalRounds: 7 },
    log: (line) => lines.push(line),
  });
  assert.equal(status.installed, true);
  insert(ctx, {}, human("write the migration and run it"));
  assert.equal(goals.created.length, 1);
  assert.deepEqual(goals.created[0].request, {
    objective: "write the migration and run it",
    maxGoalRounds: 7,
  });
  assert.ok(lines.some((line) => line.includes("armed goal-1")));
});

test("the default cap keeps an unattended goal going without a small budget", () => {
  assert.equal(DEFAULT_AUTO_GOAL_ROUNDS, 99999);
  const goals = goalsService();
  const ctx = context({ [GOALS_SERVICE]: goals });
  installAutoGoal({ ctx, resolved: { autoGoal: true }, log: () => {} });
  insert(ctx, {}, human("answer this"));
  assert.equal(goals.created[0].request.maxGoalRounds, DEFAULT_AUTO_GOAL_ROUNDS);
});

test("a goal round never feeds the switch, and neither does a subagent", () => {
  const goals = goalsService();
  const ctx = context({ [GOALS_SERVICE]: goals });
  installAutoGoal({ ctx, resolved: { autoGoal: true }, log: () => {} });
  insert(ctx, {}, { id: "r1", content: "keep going", source: { kind: "goal", round: 1 } });
  insert(ctx, { parentAgent: { id: "parent" } }, human("delegate this", "m2"));
  insert(ctx, { session: { parentSession: "parent" } }, human("delegate that", "m3"));
  insert(ctx, { session: { delegationDepth: 1 } }, human("delegate more", "m4"));
  insert(ctx, {}, { id: "m5", content: [], source: { kind: "user" } });
  assert.equal(goals.created.length, 0);
});

test("an unfinished goal is left alone, a completed one is replaced", () => {
  for (const phase of ["active", "paused", "blocked"]) {
    const goals = goalsService({ current: { id: "goal-old", phase } });
    const ctx = context({ [GOALS_SERVICE]: goals });
    installAutoGoal({ ctx, resolved: { autoGoal: true }, log: () => {} });
    insert(ctx, {}, human("steer it"));
    assert.equal(goals.created.length, 0, `phase ${phase}`);
  }
  const goals = goalsService({ current: { id: "goal-old", phase: "complete" } });
  const ctx = context({ [GOALS_SERVICE]: goals });
  installAutoGoal({ ctx, resolved: { autoGoal: true }, log: () => {} });
  insert(ctx, {}, human("the next thing"));
  assert.equal(goals.created.length, 1);
});

test("a refusing service is logged, not thrown into the turn", () => {
  const lines = [];
  const ctx = context({
    [GOALS_SERVICE]: {
      get: () => undefined,
      create: () => {
        throw new Error('goal "goal-1" already exists with phase "active"');
      },
    },
  });
  installAutoGoal({ ctx, resolved: { autoGoal: true }, log: (line) => lines.push(line) });
  insert(ctx, {}, human("anything"));
  assert.ok(lines.some((line) => line.includes("could not arm a goal")));
});

test("service resolution is deferred through inject when it is not live yet", () => {
  const services = {};
  const ctx = context(services);
  const goals = goalsService();
  ctx.inject = (names, run) => {
    assert.deepEqual(names, [GOALS_SERVICE]);
    services[GOALS_SERVICE] = goals; // the service activates after this row
    return run(ctx);
  };
  const status = installAutoGoal({ ctx, resolved: { autoGoal: true }, log: () => {} });
  assert.deepEqual(status, { installed: true, deferred: true, rounds: DEFAULT_AUTO_GOAL_ROUNDS });
  insert(ctx, {}, human("deferred but working"));
  assert.equal(goals.created.length, 1);
});

test("a live goal service is used without deferring", () => {
  const ctx = context({ [GOALS_SERVICE]: goalsService() });
  let injected = 0;
  ctx.inject = () => {
    injected += 1;
  };
  const status = installAutoGoal({ ctx, resolved: { autoGoal: true }, log: () => {} });
  assert.equal(status.installed, true);
  assert.equal(status.deferred, undefined);
  assert.equal(injected, 0);
});

test("resolveConfig defaults the switch off and validates the cap", () => {
  const base = resolveConfig({ log: () => {} });
  assert.equal(base.autoGoal, false);
  assert.equal(base.autoGoalRounds, DEFAULT_AUTO_GOAL_ROUNDS);
  assert.equal(resolveConfig({ autoGoal: true, log: () => {} }).autoGoal, true);
  assert.equal(resolveConfig({ autoGoal: "true", log: () => {} }).autoGoal, false);
  assert.equal(
    resolveConfig({ autoGoal: true, autoGoalRounds: 30, log: () => {} }).autoGoalRounds,
    30,
  );
  assert.throws(() => resolveConfig({ autoGoalRounds: 0, log: () => {} }), /positive integer/);
  assert.throws(() => resolveConfig({ autoGoalRounds: -3, log: () => {} }), /positive integer/);
  assert.throws(() => resolveConfig({ autoGoalRounds: "many", log: () => {} }), /positive integer/);
});

test("resolveConfig carries the autonomy switches and validates their budget", () => {
  const base = resolveConfig({ log: () => {} });
  assert.equal(base.autonomy, false);
  assert.equal(base.autonomyRounds, AUTONOMY_DEFAULT_ROUNDS);
  assert.equal(base.autonomySuppressQuestions, false);
  const on = resolveConfig({
    autonomy: true,
    autonomyRounds: 30,
    autonomySuppressQuestions: true,
    log: () => {},
  });
  assert.equal(on.autonomy, true);
  assert.equal(on.autonomyRounds, 30);
  assert.equal(on.autonomySuppressQuestions, true);
  assert.equal(resolveConfig({ autonomy: "true", log: () => {} }).autonomy, false);
  assert.equal(
    resolveConfig({ autonomySuppressQuestions: "yes", log: () => {} }).autonomySuppressQuestions,
    false,
  );
  assert.throws(() => resolveConfig({ autonomyRounds: 0, log: () => {} }), /positive integer/);
  assert.throws(() => resolveConfig({ autonomyRounds: "lots", log: () => {} }), /positive integer/);
});
