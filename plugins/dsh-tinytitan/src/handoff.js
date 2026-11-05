/**
 * `dsh-tinytitan/handoff` — continue a task in a fresh context before the
 * session reaches its wall.
 *
 * `autoGoal` keeps one session working; two things still stop it — a turn that
 * ends on `max-tokens`, and a goal blocked because its round cap ran out. With
 * `handoff: true`, this module moves the work to a fresh context *while a turn
 * is running*, which is the only moment the harness lets a plugin do it: the
 * subagent provider runs under an initiating agent whose loop must be active.
 * Measured 2026-10-04 — at `turn/start` it publishes a `fork` child seeded from
 * the parent's history, with our prompt delivered verbatim; at
 * `assistant/message` (which commits after the turn) or `turn/end` it fails with
 * `agent loop is not active`.
 *
 * So the trigger is a budget, judged when a turn opens, against the prompt
 * tokens the previous step reported. The objective moves on before the window
 * fills, and a child that reaches the same budget moves it again. The budget
 * follows the route: with `handoffAtTokens` unset it is `handoffWindowRatio` of
 * the context window the routed model declares, but never within
 * `DEFAULT_HANDOFF_RESERVE_TOKENS` of the window's end — 131,072 on a 256K
 * route (the reserve binds), 600,000 on a 1,000,000-token route and 629,145 on a
 * 1,048,576-token one (the ratio binds) — because "still has room" is a fraction
 * of the route rather than a constant. An explicit `handoffAtTokens` pins one
 * number, and a route whose window cannot be resolved falls back to 120,000.
 *
 * The budget is the normal path; the wall itself is the fallback. A single turn
 * can jump past the window (a huge tool result) and end on `max-tokens`, and the
 * round driver's own disarm for that event leaves no next `turn/start` to hand
 * off from. The driver therefore watches `turn/end` and, when the wall was a
 * context wall (the reported prompt tokens reached the budget) and a hop is
 * still available, re-arms the goal with `goals.resume` — the only lever that
 * creates another turn, because `start` needs a running loop. That is bounded:
 * the disarm must be a context wall, not just an output cap; a hop must be
 * available; and each goal gets at most two re-arms, so a chain that cannot
 * start cannot loop on the wall forever.
 *
 * ## What a hop actually does
 *
 * Measured on a live run (2026-10-04), a forked child inherits the parent's
 * `goal/change` events — but *not* its activation, because activation is
 * process-local and never persisted (`GoalView.activation` is documented as
 * "process-local continuation eligibility"). The child therefore starts with the
 * same objective **disarmed**, its round driver queues nothing, and the hop
 * would buy exactly one turn. A handoff is only useful if it moves the work, so
 * one hop does three things atomically from the plugin's side:
 *
 * 1. disarms the handing-off session, which stops its rounds after the turn
 *    already in flight (otherwise the parent and every child would work the same
 *    objective in parallel);
 * 2. arms the child's inherited goal, so the child's own round driver continues
 *    the objective there;
 * 3. holds the child while its goal is actively continuing and releases it once
 *    that stops — completed, blocked, disarmed, or handed on further. The release
 *    rule is not "when the run settles": a fork child is a **one-shot** run whose
 *    `result` resolves at the end of its prompt turn, and disposing there (the
 *    first version did) killed the child before its round driver could queue the
 *    continuation, so a hop bought exactly one turn. The child is therefore
 *    released only at **idle**, and only when its goal is no longer active and
 *    armed. That also keeps a finished child from holding the app open — a
 *    headless run hung for its full timeout before any release rule existed.
 *
 * A hop that never starts rolls its claim back, so a refused start does not
 * spend the objective's budget.
 *
 * ## Why the hop count is keyed on the goal
 *
 * The first version keyed it on the child session and recorded it only after
 * `start()` resolved. A child's opening turn races that write, so every child
 * read "hop 1" and a zero budget forked until the process aborted (100+
 * sessions, SIGABRT). The count is therefore keyed on the **goal id** — one
 * identity that survives the fork seed — and the hop is **claimed before** the
 * await: the guard is a property of the work, not of a session that does not
 * exist yet. An in-flight set stops two triggers for the same goal starting two
 * children, and a live-children cap bounds the whole process.
 *
 * What this deliberately is **not**:
 *
 * - **Not a successor root session.** The successor is a child session: real,
 *   with its own log and context, but not one you steer like a new chat.
 * - **Not unbounded.** `handoffHops` caps the chain per goal, `0` is not a legal
 *   budget, `handoffMaxChildren` caps live children, and every hop is logged.
 * - **Not a way around a real blocker.** A goal the model blocked stays blocked;
 *   only an active, armed goal is handed on.
 *
 * A missing service, provider or live agent is a logged no-op.
 *
 * @module dsh-tinytitan/handoff
 */

/** The subagent provider a handoff uses: the child is seeded from the parent. */
export const HANDOFF_PROVIDER = "fork";

/** How many times one goal may be handed on before the chain stops. */
export const DEFAULT_HANDOFF_HOPS = 3;

/**
 * How many handoff children may be alive at once across the whole process.
 *
 * The per-goal cap bounds one chain; this bounds the process, which is the
 * shape the runaway took (many children, one abort). It counts every live
 * handoff agent, including the ancestors of a chain that is still running —
 * releasing an ancestor early would tear its own child down — so it has to sit
 * above `handoffHops`. The default matches the harness's own concurrent-subagent
 * default of 8.
 */
export const DEFAULT_HANDOFF_MAX_CHILDREN = 8;

/**
 * The fallback prompt-token budget, used only when the routed model's context
 * window cannot be resolved.
 *
 * The normal default is derived from the window (see
 * {@link DEFAULT_HANDOFF_WINDOW_RATIO}); this number is what a harness with no
 * `llm` service, or an adapter that declines to report a window, gets instead.
 * It is tuned for the 262,144-token route this project declares.
 */
export const DEFAULT_HANDOFF_AT_TOKENS = 120000;

/**
 * The fraction of the routed model's context window that starts a handoff.
 *
 * The budget has to stay below the harness's compaction trigger, or compaction
 * continues the session in place and the handoff never fires. That trigger is
 * `min(window x thresholdRatio, window - maxTokens - headroomTokens)`, and it is
 * *not* the same fraction on every shape: with this plugin's preset (32,768
 * output, 65,536 headroom, the harness's 0.8 ratio) a 262,144-token window
 * compacts at ~0.625 of it, while a 1M window compacts at 0.8. A ratio that
 * clears the narrowest shape therefore clears them all, and 0.6 is the highest
 * round one that does.
 *
 * It is also what a live 1M route was pinned to after measuring its sessions at
 * ~400,000 prompt tokens with compaction at ~800,000: `0.6 x 1,000,000` is
 * exactly that pin's 600,000. The reserve below is the second bound, and it is
 * the one that binds on the 256K shape.
 *
 * | declared window | auto budget | compaction trigger |
 * | --------------- | ----------- | ------------------ |
 * | 262,144 (256K)  | 131,072     | ~163,840           |
 * | 1,000,000       | 600,000     | ~800,000           |
 * | 1,048,576 (1M)  | 629,145     | ~838,860           |
 */
export const DEFAULT_HANDOFF_WINDOW_RATIO = 0.6;

/**
 * The room a hop must leave at the bottom of the window, in prompt tokens.
 *
 * {@link budgetForWindow} never spends this, so the compaction trigger always
 * has at least one full output turn in hand: a step whose prompt is below the
 * budget can still grow past the trigger before the next turn opens, and the
 * hop is only judged at that opening. The number is the preset's own
 * arithmetic — `2 x maxTokens (32,768) + headroom (65,536)` — which is exactly
 * what the 256K shape leaves between its 131,072 budget and its ~163,840
 * trigger. A window smaller than the reserve cannot host a safe hop at all; it
 * gets {@link DEFAULT_HANDOFF_AT_TOKENS}, which never fires on it.
 */
export const DEFAULT_HANDOFF_RESERVE_TOKENS = 131072;

/**
 * The budget one context window implies.
 *
 * Pure so the arithmetic can be pinned without a harness. The budget is the
 * smaller of `ratio` of the window and the window minus
 * {@link DEFAULT_HANDOFF_RESERVE_TOKENS}: the ratio keeps the parent working
 * while it still has room, and the reserve keeps the hop from being outrun by
 * one long step. An unknown or nonsensical window, and a window too small to
 * hold the reserve, fall back to {@link DEFAULT_HANDOFF_AT_TOKENS} rather than
 * handing off at the first turn.
 *
 * @param options - `window` in tokens (may be absent), `ratio`, `reserve`, `fallback`.
 * @returns a positive integer token budget.
 */
export function budgetForWindow({
  window,
  ratio,
  reserve = DEFAULT_HANDOFF_RESERVE_TOKENS,
  fallback = DEFAULT_HANDOFF_AT_TOKENS,
} = {}) {
  if (typeof window !== "number" || !Number.isFinite(window) || window <= 0) return fallback;
  const room = Math.floor(window - reserve);
  if (room < 1) return fallback;
  return Math.min(Math.floor(window * ratio), room);
}

/**
 * The user message a handoff child receives.
 *
 * The objective comes first and verbatim: the child's own goal is armed from
 * this text, and a goal whose first line is not the objective reads badly in
 * `/goal` and in the session list.
 *
 * @param objective - the objective the parent was working to.
 * @returns the prompt text.
 */
export function handoffPrompt(objective) {
  return [
    objective,
    "",
    "You are continuing this objective in a fresh context: the previous session reached its budget.",
    "The shared workspace is the authority — re-read the code and the task state rather than trusting any summary.",
    "Keep working until it is solved, tested and verified, and show the evidence.",
  ].join("\n");
}

/**
 * The prompt tokens one step's usage report represents.
 *
 * `TokenUsage` counts are disjoint: `inputTokens` is uncached input only and
 * cached input rides in `cacheReadTokens`/`cacheWriteTokens`, so reading
 * `inputTokens` alone undercounts a warmed context by the whole cache — measured
 * live as 137 against 5,454 cached. `totalTokens` is the fallback for an adapter
 * that reports only a total; it includes output, which is close enough for a
 * policy budget.
 *
 * @param usage - an `assistant/message` event's `data.usage`, if any.
 * @returns the token count, or 0 when the report carries none.
 */
export function usageTokens(usage) {
  if (!usage || typeof usage !== "object") return 0;
  const parts = [usage.inputTokens, usage.cacheReadTokens, usage.cacheWriteTokens].filter(
    (value) => typeof value === "number",
  );
  if (parts.length > 0) return parts.reduce((total, value) => total + value, 0);
  return typeof usage.totalTokens === "number" ? usage.totalTokens : 0;
}

/**
 * Start a fresh-context round for an unfinished objective.
 *
 * @param options - the harness context, the objective, the parent agent, a
 *   logger, and an abort controller for cancellation.
 * @returns a result naming the child session and carrying the run, or why
 *   nothing started.
 */
export async function startHandoff({
  ctx,
  objective,
  parent,
  log = () => {},
  controller = new AbortController(),
} = {}) {
  const agents = ctx?.get?.("agents");
  const subagents = ctx?.get?.("subagents");
  if (!agents || typeof agents.withInitiator !== "function") {
    return { started: false, reason: "no-agents-service" };
  }
  if (!subagents || typeof subagents.startActivation !== "function") {
    return { started: false, reason: "no-subagents-service" };
  }
  if (!objective || !parent) return { started: false, reason: "no-objective" };

  // `startActivation` runs under an initiating Agent, and its provider requires
  // the agent's loop to be active — which it is while a turn is running.
  return agents.withInitiator(parent, async () => {
    // 0.2.2-rc.1 replaced `start(provider, options)` with `startActivation(spec)`
    // and dropped `SubagentRun.localAgent`: the activation carries the child id,
    // and the live in-process Agent is read back from `agents` by that id. The
    // adapter below keeps the rest of this driver on the fields it already
    // tracks (id, result, dispose) instead of spreading the rename through it.
    const activation = await subagents.startActivation({
      provider: HANDOFF_PROVIDER,
      label: `handoff: ${objective.slice(0, 60)}`,
      request: {
        parent,
        prompt: [{ type: "text", text: handoffPrompt(objective) }],
      },
      signal: controller.signal,
      // The handing-off session is about to be disarmed; a settled message in
      // its inbox would wake work it no longer owns. The caller keeps the result.
      delivery: "caller",
    });
    const child = agents.get?.(activation.childId);
    const run = {
      id: activation.childId,
      localAgent: child,
      result: activation.result,
      dispose: () => activation.dispose(),
    };
    const childSession = run.id ?? child?.id ?? child?.session?.id;
    log(
      `dsh-tinytitan: handed the objective to child session ${childSession ?? "(unknown)"}` +
        `${child ? "" : " (remote run: no in-process child to arm)"}`,
    );
    return { started: true, child, childSession, run };
  });
}

/**
 * Install the handoff driver, when the profile asks for it.
 *
 * @param options - the harness context, the resolved plugin config, and a logger.
 * @returns a status object; `installed: false` carries the reason.
 */
export function installHandoff({ ctx, resolved, log = () => {} } = {}) {
  if (resolved?.handoff !== true) return { installed: false, reason: "disabled" };
  if (typeof ctx?.on !== "function") return { installed: false, reason: "no-context" };
  const hopBudget = resolved.handoffHops ?? DEFAULT_HANDOFF_HOPS;
  // `handoffAtTokens` pins one number; without it the budget follows the window
  // the routed model declares.
  const fixedBudget =
    Number.isSafeInteger(resolved.handoffAtTokens) && resolved.handoffAtTokens > 0
      ? resolved.handoffAtTokens
      : null;
  const windowRatio =
    typeof resolved.handoffWindowRatio === "number" && resolved.handoffWindowRatio > 0
      ? resolved.handoffWindowRatio
      : DEFAULT_HANDOFF_WINDOW_RATIO;
  const childCap = resolved.handoffMaxChildren ?? DEFAULT_HANDOFF_MAX_CHILDREN;

  /** Hops already claimed per goal, and the goals currently starting one. */
  const hops = new Map();
  const inFlight = new Set();
  /** The last prompt-token count each session reported. */
  const usage = new Map();
  /** Sessions that handed their objective on; they never work it again. */
  const retired = new Set();
  /** Wall recoveries per goal, so a failing chain cannot loop on `max-tokens`. */
  const recoveries = new Map();
  /** Live children, so the process cap and plugin disposal can see them. */
  const runs = new Set();
  /** Child session id -> { key, run, agent }, for the release rule. */
  const children = new Map();
  const controllers = new Set();

  const attach = (scoped = ctx) => {
    const agents = scoped.get?.("agents");
    const goals = scoped.get?.("goals");
    const subagents = scoped.get?.("subagents");
    if (!agents || !goals || !subagents) {
      log(
        "dsh-tinytitan: handoff is on but the harness has no agents/goals/subagents service; " +
          "mount the standard composition",
      );
      return { installed: false, reason: "no-services" };
    }

    const describe = (error) => (error instanceof Error ? error.message : String(error));

    /** Resolved context windows per route, routes in flight, routes announced. */
    const windows = new Map();
    const resolving = new Set();
    const announced = new Set();

    /**
     * The route one agent uses, or `null` when it cannot be named.
     *
     * 0.2.2-rc.1 fixes the step's route through `agent.model.current()`; the
     * live `agent.options` are only the fallback before a session records a
     * selection. Read the durable accessor first so a model switched in the
     * picker sizes this budget against the route the step actually uses.
     */
    const routeOf = (agent) => {
      let selected;
      try {
        selected = agent?.model?.current?.();
      } catch {
        selected = undefined;
      }
      const provider = selected?.provider ?? agent?.options?.provider ?? resolved.provider;
      const model = selected?.model ?? agent?.options?.model;
      if (typeof provider !== "string" || provider.length === 0) return null;
      if (typeof model !== "string" || model.length === 0) return null;
      return { provider, model, key: `${provider}/${model}` };
    };

    /** Ask the adapter for one route's window; cache `null` when it declines. */
    const resolveWindow = (route) => {
      if (windows.has(route.key) || resolving.has(route.key)) return;
      const llm = scoped.get?.("llm");
      if (!llm || typeof llm.resolveModelInfo !== "function") {
        windows.set(route.key, null);
        return;
      }
      resolving.add(route.key);
      void Promise.resolve()
        .then(() => llm.resolveModelInfo(route.provider, route.model))
        .then((info) => {
          const window = info?.context?.contextWindow;
          windows.set(route.key, typeof window === "number" && window > 0 ? window : null);
        })
        .catch(() => windows.set(route.key, null))
        .finally(() => resolving.delete(route.key));
    };

    /**
     * The prompt-token budget this agent's route implies.
     *
     * A route's first trigger may land before the adapter answered, so an
     * unresolved window uses the fallback for that check and the derived number
     * from the next one — the trigger is re-judged at every turn start.
     */
    const budgetFor = (agent) => {
      if (fixedBudget !== null) return fixedBudget;
      const route = routeOf(agent);
      if (route === null) return DEFAULT_HANDOFF_AT_TOKENS;
      if (!windows.has(route.key)) {
        resolveWindow(route);
        return DEFAULT_HANDOFF_AT_TOKENS;
      }
      const window = windows.get(route.key);
      if (typeof window !== "number" || window <= 0) return DEFAULT_HANDOFF_AT_TOKENS;
      const budget = budgetForWindow({ window, ratio: windowRatio });
      if (!announced.has(route.key)) {
        announced.add(route.key);
        log(
          `dsh-tinytitan: handoff budget for ${route.key} is ${budget} tokens ` +
            `(${windowRatio} of its ${window}-token window, ${DEFAULT_HANDOFF_RESERVE_TOKENS} held back)`,
        );
      }
      return budget;
    };

    /** Stop the handing-off session; the objective now lives in the child. */
    const stopParent = (agent) => {
      // Retired for good, not just disarmed: a wall-recovery or a person's
      // `/goal resume` must not turn this session into a second handoff source
      // and fork a sibling chain beside the one already running.
      retired.add(agent.id);
      try {
        goals.disarm(agent);
        log("dsh-tinytitan: disarmed the handing-off session so only the child continues");
      } catch (error) {
        log(`dsh-tinytitan: could not disarm the handing-off session: ${describe(error)}`);
      }
    };

    /** Arm the goal the child inherited through the fork seed. */
    const startChild = (child) => {
      if (!child) return;
      let inherited;
      try {
        inherited = goals.get(child);
      } catch (error) {
        log(`dsh-tinytitan: could not read the child's goal: ${describe(error)}`);
        return;
      }
      if (inherited === undefined) {
        log("dsh-tinytitan: the child inherited no goal, so nothing was armed for it");
        return;
      }
      if (inherited.phase === "complete" || inherited.activation === "armed") return;
      try {
        goals.resume(child, { id: inherited.id, revision: inherited.revision });
        log(`dsh-tinytitan: armed "${inherited.objective.slice(0, 60)}" in the child session`);
      } catch (error) {
        log(`dsh-tinytitan: could not arm the child's goal: ${describe(error)}`);
      }
    };

    /**
     * Release a child that has stopped continuing the objective.
     *
     * A fork child is a **one-shot** run: its `result` settles when its prompt
     * turn ends, measured 2026-10-04. Disposing it there is what killed the
     * continuation — the child's goal round had not been queued yet, so the
     * chain stopped after one turn. A child is therefore held while its goal is
     * actively continuing and released only when it is idle *and* its goal is
     * no longer active-and-armed (completed, blocked, disarmed, or handed on).
     * Idle matters: a child that just handed off finishes the turn in flight
     * before it is torn down.
     */
    const releaseEntry = (entry) => {
      children.delete(entry.key);
      if (!runs.delete(entry.run)) return;
      log(`dsh-tinytitan: released child ${entry.key} (nothing left to continue there)`);
      void Promise.resolve()
        .then(() => entry.run.dispose?.())
        .catch((error) =>
          log(`dsh-tinytitan: disposing a handoff child failed: ${describe(error)}`),
        );
    };

    /**
     * Ask whether one tracked child is finished; never dispose inside an append.
     *
     * A child whose own handoff child is still live is *not* finished, even when
     * it is idle with a disarmed goal: disposing a parent disposes its subagent
     * children (measured 2026-10-04 — the grandchild's turn ended
     * `{kind: "aborted", reason: {kind: "disposed"}}` 3 ms after its parent was
     * released). The chain is therefore released leaf-first, and releasing a leaf
     * re-checks the hop above it.
     */
    const maybeRelease = (agent) => {
      if (!agent || agent.status !== "idle") return;
      const entry = children.get(agent.id);
      if (entry === undefined) return;
      let goal;
      try {
        goal = goals.get(agent);
      } catch (error) {
        log(`dsh-tinytitan: could not read a child's goal: ${describe(error)}`);
        return;
      }
      if (goal !== undefined && goal.phase === "active" && goal.activation === "armed") return;
      for (const other of children.values()) {
        if (other.parent === agent.id) return;
      }
      releaseEntry(entry);
      const above = entry.parent === undefined ? undefined : agents.get(entry.parent);
      if (above !== undefined) maybeRelease(above);
    };

    /** Track one child for the process cap, the release rule and unload. */
    const track = (run, child, parent) => {
      runs.add(run);
      const key = child?.id ?? `run:${run?.id ?? "unknown"}`;
      const entry = { key, run, agent: child, parent };
      if (child) children.set(child.id, entry);
      if (!child) {
        // A remote run publishes no local agent to watch, so its settlement is
        // the only signal available.
        const settle = () => releaseEntry(entry);
        void Promise.resolve(run.result).then(settle, settle);
      }
      return entry;
    };

    /** Why this goal cannot hand on right now, or `null` when it can. */
    const blockReason = (key) => {
      if ((hops.get(key) ?? 0) >= hopBudget) return "hops";
      if (inFlight.has(key)) return "in-flight";
      if (runs.size >= childCap) return "cap";
      return null;
    };

    /**
     * Move an unfinished objective on, if its budget is spent and hops remain.
     *
     * @param sessionId - the session whose turn just opened.
     * @param inputTokens - the prompt tokens its previous step reported.
     */
    const handoff = (sessionId, inputTokens) => {
      if (retired.has(sessionId)) return;
      const agent = agents.get(sessionId);
      if (!agent) return;
      if (!(inputTokens >= budgetFor(agent))) return;
      const goal = goals.get(agent);
      // Only an actively continuing goal moves: a complete, paused, blocked or
      // disarmed one is a decision someone already made.
      if (!goal || goal.phase !== "active" || goal.activation !== "armed") return;
      const key = goal.id;
      const claimed = hops.get(key) ?? 0;
      const blocked = blockReason(key);
      if (blocked === "hops") {
        log(
          `dsh-tinytitan: handoff chain for "${goal.objective.slice(0, 60)}" reached ` +
            `${hopBudget} hops; leaving it to a person`,
        );
        return;
      }
      if (blocked === "in-flight") return;
      if (blocked === "cap") {
        log(
          `dsh-tinytitan: ${runs.size} handoff children are already live (cap ${childCap}); ` +
            `not handing on "${goal.objective.slice(0, 60)}"`,
        );
        return;
      }
      // Claim before awaiting: a child's opening turn can run before the start
      // promise settles, and this is the guard that has to hold then.
      const hop = claimed + 1;
      hops.set(key, hop);
      inFlight.add(key);
      const controller = new AbortController();
      controllers.add(controller);
      void startHandoff({ ctx: scoped, objective: goal.objective, parent: agent, log, controller })
        .then((result) => {
          if (!result?.started) {
            // A refused start must not spend the objective's budget.
            if (hops.get(key) === hop) hops.set(key, claimed);
            log(`dsh-tinytitan: handoff skipped (${result?.reason ?? "unknown"})`);
            return;
          }
          log(
            `dsh-tinytitan: handoff hop ${hop}/${hopBudget} at ${inputTokens} tokens for ` +
              `"${goal.objective.slice(0, 60)}"`,
          );
          stopParent(agent);
          startChild(result.child);
          if (result.run) track(result.run, result.child, agent.id);
        })
        .catch((error) => {
          if (hops.get(key) === hop) hops.set(key, claimed);
          log(`dsh-tinytitan: handoff failed: ${describe(error)}`);
        })
        .finally(() => {
          inFlight.delete(key);
          controllers.delete(controller);
        });
    };

    /**
     * Re-arm a goal the harness disarmed on a `max-tokens` turn.
     *
     * The budget trigger normally moves the objective long before the window
     * fills, but one turn can jump past it (a huge tool result) and end on
     * `max-tokens` — and that disarm (the round driver's own, on this event)
     * leaves no next `turn/start` to hand off from. Re-arming is the only lever
     * that creates one, because `start` needs a running loop. It is bounded
     * three ways: the wall must be a *context* wall (the reported prompt tokens
     * reached the budget, not just the output cap), a hop must actually be
     * available, and each goal gets at most two attempts — otherwise a chain
     * that cannot start would loop on the wall forever.
     */
    const recoverFromWall = (session) => {
      const agent = agents.get(session?.id);
      if (!agent || retired.has(agent.id)) return;
      const tokens = usage.get(session.id) ?? 0;
      if (!(tokens >= budgetFor(agent))) return;
      const goal = goals.get(agent);
      if (!goal || goal.phase !== "active" || goal.activation === "armed") return;
      const blocked = blockReason(goal.id);
      if (blocked !== null) {
        log(
          `dsh-tinytitan: "${goal.objective.slice(0, 60)}" hit its wall at ${tokens} tokens, ` +
            `but no hop is available (${blocked}); leaving it to a person`,
        );
        return;
      }
      const attempts = recoveries.get(goal.id) ?? 0;
      if (attempts >= 2) {
        log(
          `dsh-tinytitan: "${goal.objective.slice(0, 60)}" hit its wall again after ` +
            `${attempts} attempts; leaving it to a person`,
        );
        return;
      }
      recoveries.set(goal.id, attempts + 1);
      try {
        goals.resume(agent, { id: goal.id, revision: goal.revision });
        log(
          `dsh-tinytitan: the session hit its token wall at ${tokens} tokens; re-armed the goal ` +
            `so the next turn hands the objective on`,
        );
      } catch (error) {
        log(`dsh-tinytitan: could not re-arm a walled goal: ${describe(error)}`);
      }
    };

    // Remember what each step cost; judge the budget when the next turn opens,
    // which is the moment `start` is legal.
    ctx.on("session/event", (session, event) => {
      try {
        if (event?.type === "assistant/message") {
          const tokens = usageTokens(event.data?.usage);
          if (tokens > 0) usage.set(session?.id, tokens);
          return;
        }
        if (event?.type === "turn/end") {
          if (event.data?.reason?.kind !== "max-tokens") return;
          // Deferred by a microtask: the round driver disarms the goal in its
          // own listener for this very event, and the recovery has to see that.
          void Promise.resolve().then(() => {
            try {
              recoverFromWall(session);
            } catch (error) {
              log(`dsh-tinytitan: wall recovery failed: ${describe(error)}`);
            }
          });
          return;
        }
        if (event?.type !== "turn/start") return;
        handoff(session?.id, usage.get(session?.id) ?? 0);
      } catch (error) {
        // A stale view or a race with another creator must never take the turn
        // down with it.
        log(`dsh-tinytitan: handoff check failed: ${describe(error)}`);
      }
    });

    // A tracked child is released when it is idle and its goal is no longer
    // actively continuing: both edges are needed, because a round is queued
    // *after* the idle status and *because of* the armed goal.
    ctx.on("agent/status", ({ agent, status }) => {
      if (status === "idle") maybeRelease(agent);
    });
    ctx.on("goal/activation-changed", ({ sessionId }) => {
      const agent = agents.get(sessionId);
      if (agent !== undefined) maybeRelease(agent);
    });

    ctx.on("dispose", () => {
      for (const controller of controllers) controller.abort();
      controllers.clear();
      for (const entry of [...children.values()]) {
        children.delete(entry.key);
        void Promise.resolve()
          .then(() => entry.run.dispose?.())
          .catch((error) =>
            log(`dsh-tinytitan: child disposal failed: ${describe(error)} key=${entry.key}`),
          );
      }
      for (const run of [...runs]) {
        void Promise.resolve()
          .then(() => run.dispose?.())
          .catch((error) => log(`dsh-tinytitan: run disposal failed: ${describe(error)}`));
      }
      runs.clear();
    });
    return {
      installed: true,
      hops: hopBudget,
      atTokens: fixedBudget,
      windowRatio,
      maxChildren: childCap,
    };
  };

  try {
    if (typeof ctx.inject === "function") {
      ctx.inject(["agents", "goals", "subagents"], (scoped) => attach(scoped ?? ctx));
      return {
        installed: true,
        deferred: true,
        hops: hopBudget,
        atTokens: fixedBudget,
        windowRatio,
        maxChildren: childCap,
      };
    }
    return attach();
  } catch (error) {
    log(`dsh-tinytitan: handoff setup threw: ${error instanceof Error ? error.message : error}`);
    return { installed: false, reason: "setup-threw" };
  }
}
