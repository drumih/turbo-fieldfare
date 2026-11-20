/**
 * `dsh-tinytitan/keep-going` — an ordinary manual prompt runs to completion.
 *
 * The harness already owns the machinery: `@deepseek-ai/dsh-goal` keeps one
 * long-running objective per session, `@deepseek-ai/dsh-goal-round-driver`
 * queues the next round while the agent is idle and the goal is armed, and
 * `@deepseek-ai/dsh-tool-goal` lets the model complete or block that goal. What
 * a person does not get for free is the *start*: they either type `/goal …`, or
 * the model has to decide the request is goal-shaped.
 *
 * This module adds the third way, for a profile that asks for it. With
 * `autoGoal: true`, a direct human message that arrives while no unfinished goal
 * is current creates and arms one whose objective is that message. Nothing else
 * is reproduced here — no rounds are scheduled, no completion is judged, no
 * session is spawned — so the behaviour is exactly `/goal <that message>`, and
 * every rule the harness already applies (round cap, blocking policy, authority
 * checks, arming rules after resume or fork) still applies to it.
 *
 * Two consequences worth knowing before turning it on:
 *
 * - **Every** manual prompt becomes a goal, including a question a person meant
 *   as a chat turn. The model is told to continue until it judges the goal
 *   complete, so it answers and then calls `update_goal ... complete`; a model
 *   that instead keeps working spends rounds from the cap below.
 * - Completion needs `@deepseek-ai/dsh-tool-goal` mounted; the shipped standard
 *   composition mounts it, and a deployment that does not would run an
 *   auto-created goal into its round cap instead of letting the model finish it.
 *
 * A missing goal service is not an error: the switch stays a no-op and the
 * reason is logged once, because a harness composed without goals must boot
 * exactly as it did before this option existed.
 *
 * @module dsh-tinytitan/keep-going
 */

/** The cordis name of the harness's goal service (`@deepseek-ai/dsh-goal`). */
export const GOALS_SERVICE = "goals";

/**
 * Rounds an auto-created goal may run before the harness blocks it.
 *
 * Effectively unbounded, so an unattended goal keeps working through the night
 * instead of stopping at a small round budget. A profile that turns `autoGoal`
 * on can lower it with `autoGoalRounds` when it wants prompts a person did not
 * mark as long-running work to stay bounded.
 */
export const DEFAULT_AUTO_GOAL_ROUNDS = 99999;

/**
 * The plain text of a message's content.
 *
 * A human message may be a bare string or a list of parts (text, images,
 * files); only the text can be an objective. Non-text parts contribute nothing
 * rather than a placeholder, because an objective made of `[image]` would send
 * the model looking for work nobody described.
 *
 * @param content - a message's `content` field.
 * @returns the trimmed text, or an empty string when there is none.
 */
export function messageText(content) {
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part.text === "string") return part.text;
        return "";
      })
      .join("\n")
      .trim();
  }
  if (content && typeof content.text === "string") return content.text.trim();
  return "";
}

/**
 * Whether a queued message is a person's own prompt.
 *
 * The harness labels a goal round's synthetic prompt with `source.kind ===
 * "goal"` (and a `goalId`), and other producers label theirs differently, so the
 * check is exact rather than a list of exclusions: an auto-goal must never feed
 * on the driver's own round, or a single prompt would restart itself forever.
 *
 * @param message - a queued agent message.
 * @returns true for a direct human message.
 */
export function isDirectHuman(message) {
  const source = message?.source;
  return Boolean(source) && source.kind === "user" && source.goalId === undefined;
}

/**
 * Arm a goal for every direct human prompt, when the profile asks for it.
 *
 * Resolution is deferred through `ctx.inject` where the context offers it: the
 * goal service injects `agents` and `sessionProjections` and may activate after
 * this plugin's row, which is exactly the ordering that broke the route refresh
 * on 0.2.0. A context without `inject` (the test fakes, and any older harness)
 * resolves immediately instead.
 *
 * @param options - the harness context, the resolved plugin config, and a logger.
 * @returns a status object; `installed: false` carries the reason.
 */
/**
 * Whether a live agent is a subagent, across the 0.2.2 shapes.
 *
 * @param agent - the live agent from the inserted-message event.
 * @returns true for a child session, false for a root.
 */
function isSubagent(agent) {
  if (!agent) return true;
  if (agent.parentAgent !== undefined) return true;
  const session = agent.session;
  if (session?.parentSession !== undefined) return true;
  if ((session?.delegationDepth ?? 0) > 0) return true;
  return false;
}

export function installAutoGoal({ ctx, resolved, log = () => {} } = {}) {
  if (resolved?.autoGoal !== true) return { installed: false, reason: "disabled" };
  if (typeof ctx?.on !== "function") return { installed: false, reason: "no-context" };
  const rounds = resolved.autoGoalRounds ?? DEFAULT_AUTO_GOAL_ROUNDS;

  const attach = (scoped = ctx) => {
    const goals = scoped?.get?.(GOALS_SERVICE);
    if (!goals || typeof goals.create !== "function" || typeof goals.get !== "function") {
      log(
        "dsh-tinytitan: autoGoal is on but the harness has no goal service; " +
          "mount @deepseek-ai/dsh-goal (the standard composition does)",
      );
      return { installed: false, reason: "no-goals-service" };
    }

    /** Create and arm the goal this prompt implies, if it implies one. */
    const onInserted = ({ agent, message } = {}) => {
      try {
        // Subagents carry a live parent; only a session a person is driving
        // should turn a prompt into a persistent objective. 0.2.2-rc.5 removed
        // the `parentAgent` creation option and property; a child now carries
        // `parentSession`/`delegationDepth` on its session metadata, so both
        // shapes are checked (the older one is still present in some profiles).
        if (!agent || isSubagent(agent)) return;
        if (!isDirectHuman(message)) return;
        const current = goals.get(agent);
        if (current !== undefined && current.phase !== "complete") return;
        const objective = messageText(message.content);
        if (objective.length === 0) return;
        const view = goals.create(agent, { objective, maxGoalRounds: rounds });
        log(
          `dsh-tinytitan: armed ${view?.id ?? "a goal"} from a manual prompt ` +
            `(${objective.length} chars, cap ${view?.maxGoalRounds ?? rounds})`,
        );
      } catch (error) {
        // A stale view, a race with another creator, or a service that refuses
        // this agent must never take the turn down with it.
        log(
          `dsh-tinytitan: autoGoal could not arm a goal: ` +
            `${error instanceof Error ? error.message : error}`,
        );
      }
    };

    ctx.on("agent/inbox/inserted", onInserted);
    return { installed: true, rounds };
  };

  try {
    // The service may already be live, or it may activate after this row (it
    // injects `agents` and `sessionProjections`). Ask now, and defer through
    // `inject` only when the answer is "not yet" — a `get` that throws counts as
    // "not yet" too, because a harness without the service never registers it.
    let present;
    try {
      present = scopedService(ctx);
    } catch {
      present = undefined;
    }
    if (present !== undefined) return attach();
    if (typeof ctx.inject === "function") {
      ctx.inject([GOALS_SERVICE], (scoped) => attach(scoped ?? ctx));
      return { installed: true, deferred: true, rounds };
    }
    return attach();
  } catch (error) {
    log(`dsh-tinytitan: autoGoal setup threw: ${error instanceof Error ? error.message : error}`);
    return { installed: false, reason: "setup-threw" };
  }
}

/**
 * Read the goal service without committing to its absence.
 *
 * @param ctx - the context to ask.
 * @returns the service, or undefined when it is not registered yet.
 */
function scopedService(ctx) {
  return ctx?.get?.(GOALS_SERVICE);
}
