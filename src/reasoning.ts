/**
 * Reasoning-effort vocabulary and the reasoning-only recovery contract.
 *
 * The free lane serves several always-think models: they emit
 * `reasoning_content` before any answer and charge it to the same output
 * budget as the answer. Under a small `max_tokens` the thinking consumes the
 * whole budget, the reply arrives with zero answer tokens, and the stream ends
 * with `finish: length` — what the harness surfaces as "output token limit
 * reached". Two things must therefore exist:
 *
 * 1. The picker must offer `off`, or a caller has no way to ask the model to
 *    stop thinking. models.dev rarely lists it, so it is injected here.
 * 2. `off` must travel as the wire value `none`. Omitting the field keeps the
 *    provider default (the model still thinks); `none` is the only spelling
 *    that actually stops it.
 *
 * @module dsh-opencode-connect/reasoning
 */

/** Canonical reasoning-effort ladder, weakest to strongest. */
export const REASONING_EFFORT_LADDER = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const

/**
 * Levels offered to a reasoning model whose metadata declares no ladder.
 *
 * `off` leads so a caller can always stop the thinking that causes the
 * reasoning-only truncation.
 */
export const DEFAULT_EFFORT_LADDER: readonly string[] = ['off', 'minimal', 'low', 'medium', 'high']

/**
 * Turn a metadata capability into the effort ids the harness picker shows.
 *
 * A declared ladder wins, because its spellings are the ones upstream honors.
 * `off` is always offered for a reasoning model, even when the metadata omits
 * it: models.dev lists only `low|high|max` for the always-think free models,
 * yet the gateway accepts `none` (live-probed: 200 with zero reasoning chunks)
 * and that is the only way to stop thinking that would otherwise consume the
 * entire output budget.
 *
 * @param declared - models.dev `reasoning` flag and `options[type=effort].values`.
 * @returns sorted effort ids, or undefined for a non-reasoning model.
 */
export function reasoningEfforts(
  declared: { reasoning: boolean; effortValues: readonly string[] } | undefined,
): readonly string[] | undefined {
  if (declared?.reasoning !== true) return undefined
  const seen: string[] = []
  for (const value of declared.effortValues) {
    // models.dev spells "stop thinking" as none on some providers.
    const level = value === 'none' ? 'off' : value
    if ((REASONING_EFFORT_LADDER as readonly string[]).includes(level) && !seen.includes(level)) seen.push(level)
  }
  // No declared ladder means the model exposes no preference, so open the
  // standard set the gateway accepts rather than offering "off" alone.
  if (seen.length === 0) return [...DEFAULT_EFFORT_LADDER]
  if (!seen.includes('off')) seen.push('off')
  return seen.sort(
    (a, b) => REASONING_EFFORT_LADDER.indexOf(a as never) - REASONING_EFFORT_LADDER.indexOf(b as never),
  )
}

/**
 * The `reasoning_effort` wire value for an advertised effort id.
 *
 * The gateway validates the field against `minimal|low|medium|high|xhigh|max`
 * and rejects anything else (a literal "off" is a 422), so unknown ids send
 * nothing rather than risk a hard failure. `off` also sends nothing — see the
 * body for the measurement that makes omission the only working spelling.
 *
 * @param id - an effort id previously advertised by {@link reasoningEfforts}.
 * @returns the wire spelling, or undefined to omit the field.
 */
export function reasoningEffortWire(id: string | undefined): string | undefined {
  if (id === undefined) return undefined
  // `off` means OMIT the field. Measured on the free lane: omitting stops the
  // reasoning phase entirely, while whatever value is sent (even "none") still
  // lets an always-think model think. The literal "off" is a hard 422.
  if (id === 'off') return undefined
  return (REASONING_EFFORT_LADDER as readonly string[]).includes(id) ? id : undefined
}

/**
 * Whether a finished attempt is a reasoning-only truncation worth retrying.
 *
 * This is the exact shape the free lane produces when thinking eats the whole
 * budget: reasoning arrived, no answer or tool call did, and the stream ended
 * on the token cap. Retrying with thinking off is the only in-band recovery;
 * otherwise the caller sees an empty reply.
 *
 * @param shape - what the attempt delivered.
 * @returns true when a `reasoningEffort: 'off'` retry is justified.
 */
export function shouldRecoverFromReasoningTruncation(shape: {
  sawReasoning: boolean
  sawAnswerOrTool: boolean
  finishKind: string | undefined
  canDisableThinking: boolean
  aborted: boolean
}): boolean {
  return shape.canDisableThinking
    && shape.sawReasoning
    && !shape.sawAnswerOrTool
    && shape.finishKind === 'max-tokens'
    && !shape.aborted
}