/**
 * The reasoning contract: `off` must be offered, must travel as `none`, and a
 * reasoning-only truncation must be recoverable.
 */
import { describe, expect, it } from 'vitest'
import {
  DEFAULT_EFFORT_LADDER,
  reasoningEfforts,
  reasoningEffortWire,
  shouldRecoverFromReasoningTruncation,
} from '../src/reasoning.ts'

describe('reasoningEfforts', () => {
  it('offers nothing for a non-reasoning model', () => {
    expect(reasoningEfforts({ reasoning: false, effortValues: ['high'] })).toBeUndefined()
    expect(reasoningEfforts(undefined)).toBeUndefined()
  })

  it('keeps the declared ladder and injects off, sorted by strength', () => {
    expect(reasoningEfforts({ reasoning: true, effortValues: ['max', 'low', 'high'] }))
      .toEqual(['off', 'low', 'high', 'max'])
  })

  it('folds the metadata spelling none onto off', () => {
    expect(reasoningEfforts({ reasoning: true, effortValues: ['none', 'medium'] }))
      .toEqual(['off', 'medium'])
  })

  it('falls back to the default ladder when metadata declares no effort values', () => {
    expect(reasoningEfforts({ reasoning: true, effortValues: [] })).toEqual([...DEFAULT_EFFORT_LADDER])
    expect(DEFAULT_EFFORT_LADDER).toContain('off')
  })

  it('drops ids the gateway does not accept but still offers off', () => {
    expect(reasoningEfforts({ reasoning: true, effortValues: ['turbo', 'high'] })).toEqual(['off', 'high'])
  })
})

describe('reasoningEffortWire', () => {
  it('maps off to omission: the field itself is what keeps thinking alive', () => {
    expect(reasoningEffortWire('off')).toBeUndefined()
  })

  it('passes ladder ids through verbatim', () => {
    for (const id of ['minimal', 'low', 'medium', 'high', 'xhigh', 'max']) {
      expect(reasoningEffortWire(id)).toBe(id)
    }
  })

  it('omits unknown and absent ids rather than risking a 400', () => {
    expect(reasoningEffortWire('turbo')).toBeUndefined()
    expect(reasoningEffortWire(undefined)).toBeUndefined()
  })
})

describe('shouldRecoverFromReasoningTruncation', () => {
  const base = {
    sawReasoning: true,
    sawAnswerOrTool: false,
    finishKind: 'max-tokens',
    canDisableThinking: true,
    aborted: false,
  }

  it('recovers when thinking ate the whole budget', () => {
    expect(shouldRecoverFromReasoningTruncation(base)).toBe(true)
  })

  it('does not recover when an answer already landed', () => {
    expect(shouldRecoverFromReasoningTruncation({ ...base, sawAnswerOrTool: true })).toBe(false)
  })

  it('does not recover without reasoning', () => {
    expect(shouldRecoverFromReasoningTruncation({ ...base, sawReasoning: false })).toBe(false)
  })

  it('does not recover on a normal stop', () => {
    expect(shouldRecoverFromReasoningTruncation({ ...base, finishKind: 'stop' })).toBe(false)
    expect(shouldRecoverFromReasoningTruncation({ ...base, finishKind: undefined })).toBe(false)
  })

  it('does not recover when the model cannot switch thinking off', () => {
    expect(shouldRecoverFromReasoningTruncation({ ...base, canDisableThinking: false })).toBe(false)
  })

  it('does not recover after an abort', () => {
    expect(shouldRecoverFromReasoningTruncation({ ...base, aborted: true })).toBe(false)
  })
})