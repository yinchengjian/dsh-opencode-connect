/**
 * The free-lane body gate and the free-model directory.
 */
import { describe, expect, it } from 'vitest'
import { ensureFreeLaneShape, freeLaneGateTool, FREE_LANE_GATE_TOOL_NAMES } from '../src/messages.ts'
import {
  FREE_LANE_BLOCKLIST,
  parseLiveModels,
  parseModelsDev,
  selectFreeModels,
  staticModels,
  toFreeModel,
} from '../src/catalog.ts'

describe('ensureFreeLaneShape', () => {
  it('appends the gate tools a tool-less chat is missing', () => {
    // A plain chat carries no tools and the lane refuses that body, so the
    // stubs must be injected for the request to be admitted at all.
    const shaped = ensureFreeLaneShape({ messages: [], stream: true }) as Record<string, unknown>
    const names = (shaped['tools'] as Array<{ function: { name: string } }>).map(t => t.function.name)
    expect(names).toEqual([...FREE_LANE_GATE_TOOL_NAMES])
  })

  it('pins tool_choice to none when the caller declared no tools', () => {
    // The stubs exist only to satisfy a shape check; letting the model call
    // one would send a request for a tool that does not exist.
    const shaped = ensureFreeLaneShape({ messages: [], stream: true }) as Record<string, unknown>
    expect(shaped['tool_choice']).toBe('none')
  })

  it('leaves a caller tool_choice alone when tools were declared', () => {
    const shaped = ensureFreeLaneShape({
      messages: [],
      stream: true,
      tools: [{ type: 'function', function: { name: 'bash', description: 'd', parameters: {} } }],
      tool_choice: 'auto',
    }) as Record<string, unknown>
    const names = (shaped['tools'] as Array<{ function: { name: string } }>).map(t => t.function.name)
    expect(names).toContain('bash')
    expect(names).toContain('read')
    expect(shaped['tool_choice']).toBe('auto')
  })

  it('returns undefined when the body already satisfies the gate', () => {
    // undefined is how pi-ai is told to keep the original payload.
    const body = { messages: [], tools: FREE_LANE_GATE_TOOL_NAMES.map(freeLaneGateTool) }
    expect(ensureFreeLaneShape(body)).toBeUndefined()
  })

  it('ignores a body it does not own', () => {
    expect(ensureFreeLaneShape(null)).toBeUndefined()
    expect(ensureFreeLaneShape('nope')).toBeUndefined()
    expect(ensureFreeLaneShape({ input: [] })).toBeUndefined()
  })
})

describe('parseLiveModels', () => {
  it('extracts ids in order and dedupes', () => {
    const ids = parseLiveModels({ data: [{ id: 'a' }, { id: 'b' }, { id: 'a' }] })
    expect(ids).toEqual(['a', 'b'])
  })

  it('tolerates a malformed reply', () => {
    expect(parseLiveModels(null)).toEqual([])
    expect(parseLiveModels({ data: 'no' })).toEqual([])
    expect(parseLiveModels({ data: [{}, { id: 7 }] })).toEqual([])
  })
})

describe('parseModelsDev', () => {
  const feed = {
    opencode: {
      models: {
        'free-one': { cost: { input: 0, output: 0 }, limit: { context: 200000, output: 32000 } },
        'paid-one': { cost: { input: 3, output: 15 } },
        'deprecated-free': { cost: { input: 0, output: 0 }, deprecated: true },
        'thinking': {
          cost: { input: 0, output: 0 },
          reasoning: true,
          reasoning_options: [{ type: 'effort', values: ['low', 'high'] }],
          modalities: { input: ['text', 'image'] },
        },
      },
    },
  }

  it('marks only zero-priced, non-deprecated models free', () => {
    const out = parseModelsDev(feed)
    expect(out.get('free-one')?.free).toBe(true)
    expect(out.get('paid-one')?.free).toBe(false)
    expect(out.get('deprecated-free')?.free).toBe(false)
  })

  it('reads limits, reasoning efforts, and modalities', () => {
    const out = parseModelsDev(feed).get('thinking')
    expect(out?.reasoning).toBe(true)
    expect(out?.effortValues).toEqual(['low', 'high'])
    expect(out?.modalities).toEqual(['text', 'image'])
    expect(parseModelsDev(feed).get('free-one')?.contextWindow).toBe(200000)
  })

  it('refuses a model whose price is unknown', () => {
    // Absent pricing is not evidence of free: over-claiming spends money.
    const out = parseModelsDev({ opencode: { models: { mystery: {} } } })
    expect(out.get('mystery')?.free).toBe(false)
  })

  it('tolerates a malformed feed', () => {
    expect(parseModelsDev(null).size).toBe(0)
    expect(parseModelsDev({}).size).toBe(0)
    expect(parseModelsDev({ opencode: null }).size).toBe(0)
  })
})

describe('selectFreeModels', () => {
  const metadata = parseModelsDev({
    opencode: {
      models: {
        'free-one': { cost: { input: 0, output: 0 } },
        'paid-one': { cost: { input: 3, output: 15 } },
        'exo-free': { cost: { input: 0, output: 0 } },
      },
    },
  })

  it('keeps free models and drops paid ones', () => {
    const models = selectFreeModels(['free-one', 'paid-one'], metadata)
    expect(models.map(m => m.id)).toEqual(['free-one'])
  })

  it('drops blocklisted ids even when metadata says free', () => {
    // The lane lists them but can never serve them.
    const models = selectFreeModels(['exo-free', 'free-one'], metadata)
    expect(models.map(m => m.id)).toEqual(['free-one'])
    expect(FREE_LANE_BLOCKLIST.has('exo-free')).toBe(true)
  })

  it('drops a model with no metadata at all', () => {
    expect(selectFreeModels(['unknown-free'], metadata)).toEqual([])
  })

  it('preserves the live listing order', () => {
    const feed = parseModelsDev({
      opencode: { models: { b: { cost: { input: 0, output: 0 } }, a: { cost: { input: 0, output: 0 } } } },
    })
    expect(selectFreeModels(['b', 'a'], feed).map(m => m.id)).toEqual(['b', 'a'])
  })
})

describe('toFreeModel', () => {
  it('declares text only when metadata is silent', () => {
    expect(toFreeModel('mystery-free', undefined).inputModalities).toEqual(['text'])
  })

  it('declares image input when metadata says so', () => {
    const metadata = parseModelsDev({
      opencode: { models: { v: { cost: { input: 0, output: 0 }, modalities: { input: ['text', 'image'] } } } },
    })
    expect(toFreeModel('v', metadata.get('v')).inputModalities).toEqual(['text', 'image'])
  })
})

describe('staticModels', () => {
  it('offers a usable bootstrap set', () => {
    const models = staticModels()
    expect(models.length).toBeGreaterThan(0)
    expect(models.some(m => m.id === 'deepseek-v4.1-flash-free')).toBe(true)
  })

  it('never bootstraps a blocklisted id', () => {
    for (const model of staticModels()) expect(FREE_LANE_BLOCKLIST.has(model.id)).toBe(false)
  })
})

