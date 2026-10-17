/**
 * pi-ai assistant event stream to the harness StreamChunk protocol.
 *
 * The harness protocol has fixed ordering: block-start opens a block, deltas
 * extend it, block-end closes it with the assembled block, and the stream ends
 * with usage before exactly one terminal finish. Tool arguments stay raw JSON
 * strings. pi-ai never throws mid-stream: failures arrive as error events and
 * become error or aborted finish chunks.
 *
 * @module dsh-opencode-connect/events
 */

import type { ContentBlock, FinishReason, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'

/** One pi-ai content part, narrowed to the fields this module reads. */
export type PiContentPart =
  | { type: 'text'; text?: string }
  | { type: 'thinking'; thinking?: string }
  | { type: 'toolCall'; id?: string; name?: string; arguments?: unknown }
  | { type: string; [key: string]: unknown }

/** Subset of a pi-ai assistant message this module reads. */
export interface PiAssistantMessage {
  content?: readonly PiContentPart[]
  usage?: PiUsage
  stopReason?: string
  errorMessage?: string
}

/** pi-ai token accounting. */
export interface PiUsage {
  input?: number
  output?: number
  cacheRead?: number
  cacheWrite?: number
  reasoning?: number
  totalTokens?: number
}

/** One pi-ai assistant event. */
export type PiEvent =
  | { type: 'start'; partial?: PiAssistantMessage }
  | { type: 'text_start'; contentIndex: number }
  | { type: 'text_delta'; contentIndex: number; delta: string }
  | { type: 'text_end'; contentIndex: number; content: string }
  | { type: 'thinking_start'; contentIndex: number }
  | { type: 'thinking_delta'; contentIndex: number; delta: string }
  | { type: 'thinking_end'; contentIndex: number; content: string }
  | { type: 'toolcall_start'; contentIndex: number; partial?: PiAssistantMessage }
  | { type: 'toolcall_delta'; contentIndex: number; delta: string }
  | { type: 'toolcall_end'; contentIndex: number; toolCall?: { id?: string; name?: string; arguments?: unknown } }
  | { type: 'done'; reason: string; message: PiAssistantMessage }
  | { type: 'error'; reason?: string; error?: PiAssistantMessage }

/** Stable failure codes this adapter routes on. */
export const CONTEXT_WINDOW_EXCEEDED = 'CONTEXT_WINDOW_EXCEEDED'

/**
 * Classify an upstream failure into the provider-neutral taxonomy.
 *
 * The anonymous lane has two distinct throttling faces and they are NOT the
 * same failure: FreeTierError means the request shape was refused, while
 * FreeUsageLimitError means the shape was accepted and the egress IP's quota
 * is spent. Keeping them apart is what lets a caller tell 'this plugin is
 * misconfigured' from 'try again later or from another exit'.
 */
export function classifyError(text: string, status?: number): string {
  const lower = text.toLowerCase()
  if (status === 429 || lower.includes('freeusagelimit') || lower.includes('rate limit')) return 'RATE_LIMIT'
  if (lower.includes('freetier') || lower.includes('only be used from within opencode')) return 'FREE_TIER_RESTRICTED'
  if (lower.includes('context') && (lower.includes('length') || lower.includes('window'))) return CONTEXT_WINDOW_EXCEEDED
  if (status === 401 || status === 403) return 'AUTH'
  return 'PROVIDER_ERROR'
}

/**
 * Map pi-ai usage onto the harness's disjoint-count vocabulary.
 *
 * The harness counts are disjoint: inputTokens is UNCACHED input only, with
 * cached input reported separately. pi-ai reports input as the full prompt
 * count, so the cache reads are subtracted back out here.
 */
export function mapUsage(usage: PiUsage | undefined): TokenUsage | undefined {
  if (usage === undefined) return undefined
  const input = usage.input ?? 0
  const output = usage.output ?? 0
  const cacheRead = usage.cacheRead ?? 0
  const cacheWrite = usage.cacheWrite ?? 0
  const reasoning = usage.reasoning ?? 0
  const mapped: TokenUsage = {
    inputTokens: Math.max(0, input - cacheRead - cacheWrite),
    outputTokens: output,
    totalTokens: usage.totalTokens ?? input + output,
  }
  if (cacheRead > 0) mapped.cacheReadTokens = cacheRead
  if (cacheWrite > 0) mapped.cacheWriteTokens = cacheWrite
  if (reasoning > 0) mapped.reasoningTokens = reasoning
  return mapped
}

/** Map a finished pi-ai assistant message onto the harness finish vocabulary. */
export function mapStopReason(message: PiAssistantMessage): FinishReason {
  switch (message.stopReason) {
    case 'toolUse':
      return { kind: 'tool-calls' }
    case 'length':
      return { kind: 'max-tokens' }
    case 'aborted':
      return { kind: 'aborted', failure: { message: message.errorMessage ?? 'request aborted', code: 'ABORTED' } }
    case 'error': {
      const text = message.errorMessage ?? 'provider error'
      return { kind: 'error', failure: { message: text, code: classifyError(text) } }
    }
    default:
      return { kind: 'stop' }
  }
}

/** The raw JSON argument string for one tool call. */
function rawArguments(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value ?? {})
}

/**
 * Translate one pi-ai event stream into harness chunks.
 *
 * The harness assembler correlates deltas to their block by index and takes
 * block order from the block-start sequence, so pi-ai's own contentIndex is
 * the correct index to publish. A stream that ends without a terminal event
 * is a transport fault and fails loudly rather than yielding a silent finish.
 */
export async function* toStreamChunks(events: AsyncIterable<PiEvent>): AsyncGenerator<StreamChunk> {
  const toolIds = new Map<number, { id: string; name: string }>()
  for await (const event of events) {
    switch (event.type) {
      case 'start':
        break
      case 'text_start':
        yield { type: 'block-start', index: event.contentIndex, blockType: 'text' }
        break
      case 'text_delta':
        yield { type: 'text-delta', index: event.contentIndex, text: event.delta }
        break
      case 'text_end':
        yield { type: 'block-end', index: event.contentIndex, block: { type: 'text', text: event.content } }
        break
      case 'thinking_start':
        yield { type: 'block-start', index: event.contentIndex, blockType: 'reasoning' }
        break
      case 'thinking_delta':
        yield { type: 'reasoning-delta', index: event.contentIndex, text: event.delta }
        break
      case 'thinking_end':
        yield { type: 'block-end', index: event.contentIndex, block: { type: 'reasoning', text: event.content } }
        break
      case 'toolcall_start': {
        const part = event.partial?.content?.[event.contentIndex]
        const id = part !== undefined && part.type === 'toolCall' && typeof part.id === 'string' ? part.id : ''
        const name = part !== undefined && part.type === 'toolCall' && typeof part.name === 'string' ? part.name : ''
        toolIds.set(event.contentIndex, { id, name })
        yield { type: 'block-start', index: event.contentIndex, blockType: 'tool-call' }
        break
      }
      case 'toolcall_delta': {
        const known = toolIds.get(event.contentIndex)
        const chunk: StreamChunk = {
          type: 'tool-call-delta',
          index: event.contentIndex,
          id: (known?.id ?? '') as never,
          argumentsDelta: event.delta,
        }
        if (known !== undefined && known.name.length > 0) chunk.name = known.name
        yield chunk
        break
      }
      case 'toolcall_end': {
        const call = event.toolCall ?? {}
        const block: ContentBlock = {
          type: 'tool-call',
          id: (typeof call.id === 'string' ? call.id : '') as never,
          name: typeof call.name === 'string' ? call.name : '',
          arguments: rawArguments(call.arguments),
        }
        yield { type: 'block-end', index: event.contentIndex, block }
        break
      }
      case 'done':
        yield { type: 'usage', usage: mapUsage(event.message.usage) ?? { inputTokens: 0, outputTokens: 0 } }
        yield { type: 'finish', reason: mapStopReason(event.message) }
        return
      case 'error':
        yield { type: 'usage', usage: mapUsage(event.error?.usage) ?? { inputTokens: 0, outputTokens: 0 } }
        yield { type: 'finish', reason: mapStopReason(event.error ?? { stopReason: 'error' }) }
        return
      default:
        break
    }
  }
  throw new Error('dsh-opencode-connect: pi-ai event stream ended without a terminal event')
}
