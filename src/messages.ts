/**
 * Harness vocabulary to pi-ai `Context` conversion, plus the free-lane body
 * shape the anonymous gate requires.
 *
 * The gate reads the request body, not just the headers (live-probed
 * 2026-10-17): a body is admitted only when it is BOTH
 *
 *   1. streaming (`stream: true` — a non-stream body is refused), and
 *   2. agent-shaped: the `tools` array contains `bash` and `read`.
 *
 * A plain chat with no tools is refused, so {@link ensureFreeLaneShape}
 * appends the missing gate stubs through pi-ai's `onPayload` seam. The
 * stubs carry a description telling the model not to call them, and when the
 * request declared no tools at all `tool_choice` is pinned to `none` so a
 * stub can never be invoked.
 *
 * @module dsh-opencode-connect/messages
 */

import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { AssistantMessage, Context, Message, TextContent, ThinkingContent, Tool, ToolCall } from '@earendil-works/pi-ai'

/** A durable harness message, structurally typed to avoid a hard host import. */
export interface HarnessMessage {
  role: 'system' | 'developer' | 'user' | 'assistant' | 'tool'
  content: readonly ContentBlock[] | readonly unknown[]
  /** Tool-role messages carry the answered call id at the message level. */
  toolCallId?: string
  isError?: boolean
  source?: { kind?: string; [key: string]: unknown }
}

/** A harness tool declaration. */
export interface HarnessTool {
  name: string
  description: string
  parameters: Record<string, unknown>
}

/** The request fields this adapter reads. */
export interface HarnessGenerateOptions {
  provider: string
  model: string
  messages: readonly HarnessMessage[]
  system?: string
  tools?: readonly HarnessTool[]
  temperature?: number
  maxTokens?: number
  reasoningEffort?: string
  signal?: AbortSignal
}

/** Gate tool names. Any one known pair is accepted; this is the verified minimal set. */
export const FREE_LANE_GATE_TOOL_NAMES = ['bash', 'read'] as const

/** One injected gate stub, in OpenAI function-tool shape. */
export interface FreeLaneGateTool {
  type: 'function'
  function: { name: string; description: string; parameters: Record<string, unknown> }
}

/** Build one gate stub. The description is the only guard against a call. */
export function freeLaneGateTool(name: (typeof FREE_LANE_GATE_TOOL_NAMES)[number]): FreeLaneGateTool {
  return {
    type: 'function',
    function: {
      name,
      description: 'Reserved for the host runtime; do not call it.',
      parameters: { type: 'object', properties: {} },
    },
  }
}

/** True when `value` is a plain object usable as a body. */
function isBody(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Names already declared in an OpenAI-shaped `tools` array. */
function declaredToolNames(tools: readonly unknown[]): Set<string> {
  const names = new Set<string>()
  for (const tool of tools) {
    if (!isBody(tool)) continue
    const fn = tool['function']
    if (!isBody(fn)) continue
    const name = fn['name']
    if (typeof name === 'string') names.add(name)
  }
  return names
}

/**
 * Append any missing gate stubs to an outgoing chat-completions body.
 *
 * Returns `undefined` when the body already satisfies the gate or is not a
 * chat-completions body at all, which is how pi-ai is told to keep the
 * original payload. When the caller declared no tools, `tool_choice` is
 * pinned to `none`: the stubs exist purely to satisfy the shape check and
 * must never be invoked. A caller that supplied its own tools keeps its own
 * tool choice untouched.
 */
export function ensureFreeLaneShape(payload: unknown): unknown | undefined {
  if (!isBody(payload)) return undefined
  if (!Array.isArray(payload['messages'])) return undefined

  const tools = Array.isArray(payload['tools']) ? (payload['tools'] as unknown[]) : []
  const declared = declaredToolNames(tools)
  const missing = FREE_LANE_GATE_TOOL_NAMES.filter(name => !declared.has(name))
  if (missing.length === 0) return undefined

  const next: Record<string, unknown> = { ...payload }
  next['tools'] = [...tools, ...missing.map(name => freeLaneGateTool(name))]
  if (tools.length === 0) next['tool_choice'] = 'none'
  return next
}

/** Parse a raw JSON argument string into an object, tolerating garbage. */
function parseArguments(raw: unknown): Record<string, unknown> {
  if (typeof raw !== 'string' || raw.length === 0) return {}
  try {
    const parsed: unknown = JSON.parse(raw)
    return isBody(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

/** Flatten a message's text blocks into one string. */
function flattenText(message: HarnessMessage): string {
  const parts: string[] = []
  for (const block of message.content as readonly unknown[]) {
    if (!isBody(block)) continue
    if (block['type'] === 'text' && typeof block['text'] === 'string') parts.push(block['text'])
  }
  return parts.join('')
}

/**
 * Build one pi-ai assistant message from a durable harness assistant message.
 *
 * Replayed history must carry its tool calls back to the provider, and the
 * OpenAI wire requires an `assistant` turn with `tool_calls` before any
 * `tool` result, so the blocks are rebuilt in order.
 */
function toPiAssistant(message: HarnessMessage, providerId: string): AssistantMessage {
  const content: Array<TextContent | ThinkingContent | ToolCall> = []
  for (const block of message.content as readonly unknown[]) {
    if (!isBody(block)) continue
    switch (block['type']) {
      case 'text':
        if (typeof block['text'] === 'string' && block['text'].length > 0) {
          content.push({ type: 'text', text: block['text'] })
        }
        break
      case 'reasoning':
        if (typeof block['text'] === 'string' && block['text'].length > 0) {
          content.push({ type: 'thinking', thinking: block['text'] })
        }
        break
      case 'tool-call': {
        const id = block['id']
        const name = block['name']
        if (typeof id !== 'string' || typeof name !== 'string') break
        content.push({ type: 'toolCall', id, name, arguments: parseArguments(block['arguments']) as never })
        break
      }
      default:
        break
    }
  }
  return {
    role: 'assistant',
    content,
    api: 'openai-completions',
    provider: providerId,
    model: '',
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: content.some(part => part.type === 'toolCall') ? 'toolUse' : 'stop',
    timestamp: 0,
  }
}

/**
 * Convert a harness request into a pi-ai `Context`.
 *
 * Mirrors the host's own text-only projection, and the conventions are load
 * bearing:
 *
 *   - a leading `system` message becomes `systemPrompt` and leaves the
 *     history; any LATER system message folds into a `user` message. Sending
 *     the prompt as an ordinary first user turn would make the model read its
 *     own instructions as user input.
 *   - an explicit `options.system` wins, and every history message then
 *     converts in place.
 *   - a tool result lands in the `toolResult` slot with the name its call
 *     declared: the OpenAI wire requires the assistant's `tool_calls` turn to
 *     precede it and the name to match, or the provider rejects the request.
 *
 * Text only: this adapter declares text input, and a file block is already
 * projected to handle text by the host before an adapter sees it.
 */
export function toPiContext(options: HarnessGenerateOptions, providerId: string): Context {
  const messages: Message[] = []
  // The name a tool call declared, so its result can be labelled correctly.
  const toolNames = new Map<string, string>()

  // An explicit system prompt wins and leaves the whole history to convert;
  // otherwise a LEADING system message supplies it and is consumed.
  let systemPrompt: string | undefined = options.system
  let history: readonly HarnessMessage[] = options.messages
  if (systemPrompt === undefined) {
    const [first, ...rest] = options.messages
    if (first !== undefined && first.role === 'system') {
      const text = flattenText(first)
      systemPrompt = text.length > 0 ? text : undefined
      history = rest
    }
  }

  for (const message of history) {
    if (message.role === 'system') {
      // A non-leading system message has no dedicated slot on this wire, so it
      // travels as user content in place.
      messages.push({ role: 'user', content: flattenText(message), timestamp: 0 })
      continue
    }
    if (message.role === 'assistant') {
      const assistant = toPiAssistant(message, providerId)
      for (const part of assistant.content) {
        if (part.type === 'toolCall') toolNames.set(part.id, part.name)
      }
      messages.push(assistant)
      continue
    }
    if (message.role === 'tool') {
      const callId = message.toolCallId ?? ''
      messages.push({
        role: 'toolResult',
        toolCallId: callId,
        toolName: toolNames.get(callId) ?? 'unknown',
        content: [{ type: 'text', text: flattenText(message) }],
        isError: message.isError ?? false,
        timestamp: 0,
      } as Message)
      continue
    }
    messages.push({ role: 'user', content: flattenText(message), timestamp: 0 })
  }

  const context: Context = { messages }
  if (systemPrompt !== undefined && systemPrompt.length > 0) context.systemPrompt = systemPrompt
  if (options.tools !== undefined && options.tools.length > 0) {
    context.tools = options.tools.map(tool => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    })) as unknown as Tool[]
  }
  return context
}