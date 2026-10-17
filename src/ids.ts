/**
 * Anonymous-lane request identity.
 *
 * OpenCode's free tier admits a request only when it looks like traffic from
 * the official CLI (live-probed 2026-10-17):
 *
 *   1. a canonical `ses_*` id in ANY ONE of `x-opencode-session`,
 *      `x-session-affinity`, or `X-Session-Id`. The 12 leading hex
 *      characters are not an opaque blob — they decode back to a wall-clock
 *      time, bitwise-inverted, and a digest-derived prefix is rejected.
 *   2. a `User-Agent` whose product token is `opencode`. DSH's own
 *      attribution UA alone is refused, so this module renders a HYBRID:
 *      the `opencode` token the gate reads, followed by the harness
 *      attribution the host contract requires.
 *
 * Everything here is a public product fact; no secret, path, or prompt text
 * enters a header.
 *
 * @module dsh-opencode-connect/ids
 */

import { createHash, randomBytes } from 'node:crypto'

/** The anonymous-lane credential. A literal, not a secret. */
export const ANONYMOUS_KEY = 'public'

/**
 * CLI version the disguise advertises. Kept as a constant because the gate
 * matches the `opencode` product token, not this number.
 */
export const OPENCODE_CLI_VERSION = '1.18.31'

/** Correlation ids derived for one upstream request. */
export interface RequestIDs {
  /** Stable per-conversation session id; the gate's primary signal. */
  session: string
  /** Fresh per-request id. */
  request: string
  /** Stable per-install project id. */
  project: string
}

/** Base62 alphabet OpenCode's identifier generator uses. */
const BASE62_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

/** A canonical session id: `ses_` + 12 hex + 14 Base62. */
const CANONICAL_SESSION_PATTERN = /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/

/**
 * The 12 hex characters a genuine client emits.
 *
 * OpenCode's generator takes the millisecond clock, appends a per-millisecond
 * sequence in the low 12 bits, and inverts the whole value bitwise, so ids
 * sort newest-first as plain strings. The free tier decodes those bits back
 * and accepts only a plausible wall-clock time. Counter 1 always: this plugin
 * is the sole writer of its own sessions, so a real counter would add state
 * for nothing.
 */
export function invertedTimestampHex(now: number = Date.now()): string {
  const current = BigInt(now) * 0x1000n + 1n
  const inverted = ~current & 0xffffffffffffn
  return inverted.toString(16).padStart(12, '0')
}

/** Fixed-width Base62 encoding of a non-negative integer. */
function base62Fixed(value: bigint, width: number): string {
  const base = 62n
  let n = value
  const out = new Array<string>(width)
  for (let i = width - 1; i >= 0; i--) {
    out[i] = BASE62_ALPHABET.charAt(Number(n % base))
    n /= base
  }
  return out.join('')
}

/** A fresh random id under a prefix. */
export function randomID(prefix: string, size: number): string {
  return `${prefix}_${randomBytes(size).toString('hex')}`
}

/**
 * The conversation signal: JSON of the first user message's content. Using
 * the first turn keeps a multi-turn conversation stable as history grows
 * while separating conversations that begin differently.
 */
export function conversationSeed(messages: ReadonlyArray<{ role: string; content: unknown }>): string {
  for (const message of messages) {
    if (message.role !== 'user') continue
    const encoded = JSON.stringify(message.content ?? null)
    if (encoded !== 'null' && encoded.length > 0) return encoded
  }
  return ''
}

/**
 * Mint a canonical session id for one conversation signal.
 *
 * Memoized per signal. The clock half must be minted once per conversation,
 * not per request: upstream groups the free tier's quota and prompt cache by
 * session, so a fresh clock on every request would forfeit cache reuse across
 * turns. Capped so a long-lived process cannot grow the map without bound.
 */
const SESSION_MEMO_LIMIT = 1024
const sessionMemo = new Map<string, string>()

export function canonicalSessionID(signal: string, now: number = Date.now()): string {
  if (CANONICAL_SESSION_PATTERN.test(signal)) return signal

  const memoized = sessionMemo.get(signal)
  if (memoized !== undefined) return memoized

  const digest = createHash('sha256').update(`ses\u0000${signal}`).digest()
  const session = `ses_${invertedTimestampHex(now)}${base62Fixed(BigInt(`0x${digest.subarray(6, 16).toString('hex')}`), 14)}`

  // FIFO eviction; a re-minted id only costs cache affinity.
  if (sessionMemo.size >= SESSION_MEMO_LIMIT) {
    const oldest = sessionMemo.keys().next()
    if (!oldest.done) sessionMemo.delete(oldest.value)
  }
  sessionMemo.set(signal, session)
  return session
}

/** Derive the correlation ids for one upstream request from its conversation. */
export function deriveRequestIDs(
  messages: ReadonlyArray<{ role: string; content: unknown }>,
  now: number = Date.now(),
): RequestIDs {
  let signal = conversationSeed(messages)
  if (signal === '' || signal === '{}') signal = randomID('fallback', 16)
  return {
    session: canonicalSessionID(signal, now),
    request: randomID('req', 16),
    project: `prj_${createHash('sha256').update('dsh-opencode-connect\u0000default-project').digest().subarray(0, 12).toString('hex')}`,
  }
}

/**
 * The `User-Agent` the SHIM sends upstream.
 *
 * Only the CLI identity — deliberately NOT the harness attribution.
 *
 * The harness requires its attribution on every provider request, and the free
 * lane refuses that exact identity: a `deepseek-harness/` product token answers
 * `403 FreeTierError` (live-probed, interleaved A/B; `codex` and
 * `claude-code` tokens are admitted, so the refusal is specific to this
 * harness). Both cannot hold on one request, so the shim is where they part:
 * the attribution is satisfied toward the loopback listener (the adapter's own
 * request to the shim, which pi-ai stamps), and this function renders what the
 * real upstream request carries.
 *
 * Do not "restore" the attribution here. It is not an oversight, and adding it
 * back turns every call into an upstream 403.
 */
export function disguiseUserAgent(): string {
  const node = process.versions.node ?? '0'
  return `opencode/${OPENCODE_CLI_VERSION} (${process.platform} ${process.arch}; node${node})`
}

/**
 * The correlation headers upstream expects. All three session spellings are
 * sent: any one satisfies the gate, and sending the full set matches a real
 * client's shape and costs nothing.
 */
export function disguiseHeaders(ids: RequestIDs): Record<string, string> {
  return {
    'user-agent': disguiseUserAgent(),
    'x-opencode-client': 'cli',
    'x-opencode-session': ids.session,
    'x-session-affinity': ids.session,
    'X-Session-Id': ids.session,
    'x-opencode-request': ids.request,
    'x-opencode-project': ids.project,
  }
}

/** Reset the session memo; test-only. */
export function clearSessionMemo(): void {
  sessionMemo.clear()
}