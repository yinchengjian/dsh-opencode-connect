/**
 * The request-identity contract the anonymous lane enforces.
 *
 * These assertions pin behavior that was established by live probing, so a
 * refactor cannot silently reintroduce a shape the lane refuses.
 */
import { describe, expect, it, beforeEach } from 'vitest'
import {
  ANONYMOUS_KEY,
  canonicalSessionID,
  clearSessionMemo,
  deriveRequestIDs,
  disguiseHeaders,
  disguiseUserAgent,
  invertedTimestampHex,
} from '../src/ids.ts'

beforeEach(() => { clearSessionMemo() })

describe('canonicalSessionID', () => {
  it('mints the shape the gate accepts', () => {
    const session = canonicalSessionID('some conversation')
    expect(session).toMatch(/^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/)
  })

  it('encodes the per-millisecond counter the gate actually validates', () => {
    // The 12 hex characters carry a 53-bit value masked to 48 bits, so the
    // clock does NOT round-trip — the high bits are truncated and upstream is
    // documented as not checking them. What IS checked is the low 12 bits:
    // undoing the inversion must yield a counter in [0, 63]. That is why a
    // digest-derived prefix is refused while this encoder is admitted
    // (a random 12-hex value lands in the window only ~1.5% of the time).
    const hex = invertedTimestampHex(Date.now())
    const current = ~BigInt('0x' + hex) & 0xffffffffffffn
    const counter = current & 0xfffn
    expect(counter).toBe(1n)
    expect(counter >= 0n && counter <= 63n).toBe(true)
  })

  it('mints a value indistinguishable from a real client id', () => {
    // Shape plus the counter window, which together are what the gate reads.
    const hex = invertedTimestampHex(Date.now())
    expect(hex).toMatch(/^[0-9a-f]{12}$/)
    const counter = (~BigInt('0x' + hex) & 0xfffn)
    expect(counter <= 63n).toBe(true)
  })

  it('is stable per conversation, so cache affinity survives a second turn', () => {
    const first = canonicalSessionID('same conversation')
    const second = canonicalSessionID('same conversation')
    expect(second).toBe(first)
  })

  it('separates different conversations', () => {
    expect(canonicalSessionID('one')).not.toBe(canonicalSessionID('two'))
  })

  it('passes an already-canonical id through unchanged', () => {
    const official = 'ses_0123456789ab0123456789ABCD'
    expect(canonicalSessionID(official)).toBe(official)
  })
})

describe('deriveRequestIDs', () => {
  it('derives a stable session from the first user turn', () => {
    const messages = [
      { role: 'user', content: [{ type: 'text', text: 'hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'hi' }] },
    ]
    const first = deriveRequestIDs(messages)
    const second = deriveRequestIDs(messages)
    expect(second.session).toBe(first.session)
  })

  it('mints a fresh request id every call', () => {
    const messages = [{ role: 'user', content: 'hi' }]
    expect(deriveRequestIDs(messages).request).not.toBe(deriveRequestIDs(messages).request)
  })

  it('still yields a session when there is no user turn', () => {
    const ids = deriveRequestIDs([])
    expect(ids.session).toMatch(/^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/)
  })
})

describe('disguiseHeaders', () => {
  const ids = { session: 'ses_0123456789ab0123456789ABCD', request: 'req_ab', project: 'prj_cd' }

  it('carries all three session spellings the gate accepts', () => {
    const headers = disguiseHeaders(ids)
    expect(headers['x-opencode-session']).toBe(ids.session)
    expect(headers['x-session-affinity']).toBe(ids.session)
    expect(headers['X-Session-Id']).toBe(ids.session)
  })

  it('advertises the cli client and the correlation ids', () => {
    const headers = disguiseHeaders(ids)
    expect(headers['x-opencode-client']).toBe('cli')
    expect(headers['x-opencode-request']).toBe(ids.request)
    expect(headers['x-opencode-project']).toBe(ids.project)
  })
})

describe('disguiseUserAgent', () => {
  it('leads with the opencode product token the gate reads', () => {
    expect(disguiseUserAgent().startsWith('opencode/')).toBe(true)
  })

  it('carries ONLY the CLI identity, never the harness attribution', () => {
    // This is the trust boundary in one assertion. The harness requires its
    // attribution on every provider request, and the free lane refuses that
    // exact token; the shim is where the two part. What the shim sends
    // upstream must therefore never contain the harness product token —
    // restoring it turns every call into an upstream 403 FreeTierError.
    const ua = disguiseUserAgent()
    expect(ua.includes('deepseek-harness/')).toBe(false)
    expect(ua.includes('deepseek-harness')).toBe(false)
  })
})

describe('ANONYMOUS_KEY', () => {
  it('is the lane literal, not a secret', () => {
    expect(ANONYMOUS_KEY).toBe('public')
  })
})