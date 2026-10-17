/**
 * The loopback shim: the trust boundary that keeps the harness attribution
 * and the upstream CLI identity on separate requests.
 *
 * These tests drive the real listener over HTTP with a stubbed upstream, so
 * they pin the behavior that matters without touching the network.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createOpenCodeShim, shapeChatBody, upstreamHeaders, type OpenCodeShim } from '../src/shim.ts'
import type { FreeModel } from '../src/catalog.ts'

const MODELS: FreeModel[] = [
  { id: 'deepseek-v4.1-flash-free', name: 'deepseek-v4.1-flash-free', contextWindow: 200000, maxOutput: 32000, inputModalities: ['text'] },
]

/** A shim over a stubbed upstream, plus the fetch probe it records. */
async function startShim(upstream: (url: string, init: RequestInit) => Response | Promise<Response>) {
  const calls: Array<{ url: string; init: RequestInit }> = []
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    calls.push({ url, init: init ?? {} })
    return await upstream(url, init ?? {})
  }) as unknown as typeof fetch
  const shim = createOpenCodeShim({
    models: () => MODELS,
    upstreamBaseUrl: 'https://opencode.ai/zen/v1',
    fetchImpl,
  })
  await shim.ready
  return { shim, calls }
}

let active: OpenCodeShim | undefined
afterEach(async () => { if (active !== undefined) { await active.close(); active = undefined } })

/** A 200 SSE response carrying one text delta and a stop. */
function sseResponse(): Response {
  const body = 'data: ' + JSON.stringify({ choices: [{ index: 0, delta: { content: 'hi' }, finish_reason: null }] }) + '\n\n' +
    'data: [DONE]\n\n'
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

describe('upstreamHeaders', () => {
  it('sends the CLI identity the lane admits, never the harness attribution', () => {
    const headers = upstreamHeaders({ session: 'ses_0123456789ab0123456789ABCD', request: 'req_a', project: 'prj_b' })
    expect(headers['user-agent']!.startsWith('opencode/')).toBe(true)
    // The whole point of the boundary: the token the harness requires and the
    // lane refuses must not reach upstream.
    expect(headers['user-agent']!.includes('deepseek-harness/')).toBe(false)
    expect(headers['Authorization']).toBe('Bearer public')
    expect(headers['x-opencode-client']).toBe('cli')
  })
})

describe('shapeChatBody', () => {
  it('forces streaming on', () => {
    const shaped = JSON.parse(shapeChatBody(JSON.stringify({ messages: [], stream: false }))!)
    expect(shaped.stream).toBe(true)
  })

  it('injects the gate tools a tool-less body is missing', () => {
    const shaped = JSON.parse(shapeChatBody(JSON.stringify({ messages: [] }))!)
    expect(shaped.tools.map((t: { function: { name: string } }) => t.function.name)).toEqual(['bash', 'read'])
  })

  it('rejects a body that is not a chat-completions object', () => {
    expect(shapeChatBody('not json')).toBeUndefined()
    expect(shapeChatBody('[]')).toBeUndefined()
    expect(shapeChatBody(JSON.stringify({ input: [] }))).toBeUndefined()
  })
})

describe('shim listener', () => {
  it('refuses a request without the shared secret', async () => {
    const { shim } = await startShim(() => sseResponse())
    active = shim
    const response = await fetch(shim.baseUrl() + '/v1/models')
    expect(response.status).toBe(401)
  })

  it('refuses a request carrying the wrong secret', async () => {
    const { shim } = await startShim(() => sseResponse())
    active = shim
    const response = await fetch(shim.baseUrl() + '/v1/models', { headers: { authorization: 'Bearer nope' } })
    expect(response.status).toBe(401)
  })

  it('serves the advertised models behind the secret', async () => {
    const { shim } = await startShim(() => sseResponse())
    active = shim
    const response = await fetch(shim.baseUrl() + '/v1/models', { headers: { authorization: 'Bearer ' + shim.token() } })
    expect(response.status).toBe(200)
    const body = await response.json() as { data: Array<{ id: string }> }
    expect(body.data.map(m => m.id)).toEqual(['deepseek-v4.1-flash-free'])
  })

  it('binds to loopback on a random port', async () => {
    const { shim } = await startShim(() => sseResponse())
    active = shim
    expect(shim.baseUrl()).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/)
  })

  it('forwards chat to upstream and streams the response back', async () => {
    const { shim, calls } = await startShim(() => sseResponse())
    active = shim
    const response = await fetch(shim.baseUrl() + '/v1/chat/completions', {
      method: 'POST',
      headers: { authorization: 'Bearer ' + shim.token(), 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'deepseek-v4.1-flash-free', messages: [{ role: 'user', content: 'hi' }] }),
    })
    expect(response.status).toBe(200)
    expect(await response.text()).toContain('hi')
    expect(calls).toHaveLength(1)
    expect(calls[0]!.url).toBe('https://opencode.ai/zen/v1/chat/completions')
  })

  it('shapes the forwarded body for the gate', async () => {
    const { shim, calls } = await startShim(() => sseResponse())
    active = shim
    await fetch(shim.baseUrl() + '/v1/chat/completions', {
      method: 'POST',
      headers: { authorization: 'Bearer ' + shim.token(), 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'm', messages: [{ role: 'user', content: 'hi' }] }),
    })
    const forwarded = JSON.parse(String(calls[0]!.init.body)) as { stream: boolean; tools: unknown[] }
    expect(forwarded.stream).toBe(true)
    expect(forwarded.tools).toHaveLength(2)
  })

  it('rejects a non-JSON content type', async () => {
    const { shim } = await startShim(() => sseResponse())
    active = shim
    const response = await fetch(shim.baseUrl() + '/v1/chat/completions', {
      method: 'POST',
      headers: { authorization: 'Bearer ' + shim.token(), 'content-type': 'text/plain' },
      body: 'hi',
    })
    expect(response.status).toBe(415)
  })

  it('surfaces an upstream refusal as an OpenAI-shaped error', async () => {
    const { shim } = await startShim(() => new Response('{"type":"FreeUsageLimitError"}', { status: 429 }))
    active = shim
    const response = await fetch(shim.baseUrl() + '/v1/chat/completions', {
      method: 'POST',
      headers: { authorization: 'Bearer ' + shim.token(), 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    })
    expect(response.status).toBe(429)
    const body = await response.json() as { error: { message: string } }
    expect(body.error.message).toContain('FreeUsageLimitError')
  })

  it('reports an upstream transport failure as 502', async () => {
    const { shim } = await startShim(() => { throw new Error('boom') })
    active = shim
    const response = await fetch(shim.baseUrl() + '/v1/chat/completions', {
      method: 'POST',
      headers: { authorization: 'Bearer ' + shim.token(), 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    })
    expect(response.status).toBe(502)
  })

  it('answers healthz behind the secret', async () => {
    const { shim } = await startShim(() => sseResponse())
    active = shim
    const response = await fetch(shim.baseUrl() + '/healthz', { headers: { authorization: 'Bearer ' + shim.token() } })
    expect(response.status).toBe(200)
  })

  it('404s an unknown route', async () => {
    const { shim } = await startShim(() => sseResponse())
    active = shim
    const response = await fetch(shim.baseUrl() + '/nope', { headers: { authorization: 'Bearer ' + shim.token() } })
    expect(response.status).toBe(404)
  })
})

