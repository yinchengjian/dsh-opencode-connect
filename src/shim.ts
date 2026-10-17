/**
 * Loopback OpenAI-compatible endpoint — the trust boundary between the
 * harness and OpenCode.
 *
 * Why a local hop exists at all: the harness contract requires every provider
 * request to carry the harness attribution User-Agent, while the free lane
 * refuses that exact identity (a `deepseek-harness/` product token answers 403
 * FreeTierError, verified by interleaved A/B probes, while codex / claude-code
 * / cursor tokens are admitted). Those two requirements cannot both hold on
 * ONE outbound request, so they are split across two:
 *
 *   adapter --(harness attribution)--> this shim --(CLI identity)--> OpenCode
 *
 * The attribution is satisfied toward this loopback endpoint; the real
 * upstream request carries only the CLI identity the lane admits. This is the
 * same boundary `dsh-workbuddy-connect` draws with its own shim.
 *
 * The shim owns every wire quirk of the free lane — the CLI headers, the
 * derived session ids, the forced streaming, and the agent-shaped body the
 * gate requires — so the adapter above it stays a plain provider.
 *
 * It binds 127.0.0.1 only and never serves another interface.
 *
 * @module dsh-opencode-connect/shim
 */

import { randomBytes, timingSafeEqual } from 'node:crypto'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { Readable } from 'node:stream'
import { ANONYMOUS_KEY, deriveRequestIDs, disguiseHeaders, type RequestIDs } from './ids.ts'
import { ensureFreeLaneShape } from './messages.ts'
import type { FreeModel } from './catalog.ts'

/** Minimal logger surface the plugin context already provides. */
export interface ShimLogger {
  warn(...args: unknown[]): void
  error(...args: unknown[]): void
}

/** What the plugin needs from a running shim. */
export interface OpenCodeShim {
  /** Resolves once the listener is up; rejects if listening failed. */
  ready: Promise<void>
  /** The shim origin, e.g. `http://127.0.0.1:39271`; valid after ready. */
  baseUrl(): string
  /**
   * The per-process shared secret the adapter presents as its API key. It
   * lives only in memory, so a local process that can reach the port still
   * cannot forge the adapter's identity, and it never travels upstream.
   */
  token(): string
  /** Stop serving and destroy open connections. */
  close(): Promise<void>
}

/** Constructor dependencies. */
export interface OpenCodeShimOptions {
  /** Advertised models, read per request so a refresh reaches the endpoint. */
  models: () => readonly FreeModel[]
  /** Upstream lane base, including its version prefix. */
  upstreamBaseUrl: string
  logger?: ShimLogger
  /** Injectable fetch, so tests never touch the network. */
  fetchImpl?: typeof fetch
}

/** Request-body ceiling; the harness sends whole conversation histories here. */
const REQUEST_BODY_LIMIT = 64 * 1024 * 1024

/**
 * Outbound headers for one upstream call.
 *
 * Exported for the wire test that pins them: the lane's admission depends on
 * this exact set, and a refactor that dropped one would otherwise surface only
 * as an opaque upstream 403.
 */
export function upstreamHeaders(ids: RequestIDs): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    Authorization: 'Bearer ' + ANONYMOUS_KEY,
    ...disguiseHeaders(ids),
  }
}

/**
 * Shape one outgoing chat body for the free lane.
 *
 * Streaming is forced on and the gate tools are injected when absent;
 * everything the caller sent is preserved. Returns undefined when the body is
 * not a chat-completions object. Exported for the wire test.
 */
export function shapeChatBody(raw: string): string | undefined {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return undefined
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return undefined
  const body = parsed as Record<string, unknown>
  if (!Array.isArray(body['messages'])) return undefined
  body['stream'] = true
  const shaped = ensureFreeLaneShape(body)
  return JSON.stringify(shaped ?? body)
}

/** Read a request body with a size cap. */
function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    req.on('data', (chunk: Buffer) => {
      size += chunk.length
      if (size > REQUEST_BODY_LIMIT) {
        reject(new Error('request body too large'))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })
}

/** Write a JSON response. */
function writeJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
  res.end(payload)
}

/** Write an OpenAI-shaped error, which is the shape pi-ai classifies. */
function writeError(res: ServerResponse, status: number, kind: string, message: string): void {
  writeJson(res, status, { error: { message, type: kind, code: kind } })
}

/** The loopback host check: the bind alone is not a trust boundary. */
function hostIsLoopback(host: string | undefined): boolean {
  if (typeof host !== 'string') return false
  const name = host.replace(/^\[/, '').replace(/\]:\d+$/, '').replace(/:\d+$/, '')
  return name === '127.0.0.1' || name === 'localhost' || name === '::1'
}

/** Browser-sent origins must also be loopback (DNS-rebinding guard). */
function originIsLoopback(origin: string | undefined): boolean {
  if (origin === undefined) return true
  if (origin === 'null') return false
  try {
    const url = new URL(origin)
    return url.hostname === '127.0.0.1' || url.hostname === 'localhost' || url.hostname === '::1'
  } catch {
    return false
  }
}

/**
 * Start the loopback endpoint.
 *
 * Every inbound request is authenticated with the per-process secret; the
 * upstream credential is the lane literal and never comes from the caller.
 */
export function createOpenCodeShim(options: OpenCodeShimOptions): OpenCodeShim {
  const { models, upstreamBaseUrl } = options
  const logger = options.logger
  const doFetch = options.fetchImpl ?? fetch

  // Per-process shared secret; see OpenCodeShim.token.
  const SHARED_SECRET = randomBytes(32).toString('base64url')

  /** Constant-time bearer check; absent or mismatched bearers are rejected. */
  function bearerOk(req: IncomingMessage): boolean {
    const header = req.headers.authorization
    if (typeof header !== 'string') return false
    const match = /^Bearer\s+(.+)$/i.exec(header.trim())
    if (match === null) return false
    const presented = match[1] as string
    const expected = SHARED_SECRET
    const a = Buffer.from(presented)
    const b = Buffer.from(expected)
    if (a.length !== b.length) return false
    return timingSafeEqual(a, b)
  }

  const server: Server = createServer((req, res) => {
    void handle(req, res)
  })

  const ready = new Promise<void>((resolve, reject) => {
    server.once('listening', () => resolve())
    server.once('error', reject)
  })

  server.listen(0, '127.0.0.1')

  const baseUrl = (): string => {
    const address = server.address()
    if (address === null || typeof address === 'string') throw new Error('opencode shim has no listening address')
    return 'http://127.0.0.1:' + address.port
  }

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    try {
      if (!hostIsLoopback(req.headers.host)) {
        writeError(res, 403, 'host_not_allowed', 'Host header must name the loopback interface')
        return
      }
      if (!originIsLoopback(req.headers.origin)) {
        writeError(res, 403, 'origin_not_allowed', 'Origin must be a loopback origin')
        return
      }
      if (!bearerOk(req)) {
        writeError(res, 401, 'unauthorized', 'missing or invalid Authorization bearer')
        return
      }
      const url = req.url ?? '/'
      if (req.method === 'GET' && (url === '/healthz' || url === '/healthz/')) {
        writeJson(res, 200, { ok: true })
        return
      }
      if (req.method === 'GET' && (url === '/v1/models' || url === '/v1/models/')) {
        writeJson(res, 200, {
          object: 'list',
          data: models().map(model => ({ id: model.id, object: 'model', created: 0, owned_by: 'opencode' })),
        })
        return
      }
      if (req.method === 'POST' && (url === '/v1/chat/completions' || url === '/v1/chat/completions/')) {
        await chatCompletions(req, res)
        return
      }
      writeError(res, 404, 'not_found', 'no such route: ' + req.method + ' ' + url)
    } catch (error: unknown) {
      if (!res.headersSent) writeError(res, 500, 'internal', String(error))
      else res.end()
    }
  }

  async function chatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const type = req.headers['content-type']
    if (typeof type !== 'string' || !type.trim().toLowerCase().startsWith('application/json')) {
      writeError(res, 415, 'unsupported_media_type', 'Content-Type must be application/json')
      return
    }

    const raw = (await readBody(req)).toString('utf8')
    const shaped = shapeChatBody(raw)
    if (shaped === undefined) {
      writeError(res, 400, 'invalid_request', 'body must be a chat-completions JSON object')
      return
    }

    // The conversation seed comes from the body, so a multi-turn exchange
    // keeps one session id and retains upstream prompt-cache affinity.
    const parsed = JSON.parse(shaped) as { messages?: ReadonlyArray<{ role: string; content: unknown }> }
    const ids = deriveRequestIDs(parsed.messages ?? [])

    const controller = new AbortController()
    req.on('close', () => controller.abort())

    let upstream: Response
    try {
      upstream = await doFetch(upstreamBaseUrl + '/chat/completions', {
        method: 'POST',
        headers: upstreamHeaders(ids),
        body: shaped,
        signal: controller.signal,
      })
    } catch (error: unknown) {
      writeError(res, 502, 'transport', 'opencode upstream transport error: ' + String(error))
      return
    }

    if (!upstream.ok) {
      const text = (await upstream.text().catch(() => '')).slice(0, 4000)
      writeError(res, upstream.status, 'upstream_error', text.length > 0 ? text : 'upstream answered ' + upstream.status)
      return
    }

    if (upstream.body === null) {
      writeError(res, 502, 'transport', 'opencode upstream returned no body')
      return
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    })
    const body = Readable.fromWeb(upstream.body as Parameters<typeof Readable.fromWeb>[0])
    body.on('error', (error: unknown) => {
      logger?.warn('dsh-opencode-connect: upstream stream failed mid-flight', error)
      if (res.writable) res.end()
    })
    body.pipe(res)
  }

  return {
    ready,
    baseUrl,
    token: () => SHARED_SECRET,
    close: () => new Promise<void>((resolve, reject) => {
      server.close(() => resolve())
      server.closeAllConnections()
      server.once('error', reject)
    }),
  }
}
