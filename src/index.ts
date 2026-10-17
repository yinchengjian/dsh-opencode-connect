/**
 * dsh-opencode-connect: OpenCode's free models for DeepSeek Harness.
 *
 * Registers one native `LlmAdapter` on the `opencode` route. The adapter
 * streams through a loopback shim, which owns the free lane's wire quirks and
 * is the boundary that keeps the harness attribution and the CLI identity on
 * separate requests. No API key, no sign-in, no child process, and no binary —
 * the shim is an in-process listener on a random loopback port.
 *
 * @module dsh-opencode-connect
 */

import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { createOpenCodeAdapter, OPENCODE_PROVIDER } from './adapter.ts'
import { createOpenCodeShim } from './shim.ts'
import {
  CACHE_TTL_MS,
  MODELS_DEV_URL,
  ZEN_BASE_URL,
  catalogCachePath,
  parseLiveModels,
  parseModelsDev,
  selectFreeModels,
  staticModels,
  type CatalogSnapshot,
  type FreeModel,
} from './catalog.ts'

/** Stable Cordis plugin name. */
export const name = 'llm-opencode'

/** The model registry required before the provider can register. */
export const inject = ['llm']

/** How often the live directory is refreshed. */
const DEFAULT_REFRESH_MS = 300_000

/** Bounded first fetch: a hung network must not hold the picker hostage. */
const FETCH_TIMEOUT_MS = 10_000

/**
 * A GET with a bounded wait.
 *
 * Returns undefined instead of throwing: both directory reads are best-effort,
 * and a failure has a defined fallback one layer up.
 */
async function getJson(url: string): Promise<unknown | undefined> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)
  try {
    const response = await fetch(url, { signal: controller.signal })
    if (!response.ok) return undefined
    return (await response.json()) as unknown
  } catch {
    return undefined
  } finally {
    clearTimeout(timer)
  }
}

/** Read a cached snapshot, or undefined when absent, unusable, or stale. */
async function readCache(path: string, now: number): Promise<CatalogSnapshot | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, 'utf8'))
    if (typeof parsed !== 'object' || parsed === null) return undefined
    const snapshot = parsed as CatalogSnapshot
    if (!Array.isArray(snapshot.models) || snapshot.models.length === 0) return undefined
    if (typeof snapshot.updatedAt !== 'number') return undefined
    if (now - snapshot.updatedAt > CACHE_TTL_MS) return undefined
    return snapshot
  } catch {
    return undefined
  }
}

/**
 * Persist a snapshot without risking the live file.
 *
 * Written to a sibling and renamed, so a crash mid-write cannot leave a
 * truncated document the next boot would read as corrupt.
 */
async function writeCache(path: string, snapshot: CatalogSnapshot): Promise<void> {
  try {
    await mkdir(dirname(path), { recursive: true })
    const temporary = path + '.tmp'
    await writeFile(temporary, JSON.stringify(snapshot), 'utf8')
    await rename(temporary, path)
  } catch {
    // A cache that cannot be written is not a failure: the live layer still
    // serves this process, and the next boot simply refetches.
  }
}

/**
 * Owns the advertised model list and its refresh loop.
 *
 * The list always has an answer — a cold start with no network advertises the
 * verified static set — because a model picker with no group is worse than one
 * showing a few models that may be stale.
 */
class Directory {
  #models: readonly FreeModel[] = staticModels()
  #source: CatalogSnapshot['source'] = 'static'
  #timer: ReturnType<typeof setInterval> | undefined

  constructor(
    private readonly log: (message: string) => void,
    private readonly cachePath: string,
    private readonly refreshMs: number = DEFAULT_REFRESH_MS,
  ) {}

  /** The current advertised models. */
  current(): readonly FreeModel[] {
    return this.#models
  }

  /** Which layer produced the current set; diagnostics only. */
  source(): CatalogSnapshot['source'] {
    return this.#source
  }

  /** Adopt a cached snapshot, run one refresh, and start the periodic loop. */
  async start(): Promise<void> {
    const cached = await readCache(this.cachePath, Date.now())
    if (cached !== undefined) {
      this.#models = cached.models
      this.#source = 'cache'
    }
    await this.refresh()
    this.#timer = setInterval(() => { void this.refresh() }, this.refreshMs)
    this.#timer.unref?.()
  }

  /** Stop the refresh loop. */
  stop(): void {
    if (this.#timer !== undefined) clearInterval(this.#timer)
    this.#timer = undefined
  }

  /**
   * One refresh: the live catalog filtered by pricing metadata.
   *
   * The two reads are independent and both may fail. Metadata is what makes the
   * free decision, so a live list without it cannot be filtered; the existing
   * set is kept rather than replaced with models whose price is unknown.
   */
  async refresh(): Promise<void> {
    const [live, metadataBody] = await Promise.all([
      getJson(ZEN_BASE_URL + '/v1/models'),
      getJson(MODELS_DEV_URL),
    ])
    const liveIds = parseLiveModels(live)
    const metadata = parseModelsDev(metadataBody)

    if (liveIds.length === 0 || metadata.size === 0) {
      this.log(
        'dsh-opencode-connect: directory refresh incomplete (live=' + liveIds.length +
        ', metadata=' + metadata.size + '); keeping ' + this.#models.length + ' model(s)',
      )
      return
    }

    const models = selectFreeModels(liveIds, metadata)
    if (models.length === 0) {
      this.log('dsh-opencode-connect: refresh found no free models; keeping the previous set')
      return
    }

    this.#models = models
    this.#source = 'live'
    await writeCache(this.cachePath, { models, source: 'live', updatedAt: Date.now() })
  }
}

/**
 * Register the OpenCode provider.
 *
 * The registration must wait for the shim's listener, because every model
 * carries the loopback base URL and only the bound port provides it. That wait
 * is why registration happens in the startup task rather than synchronously:
 * there is no model route to publish before a port exists. Until it lands the
 * plugin contributes no provider, which is the honest state — the models do not
 * work yet either.
 */
export function apply(ctx: Context): void {
  const directory = new Directory(
    message => { ctx.logger.info(message) },
    catalogCachePath(),
  )

  const shim = createOpenCodeShim({
    models: () => directory.current(),
    upstreamBaseUrl: ZEN_BASE_URL + '/v1',
    logger: ctx.logger,
  })

  let stopped = false
  const start = async (): Promise<void> => {
    // A listener failure must not reject this task: the plugin simply serves
    // nothing, and the reason reaches the log instead of an unhandled promise.
    await shim.ready
    if (stopped) {
      await shim.close()
      return
    }

    const opencode = createOpenCodeAdapter({ shim, models: () => directory.current() })
    const release = ctx.llm.registerAdapter([OPENCODE_PROVIDER], opencode.adapter)
    ctx.effect(() => () => {
      release()
      directory.stop()
      void shim.close()
    })

    await directory.start()
    ctx.emit('llm/adapters-updated')
    ctx.logger.info(
      'dsh-opencode-connect: ' + directory.current().length +
      ' free model(s) on "' + OPENCODE_PROVIDER + '" (' + directory.source() + ')',
    )
  }

  void start().catch((error: unknown) => {
    stopped = true
    ctx.logger.error('dsh-opencode-connect: startup failed; no model route was registered', error)
    void shim.close()
  })
}

export { createOpenCodeAdapter, OPENCODE_PROVIDER } from './adapter.ts'
export { createOpenCodeShim } from './shim.ts'
export {
  ZEN_BASE_URL,
  MODELS_DEV_URL,
  catalogCachePath,
  parseLiveModels,
  parseModelsDev,
  selectFreeModels,
  staticModels,
} from './catalog.ts'