/**
 * The free-model directory, resolved through three layers:
 *
 *   S1  GET {base}/v1/models             live catalog (free AND paid ids)
 *   S2  GET https://models.dev/api.json  pricing metadata -> the free decision
 *   S3  a compile-time verified list        bootstrap when neither read works
 *
 * The exposed set is S1 filtered by S2: only models whose metadata prices
 * input and output at zero, and which are not deprecated, are advertised —
 * the picker must never offer an id that costs money. S3 keeps the group
 * usable on a first launch with no network, and a cached snapshot covers a
 * later outage.
 *
 * @module dsh-opencode-connect/catalog
 */

import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { homedir } from 'node:os'

/** Anonymous-lane base; the OpenAI client appends /chat/completions. */
export const ZEN_BASE_URL = 'https://opencode.ai/zen'

/** Pricing metadata feed. */
export const MODELS_DEV_URL = 'https://models.dev/api.json'

/** The provider key this directory reads in that feed. */
export const MODELS_DEV_PROVIDER = 'opencode'

/** How long a cached snapshot stays usable. */
export const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000

/** A model input kind. */
export type ModelModality = 'text' | 'image'

/** Where the snapshot is cached. */
export function catalogCachePath(home: string = homedir()): string {
  return join(home, '.dsh-opencode-connect', 'catalog.json')
}

/**
 * Ids this lane can never serve — permanently, for every caller.
 *
 * Deliberately tiny, and the criterion is narrow: an id belongs here only when
 * it is dead no matter who asks.
 *
 * Two kinds of failure that look similar do NOT belong here:
 *
 *   - region blocks (`403 RegionError`). The lane decides these from the
 *     caller's egress, so an id refused from one network is serviceable from
 *     another. Listing one here would let this machine's location remove a
 *     model the user could actually run.
 *   - transient unavailability (`400 Model is unavailable`). Those recover.
 *
 * Both stay in the roster and the user chooses, which is the same call
 * `dsh-workbuddy-connect` makes with its own list. Removing an entry still
 * requires a re-probe, never a guess.
 */
export const FREE_LANE_BLOCKLIST: ReadonlySet<string> = new Set([
  'exo-free', // delisted upstream (410 ModelDeprecated)
  'jev-1.13-free', // protocol unsupported (SystemOne-only)
  'fledge-alpha-free', // delisted upstream (410 ModelDeprecated)
  'll-alpha-free', // delisted upstream (410 ModelDeprecated)
  'mimo-v2.5-free', // delisted upstream, superseded by mimo-v2.6
])

/** One bootstrap entry, carrying the window it was verified at. */
export interface StaticFreeModel {
  id: string
  contextWindow: number
  maxOutput: number
}

/**
 * Last-resort models, each verified on the anonymous chat lane.
 *
 * Deliberately short: the live directory republishes the picker on every
 * refresh, so this only makes a cold start usable.
 *
 * Each entry carries its OWN window rather than sharing a default. A single
 * default silently understates the outliers — `nemotron-3.5-lightning-free`
 * really offers 262K, and flattening that to a 200K shared default would make
 * compaction fire a third earlier than it needs to.
 */
export const STATIC_FREE_MODELS: readonly StaticFreeModel[] = [
  { id: 'deepseek-v4.1-flash-free', contextWindow: 200_000, maxOutput: 32_000 },
  { id: 'mimo-v2.6-flash-free', contextWindow: 200_000, maxOutput: 32_000 },
  { id: 'nemotron-3.5-lightning-free', contextWindow: 262_144, maxOutput: 262_144 },
  { id: 'glm-5.3-flash-free', contextWindow: 200_000, maxOutput: 32_000 },
  { id: 'big-pickle', contextWindow: 200_000, maxOutput: 32_000 },
]

/** Per-model facts resolved from pricing metadata. */
export interface ModelMetadata {
  /** Metadata prices input AND output at zero, and the model is not deprecated. */
  free: boolean
  contextWindow?: number
  maxOutput?: number
  reasoning: boolean
  effortValues: readonly string[]
  modalities: readonly string[]
}

/** One advertised model. */
export interface FreeModel {
  id: string
  name: string
  contextWindow: number
  maxOutput: number
  inputModalities: readonly ModelModality[]
  reasoningEfforts?: readonly string[]
}

/** One directory snapshot. */
export interface CatalogSnapshot {
  models: readonly FreeModel[]
  /** Which layer produced this set; diagnostics only. */
  source: 'live' | 'cache' | 'static'
  updatedAt: number
}

/**
 * Context window assumed for a model the metadata does not size.
 *
 * Deliberately conservative, and the asymmetry is why. This value reaches
 * `dsh-compaction-basic`, which budgets `contextWindow - reservedCompletion`:
 *
 *   - claiming LESS than the truth only makes compaction fire early — the
 *     conversation loses depth but keeps working;
 *   - claiming MORE than the truth means compaction never fires and the
 *     request eventually exceeds what upstream accepts, which fails the turn.
 *
 * Since the two costs are not symmetric, an unknown model is sized at the
 * floor of what this lane is known to serve rather than at the optimistic
 * 200K that most models happen to declare.
 */
export const DEFAULT_CONTEXT_WINDOW = 131_072

/** Default output cap when metadata does not size a model. */
export const DEFAULT_MAX_OUTPUT = 32_000

/** Models known to accept image input; used only when metadata is silent. */
const VISION_PATTERN = /(?:vl|vision|omni)/i

/** One model entry as the pricing feed declares it. */
interface RawModelEntry {
  reasoning?: unknown
  deprecated?: unknown
  reasoning_options?: unknown
  modalities?: unknown
  limit?: unknown
  cost?: unknown
}

/** One reasoning option the feed declares. */
interface RawReasoningOption {
  type?: unknown
  values?: unknown
}

/** Read a positive finite number, or undefined. */
function positiveNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

/** Read a non-negative finite number, or undefined. */
function nonNegativeNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

/** A plain object usable as a record. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * The free decision for one entry, from positive evidence only.
 *
 * Both prices must be present and zero. Absent pricing is NOT free:
 * over-claiming spends the user money, while under-claiming only hides a
 * model until the feed describes it.
 */
export function isFreeEntry(raw: RawModelEntry): boolean {
  if (raw.deprecated === true) return false
  const cost = raw.cost
  if (!isRecord(cost)) return false
  const input = nonNegativeNumber(cost['input'])
  const output = nonNegativeNumber(cost['output'])
  if (input === undefined || output === undefined) return false
  return input === 0 && output === 0
}

/**
 * Parse the pricing feed into per-model metadata.
 *
 * Tolerant by construction: the feed is an external document this plugin
 * does not control, so a malformed entry is skipped rather than allowed to
 * fail the whole catalog.
 */
export function parseModelsDev(body: unknown): Map<string, ModelMetadata> {
  const out = new Map<string, ModelMetadata>()
  if (!isRecord(body)) return out
  const provider = body[MODELS_DEV_PROVIDER]
  if (!isRecord(provider)) return out
  const models = provider['models']
  if (!isRecord(models)) return out

  for (const [id, value] of Object.entries(models)) {
    if (!isRecord(value)) continue
    const raw = value as RawModelEntry
    const limit = isRecord(raw.limit) ? raw.limit : {}
    const modalitiesContainer = isRecord(raw.modalities) ? raw.modalities : {}
    const declaredModalities = modalitiesContainer['input']
    const effortValues: string[] = []
    if (Array.isArray(raw.reasoning_options)) {
      for (const option of raw.reasoning_options) {
        if (!isRecord(option)) continue
        if (option['type'] !== 'effort') continue
        const values = option['values']
        if (!Array.isArray(values)) continue
        for (const v of values) if (typeof v === 'string') effortValues.push(v)
      }
    }
    const metadata: ModelMetadata = {
      free: isFreeEntry(raw),
      reasoning: raw.reasoning === true,
      effortValues,
      modalities: Array.isArray(declaredModalities)
        ? declaredModalities.filter((m): m is string => typeof m === 'string')
        : [],
    }
    const contextWindow = positiveNumber(limit['context'])
    const maxOutput = positiveNumber(limit['output'])
    if (contextWindow !== undefined) metadata.contextWindow = contextWindow
    if (maxOutput !== undefined) metadata.maxOutput = maxOutput
    out.set(id, metadata)
  }
  return out
}

/** Shape of the live /v1/models reply. */
/**
 * Extract the ids a live catalog reply carries, in order, deduped.
 */
export function parseLiveModels(body: unknown): string[] {
  const ids: string[] = []
  const seen = new Set<string>()
  if (!isRecord(body)) return ids
  const data = body['data']
  if (!Array.isArray(data)) return ids
  for (const entry of data) {
    if (!isRecord(entry)) continue
    const id = entry['id']
    if (typeof id !== 'string' || id.length === 0) continue
    if (seen.has(id)) continue
    seen.add(id)
    ids.push(id)
  }
  return ids
}

/** Resolve one advertised model from its id and metadata. */
export function toFreeModel(id: string, metadata: ModelMetadata | undefined): FreeModel {
  const declared = metadata?.modalities ?? []
  const image = declared.length > 0 ? declared.includes('image') : VISION_PATTERN.test(id)
  const inputModalities: ModelModality[] = image ? ['text', 'image'] : ['text']
  const model: FreeModel = {
    id,
    name: id,
    contextWindow: metadata?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxOutput: metadata?.maxOutput ?? DEFAULT_MAX_OUTPUT,
    inputModalities,
  }
  if (metadata !== undefined && metadata.reasoning && metadata.effortValues.length > 0) {
    model.reasoningEfforts = metadata.effortValues
  }
  return model
}

/**
 * Build the advertised set from a live id list and the pricing metadata.
 *
 * Order follows the live listing. The blocklist always wins, and an id is
 * kept only on a positive free decision — never on its name alone, because
 * a name is not evidence of price.
 */
export function selectFreeModels(
  liveIds: readonly string[],
  metadata: ReadonlyMap<string, ModelMetadata>,
): FreeModel[] {
  const out: FreeModel[] = []
  for (const id of liveIds) {
    if (FREE_LANE_BLOCKLIST.has(id)) continue
    const known = metadata.get(id)
    if (known === undefined || !known.free) continue
    out.push(toFreeModel(id, known))
  }
  return out
}

/**
 * Build the S3 fallback set, whose windows are trusted by construction.
 *
 * Each entry supplies its own verified window, so the fallback never flattens
 * a 262K model down to a shared default.
 */
export function staticModels(): FreeModel[] {
  return STATIC_FREE_MODELS.map(entry => ({
    id: entry.id,
    name: entry.id,
    contextWindow: entry.contextWindow,
    maxOutput: entry.maxOutput,
    inputModalities: ['text'] as const,
  }))
}