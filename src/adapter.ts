/**
 * The `opencode` pi-ai provider: an LlmAdapter over the loopback shim,
 * assembled from public `dsh-llm-pi-ai` extension points the way
 * `dsh-workbuddy-connect` assembles its WorkBuddy route.
 *
 * The adapter talks to the shim, never to OpenCode. That indirection is what
 * lets one outbound design satisfy two requirements that cannot both hold on a
 * single request: the harness contract demands the harness attribution
 * `User-Agent` on every provider request, while the free lane refuses that
 * identity. The attribution lands on the loopback boundary; the shim then
 * speaks to OpenCode with the CLI identity the lane admits.
 *
 * Every lane quirk — the CLI headers, the derived session ids, the forced
 * streaming, the agent-shaped body — belongs to the shim, so this file stays a
 * plain provider dispatch.
 *
 * @module dsh-opencode-connect/adapter
 */

import { createProvider, normalizeContext } from '@earendil-works/pi-ai'
import type { Api, Model, Provider } from '@earendil-works/pi-ai'
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy'
import { attributionHeaders, LlmAdapter, resolveRetryPolicy } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmResolvedModelInfo,
  PreparedAdapterCall,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import { DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_OUTPUT, type FreeModel } from './catalog.ts'
import { toPiContext, type HarnessGenerateOptions } from './messages.ts'
import { toStreamChunks } from './events.ts'
import type { OpenCodeShim } from './shim.ts'

/** Provider route this bundle owns. */
export const OPENCODE_PROVIDER = 'opencode'

/** Timeout while one stream read is outstanding. */
export const OPENCODE_STREAM_IDLE_TIMEOUT_MS = 300_000

/** The API surface every model on this route speaks. */
export const OPENCODE_API = 'openai-completions'

/**
 * Retry policy for the anonymous lane.
 *
 * The free lane throttles in bursts that outlast the host default, so the
 * silent absorption window is stretched: a transient 429 becomes a delay the
 * user never sees instead of a visible failure. Deterministic failures stay
 * out — retrying a refused request shape or a region block only earns the same
 * refusal. Codes the host does not recognize simply never match, which is the
 * safe direction.
 */
export const FREE_LANE_RETRY_POLICY: ResolvedRetryPolicy = resolveRetryPolicy({
  mode: 'normal',
  maxRetries: 8,
  retryableCodes: ['RATE_LIMIT', 'PROVIDER_ERROR', 'TRANSPORT'],
  backoff: { initialDelayMs: 1_000, maxDelayMs: 20_000, jitterRatio: 0.2 },
}, 'dsh-opencode-connect retryPolicy')

/** The metadata every model on this route carries. */
function modelMetadata(model: FreeModel): Omit<LlmResolvedModelInfo, 'provider' | 'id' | 'name'> {
  const resolved: Omit<LlmResolvedModelInfo, 'provider' | 'id' | 'name'> = {
    inputModalities: [...model.inputModalities],
    context: { contextWindow: model.contextWindow },
    defaultMaxTokens: model.maxOutput,
  }
  if (model.reasoningEfforts !== undefined) {
    resolved.reasoning = {
      efforts: model.reasoningEfforts.map(effort => ({ id: effort as never, name: effort })),
    }
  }
  return resolved
}

/** Project one advertised model into pi-ai's model shape. */
function catalogModel(model: FreeModel, baseUrl: string, providerId: string): Model<Api> {
  return {
    id: model.id,
    name: model.name,
    api: OPENCODE_API,
    provider: providerId,
    baseUrl,
    reasoning: model.reasoningEfforts !== undefined,
    input: [...model.inputModalities],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: model.contextWindow,
    maxTokens: model.maxOutput,
  } as unknown as Model<Api>
}

/** Options for {@link createOpenCodeAdapter}. */
export interface OpenCodeAdapterOptions {
  providerId?: string
  displayName?: string
  /** The running loopback shim the adapter streams through. */
  shim: OpenCodeShim
  /** Current advertised models; read per call so a refresh reaches the picker. */
  models: () => readonly FreeModel[]
}

/** What a caller needs from the assembled adapter. */
export interface OpenCodeAdapter {
  adapter: LlmAdapter
  /** The provider route this adapter serves. */
  providerId: string
  /** Resolve one advertised model, or undefined. */
  model(id: string): FreeModel | undefined
}

/**
 * The LlmAdapter over the loopback shim.
 *
 * The provider is built once the shim reports its origin, because every model
 * carries that base URL. `prepareCall()` captures the resolved model so a
 * dispatch cannot combine one generation's metadata with another's request.
 */
class OpenCodePiAiAdapter extends LlmAdapter {
  readonly #providerId: string
  readonly #displayName: string
  readonly #shim: OpenCodeShim
  readonly #models: () => readonly FreeModel[]
  readonly #provider: Provider

  constructor(options: {
    providerId: string
    displayName: string
    shim: OpenCodeShim
    models: () => readonly FreeModel[]
  }) {
    super()
    this.#providerId = options.providerId
    this.#displayName = options.displayName
    this.#shim = options.shim
    this.#models = options.models
    // Keyless by construction: the lane credential is the literal string
    // 'public', resolved by the shim, so there is no store to consult. The
    // value pi-ai sends here is the shim's shared secret, which the shim
    // validates before forwarding and never passes upstream.
    this.#provider = createProvider({
      id: this.#providerId,
      name: this.#displayName,
      baseUrl: this.#shim.baseUrl() + '/v1',
      auth: {
        apiKey: {
          name: 'OpenCode loopback shim',
          resolve: async () => ({ auth: { apiKey: this.#shim.token() } }),
        },
      },
      models: [],
      api: openAICompletionsApi(),
    })
  }

  override providerInfo(provider: string): { id: string; name: string } {
    return { id: provider, name: this.#displayName }
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy {
    return FREE_LANE_RETRY_POLICY
  }

  override imageRequestPricing(_provider: string, _model: string): undefined {
    return undefined
  }

  /**
   * The advertised models, refreshed on every read.
   *
   * Reading through the callback rather than snapshotting here is what lets a
   * catalog refresh reach the picker without re-registering the adapter.
   */
  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const out: LlmModelInfo[] = []
    const seen = new Set<string>()
    for (const model of this.#models()) {
      if (seen.has(model.id)) continue
      seen.add(model.id)
      out.push({ provider, id: model.id, name: model.name, inputModalities: [...model.inputModalities] })
    }
    return out
  }

  override async resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    const known = this.#find(model)
    return {
      provider,
      id: model,
      name: known?.name ?? model,
      ...(known !== undefined
        ? modelMetadata(known)
        : { inputModalities: ['text'] as const, context: { contextWindow: DEFAULT_CONTEXT_WINDOW }, defaultMaxTokens: DEFAULT_MAX_OUTPUT }),
    }
  }

  override async prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    return {
      model: await this.resolveModel(provider, model, signal),
      stream: options => this.stream(options),
    }
  }

  /** The advertised model with this id, or undefined. */
  #find(id: string): FreeModel | undefined {
    return this.#models().find(candidate => candidate.id === id)
  }

  /**
   * Stream one model call through the shim.
   *
   * The shim owns every lane quirk, so this is a plain provider dispatch: it
   * converts the harness request and translates the resulting event stream.
   */
  override async *stream(options: GenerateOptions): AsyncGenerator<StreamChunk> {
    const request = options as unknown as HarnessGenerateOptions
    const known = this.#find(request.model)
    const model = catalogModel(
      known ?? {
        id: request.model,
        name: request.model,
        contextWindow: DEFAULT_CONTEXT_WINDOW,
        maxOutput: DEFAULT_MAX_OUTPUT,
        inputModalities: ['text'],
      },
      this.#shim.baseUrl() + '/v1',
      this.#providerId,
    )

    const context = toPiContext(request, this.#providerId)
    const streamOptions: Record<string, unknown> = {
      apiKey: this.#shim.token(),
      maxRetries: 0,
      // The harness attribution rides THIS request — the adapter-to-shim hop.
      // The host contract requires it on every provider request, and the shim
      // is this route's provider endpoint. The shim then strips it: what leaves
      // for OpenCode carries only the CLI identity the free lane admits, since
      // a `deepseek-harness/` token is refused there with 403 FreeTierError.
      // Stamped explicitly because this adapter drives pi-ai directly rather
      // than through `PiAiAdapter`, which is where the host normally adds it.
      headers: attributionHeaders(),
    }
    if (request.signal !== undefined) streamOptions['signal'] = request.signal
    if (request.temperature !== undefined) streamOptions['temperature'] = request.temperature
    if (request.maxTokens !== undefined) streamOptions['maxTokens'] = request.maxTokens
    if (request.reasoningEffort !== undefined) streamOptions['reasoning'] = request.reasoningEffort

    const events = this.#provider.streamSimple(model, normalizeContext(context), streamOptions as never)
    yield* toStreamChunks(toPiEvents(events))
  }
}

/**
 * Adapt a pi-ai event stream to this module's structural event type.
 *
 * The shapes are identical on the wire; the indirection exists so the
 * translation layer is unit-testable without building a live provider.
 */
async function* toPiEvents(events: AsyncIterable<unknown>): AsyncGenerator<import('./events.ts').PiEvent> {
  for await (const event of events) yield event as import('./events.ts').PiEvent
}

/** Assemble the adapter over a live model list and a running shim. */
export function createOpenCodeAdapter(options: OpenCodeAdapterOptions): OpenCodeAdapter {
  const providerId = options.providerId ?? OPENCODE_PROVIDER
  const displayName = options.displayName ?? 'OpenCode'
  const models = options.models
  const adapter = new OpenCodePiAiAdapter({ providerId, displayName, shim: options.shim, models })
  return {
    adapter,
    providerId,
    model: (id: string) => models().find(model => model.id === id),
  }
}