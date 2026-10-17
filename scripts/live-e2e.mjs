/**
 * Live end-to-end check through the real shim boundary.
 *
 * Drives the BUILT plugin the way the harness does and asserts a real answer
 * comes back, so a passing unit suite cannot hide a broken wire. Also asserts
 * the boundary itself: the request the ADAPTER makes carries the harness
 * attribution, and the request the SHIM sends upstream does not.
 *
 * Requires network reach to opencode.ai; run with:
 *
 *   node scripts/live-e2e.mjs
 */
import { createOpenCodeShim } from '../lib/index.js'
import { createOpenCodeAdapter } from '../lib/index.js'

const providerId = 'opencode'
const models = [
  { id: 'deepseek-v4.1-flash-free', name: 'deepseek-v4.1-flash-free', contextWindow: 200000, maxOutput: 32000, inputModalities: ['text'] },
]

function assert(condition, message) {
  if (!condition) throw new Error('FAIL: ' + message)
  console.log('  ok  ' + message)
}

// Observe both hops by wrapping fetch before anything starts.
const realFetch = globalThis.fetch
const seen = []
globalThis.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url
  const headers = new Headers(init?.headers ?? {})
  seen.push({ url, ua: headers.get('user-agent') ?? '' })
  return realFetch(input, init)
}

const shim = createOpenCodeShim({
  models: () => models,
  upstreamBaseUrl: 'https://opencode.ai/zen/v1',
})
await shim.ready
console.log('shim listening at', shim.baseUrl())

const { adapter } = createOpenCodeAdapter({ providerId, shim, models: () => models })

async function collect(options) {
  const chunks = []
  for await (const chunk of adapter.stream(options)) chunks.push(chunk)
  return chunks
}
const textOf = chunks => chunks.filter(c => c.type === 'text-delta').map(c => c.text).join('')

console.log('--- 1. catalog ---')
const listed = await adapter.listModels(providerId)
assert(listed.length > 0, 'advertises at least one model (' + listed.length + ')')
assert(listed.some(m => m.id === 'deepseek-v4.1-flash-free'), 'advertises deepseek-v4.1-flash-free')

console.log('--- 2. resolveModel ---')
const resolved = await adapter.resolveModel(providerId, 'deepseek-v4.1-flash-free')
assert(resolved.context?.contextWindow === 200000, 'context window is 200000 (got ' + resolved.context?.contextWindow + ')')
assert(resolved.defaultMaxTokens === 32000, 'default max tokens is 32000')

console.log('--- 3. prepareCall ---')
const prepared = await adapter.prepareCall(providerId, 'deepseek-v4.1-flash-free')
assert(prepared.model.id === 'deepseek-v4.1-flash-free', 'prepared call carries the resolved model')

console.log('--- 4. plain chat over the real wire ---')
seen.length = 0
const chunks = await collect({
  provider: providerId,
  model: 'deepseek-v4.1-flash-free',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'Reply with exactly: pong' }] }],
})
const finish = chunks.find(c => c.type === 'finish')
const usage = chunks.find(c => c.type === 'usage')
const text = textOf(chunks)
console.log('  text  :', JSON.stringify(text.slice(0, 80)))
console.log('  finish:', JSON.stringify(finish?.reason))
console.log('  usage :', JSON.stringify(usage?.usage))
assert(finish !== undefined, 'stream terminated with a finish chunk')
assert(finish.reason.kind === 'stop', 'finished with stop (got ' + finish.reason.kind + ')')
assert(text.toLowerCase().includes('pong'), 'the model answered the instruction')
assert(usage !== undefined, 'usage was reported')
assert(chunks.some(c => c.type === 'block-start' && c.blockType === 'text'), 'emitted a text block-start')
assert(chunks.some(c => c.type === 'block-end' && c.block.type === 'text'), 'emitted a text block-end')
assert(chunks.filter(c => c.type !== 'finish').pop()?.type === 'usage', 'usage precedes the terminal finish')

console.log('--- 5. the boundary held ---')
const toShim = seen.filter(s => s.url.includes('127.0.0.1'))
const toUpstream = seen.filter(s => s.url.includes('opencode.ai'))
assert(toShim.length > 0, 'the adapter called the loopback shim')
assert(toUpstream.length > 0, 'the shim called upstream')
assert(toShim.every(s => s.ua.includes('deepseek-harness/')), 'the shim hop carried the harness attribution')
assert(toUpstream.every(s => !s.ua.includes('deepseek-harness')), 'the upstream hop did NOT carry the harness token')
assert(toUpstream.every(s => s.ua.startsWith('opencode/')), 'the upstream hop carried the CLI identity')

console.log('--- 6. system prompt ---')
const sysChunks = await collect({
  provider: providerId,
  model: 'deepseek-v4.1-flash-free',
  messages: [
    { role: 'system', content: [{ type: 'text', text: 'You answer with one word only.' }] },
    { role: 'user', content: [{ type: 'text', text: 'Say yes' }] },
  ],
})
const sysText = textOf(sysChunks)
console.log('  text:', JSON.stringify(sysText.slice(0, 80)))
assert(sysText.length > 0, 'a leading system message still produces an answer')

console.log('--- 7. declared tools pass through ---')
const toolChunks = await collect({
  provider: providerId,
  model: 'deepseek-v4.1-flash-free',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'What is 2+2? Answer briefly.' }] }],
  tools: [{ name: 'get_time', description: 'Return the current time', parameters: { type: 'object', properties: {} } }],
})
const toolFinish = toolChunks.find(c => c.type === 'finish')
console.log('  finish:', JSON.stringify(toolFinish?.reason))
assert(toolFinish !== undefined, 'a request that declares tools completes')
assert(['stop', 'tool-calls'].includes(toolFinish.reason.kind), 'finish is stop or tool-calls')

await shim.close()
console.log('\nALL LIVE CHECKS PASSED')
