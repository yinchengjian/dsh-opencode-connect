import type { UserConfig } from 'tsdown'

/** Host-provided modules kept external so the plugin shares the host's instances. */
const HOST_EXTERNALS = [
  '@earendil-works/pi-ai',
  '@deepseek-ai/cordis',
  '@deepseek-ai/dsh-llm',
  '@deepseek-ai/schemastery',
] as const

export default {
  entry: { index: 'src/index.ts' },
  outDir: 'lib',
  format: ['esm'],
  platform: 'node',
  target: 'es2024',
  fixedExtension: false,
  dts: true,
  clean: true,
  deps: { neverBundle: [...HOST_EXTERNALS] },
} satisfies UserConfig
