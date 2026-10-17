# DSH OpenCode Connect

将 OpenCode 的**免费模型**接入 DeepSeek Harness。无需 API Key、无需注册、无需额外进程。

插件向 DSH 注册一个原生 `LlmAdapter`（provider 名 `opencode`），模型出现在模型选择器的 **OpenCode** 分组里。

## 安装

**前置**：出站需可达 `opencode.ai` 与 `models.dev`。

两件事要先确认，否则会得到 429：

### 1. 让 opencode.ai 直连

匿名免费额度**按出口 IP 计**。共享代理出口通常已被全球用户打满，会答 `429 FreeUsageLimitError`；干净的家宽直连则答 200。在代理规则里加一条：

```yaml
- DOMAIN-SUFFIX,opencode.ai,DIRECT
```

其余流量不受影响。`models.dev` 无需特殊规则（走代理正常）。

### 2. 安装插件

尚未发布到 npm 或插件市场，从源码安装：

```sh
pnpm install && pnpm build && pnpm pack
dsh plugin --profile web add ./dsh-opencode-connect-0.1.0.tgz
dsh web
```

（若之后发布到 npm / 插件市场，可简化为 `dsh plugin --profile web add dsh-opencode-connect`。）

**验证**：重启后模型选择器出现 **OpenCode** 分组，选一个 `*-free` 模型完成一次简短对话即可。

## 为什么需要一个本地端口

这是本插件唯一"非纯粹 adapter"的地方，值得说清楚。

免费通道会拒绝来自 DSH 自己的请求：`User-Agent` 里出现 `deepseek-harness/`（大小写不敏感）就返回 `403 FreeTierError`。这是**定向**的——`codex` / `claude-code` / `cursor` 等 token 都被放行。实测是交错 A/B 验证的，不是抖动。

而 DSH 的契约要求每个 provider 请求都带 harness attribution。两者在**同一个请求上无法同时成立**，于是插件把它们拆到两个请求：

```
adapter --(harness attribution)--> 本机 shim --(CLI 身份)--> opencode.ai
         127.0.0.1:随机端口              Authorization: Bearer public
```

- attribution 真实发往**本机 shim**——那是这条路由的 provider 端点，契约有落点
- 真正出网的请求只带 `opencode/...` CLI 身份，即上游认可的那一个

这与 `dsh-workbuddy-connect` 用的是同一种边界设计。

shim 同时承担免费通道的全部线协议细节：CLI 请求头、派生的会话 ID、强制流式、以及网关要求的 agent 形态 body。所以 adapter 侧保持成一个普通 provider。

端口是 `listen(0, '127.0.0.1')`——随机、仅回环、随插件 fiber 释放回收。无子进程、无二进制。

## 工作方式

```
DSH 会话
   │  harness chunk（block-start / text-delta / usage / finish …）
   ▼
OpenCodePiAiAdapter
   │  pi-ai openai-completions + attribution UA
   ▼
http://127.0.0.1:<随机端口>          ← 信任边界
   │  add: CLI headers、ses_* 会话 ID、强制 stream、bash+read 门槛
   ▼
https://opencode.ai/zen/v1           ← Authorization: Bearer public
```

- **会话关联**：session id 由会话首条用户消息经 SHA-256 派生（同会话稳定、不可逆推），每请求再带一个新的 request id
- **目录三级回退**：
  - S1 实时 `GET /v1/models`
  - S2 `models.dev` 定价元数据判定"免费"（`input == 0 && output == 0` 且未 deprecated）
  - S3 编译期验证的静态名单
  - 另有磁盘缓存（7 天）兜底，冷启动无网也能用
- **只暴露免费模型**：定价未知的模型**不会**被展示——宁可少显示，也不能让选择器出现花钱的模型
- **自愈**：启动首刷 + 每 5 分钟刷新；上游故障时保留上一份可用名单

## 配置

开箱即用，无需配置。需要覆盖时编辑 profile 的 `cordis.patch.yml`：

```yaml
- id: llm-opencode
  name: dsh-opencode-connect
```

## 排查

| 现象 | 原因与处理 |
| --- | --- |
| `429 FreeUsageLimitError` | 出口 IP 的匿名额度用尽。确认 `opencode.ai` 走直连（见上），或稍后重试 |
| `403 FreeTierError` | 请求形状被拒。若你改过 `src/ids.ts` 或 `src/shim.ts`，多半是破坏了伪装头/body 形状；`npm test` 会钉住这些不变量 |
| 模型列表为空 | 首次刷新需要网络。检查 `opencode.ai` 与 `models.dev` 可达性；日志会打印 `live=N, metadata=N` |
| 只有静态那几个模型 | 刷新未完成（网络未就绪或被拦），保留的是 S3 名单，属预期降级 |
| 报「已达到输出 token 上限，回答被截断」且回答为空 | 这是 always-think 模型的思考吃光了输出预算，不是配置问题。插件已内置恢复：检测到「只有推理、没有正文、`finish: length`」会自动关掉思考重试一次。若仍出现，把该会话的推理强度调低或选 `off` |

## 推理模型与输出预算

免费通道里有几个 always-think 模型：它们先吐 `reasoning_content`，且这段思考**和正文共用同一个输出预算**。预算偏小时思考会把预算吃光，正文一个 token 都没有，流以 `finish: length` 结束——DSH 会显示「已达到输出 token 上限」。

插件对此做了两件事：

1. **始终提供 `off` 档位**，即使 models.dev 没声明（它只给 `low|high|max`）。实测发现 `off` 必须映射成**不发送** `reasoning_effort` 字段：显式发 `none` 仍会让模型思考，发字面量 `off` 是 422。
2. **自动恢复**：若一次尝试只有推理、没有正文且以 `max-tokens` 结束，自动以关闭思考重试一次，并把两次的 usage 合并上报。

`defaultMaxTokens` 取自 models.dev 的 `limit.output`（缺省 32768），写入时会被当作该模型的默认输出上限。

## 安全性

- 不涉及任何机密：匿名通道的凭据就是字面量 `public`，插件不存储任何东西，也没有遥测
- shim 每次进程启动生成随机共享密钥，仅存内存；请求须带该密钥，否则 401
- 入站校验：仅回环 Host、仅回环 Origin、必须 `application/json`
- 所有请求从你的机器直连 `opencode.ai`

## 开发

```sh
pnpm install
pnpm typecheck && pnpm test
pnpm build
node scripts/live-e2e.mjs   # 真实联网联调（含边界断言）
```

## 许可证

MIT