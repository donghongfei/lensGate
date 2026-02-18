计划：OpenAI 兼容代理服务器（packages/openai-proxy）
Context
用户希望将本项目中已有的 openai-codex OAuth 登录能力封装成一个对外兼容 OpenAI Chat Completions API 的 HTTP 代理服务。
这样任何支持配置 OpenAI baseURL 的客户端（Cursor、Claude Code、各类 SDK）都可以直接接入，背后使用 ChatGPT Plus/Pro 订阅，无需 API Key。

实际调用链：


客户端 POST /v1/chat/completions
  ↓ （本代理，格式转换）
streamSimple(codex model, context, { apiKey: oauthToken })
  ↓
https://chatgpt.com/backend-api/codex/responses（OAuth Bearer Token）
  ↓
ChatGPT Plus/Pro 账号
兼容范围
功能	支持
POST /v1/chat/completions 文本对话（流式+非流式）	✅
GET /v1/models	✅
system prompt	✅
Tool/Function Calling（流式+非流式）	✅
temperature	✅
图片内容 (image_url)	❌ 返回 400
Embeddings / Audio / Images API	❌ 不适用
top_p, frequency_penalty 等高级参数	忽略（不透传）
文件结构
新建 packages/openai-proxy/（私有包，不发布）：


packages/openai-proxy/
├── package.json
├── tsconfig.build.json
└── src/
    ├── types.ts       # OpenAI 请求/响应 wire 格式类型定义
    ├── models.ts      # 获取 codex 模型 + 名称映射表
    ├── auth.ts        # AuthStorage 包装，取 token（自动刷新）
    ├── convert.ts     # 双向格式转换 + 工具函数
    ├── handler.ts     # HTTP 路由和请求处理
    ├── server.ts      # 入口：启动 HTTP 服务器
    └── login.ts       # 入口：一次性 OAuth 登录 CLI
关键文件详解
package.json

{
  "name": "@mariozechner/pi-openai-proxy",
  "version": "0.53.0",
  "private": true,
  "type": "module",
  "scripts": {
    "build": "tsgo -p tsconfig.build.json && shx chmod +x dist/server.js dist/login.js",
    "login": "node dist/login.js",
    "start": "node dist/server.js"
  },
  "dependencies": {
    "@mariozechner/pi-ai": "^0.53.0",
    "@mariozechner/pi-coding-agent": "^0.53.0"
  },
  "devDependencies": {
    "@types/node": "^24.3.0",
    "shx": "^0.4.0"
  }
}
tsconfig.build.json

{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": { "outDir": "./dist", "rootDir": "./src" },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
注意：所有包内相对引用必须用 .js 后缀（Node16 模块解析）。

src/types.ts — OpenAI Wire 格式类型
定义以下接口（无外部依赖）：

OpenAIChatRequest：包含 model, messages, stream?, temperature?, max_tokens?, tools?, tool_choice?
OpenAIMessage：role: "system"|"user"|"assistant"|"tool", content, tool_calls?, tool_call_id?
OpenAIContentPart：{type:"text", text} | {type:"image_url", image_url:{url}}
OpenAITool / OpenAIToolCall
OpenAIChatResponse：非流式完整响应
OpenAIChatChunk：流式 SSE chunk（object: "chat.completion.chunk"）
OpenAIModelsResponse / OpenAIModelObject
OpenAIErrorResponse
src/models.ts — 模型解析
可用 Codex 模型（来自 models.generated.ts）：
gpt-5.1, gpt-5.1-codex-max, gpt-5.1-codex-mini, gpt-5.2, gpt-5.2-codex, gpt-5.3-codex, gpt-5.3-codex-spark

关键函数：


// 从 @mariozechner/pi-ai 导入
import { getModel, getModels } from "@mariozechner/pi-ai";

// 客户端发来标准 OpenAI 名称的映射表
const MODEL_ALIAS_MAP: Record<string, string> = {
  "gpt-4o": "gpt-5.1",
  "gpt-4o-mini": "gpt-5.1-codex-mini",
  "gpt-4": "gpt-5.1",
  "gpt-4-turbo": "gpt-5.1",
  "gpt-3.5-turbo": "gpt-5.1-codex-mini",
  "o1": "gpt-5.1",
  "o3": "gpt-5.2",
  "o4-mini": "gpt-5.1-codex-mini",
};

// 解析顺序：精确匹配 codex ID → 别名映射 → 默认 gpt-5.1
export function resolveCodexModel(requested: string): Model<"openai-codex-responses">

export function getAllCodexModels(): Model<"openai-codex-responses">[]
src/auth.ts — Token 管理

import { AuthStorage } from "@mariozechner/pi-coding-agent";

// 单例，避免重复初始化（共享 file-lock 状态）
const authStorage = AuthStorage.create(); // 默认读 ~/.pi/agent/auth.json

// 返回 access token，过期自动刷新（AuthStorage 内部有 file-lock 防并发）
// 未登录时抛出错误提示用户运行 pnpm run login
export async function getAccessToken(): Promise<string>

export function hasCredentials(): boolean
src/convert.ts — 双向格式转换
① OpenAI 请求 → pi-ai Context


import type { Context, UserMessage, AssistantMessage, ToolResultMessage, TextContent, ToolCall, Tool } from "@mariozechner/pi-ai";

export class ImageNotSupportedError extends Error {}

export function openAIRequestToContext(req: OpenAIChatRequest): Context
转换规则：

role:"system" 消息 → context.systemPrompt（多条拼接为 \n\n）
role:"user" 文本 → UserMessage { role:"user", content: string, timestamp }；含 image_url 则抛 ImageNotSupportedError
role:"assistant" → 合成 AssistantMessage，文本→TextContent[]，tool_calls→ToolCall[]（arguments 做 JSON.parse）；api/provider/model 设为占位值
role:"tool" → ToolResultMessage { role:"toolResult", toolCallId, toolName: msg.name??"", content: [{type:"text", text}], isError: false }
tools[] → Tool[] { name, description, parameters }（JSON Schema 直接透传）
② pi-ai 流式事件 → OpenAI SSE chunks


export interface StreamState {
  completionId: string;   // "chatcmpl-xxx"
  model: string;
  created: number;        // Unix seconds
  toolCallIndexMap: Map<number, number>;  // contentIndex → openai tool_calls index
  nextToolCallIndex: number;
}

export function createStreamState(modelId: string): StreamState

// 每个事件 → 0 或多个 SSE 数据行（"data: {...}\n\n"）
export function eventToSSEChunks(event: AssistantMessageEvent, state: StreamState): string[]
事件映射：

pi-ai 事件	OpenAI SSE
start	delta: {role:"assistant"}
text_delta	delta: {content: delta}
toolcall_start	delta: {tool_calls:[{index, id, type:"function", function:{name, arguments:""}}]}
toolcall_delta	delta: {tool_calls:[{index, function:{arguments: delta}}]}
toolcall_end	无需额外输出
done	`delta:{}, finish_reason:"stop"
error	data: [DONE]（流已开始，无法改状态码）
thinking_* / text_start/end	忽略
③ pi-ai AssistantMessage → OpenAI 非流式响应


export function assistantMessageToResponse(
  msg: AssistantMessage,
  requestedModelId: string,
  completionId: string,
): OpenAIChatResponse
文本：提取所有 TextContent 拼接
工具调用：ToolCall[] → OpenAIToolCall[]（arguments: JSON.stringify(tc.arguments)）
usage：input + cacheRead → prompt_tokens，output → completion_tokens
finish_reason：有工具调用时为 "tool_calls"，否则按 stopReason 映射
工具函数：


export function generateCompletionId(): string  // "chatcmpl-" + random
src/handler.ts — HTTP 路由
纯 node:http，无第三方框架。

路由：

OPTIONS * → 204（CORS preflight）
GET /v1/models → JSON 列出所有 codex 模型
POST /v1/chat/completions → 主处理逻辑
其他 → 404
POST /v1/chat/completions 流程：

读取并解析 JSON body
调用 getAccessToken()（401 if not logged in）
resolveCodexModel(body.model) 获取 pi-ai Model 对象
openAIRequestToContext(body) 转换（400 if image）
调用 streamSimple(model, context, { apiKey, temperature }) from @mariozechner/pi-ai
若 body.stream === true：SSE 流式输出（先写 headers，再 async for-of 事件转 chunks）
否则：await eventStream.result() 后返回完整 JSON
错误码映射：

场景	HTTP
未登录	401
图片内容	400
限额/限速（含 "usage_limit"/"rate_limit"）	429
其他上游错误	500
流式中途错误	发 [DONE] 关闭（已无法改状态码）
所有响应带 CORS header：Access-Control-Allow-Origin: *

src/server.ts — 服务器入口

#!/usr/bin/env node
import { createServer } from "node:http";
import { handleRequest } from "./handler.js";

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "127.0.0.1";

// createServer → handleRequest → graceful shutdown on SIGINT/SIGTERM
src/login.ts — OAuth 登录 CLI

#!/usr/bin/env node
import { createInterface } from "node:readline";
import { exec } from "node:child_process";   // 静态 import
import { AuthStorage } from "@mariozechner/pi-coding-agent";

// 1. AuthStorage.create()
// 2. 若已有凭证询问是否重新登录
// 3. authStorage.login("openai-codex", { onAuth, onPrompt, onProgress })
//    - onAuth: 打印 URL + 尝试 exec("open"/"xdg-open" URL)
//    - onPrompt: readline 提示粘贴授权码（回调服务器失败的备用）
// 4. 成功后打印提示：凭证已保存，可运行 pnpm run start
需修改的现有文件
package.json（根）：在 build script 末尾追加 && cd ../openai-proxy && pnpm run build，确保按依赖顺序构建。

使用方式（构建完成后）

# 首次登录
cd packages/openai-proxy && pnpm run login

# 启动代理（默认 http://127.0.0.1:3000）
pnpm run start

# 配置其他客户端时：
#   base_url = http://127.0.0.1:3000/v1
#   api_key  = any-string（代理不校验 key，使用 OAuth token）
验证步骤

# 1. 构建依赖包
cd packages/ai && pnpm run build
cd ../coding-agent && pnpm run build

# 2. 构建 proxy
cd ../openai-proxy && pnpm install && pnpm run build

# 3. 登录
pnpm run login

# 4. 启动
pnpm run start

# 5. 测试非流式
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.1","messages":[{"role":"user","content":"Say hi"}]}'

# 6. 测试流式
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.1","messages":[{"role":"user","content":"Say hi"}],"stream":true}'

# 7. 测试模型列表
curl http://localhost:3000/v1/models

# 8. 测试模型别名（发 gpt-4o，代理映射到 gpt-5.1）
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
关键依赖文件（实现时需参考）
packages/ai/src/types.ts — Context, AssistantMessage, AssistantMessageEvent 等核心类型
packages/ai/src/stream.ts — streamSimple() 主调用函数
packages/ai/src/models.generated.ts — "openai-codex" 下的 7 个模型定义
packages/coding-agent/src/core/auth-storage.ts — AuthStorage.create(), getApiKey(), login(), hasAuth()
packages/ai/src/utils/oauth/openai-codex.ts — OAuth 流程（由 AuthStorage 内部调用）
packages/coding-agent/package.json — 参考 build script 格式（tsgo + shx）