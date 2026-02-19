import { randomBytes } from "node:crypto";
import { BlockList, isIP } from "node:net";
import type {
  OpenAIChatChunk,
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAIContentPart,
  OpenAIMessage,
  OpenAITool,
  OpenAIToolCall,
  OpenAIUsage
} from "./types.js";

const SYSTEM_FINGERPRINT = "fp_lensgate";
const MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024;
const IMAGE_FETCH_TIMEOUT_MS = 20_000;
const PRIVATE_ADDRESS_BLOCKLIST = new BlockList();
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("0.0.0.0", 8, "ipv4");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("10.0.0.0", 8, "ipv4");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("100.64.0.0", 10, "ipv4");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("127.0.0.0", 8, "ipv4");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("169.254.0.0", 16, "ipv4");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("172.16.0.0", 12, "ipv4");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("192.168.0.0", 16, "ipv4");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("::", 128, "ipv6");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("::1", 128, "ipv6");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("fc00::", 7, "ipv6");
PRIVATE_ADDRESS_BLOCKLIST.addSubnet("fe80::", 10, "ipv6");

export class ImageNotSupportedError extends Error {
  constructor(message = "image_url content is not supported by this proxy.") {
    super(message);
    this.name = "ImageNotSupportedError";
  }
}

export interface StreamState {
  completionId: string;
  model: string;
  created: number;
  toolCallIndexMap: Map<number, number>;
  nextToolCallIndex: number;
  textDeltaSeen: Set<number>;
  emittedAnyOutput: boolean;
  accumulatedText: string;
  streamFinishReason: string | null;
  streamUsage: OpenAIUsage | null;
  streamMessage: unknown;
}

function asTextContent(content: string | OpenAIContentPart[] | null): string {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  const parts: string[] = [];
  for (const part of content) {
    if (part.type === "text") {
      parts.push(part.text);
    }
  }
  return parts.join("");
}

function isImageMimeType(mimeType: string): boolean {
  return mimeType.toLowerCase().startsWith("image/");
}

function inferMimeTypeFromUrl(url: URL): string | null {
  const pathname = url.pathname.toLowerCase();
  if (pathname.endsWith(".png")) return "image/png";
  if (pathname.endsWith(".jpg") || pathname.endsWith(".jpeg")) return "image/jpeg";
  if (pathname.endsWith(".webp")) return "image/webp";
  if (pathname.endsWith(".gif")) return "image/gif";
  if (pathname.endsWith(".bmp")) return "image/bmp";
  if (pathname.endsWith(".svg")) return "image/svg+xml";
  if (pathname.endsWith(".avif")) return "image/avif";
  if (pathname.endsWith(".tif") || pathname.endsWith(".tiff")) return "image/tiff";
  if (pathname.endsWith(".heic")) return "image/heic";
  if (pathname.endsWith(".heif")) return "image/heif";
  return null;
}

function isBlockedImageHostname(hostname: string): boolean {
  const normalized = hostname.trim().toLowerCase().replace(/\.$/, "");
  if (!normalized) {
    return true;
  }
  if (normalized === "localhost" || normalized.endsWith(".localhost")) {
    return true;
  }

  const ipVersion = isIP(normalized);
  if (ipVersion === 4) {
    return PRIVATE_ADDRESS_BLOCKLIST.check(normalized, "ipv4");
  }
  if (ipVersion === 6) {
    if (normalized.startsWith("::ffff:")) {
      const mapped = normalized.slice("::ffff:".length);
      if (isIP(mapped) === 4 && PRIVATE_ADDRESS_BLOCKLIST.check(mapped, "ipv4")) {
        return true;
      }
    }
    return PRIVATE_ADDRESS_BLOCKLIST.check(normalized, "ipv6");
  }
  return false;
}

function parseDataUrl(dataUrl: string): { mimeType: string; data: string } {
  const commaIndex = dataUrl.indexOf(",");
  if (commaIndex < 0) {
    throw new ImageNotSupportedError("Invalid image_url data URI.");
  }

  const metadata = dataUrl.slice(5, commaIndex);
  const payload = dataUrl.slice(commaIndex + 1);
  const metadataParts = metadata
    .split(";")
    .map((part) => part.trim().toLowerCase())
    .filter((part) => part.length > 0);
  const mimeType = metadataParts.length > 0 && !metadataParts[0].includes("=") ? metadataParts[0] : "text/plain";
  if (!isImageMimeType(mimeType)) {
    throw new ImageNotSupportedError(`Unsupported image_url media type '${mimeType}'.`);
  }

  const isBase64 = metadataParts.includes("base64");
  if (isBase64) {
    const normalized = payload.trim();
    if (normalized.length === 0) {
      throw new ImageNotSupportedError("image_url data URI is empty.");
    }
    return { mimeType, data: Buffer.from(normalized, "base64").toString("base64") };
  }

  const decoded = decodeURIComponent(payload);
  return { mimeType, data: Buffer.from(decoded, "utf8").toString("base64") };
}

async function readResponseBodyLimited(response: Response, maxBytes: number): Promise<Buffer> {
  if (!response.body) {
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    if (buffer.length > maxBytes) {
      throw new ImageNotSupportedError(`Remote image exceeds ${maxBytes} bytes.`);
    }
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let totalBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (!value) {
      continue;
    }
    totalBytes += value.byteLength;
    if (totalBytes > maxBytes) {
      await reader.cancel();
      throw new ImageNotSupportedError(`Remote image exceeds ${maxBytes} bytes.`);
    }
    chunks.push(Buffer.from(value));
  }

  return Buffer.concat(chunks, totalBytes);
}

async function fetchRemoteImageAsBase64(url: URL): Promise<{ mimeType: string; data: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), IMAGE_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      method: "GET",
      redirect: "follow",
      signal: controller.signal
    });
    if (!response.ok) {
      throw new ImageNotSupportedError(`Failed to fetch image_url. HTTP ${response.status}.`);
    }

    const contentTypeHeader = response.headers.get("content-type");
    const headerMimeType = contentTypeHeader?.split(";")[0]?.trim().toLowerCase() ?? "";
    const mimeType = isImageMimeType(headerMimeType) ? headerMimeType : inferMimeTypeFromUrl(url);
    if (!mimeType) {
      throw new ImageNotSupportedError("Could not determine image_url media type.");
    }

    const bytes = await readResponseBodyLimited(response, MAX_REMOTE_IMAGE_BYTES);
    if (bytes.length === 0) {
      throw new ImageNotSupportedError("image_url resolved to an empty body.");
    }

    return { mimeType, data: bytes.toString("base64") };
  } catch (error) {
    if (error instanceof ImageNotSupportedError) {
      throw error;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new ImageNotSupportedError(`Failed to fetch image_url: ${message}`);
  } finally {
    clearTimeout(timer);
  }
}

async function toPiImageContent(urlValue: string): Promise<{ type: "image"; data: string; mimeType: string }> {
  const trimmedUrl = urlValue.trim();
  if (!trimmedUrl) {
    throw new ImageNotSupportedError("image_url.url must be a non-empty string.");
  }

  if (trimmedUrl.toLowerCase().startsWith("data:")) {
    const parsed = parseDataUrl(trimmedUrl);
    return { type: "image", ...parsed };
  }

  let parsedUrl: URL;
  try {
    parsedUrl = new URL(trimmedUrl);
  } catch {
    throw new ImageNotSupportedError("image_url.url must be a valid URL.");
  }

  if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
    throw new ImageNotSupportedError("Only http(s) and data URI image_url values are supported.");
  }
  if (isBlockedImageHostname(parsedUrl.hostname)) {
    throw new ImageNotSupportedError("image_url host is not allowed.");
  }

  const remote = await fetchRemoteImageAsBase64(parsedUrl);
  return { type: "image", ...remote };
}

async function userContentToPiContent(
  content: string | OpenAIContentPart[] | null
): Promise<string | Array<{ type: "text"; text: string } | { type: "image"; data: string; mimeType: string }>> {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  const hasImagePart = content.some((part) => part.type === "image_url");
  if (!hasImagePart) {
    return asTextContent(content);
  }

  const parts: Array<{ type: "text"; text: string } | { type: "image"; data: string; mimeType: string }> = [];
  for (const part of content) {
    if (part.type === "text") {
      if (part.text.length > 0) {
        parts.push({ type: "text", text: part.text });
      }
      continue;
    }
    const imageUrl = part.image_url?.url;
    if (typeof imageUrl !== "string") {
      throw new ImageNotSupportedError("image_url.url must be a string.");
    }
    parts.push(await toPiImageContent(imageUrl));
  }

  return parts;
}

function parseToolArguments(argumentsText: string): unknown {
  const input = argumentsText.trim();
  if (!input) {
    return {};
  }
  try {
    return JSON.parse(input);
  } catch {
    return {
      raw: argumentsText
    };
  }
}

function messageToTool(message: OpenAITool): Record<string, unknown> {
  return {
    name: message.function.name,
    description: message.function.description ?? "",
    parameters: message.function.parameters ?? {
      type: "object",
      properties: {}
    }
  };
}

async function messageRoleToPiMessage(message: OpenAIMessage): Promise<Record<string, unknown> | null> {
  if (message.role === "system") {
    return null;
  }

  if (message.role === "user") {
    return {
      role: "user",
      content: await userContentToPiContent(message.content),
      timestamp: Date.now()
    };
  }

  if (message.role === "assistant") {
    const assistantMessage: Record<string, unknown> = {
      role: "assistant",
      content: [],
      timestamp: Date.now(),
      api: "openai-codex-responses",
      provider: "openai",
      model: "unknown"
    };

    const text = asTextContent(message.content);
    if (text.length > 0) {
      (assistantMessage.content as unknown[]).push({
        type: "text",
        text
      });
    }

    if (Array.isArray(message.tool_calls)) {
      for (const toolCall of message.tool_calls) {
        (assistantMessage.content as unknown[]).push({
          type: "toolCall",
          id: toolCall.id,
          name: toolCall.function.name,
          arguments: parseToolArguments(toolCall.function.arguments)
        });
      }
    }

    return assistantMessage;
  }

  if (message.role === "tool") {
    return {
      role: "toolResult",
      toolCallId: message.tool_call_id ?? "",
      toolName: message.name ?? "",
      content: [
        {
          type: "text",
          text: asTextContent(message.content)
        }
      ],
      isError: false,
      timestamp: Date.now()
    };
  }

  return null;
}

export async function openAIRequestToContext(req: OpenAIChatRequest): Promise<Record<string, unknown>> {
  const context: Record<string, unknown> = {
    messages: []
  };
  const messages = context.messages as unknown[];
  const systemPrompts: string[] = [];

  for (const message of req.messages) {
    if (message.role === "system") {
      const text = asTextContent(message.content);
      if (text.length > 0) {
        systemPrompts.push(text);
      }
      continue;
    }

    const converted = await messageRoleToPiMessage(message);
    if (converted) {
      messages.push(converted);
    }
  }

  // Codex API 要求 instructions 字段必须存在，没有 system message 时传空字符串
  context.systemPrompt = systemPrompts.join("\n\n");

  if (Array.isArray(req.tools) && req.tools.length > 0) {
    context.tools = req.tools.map((tool) => messageToTool(tool));
  }

  if (req.tool_choice) {
    context.toolChoice = req.tool_choice;
  }

  return context;
}

function formatSSEDataLine(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function mapStopReason(stopReason: unknown): "stop" | "length" | "tool_calls" | "content_filter" {
  if (
    stopReason === "tool_calls" ||
    stopReason === "tool_call" ||
    stopReason === "toolUse" ||
    stopReason === "tool_use"
  ) {
    return "tool_calls";
  }
  if (stopReason === "max_tokens" || stopReason === "length") {
    return "length";
  }
  if (stopReason === "content_filter" || stopReason === "error" || stopReason === "aborted") {
    return "content_filter";
  }
  return "stop";
}

function toChunk(
  state: StreamState,
  delta: Partial<OpenAIMessage>,
  finishReason: "stop" | "length" | "tool_calls" | "content_filter" | null
): OpenAIChatChunk {
  return {
    id: state.completionId,
    object: "chat.completion.chunk",
    created: state.created,
    model: state.model,
    system_fingerprint: SYSTEM_FINGERPRINT,
    choices: [
      {
        index: 0,
        delta,
        logprobs: null,
        finish_reason: finishReason
      }
    ]
  };
}

function getEventType(event: unknown): string {
  if (!event || typeof event !== "object") {
    return "";
  }
  const record = event as Record<string, unknown>;
  if (typeof record.type === "string") {
    return record.type;
  }
  if (typeof record.event === "string") {
    return record.event;
  }
  if (typeof record.kind === "string") {
    return record.kind;
  }
  return "";
}

function readString(record: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return "";
}

function readNumber(record: Record<string, unknown>, ...keys: string[]): number | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function readToolCallFromPartial(
  record: Record<string, unknown>,
  contentIndex: number
): { id: string; name: string } | null {
  const partial = record.partial;
  if (!partial || typeof partial !== "object") {
    return null;
  }

  const content = (partial as Record<string, unknown>).content;
  if (!Array.isArray(content) || contentIndex < 0 || contentIndex >= content.length) {
    return null;
  }

  const item = content[contentIndex];
  if (!item || typeof item !== "object") {
    return null;
  }
  const toolCall = item as Record<string, unknown>;
  if (toolCall.type !== "toolCall") {
    return null;
  }

  return {
    id: typeof toolCall.id === "string" ? toolCall.id : "",
    name: typeof toolCall.name === "string" ? toolCall.name : ""
  };
}

export function createStreamState(modelId: string): StreamState {
  return {
    completionId: generateCompletionId(),
    model: modelId,
    created: Math.floor(Date.now() / 1000),
    toolCallIndexMap: new Map<number, number>(),
    nextToolCallIndex: 0,
    textDeltaSeen: new Set<number>(),
    emittedAnyOutput: false,
    accumulatedText: "",
    streamFinishReason: null,
    streamUsage: null,
    streamMessage: null
  };
}

export function eventToSSEChunks(event: unknown, state: StreamState): string[] {
  const chunks: string[] = [];
  if (!event || typeof event !== "object") {
    return chunks;
  }

  const record = event as Record<string, unknown>;
  const eventType = getEventType(event);

  if (eventType === "start") {
    chunks.push(formatSSEDataLine(toChunk(state, { role: "assistant" }, null)));
    return chunks;
  }

  if (eventType === "text_delta") {
    const deltaText = readString(record, "delta", "text", "content");
    const contentIndex = readNumber(record, "contentIndex", "index") ?? 0;
    if (deltaText.length > 0) {
      state.textDeltaSeen.add(contentIndex);
      state.emittedAnyOutput = true;
      state.accumulatedText += deltaText;
      chunks.push(formatSSEDataLine(toChunk(state, { content: deltaText }, null)));
    }
    return chunks;
  }

  if (eventType === "text_end") {
    const contentIndex = readNumber(record, "contentIndex", "index") ?? 0;
    const text = readString(record, "content", "text", "delta");
    // Some providers send full text only in text_end; emit it when no prior delta was emitted.
    if (text.length > 0 && !state.textDeltaSeen.has(contentIndex)) {
      state.emittedAnyOutput = true;
      state.accumulatedText += text;
      chunks.push(formatSSEDataLine(toChunk(state, { content: text }, null)));
    }
    return chunks;
  }

  if (eventType === "toolcall_start") {
    const contentIndex = readNumber(record, "contentIndex", "index", "toolCallIndex") ?? state.nextToolCallIndex;
    let openAiToolCallIndex = state.toolCallIndexMap.get(contentIndex);
    if (openAiToolCallIndex === undefined) {
      openAiToolCallIndex = state.nextToolCallIndex;
      state.toolCallIndexMap.set(contentIndex, openAiToolCallIndex);
      state.nextToolCallIndex += 1;
    }

    const partialToolCall = readToolCallFromPartial(record, contentIndex);
    const toolCallId = partialToolCall?.id || readString(record, "id", "toolCallId") || `call_${openAiToolCallIndex}`;
    const toolName = partialToolCall?.name || readString(record, "name", "toolName") || "tool";

    const delta: Partial<OpenAIMessage> = {
      tool_calls: [
        {
          id: toolCallId,
          type: "function",
          function: {
            name: toolName,
            arguments: ""
          },
          index: openAiToolCallIndex
        }
      ]
    };

    state.emittedAnyOutput = true;
    chunks.push(formatSSEDataLine(toChunk(state, delta, null)));
    return chunks;
  }

  if (eventType === "toolcall_delta") {
    const contentIndex = readNumber(record, "contentIndex", "index", "toolCallIndex") ?? 0;
    let openAiToolCallIndex = state.toolCallIndexMap.get(contentIndex);
    if (openAiToolCallIndex === undefined) {
      openAiToolCallIndex = state.nextToolCallIndex;
      state.toolCallIndexMap.set(contentIndex, openAiToolCallIndex);
      state.nextToolCallIndex += 1;
    }
    const partialToolCall = readToolCallFromPartial(record, contentIndex);

    const argumentsDelta =
      readString(record, "delta", "argumentsDelta", "arguments", "textDelta") ||
      String(record.value ?? "");

    if (argumentsDelta.length > 0 || partialToolCall) {
      const toolCallDelta: Record<string, unknown> = {
        index: openAiToolCallIndex,
        function: { arguments: argumentsDelta }
      };
      if (partialToolCall?.id) {
        toolCallDelta.id = partialToolCall.id;
      }
      if (partialToolCall?.name) {
        (toolCallDelta.function as Record<string, unknown>).name = partialToolCall.name;
      }

      const delta: Partial<OpenAIMessage> = {
        tool_calls: [toolCallDelta as unknown as OpenAIToolCall]
      };
      state.emittedAnyOutput = true;
      chunks.push(formatSSEDataLine(toChunk(state, delta, null)));
    }
    return chunks;
  }

  if (eventType === "error") {
    const errorRecord =
      record.error && typeof record.error === "object" ? (record.error as Record<string, unknown>) : undefined;
    const message =
      readString(record, "errorMessage", "message") ||
      (errorRecord && readString(errorRecord, "errorMessage", "message")) ||
      "Upstream model error.";

    chunks.push(
      formatSSEDataLine({
        error: {
          message,
          type: "server_error",
          param: null,
          code: "upstream_model_error"
        }
      })
    );
    return chunks;
  }

  if (eventType === "done") {
    const finishReason = mapStopReason(record.reason ?? record.stopReason);
    state.streamFinishReason = finishReason;

    // Capture the full done message for Langfuse output (includes tool calls)
    if (record.message && typeof record.message === "object") {
      state.streamMessage = record.message;

      // Extract token usage from pi-ai's done event
      const msg = record.message as Record<string, unknown>;
      const usage = msg.usage && typeof msg.usage === "object" ? (msg.usage as Record<string, unknown>) : null;
      if (usage) {
        const inputTokens = safeNumber(usage.input, usage.inputTokens, usage.prompt_tokens);
        const outputTokens = safeNumber(usage.output, usage.outputTokens, usage.completion_tokens);
        const cacheRead = safeNumber(usage.cacheRead, usage.cacheReadTokens, usage.cached_tokens);
        state.streamUsage = {
          prompt_tokens: inputTokens + cacheRead,
          completion_tokens: outputTokens,
          total_tokens: safeNumber(usage.totalTokens, usage.total_tokens) || inputTokens + cacheRead + outputTokens,
          completion_tokens_details: { reasoning_tokens: 0, audio_tokens: 0, accepted_prediction_tokens: 0, rejected_prediction_tokens: 0 },
          prompt_tokens_details: { cached_tokens: cacheRead, audio_tokens: 0 }
        };
      }
    }

    chunks.push(formatSSEDataLine(toChunk(state, {}, finishReason)));
    return chunks;
  }

  return chunks;
}

export function formatUsageSSEChunk(state: StreamState): string | null {
  if (!state.streamUsage) return null;
  const chunk = {
    id: state.completionId,
    object: "chat.completion.chunk",
    created: state.created,
    model: state.model,
    system_fingerprint: SYSTEM_FINGERPRINT,
    choices: [],
    usage: state.streamUsage
  };
  return formatSSEDataLine(chunk);
}

function safeNumber(...values: unknown[]): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return 0;
}

function stringifyToolArguments(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return "{}";
  }
}

function assistantTextFromMessage(message: Record<string, unknown>): string {
  const content = message.content;
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }

  const parts: string[] = [];
  for (const item of content) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const record = item as Record<string, unknown>;
    if (record.type === "text" && typeof record.text === "string") {
      parts.push(record.text);
      continue;
    }
    if (record.type === "output_text" && typeof record.text === "string") {
      parts.push(record.text);
    }
  }
  return parts.join("");
}

function openAiToolCallsFromMessage(message: Record<string, unknown>): OpenAIToolCall[] {
  const structuredCalls =
    (Array.isArray(message.toolCalls) ? message.toolCalls : null) ??
    (Array.isArray(message.tool_calls) ? message.tool_calls : null) ??
    [];
  const contentCalls =
    Array.isArray(message.content)
      ? message.content.filter(
          (item): item is Record<string, unknown> =>
            Boolean(item) && typeof item === "object" && (item as Record<string, unknown>).type === "toolCall"
        )
      : [];
  const source = structuredCalls.length > 0 ? structuredCalls : contentCalls;

  const calls: OpenAIToolCall[] = [];
  for (let i = 0; i < source.length; i += 1) {
    const item = source[i];
    if (!item || typeof item !== "object") {
      continue;
    }
    const record = item as Record<string, unknown>;
    const functionRecord =
      record.function && typeof record.function === "object"
        ? (record.function as Record<string, unknown>)
        : undefined;

    const name =
      (typeof record.name === "string" && record.name) ||
      (typeof record.toolName === "string" && record.toolName) ||
      ((functionRecord && typeof functionRecord.name === "string" ? functionRecord.name : "") || "tool");

    calls.push({
      id:
        (typeof record.id === "string" && record.id) ||
        (typeof record.toolCallId === "string" && record.toolCallId) ||
        `call_${i}`,
      type: "function",
      function: {
        name,
        arguments: stringifyToolArguments(record.arguments ?? functionRecord?.arguments ?? record.input ?? {})
      }
    });
  }

  return calls;
}

export function assistantMessageToResponse(
  message: unknown,
  requestedModelId: string,
  completionId: string
): OpenAIChatResponse {
  const record = (message && typeof message === "object" ? message : {}) as Record<string, unknown>;
  const text = assistantTextFromMessage(record);
  const toolCalls = openAiToolCallsFromMessage(record);

  const usage = (record.usage as Record<string, unknown> | undefined) ?? {};
  const promptTokens =
    safeNumber(usage.prompt_tokens, usage.inputTokens, usage.input) +
    safeNumber(usage.cacheReadTokens, usage.cacheRead);
  const completionTokens = safeNumber(usage.completion_tokens, usage.outputTokens, usage.output);

  const responseMessage: OpenAIMessage = {
    role: "assistant",
    content: text
  };
  if (toolCalls.length > 0) {
    responseMessage.tool_calls = toolCalls;
  }

  const reasoningTokens = safeNumber(
    (usage as Record<string, unknown>).reasoning_tokens,
    (usage as Record<string, unknown>).reasoningTokens
  );
  const cachedTokens = safeNumber(
    (usage as Record<string, unknown>).cached_tokens,
    (usage as Record<string, unknown>).cacheReadTokens,
    (usage as Record<string, unknown>).cacheRead
  );

  return {
    id: completionId,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: requestedModelId,
    system_fingerprint: SYSTEM_FINGERPRINT,
    choices: [
      {
        index: 0,
        message: responseMessage,
        logprobs: null,
        finish_reason: toolCalls.length > 0 ? "tool_calls" : mapStopReason(record.stopReason ?? record.reason)
      }
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
      completion_tokens_details: {
        reasoning_tokens: reasoningTokens,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0
      },
      prompt_tokens_details: {
        cached_tokens: cachedTokens,
        audio_tokens: 0
      }
    }
  };
}

export function generateCompletionId(): string {
  return `chatcmpl-${randomBytes(12).toString("hex")}`;
}
