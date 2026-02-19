import { randomBytes } from "node:crypto";
import type {
  OpenAIChatChunk,
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAIContentPart,
  OpenAIMessage,
  OpenAITool,
  OpenAIToolCall
} from "./types.js";

const SYSTEM_FINGERPRINT = "fp_lensgate";

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
}

function asTextContent(content: string | OpenAIContentPart[] | null, allowImages: boolean): string {
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
      continue;
    }
    if (part.type === "image_url" && !allowImages) {
      throw new ImageNotSupportedError();
    }
  }
  return parts.join("");
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

function messageRoleToPiMessage(message: OpenAIMessage): Record<string, unknown> | null {
  if (message.role === "system") {
    return null;
  }

  if (message.role === "user") {
    return {
      role: "user",
      content: asTextContent(message.content, false),
      timestamp: Date.now()
    };
  }

  if (message.role === "assistant") {
    const assistantMessage: Record<string, unknown> = {
      role: "assistant",
      content: [],
      toolCalls: [],
      timestamp: Date.now(),
      api: "openai-codex-responses",
      provider: "openai",
      model: "unknown"
    };

    const text = asTextContent(message.content, true);
    if (text.length > 0) {
      (assistantMessage.content as unknown[]).push({
        type: "text",
        text
      });
    }

    if (Array.isArray(message.tool_calls)) {
      for (const toolCall of message.tool_calls) {
        (assistantMessage.toolCalls as unknown[]).push({
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
          text: asTextContent(message.content, true)
        }
      ],
      isError: false,
      timestamp: Date.now()
    };
  }

  return null;
}

export function openAIRequestToContext(req: OpenAIChatRequest): Record<string, unknown> {
  const context: Record<string, unknown> = {
    messages: []
  };
  const messages = context.messages as unknown[];
  const systemPrompts: string[] = [];

  for (const message of req.messages) {
    if (message.role === "system") {
      const text = asTextContent(message.content, true);
      if (text.length > 0) {
        systemPrompts.push(text);
      }
      continue;
    }

    const converted = messageRoleToPiMessage(message);
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
    streamFinishReason: null
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

    const toolCallId = readString(record, "id", "toolCallId") || `call_${openAiToolCallIndex}`;
    const toolName = readString(record, "name", "toolName") || "tool";

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

    const argumentsDelta =
      readString(record, "delta", "argumentsDelta", "arguments", "textDelta") ||
      String(record.value ?? "");

    if (argumentsDelta.length > 0) {
      const delta: Partial<OpenAIMessage> = {
        tool_calls: [
          { index: openAiToolCallIndex, function: { arguments: argumentsDelta } } as OpenAIToolCall
        ]
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
    chunks.push(formatSSEDataLine(toChunk(state, {}, finishReason)));
    return chunks;
  }

  return chunks;
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
  const source =
    (Array.isArray(message.toolCalls) ? message.toolCalls : null) ??
    (Array.isArray(message.tool_calls) ? message.tool_calls : null) ??
    [];

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
