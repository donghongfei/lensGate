import type { IncomingMessage, ServerResponse } from "node:http";
import { randomBytes } from "node:crypto";
import { streamSimple } from "@mariozechner/pi-ai";
import {
  createTraceId,
  propagateAttributes,
  startActiveObservation,
  startObservation,
  type LangfuseAgent,
  type LangfuseSpan
} from "@langfuse/tracing";
import { getAccessToken, NotLoggedInError } from "./auth.js";
import { BODY_LIMIT_BYTES, CORS_ALLOW_ORIGIN, LOG_CHAT_PAYLOAD, LOG_LEVEL, PROXY_API_KEY } from "./config.js";
import {
  assistantMessageToResponse,
  createStreamState,
  eventToSSEChunks,
  formatUsageSSEChunk,
  generateCompletionId,
  ImageNotSupportedError,
  openAIRequestToContext
} from "./convert.js";
import { getAllCodexModels, modelIdFromUnknown, resolveCodexModel, UnknownModelError } from "./models.js";
import type {
  OpenAIChatRequest,
  OpenAIErrorResponse,
  OpenAIModelObject,
  OpenAIModelsResponse,
  OpenAIUsage
} from "./types.js";

type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40
};

const CURRENT_LOG_LEVEL: LogLevel = normalizeLogLevel(LOG_LEVEL);

class PayloadTooLargeError extends Error {
  constructor(message = "Request body is too large.") {
    super(message);
    this.name = "PayloadTooLargeError";
  }
}

class InvalidRequestError extends Error {
  code: string | null;

  constructor(message: string, code: string | null = "invalid_request") {
    super(message);
    this.name = "InvalidRequestError";
    this.code = code;
  }
}

class InvalidJSONError extends Error {
  constructor(message = "Invalid JSON payload.") {
    super(message);
    this.name = "InvalidJSONError";
  }
}

class UpstreamModelError extends Error {
  code: string | null;

  constructor(message: string, code: string | null = "upstream_model_error") {
    super(message);
    this.name = "UpstreamModelError";
    this.code = code;
  }
}

function normalizeLogLevel(value: string | undefined): LogLevel {
  if (value === "debug" || value === "info" || value === "warn" || value === "error") {
    return value;
  }
  return "info";
}

function shouldLog(level: LogLevel): boolean {
  return LOG_LEVEL_PRIORITY[level] >= LOG_LEVEL_PRIORITY[CURRENT_LOG_LEVEL];
}

function writeLog(level: LogLevel, event: string, fields: Record<string, unknown>): void {
  if (!shouldLog(level)) {
    return;
  }

  const payload = {
    ts: new Date().toISOString(),
    level,
    event,
    ...fields
  };
  const line = `${JSON.stringify(payload)}\n`;
  if (level === "error") {
    process.stderr.write(line);
    return;
  }
  process.stdout.write(line);
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function normalizeId(value: unknown, maxLength: number): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > maxLength) {
    return null;
  }
  return trimmed;
}

function readHeaderString(request: IncomingMessage, headerName: string, maxLength: number): string | null {
  const headerValue = request.headers[headerName];
  const candidate = Array.isArray(headerValue) ? headerValue[0] : headerValue;
  return normalizeId(candidate, maxLength);
}

function getRequestId(request: IncomingMessage): string {
  const requestId = readHeaderString(request, "x-request-id", 128);
  if (requestId) {
    return requestId;
  }
  return `req_${randomBytes(8).toString("hex")}`;
}

interface Candidate {
  source: string;
  value: unknown;
}

interface ResolvedCandidate {
  source: string;
  value: string;
}

interface ConversationInfo {
  sender?: string;
  messageId?: string;
  sessionId?: string;
  turnId?: string;
  traceId?: string;
}

interface ParsedTraceparent {
  traceId: string;
  spanId: string;
  traceFlags: number;
}

interface ParentSpanContext {
  traceId: string;
  spanId: string;
  traceFlags: number;
}

interface CorrelationResolution {
  sessionId?: string;
  userId?: string;
  turnId?: string;
  traceId?: string;
  parentSpanContext?: ParentSpanContext;
  sources: {
    sessionId?: string;
    userId?: string;
    turnId?: string;
    traceId?: string;
  };
  conversationInfo: ConversationInfo;
  traceparent: ParsedTraceparent | null;
}

interface PayloadPassthroughRule {
  bodyKeys: string[];
  payloadKey: string;
}

const OPENAI_PAYLOAD_PASSTHROUGH_RULES: PayloadPassthroughRule[] = [
  { bodyKeys: ["top_p", "topP"], payloadKey: "top_p" },
  { bodyKeys: ["frequency_penalty", "frequencyPenalty"], payloadKey: "frequency_penalty" },
  { bodyKeys: ["presence_penalty", "presencePenalty"], payloadKey: "presence_penalty" },
  { bodyKeys: ["seed"], payloadKey: "seed" },
  { bodyKeys: ["stop"], payloadKey: "stop" },
  { bodyKeys: ["metadata"], payloadKey: "metadata" },
  { bodyKeys: ["user"], payloadKey: "user" },
  { bodyKeys: ["response_format", "responseFormat"], payloadKey: "response_format" },
  { bodyKeys: ["n"], payloadKey: "n" },
  { bodyKeys: ["logit_bias", "logitBias"], payloadKey: "logit_bias" },
  { bodyKeys: ["logprobs"], payloadKey: "logprobs" },
  { bodyKeys: ["top_logprobs", "topLogprobs"], payloadKey: "top_logprobs" },
  { bodyKeys: ["parallel_tool_calls", "parallelToolCalls"], payloadKey: "parallel_tool_calls" },
  { bodyKeys: ["tool_choice", "toolChoice"], payloadKey: "tool_choice" },
  { bodyKeys: ["maxCompletionTokens"], payloadKey: "maxCompletionTokens" },
  { bodyKeys: ["max_completion_tokens"], payloadKey: "max_completion_tokens" }
];

function firstValidCandidate(maxLength: number, candidates: Candidate[]): ResolvedCandidate | undefined {
  for (const candidate of candidates) {
    const value = normalizeId(candidate.value, maxLength);
    if (value) {
      return { source: candidate.source, value };
    }
  }
  return undefined;
}

function normalizeHex(value: string, length: number, options?: { allowAllZero?: boolean }): string | null {
  const normalized = value.trim().toLowerCase();
  if (normalized.length !== length) {
    return null;
  }
  if (!/^[0-9a-f]+$/.test(normalized)) {
    return null;
  }
  if (!options?.allowAllZero && /^0+$/.test(normalized)) {
    return null;
  }
  return normalized;
}

function normalizeTraceId(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  return normalizeHex(value, 32);
}

function normalizeSpanId(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  return normalizeHex(value, 16);
}

function parseTraceparent(value: string | null): ParsedTraceparent | null {
  if (!value) {
    return null;
  }

  const parts = value.trim().split("-");
  if (parts.length !== 4) {
    return null;
  }

  const version = normalizeHex(parts[0], 2, { allowAllZero: true });
  const traceId = normalizeHex(parts[1], 32);
  const spanId = normalizeSpanId(parts[2]);
  const traceFlagsHex = parts[3].trim().toLowerCase();
  if (!version || version === "ff" || !traceId || !spanId || !/^[0-9a-f]{2}$/.test(traceFlagsHex)) {
    return null;
  }

  return {
    traceId,
    spanId,
    traceFlags: Number.parseInt(traceFlagsHex, 16)
  };
}

function buildSyntheticParentSpanContext(traceId: string): ParentSpanContext {
  return {
    traceId,
    spanId: randomBytes(8).toString("hex"),
    traceFlags: 1
  };
}

function textFromMessageContent(content: OpenAIChatRequest["messages"][number]["content"]): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n");
}

function parseConversationInfoObject(candidate: unknown): ConversationInfo {
  if (!candidate || typeof candidate !== "object") {
    return {};
  }

  const record = candidate as Record<string, unknown>;
  const sender = firstValidId(200, record.sender, record.sender_id, record.senderId, record.user_id, record.userId);
  const messageId = firstValidId(200, record.message_id, record.messageId, record.id);
  const sessionId = firstValidCandidate(200, [
    { source: "conversation_info.session_id", value: record.session_id },
    { source: "conversation_info.sessionId", value: record.sessionId },
    { source: "conversation_info.conversation_id", value: record.conversation_id },
    { source: "conversation_info.conversationId", value: record.conversationId },
    { source: "conversation_info.thread_id", value: record.thread_id },
    { source: "conversation_info.threadId", value: record.threadId }
  ])?.value;
  const turnId = firstValidCandidate(200, [
    { source: "conversation_info.turn_id", value: record.turn_id },
    { source: "conversation_info.turnId", value: record.turnId },
    { source: "conversation_info.workflow_id", value: record.workflow_id },
    { source: "conversation_info.workflowId", value: record.workflowId },
    { source: "conversation_info.chat_id", value: record.chat_id },
    { source: "conversation_info.chatId", value: record.chatId }
  ])?.value;
  const traceId = firstValidCandidate(256, [
    { source: "conversation_info.trace_id", value: record.trace_id },
    { source: "conversation_info.traceId", value: record.traceId }
  ])?.value;
  return { sender, messageId, sessionId, turnId, traceId };
}

function extractConversationInfoFromText(text: string): ConversationInfo {
  const result: ConversationInfo = {};
  if (!text) {
    return result;
  }

  const codeFenceRegex = /```json\s*([\s\S]*?)```/gi;
  let match: RegExpExecArray | null;
  while ((match = codeFenceRegex.exec(text)) !== null) {
    const jsonBlock = match[1];
    try {
      const parsed = JSON.parse(jsonBlock);
      const info = parseConversationInfoObject(parsed);
      if (info.sender && !result.sender) {
        result.sender = info.sender;
      }
      if (info.messageId && !result.messageId) {
        result.messageId = info.messageId;
      }
      if (info.sessionId && !result.sessionId) {
        result.sessionId = info.sessionId;
      }
      if (info.turnId && !result.turnId) {
        result.turnId = info.turnId;
      }
      if (info.traceId && !result.traceId) {
        result.traceId = info.traceId;
      }
      if (result.sender && result.messageId && result.sessionId && result.turnId && result.traceId) {
        return result;
      }
    } catch {
      // ignore invalid json fences
    }
  }

  if (!result.sender) {
    const senderMatch = text.match(/"sender"\s*:\s*"([^"]+)"/);
    if (senderMatch) {
      result.sender = normalizeId(senderMatch[1], 200) ?? undefined;
    }
  }
  if (!result.messageId) {
    const messageIdMatch = text.match(/"message_id"\s*:\s*"([^"]+)"/);
    if (messageIdMatch) {
      result.messageId = normalizeId(messageIdMatch[1], 200) ?? undefined;
    }
  }
  if (!result.sessionId) {
    const sessionIdMatch = text.match(
      /"(session_id|sessionId|conversation_id|conversationId|thread_id|threadId)"\s*:\s*"([^"]+)"/
    );
    if (sessionIdMatch) {
      result.sessionId = normalizeId(sessionIdMatch[2], 200) ?? undefined;
    }
  }
  if (!result.turnId) {
    const turnIdMatch = text.match(/"(turn_id|turnId|workflow_id|workflowId|chat_id|chatId)"\s*:\s*"([^"]+)"/);
    if (turnIdMatch) {
      result.turnId = normalizeId(turnIdMatch[2], 200) ?? undefined;
    }
  }
  if (!result.traceId) {
    const traceIdMatch = text.match(/"(trace_id|traceId)"\s*:\s*"([^"]+)"/);
    if (traceIdMatch) {
      result.traceId = normalizeId(traceIdMatch[2], 256) ?? undefined;
    }
  }

  return result;
}

function extractConversationInfoFromMessages(messages: OpenAIChatRequest["messages"]): ConversationInfo {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "user") {
      continue;
    }
    const text = textFromMessageContent(message.content);
    const info = extractConversationInfoFromText(text);
    if (info.sender || info.messageId || info.sessionId || info.turnId || info.traceId) {
      return info;
    }
  }
  return {};
}

function firstValidId(maxLength: number, ...candidates: unknown[]): string | undefined {
  return firstValidCandidate(
    maxLength,
    candidates.map((value, index) => ({ source: `candidate_${index}`, value }))
  )?.value;
}

async function resolveCorrelationContext(
  request: IncomingMessage,
  body: OpenAIChatRequest
): Promise<CorrelationResolution> {
  const bodyRecord = body as Record<string, unknown>;
  const conversationInfo = extractConversationInfoFromMessages(body.messages);

  const sessionCandidate = firstValidCandidate(200, [
    { source: "body.session_id", value: bodyRecord.session_id },
    { source: "body.sessionId", value: bodyRecord.sessionId },
    { source: "body.conversation_id", value: bodyRecord.conversation_id },
    { source: "body.conversationId", value: bodyRecord.conversationId },
    { source: "body.thread_id", value: bodyRecord.thread_id },
    { source: "body.threadId", value: bodyRecord.threadId },
    { source: "header.x-session-id", value: readHeaderString(request, "x-session-id", 200) },
    { source: "header.x-conversation-id", value: readHeaderString(request, "x-conversation-id", 200) },
    { source: "header.x-thread-id", value: readHeaderString(request, "x-thread-id", 200) },
    { source: "conversation_info.session_id", value: conversationInfo.sessionId }
  ]);

  const userCandidate = firstValidCandidate(200, [
    { source: "body.user", value: bodyRecord.user },
    { source: "body.user_id", value: bodyRecord.user_id },
    { source: "body.userId", value: bodyRecord.userId },
    { source: "header.x-user-id", value: readHeaderString(request, "x-user-id", 200) },
    { source: "conversation_info.sender", value: conversationInfo.sender }
  ]);

  const turnCandidate = firstValidCandidate(200, [
    { source: "body.turn_id", value: bodyRecord.turn_id },
    { source: "body.turnId", value: bodyRecord.turnId },
    { source: "body.workflow_id", value: bodyRecord.workflow_id },
    { source: "body.workflowId", value: bodyRecord.workflowId },
    { source: "body.chat_id", value: bodyRecord.chat_id },
    { source: "body.chatId", value: bodyRecord.chatId },
    { source: "header.x-turn-id", value: readHeaderString(request, "x-turn-id", 200) },
    { source: "header.x-workflow-id", value: readHeaderString(request, "x-workflow-id", 200) },
    { source: "header.x-chat-id", value: readHeaderString(request, "x-chat-id", 200) },
    { source: "header.x-correlation-id", value: readHeaderString(request, "x-correlation-id", 200) },
    { source: "conversation_info.turn_id", value: conversationInfo.turnId },
    {
      source: "conversation_info.sender_message_id",
      value:
        conversationInfo.sender && conversationInfo.messageId
          ? `${conversationInfo.sender}:${conversationInfo.messageId}`
          : undefined
    },
    { source: "conversation_info.message_id", value: conversationInfo.messageId }
  ]);

  const traceparent = parseTraceparent(readHeaderString(request, "traceparent", 512));
  const explicitTraceCandidate = firstValidCandidate(256, [
    { source: "body.trace_id", value: bodyRecord.trace_id },
    { source: "body.traceId", value: bodyRecord.traceId },
    { source: "header.x-trace-id", value: readHeaderString(request, "x-trace-id", 256) },
    { source: "conversation_info.trace_id", value: conversationInfo.traceId }
  ]);

  let traceId: string | undefined;
  let traceSource: string | undefined;
  let parentSpanContext: ParentSpanContext | undefined;

  if (traceparent) {
    traceId = traceparent.traceId;
    traceSource = "header.traceparent";
    parentSpanContext = {
      traceId: traceparent.traceId,
      spanId: traceparent.spanId,
      traceFlags: traceparent.traceFlags
    };
  } else if (explicitTraceCandidate) {
    const normalizedProvidedTraceId = normalizeTraceId(explicitTraceCandidate.value);
    traceId = normalizedProvidedTraceId ?? (await createTraceId(`trace:${explicitTraceCandidate.value}`));
    traceSource = normalizedProvidedTraceId
      ? explicitTraceCandidate.source
      : `${explicitTraceCandidate.source}:seed`;
    parentSpanContext = buildSyntheticParentSpanContext(traceId);
  } else if (turnCandidate?.value) {
    traceId = await createTraceId(`turn:${turnCandidate.value}`);
    traceSource = `${turnCandidate.source}:derived_trace`;
    parentSpanContext = buildSyntheticParentSpanContext(traceId);
  }

  return {
    sessionId: sessionCandidate?.value,
    userId: userCandidate?.value,
    turnId: turnCandidate?.value,
    traceId,
    parentSpanContext,
    sources: {
      sessionId: sessionCandidate?.source,
      userId: userCandidate?.source,
      turnId: turnCandidate?.source,
      traceId: traceSource
    },
    conversationInfo,
    traceparent
  };
}

function buildCorrelationHints(
  request: IncomingMessage,
  body: OpenAIChatRequest,
  correlation: CorrelationResolution
): Record<string, unknown> {
  const bodyRecord = body as Record<string, unknown>;

  const bodyKeys = Object.keys(bodyRecord).filter((key) => !["messages", "tools"].includes(key));
  const headerKeys = Object.keys(request.headers)
    .map((key) => key.toLowerCase())
    .filter((key) => key !== "authorization");

  return {
    body_keys: bodyKeys.slice(0, 40),
    header_keys: headerKeys.slice(0, 40),
    body_candidates: {
      session_id: normalizeId(bodyRecord.session_id, 200),
      sessionId: normalizeId(bodyRecord.sessionId, 200),
      conversation_id: normalizeId(bodyRecord.conversation_id, 200),
      conversationId: normalizeId(bodyRecord.conversationId, 200),
      thread_id: normalizeId(bodyRecord.thread_id, 200),
      threadId: normalizeId(bodyRecord.threadId, 200),
      turn_id: normalizeId(bodyRecord.turn_id, 200),
      turnId: normalizeId(bodyRecord.turnId, 200),
      workflow_id: normalizeId(bodyRecord.workflow_id, 200),
      workflowId: normalizeId(bodyRecord.workflowId, 200),
      chat_id: normalizeId(bodyRecord.chat_id, 200),
      chatId: normalizeId(bodyRecord.chatId, 200),
      trace_id: normalizeId(bodyRecord.trace_id, 256),
      traceId: normalizeId(bodyRecord.traceId, 256),
      user: normalizeId(bodyRecord.user, 200),
      user_id: normalizeId(bodyRecord.user_id, 200),
      userId: normalizeId(bodyRecord.userId, 200)
    },
    header_candidates: {
      x_session_id: readHeaderString(request, "x-session-id", 200),
      x_conversation_id: readHeaderString(request, "x-conversation-id", 200),
      x_thread_id: readHeaderString(request, "x-thread-id", 200),
      x_turn_id: readHeaderString(request, "x-turn-id", 200),
      x_workflow_id: readHeaderString(request, "x-workflow-id", 200),
      x_chat_id: readHeaderString(request, "x-chat-id", 200),
      x_user_id: readHeaderString(request, "x-user-id", 200),
      x_correlation_id: readHeaderString(request, "x-correlation-id", 200),
      x_trace_id: readHeaderString(request, "x-trace-id", 200),
      traceparent: readHeaderString(request, "traceparent", 512),
      tracestate: readHeaderString(request, "tracestate", 512),
      baggage: readHeaderString(request, "baggage", 512)
    },
    derived: {
      session_id: correlation.sessionId ?? null,
      user_id: correlation.userId ?? null,
      turn_id: correlation.turnId ?? null,
      trace_id: correlation.traceId ?? null,
      source: {
        session_id: correlation.sources.sessionId ?? null,
        user_id: correlation.sources.userId ?? null,
        turn_id: correlation.sources.turnId ?? null,
        trace_id: correlation.sources.traceId ?? null
      },
      conversation_info: {
        sender: correlation.conversationInfo.sender ?? null,
        message_id: correlation.conversationInfo.messageId ?? null,
        session_id: correlation.conversationInfo.sessionId ?? null,
        turn_id: correlation.conversationInfo.turnId ?? null,
        trace_id: correlation.conversationInfo.traceId ?? null
      },
      traceparent: correlation.traceparent ?? null
    }
  };
}

function applyOpenAIParameterPassthrough(payload: unknown, body: OpenAIChatRequest): void {
  if (!payload || typeof payload !== "object") {
    return;
  }

  const payloadRecord = payload as Record<string, unknown>;
  const bodyRecord = body as Record<string, unknown>;

  for (const rule of OPENAI_PAYLOAD_PASSTHROUGH_RULES) {
    for (const bodyKey of rule.bodyKeys) {
      if (bodyRecord[bodyKey] !== undefined) {
        payloadRecord[rule.payloadKey] = bodyRecord[bodyKey];
        break;
      }
    }
  }
}

function isSensitiveHeader(headerName: string): boolean {
  return (
    headerName === "authorization" ||
    headerName === "proxy-authorization" ||
    headerName === "cookie" ||
    headerName === "set-cookie" ||
    headerName === "x-api-key"
  );
}

function sanitizeForLog(
  value: unknown,
  depth = 0,
  limits: { maxDepth: number; maxArray: number; maxKeys: number; maxString: number } = {
    maxDepth: 8,
    maxArray: 200,
    maxKeys: 200,
    maxString: 4000
  }
): unknown {
  if (value === null || value === undefined) {
    return value ?? null;
  }

  if (depth > limits.maxDepth) {
    return "[TRUNCATED_MAX_DEPTH]";
  }

  if (typeof value === "string") {
    if (value.length <= limits.maxString) {
      return value;
    }
    return `${value.slice(0, limits.maxString)}...[TRUNCATED_${value.length - limits.maxString}_CHARS]`;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return value;
  }

  if (Array.isArray(value)) {
    const result = value.slice(0, limits.maxArray).map((item) => sanitizeForLog(item, depth + 1, limits));
    if (value.length > limits.maxArray) {
      result.push(`[TRUNCATED_${value.length - limits.maxArray}_ITEMS]`);
    }
    return result;
  }

  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const entries = Object.entries(record);
    const out: Record<string, unknown> = {};
    const maxKeys = Math.min(entries.length, limits.maxKeys);
    for (let i = 0; i < maxKeys; i += 1) {
      const [key, item] = entries[i];
      out[key] = sanitizeForLog(item, depth + 1, limits);
    }
    if (entries.length > limits.maxKeys) {
      out.__truncated_keys__ = entries.length - limits.maxKeys;
    }
    return out;
  }

  return String(value);
}

function buildRequestHeadersForLog(request: IncomingMessage): Record<string, unknown> {
  const headers: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(request.headers)) {
    const normalizedKey = key.toLowerCase();
    if (isSensitiveHeader(normalizedKey)) {
      headers[key] = "[REDACTED]";
      continue;
    }
    headers[key] = sanitizeForLog(value);
  }
  return headers;
}

interface ToolContextSummary {
  advertisedTools: number;
  assistantToolCallsInInput: number;
  toolMessagesInInput: number;
  isAgentic: boolean;
}

interface NormalizedToolCall {
  id: string;
  name: string;
  argumentsText: string;
}

interface CompletedToolExecution extends NormalizedToolCall {
  output: unknown;
}

type ObservationWithChildren = Pick<LangfuseSpan, "startObservation">;

function stringifyToolCallArguments(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === undefined || value === null) {
    return "";
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function extractToolCallsFromUnknown(message: unknown): NormalizedToolCall[] {
  if (!message || typeof message !== "object") {
    return [];
  }

  const record = message as Record<string, unknown>;
  const structuredCalls =
    (Array.isArray(record.toolCalls) ? record.toolCalls : null) ??
    (Array.isArray(record.tool_calls) ? record.tool_calls : null) ??
    [];
  const contentCalls = Array.isArray(record.content)
    ? record.content.filter(
        (item): item is Record<string, unknown> =>
          Boolean(item) && typeof item === "object" && (item as Record<string, unknown>).type === "toolCall"
      )
    : [];
  const source = structuredCalls.length > 0 ? structuredCalls : contentCalls;

  const calls: NormalizedToolCall[] = [];
  for (let i = 0; i < source.length; i += 1) {
    const item = source[i];
    if (!item || typeof item !== "object") {
      continue;
    }
    const itemRecord = item as Record<string, unknown>;
    const functionRecord =
      itemRecord.function && typeof itemRecord.function === "object"
        ? (itemRecord.function as Record<string, unknown>)
        : undefined;

    const id =
      normalizeId(itemRecord.id, 200) ??
      normalizeId(itemRecord.toolCallId, 200) ??
      normalizeId(functionRecord?.id, 200) ??
      `call_${i}`;
    const name =
      normalizeId(itemRecord.name, 200) ??
      normalizeId(itemRecord.toolName, 200) ??
      normalizeId(functionRecord?.name, 200) ??
      "tool";
    const argumentsText = stringifyToolCallArguments(
      itemRecord.arguments ?? functionRecord?.arguments ?? itemRecord.input ?? ""
    );

    calls.push({
      id,
      name,
      argumentsText
    });
  }

  return calls;
}

function parseToolArgumentsForObservation(argumentsText: string): unknown {
  const input = argumentsText.trim();
  if (!input) {
    return {};
  }
  try {
    return JSON.parse(input);
  } catch {
    return { raw: argumentsText };
  }
}

function toolMessageContentToObservationOutput(
  content: OpenAIChatRequest["messages"][number]["content"]
): unknown {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return null;
  }

  const textParts = content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .filter((text) => text.length > 0);

  if (textParts.length > 0) {
    return textParts.join("\n");
  }

  return content;
}

function extractRecentCompletedToolExecutions(messages: OpenAIChatRequest["messages"]): CompletedToolExecution[] {
  const lastUserMessageIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === "user") {
        return i;
      }
    }
    return -1;
  })();

  // Only associate tool activity that happened after the latest user message (current turn).
  const currentTurnStart = lastUserMessageIndex >= 0 ? lastUserMessageIndex : messages.length;

  let lastAssistantWithToolCallsIndex = -1;
  for (let i = messages.length - 1; i > currentTurnStart; i -= 1) {
    const message = messages[i];
    if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      lastAssistantWithToolCallsIndex = i;
      break;
    }
  }

  if (lastAssistantWithToolCallsIndex < 0) {
    return [];
  }

  const assistantMessage = messages[lastAssistantWithToolCallsIndex];
  const recentToolCalls =
    assistantMessage.role === "assistant" && Array.isArray(assistantMessage.tool_calls)
      ? assistantMessage.tool_calls
      : [];
  const toolCallById = new Map<string, NormalizedToolCall>();
  for (const toolCall of recentToolCalls) {
    if (!toolCall || typeof toolCall !== "object") {
      continue;
    }
    const toolCallId = normalizeId(toolCall.id, 200);
    if (!toolCallId) {
      continue;
    }
    toolCallById.set(toolCallId, {
      id: toolCallId,
      name: normalizeId(toolCall.function?.name, 200) ?? "tool",
      argumentsText: typeof toolCall.function?.arguments === "string" ? toolCall.function.arguments : ""
    });
  }

  const completedExecutions: CompletedToolExecution[] = [];
  for (let i = lastAssistantWithToolCallsIndex + 1; i < messages.length; i += 1) {
    const message = messages[i];
    if (message.role !== "tool") {
      continue;
    }
    const toolCallId = normalizeId(message.tool_call_id, 200);
    const matchedCall = toolCallId ? toolCallById.get(toolCallId) : undefined;
    const fallbackName = normalizeId(message.name, 200) ?? "tool";
    const fallbackId = toolCallId ?? `tool_result_${i}`;

    completedExecutions.push({
      id: fallbackId,
      name: matchedCall?.name ?? fallbackName,
      argumentsText: matchedCall?.argumentsText ?? "",
      output: toolMessageContentToObservationOutput(message.content)
    });
  }

  return completedExecutions;
}

function emitCompletedToolResultObservations(
  parent: ObservationWithChildren,
  completedExecutions: CompletedToolExecution[]
): void {
  for (const execution of completedExecutions) {
    const toolObservation = parent.startObservation(
      `tool.${execution.name}`,
      {
        input: parseToolArgumentsForObservation(execution.argumentsText),
        output: execution.output,
        metadata: {
          toolCallId: execution.id,
          phase: "tool_result_from_history"
        },
        statusMessage: "Tool result observed from request history."
      },
      { asType: "tool" }
    );
    toolObservation.end();
  }
}

function emitRequestedToolCallObservations(parent: ObservationWithChildren, toolCalls: NormalizedToolCall[]): void {
  for (const toolCall of toolCalls) {
    const toolObservation = parent.startObservation(
      `tool.${toolCall.name}`,
      {
        input: parseToolArgumentsForObservation(toolCall.argumentsText),
        output: {
          status: "requested",
          detail: "Tool call requested by model; awaiting runtime result."
        },
        metadata: {
          toolCallId: toolCall.id,
          phase: "model_requested"
        },
        statusMessage: "Tool call requested by model. Runtime executes this tool outside the proxy."
      },
      { asType: "tool" }
    );
    toolObservation.end();
  }
}

function summarizeToolContext(body: OpenAIChatRequest): ToolContextSummary {
  let assistantToolCallsInInput = 0;
  let toolMessagesInInput = 0;

  for (const message of body.messages) {
    if (message.role === "assistant" && Array.isArray(message.tool_calls)) {
      assistantToolCallsInInput += message.tool_calls.length;
    }
    if (message.role === "tool") {
      toolMessagesInInput += 1;
    }
  }

  const advertisedTools = Array.isArray(body.tools) ? body.tools.length : 0;
  const isAgentic = advertisedTools > 0 || assistantToolCallsInInput > 0 || toolMessagesInInput > 0;

  return {
    advertisedTools,
    assistantToolCallsInInput,
    toolMessagesInInput,
    isAgentic
  };
}

function setCorsHeaders(response: ServerResponse): void {
  response.setHeader("Access-Control-Allow-Origin", CORS_ALLOW_ORIGIN);
  response.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

function sendJson(response: ServerResponse, statusCode: number, body: unknown): void {
  response.statusCode = statusCode;
  setCorsHeaders(response);
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.end(JSON.stringify(body));
}

function sendOpenAIError(
  response: ServerResponse,
  statusCode: number,
  message: string,
  type: string,
  code: string | null
): void {
  const body: OpenAIErrorResponse = {
    error: {
      message,
      type,
      param: null,
      code
    }
  };
  sendJson(response, statusCode, body);
}

function logAndSendOpenAIError(
  response: ServerResponse,
  requestId: string,
  statusCode: number,
  message: string,
  type: string,
  code: string | null,
  level: "warn" | "error",
  event: string
): void {
  writeLog(level, event, {
    request_id: requestId,
    status_code: statusCode,
    error_type: type,
    error_code: code,
    message
  });
  sendOpenAIError(response, statusCode, message, type, code);
}

async function readRequestBody(request: IncomingMessage): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalBytes = 0;

    request.on("data", (chunk: Buffer | string) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      totalBytes += buffer.length;
      if (totalBytes > BODY_LIMIT_BYTES) {
        reject(new PayloadTooLargeError(`Body exceeds ${BODY_LIMIT_BYTES} bytes.`));
        request.destroy();
        return;
      }
      chunks.push(buffer);
    });

    request.on("end", () => {
      resolve(Buffer.concat(chunks).toString("utf8"));
    });

    request.on("error", (error: Error) => {
      reject(error);
    });
  });
}

async function parseChatRequest(request: IncomingMessage): Promise<OpenAIChatRequest> {
  const rawBody = await readRequestBody(request);
  let parsed: unknown;
  try {
    parsed = rawBody.length > 0 ? JSON.parse(rawBody) : {};
  } catch {
    throw new InvalidJSONError();
  }

  if (!parsed || typeof parsed !== "object") {
    throw new InvalidRequestError("Request body must be a JSON object.");
  }

  const body = parsed as Partial<OpenAIChatRequest>;
  if (typeof body.model !== "string" || body.model.length === 0) {
    throw new InvalidRequestError("`model` is required and must be a non-empty string.", "missing_model");
  }
  if (!Array.isArray(body.messages)) {
    throw new InvalidRequestError("`messages` is required and must be an array.", "missing_messages");
  }
  if (body.messages.length === 0) {
    throw new InvalidRequestError("`messages` must contain at least one message.", "empty_messages");
  }

  return body as OpenAIChatRequest;
}

function asAsyncIterable(streamLike: unknown): AsyncIterable<unknown> {
  if (isAsyncIterable(streamLike)) {
    return streamLike;
  }

  if (streamLike && typeof streamLike === "object") {
    const record = streamLike as Record<string, unknown>;

    if (isAsyncIterable(record.events)) {
      return record.events;
    }

    if (isAsyncIterable(record.stream)) {
      return record.stream;
    }
  }

  throw new Error("Upstream stream object is not async-iterable.");
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  if (!value || typeof value !== "object") {
    return false;
  }

  const iterator = (value as { [Symbol.asyncIterator]?: unknown })[Symbol.asyncIterator];
  return typeof iterator === "function";
}

async function getFinalAssistantMessage(streamLike: unknown): Promise<unknown> {
  if (streamLike && typeof streamLike === "object") {
    const resultFn = (streamLike as { result?: () => Promise<unknown> }).result;
    if (typeof resultFn === "function") {
      return resultFn.call(streamLike);
    }
  }

  let finalMessage: unknown = null;
  for await (const event of asAsyncIterable(streamLike)) {
    if (!event || typeof event !== "object") {
      continue;
    }
    const record = event as Record<string, unknown>;
    if ((record.type === "done" || record.type === "assistant_done") && record.message) {
      finalMessage = record.message;
    }
  }

  return finalMessage;
}

async function writeSSEStream(
  response: ServerResponse,
  streamLike: unknown,
  modelId: string,
  requestId: string,
  includeUsage: boolean,
  abortController: AbortController
): Promise<{
  emittedAnyOutput: boolean;
  accumulatedText: string;
  streamFinishReason: string | null;
  streamMessage: unknown;
  streamUsage: OpenAIUsage | null;
  iterationErrorMessage: string | null;
  clientDisconnected: boolean;
}> {
  response.statusCode = 200;
  setCorsHeaders(response);
  response.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  response.setHeader("Cache-Control", "no-cache, no-transform");
  response.setHeader("Connection", "keep-alive");
  response.setHeader("X-Accel-Buffering", "no");
  response.flushHeaders?.();

  const state = createStreamState(modelId);
  let closed = false;
  let clientDisconnected = false;
  let iterationErrorMessage: string | null = null;

  response.on("close", () => {
    clientDisconnected = true;
    if (!abortController.signal.aborted) {
      abortController.abort();
    }
  });

  try {
    for await (const event of asAsyncIterable(streamLike)) {
      if (clientDisconnected) {
        break;
      }
      const chunks = eventToSSEChunks(event, state);
      for (const chunk of chunks) {
        response.write(chunk);
      }
    }
  } catch (error) {
    iterationErrorMessage = getErrorMessage(error);
    state.streamFinishReason = state.streamFinishReason ?? "content_filter";
    writeLog("warn", "chat.stream.iteration_error", {
      request_id: requestId,
      model: modelId,
      message: iterationErrorMessage
    });
    if (!clientDisconnected && !response.writableEnded) {
      const errorChunks = eventToSSEChunks({ type: "error", message: iterationErrorMessage }, state);
      for (const chunk of errorChunks) {
        response.write(chunk);
      }
    }
  } finally {
    if (!closed && !response.writableEnded) {
      if (includeUsage) {
        const usageChunk = formatUsageSSEChunk(state);
        if (usageChunk) {
          response.write(usageChunk);
        }
      }
      response.write("data: [DONE]\n\n");
      response.end();
      closed = true;
    }
  }

  return {
    emittedAnyOutput: state.emittedAnyOutput,
    accumulatedText: state.accumulatedText,
    streamFinishReason: state.streamFinishReason,
    streamMessage: state.streamMessage,
    streamUsage: state.streamUsage,
    iterationErrorMessage,
    clientDisconnected
  };
}

function toOpenAIModelObject(model: unknown): OpenAIModelObject {
  const id = modelIdFromUnknown(model);
  const record = model && typeof model === "object" ? (model as Record<string, unknown>) : {};
  const contextWindow = typeof record.contextWindow === "number" ? record.contextWindow : undefined;
  const maxTokens = typeof record.maxTokens === "number" ? record.maxTokens : undefined;
  return {
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "openai",
    ...(contextWindow !== undefined ? { context_window: contextWindow } : {}),
    ...(maxTokens !== undefined ? { max_tokens: maxTokens } : {})
  };
}

function isRateLimitError(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const record = error as Record<string, unknown>;
  const text = String(record.message ?? "").toLowerCase();
  const code = String(record.code ?? "").toLowerCase();
  return text.includes("rate_limit") || text.includes("usage_limit") || code.includes("rate_limit");
}

function getAssistantStopReason(message: unknown): string {
  if (!message || typeof message !== "object") {
    return "";
  }
  const record = message as Record<string, unknown>;
  const stopReason = record.stopReason ?? record.stop_reason ?? record.reason;
  return typeof stopReason === "string" ? stopReason : "";
}

function getAssistantErrorMessage(message: unknown): string {
  if (!message || typeof message !== "object") {
    return "";
  }
  const record = message as Record<string, unknown>;
  const value = record.errorMessage ?? record.error_message ?? record.message;
  return typeof value === "string" ? value : "";
}

function handleRequestError(response: ServerResponse, error: unknown, requestId: string): void {
  if (error instanceof NotLoggedInError) {
    logAndSendOpenAIError(
      response,
      requestId,
      401,
      error.message,
      "authentication_error",
      "not_logged_in",
      "warn",
      "request.error.auth"
    );
    return;
  }

  if (error instanceof ImageNotSupportedError) {
    logAndSendOpenAIError(
      response,
      requestId,
      400,
      error.message,
      "invalid_request_error",
      "image_not_supported",
      "warn",
      "request.error.image_not_supported"
    );
    return;
  }

  if (error instanceof UnknownModelError) {
    logAndSendOpenAIError(
      response,
      requestId,
      400,
      error.message,
      "invalid_request_error",
      "unknown_model",
      "warn",
      "request.error.unknown_model"
    );
    return;
  }

  if (error instanceof InvalidRequestError) {
    logAndSendOpenAIError(
      response,
      requestId,
      400,
      error.message,
      "invalid_request_error",
      error.code,
      "warn",
      "request.error.invalid_request"
    );
    return;
  }

  if (error instanceof InvalidJSONError) {
    logAndSendOpenAIError(
      response,
      requestId,
      400,
      error.message,
      "invalid_request_error",
      "invalid_json",
      "warn",
      "request.error.invalid_json"
    );
    return;
  }

  if (error instanceof PayloadTooLargeError) {
    logAndSendOpenAIError(
      response,
      requestId,
      413,
      error.message,
      "invalid_request_error",
      "payload_too_large",
      "warn",
      "request.error.payload_too_large"
    );
    return;
  }

  if (error instanceof UpstreamModelError) {
    if (isRateLimitError(error)) {
      logAndSendOpenAIError(
        response,
        requestId,
        429,
        error.message,
        "rate_limit_error",
        "rate_limit",
        "warn",
        "request.error.upstream_rate_limit"
      );
      return;
    }
    logAndSendOpenAIError(
      response,
      requestId,
      502,
      error.message,
      "server_error",
      error.code,
      "warn",
      "request.error.upstream"
    );
    return;
  }

  if (isRateLimitError(error)) {
    logAndSendOpenAIError(
      response,
      requestId,
      429,
      "Rate limit exceeded.",
      "rate_limit_error",
      "rate_limit",
      "warn",
      "request.error.rate_limit"
    );
    return;
  }

  const message = error instanceof Error ? error.message : "Unknown proxy error.";
  logAndSendOpenAIError(
    response,
    requestId,
    500,
    message,
    "server_error",
    "proxy_internal_error",
    "error",
    "request.error.internal"
  );
}

function validateApiKeyHeader(request: IncomingMessage): void {
  const expectedApiKey = PROXY_API_KEY || undefined;
  if (!expectedApiKey) {
    return;
  }

  const authHeader = request.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new InvalidRequestError("Missing Authorization header.", "missing_authorization");
  }

  const providedKey = authHeader.slice("Bearer ".length).trim();
  if (providedKey !== expectedApiKey) {
    throw new InvalidRequestError("Invalid API key.", "invalid_api_key");
  }
}

async function handleModels(response: ServerResponse, requestId: string): Promise<void> {
  const models: OpenAIModelsResponse = {
    object: "list",
    data: getAllCodexModels().map((model) => toOpenAIModelObject(model))
  };
  sendJson(response, 200, models);
  writeLog("info", "models.list.success", {
    request_id: requestId,
    status_code: 200,
    model_count: models.data.length
  });
}

async function handleChatCompletions(
  request: IncomingMessage,
  response: ServerResponse,
  requestId: string
): Promise<void> {
  const startedAt = Date.now();
  validateApiKeyHeader(request);

  const body = await parseChatRequest(request);
  const correlation = await resolveCorrelationContext(request, body);
  const { sessionId, userId, turnId, traceId, parentSpanContext } = correlation;
  const toolContext = summarizeToolContext(body);
  const requestedModelId = body.model;
  const model = resolveCodexModel(requestedModelId);
  const accessToken = await getAccessToken();
  const context = await openAIRequestToContext(body);
  const bodyRecord = body as Record<string, unknown>;
  const maxCompletionTokensCandidate =
    typeof bodyRecord.max_completion_tokens === "number"
      ? bodyRecord.max_completion_tokens
      : typeof bodyRecord.maxCompletionTokens === "number"
        ? bodyRecord.maxCompletionTokens
        : undefined;
  const effectiveMaxTokens = typeof body.max_tokens === "number" ? body.max_tokens : maxCompletionTokensCandidate;
  const upstreamAbortController = new AbortController();
  const abortUpstream = (): void => {
    if (!upstreamAbortController.signal.aborted) {
      upstreamAbortController.abort();
    }
  };

  request.on("aborted", abortUpstream);
  response.on("close", abortUpstream);

  const streamOptions: Record<string, unknown> = {
    apiKey: accessToken,
    signal: upstreamAbortController.signal,
    onPayload: (payload: unknown) => applyOpenAIParameterPassthrough(payload, body)
  };
  if (typeof body.temperature === "number") {
    streamOptions.temperature = body.temperature;
  }
  if (typeof effectiveMaxTokens === "number") {
    streamOptions.maxTokens = effectiveMaxTokens;
  }

  writeLog("info", "chat.request.start", {
    request_id: requestId,
    requested_model: body.model,
    stream: body.stream === true,
    session_id: sessionId ?? null,
    user_id: userId ?? null,
    turn_id: turnId ?? null,
    trace_id: traceId ?? null,
    source: {
      session_id: correlation.sources.sessionId ?? null,
      user_id: correlation.sources.userId ?? null,
      turn_id: correlation.sources.turnId ?? null,
      trace_id: correlation.sources.traceId ?? null
    },
    tool_context: toolContext
  });

  writeLog("debug", "chat.request.correlation_hints", {
    request_id: requestId,
    ...buildCorrelationHints(request, body, correlation)
  });

  if (LOG_CHAT_PAYLOAD) {
    writeLog("info", "chat.request.payload", {
      request_id: requestId,
      headers: buildRequestHeadersForLog(request),
      body: sanitizeForLog(body)
    });
  }

  const modelParameters: Record<string, number> = {};
  if (typeof body.temperature === "number") modelParameters.temperature = body.temperature;
  if (typeof body.max_tokens === "number") modelParameters.max_tokens = body.max_tokens;
  if (typeof maxCompletionTokensCandidate === "number") modelParameters.max_completion_tokens = maxCompletionTokensCandidate;
  if (typeof body.top_p === "number") modelParameters.top_p = body.top_p;
  if (typeof body.frequency_penalty === "number") modelParameters.frequency_penalty = body.frequency_penalty;
  if (typeof body.presence_penalty === "number") modelParameters.presence_penalty = body.presence_penalty;

  const runWithRootObservation = async (rootObservation: LangfuseSpan | LangfuseAgent): Promise<void> => {
    const executeCompletion = async (): Promise<void> => {
      rootObservation.updateTrace({
        name: toolContext.isAgentic ? "openai.chat.agent_turn" : "openai.chat.turn",
        ...(sessionId ? { sessionId } : {}),
        ...(userId ? { userId } : {}),
        tags: toolContext.isAgentic
          ? ["openai-proxy", "chat.completions", "agentic"]
          : ["openai-proxy", "chat.completions"],
        input: body.messages,
        metadata: {
          model: body.model,
          stream: body.stream === true,
          requestId,
          turnId: turnId ?? null,
          traceId: traceId ?? null,
          correlation: {
            sessionSource: correlation.sources.sessionId ?? null,
            userSource: correlation.sources.userId ?? null,
            turnSource: correlation.sources.turnId ?? null,
            traceSource: correlation.sources.traceId ?? null
          },
          toolContext
        }
      });
      rootObservation.update({ input: { messages: body.messages } });

      const generation = startObservation(
        "chat_completion",
        {
          model: body.model,
          input: { messages: body.messages },
          modelParameters
        },
        { asType: "generation", startTime: new Date(startedAt) }
      );

      const completedToolExecutions = extractRecentCompletedToolExecutions(body.messages);
      if (completedToolExecutions.length > 0) {
        emitCompletedToolResultObservations(rootObservation, completedToolExecutions);
      }

      const eventStream = streamSimple(model as never, context as never, streamOptions as never);

      if (body.stream === true) {
        const streamOutcome = await writeSSEStream(
          response,
          eventStream,
          requestedModelId,
          requestId,
          body.stream_options?.include_usage === true,
          upstreamAbortController
        );
        // Prefer structured stream message to preserve tool-call payloads for tracing.
        const streamOutput = streamOutcome.streamMessage ?? streamOutcome.accumulatedText;
        const requestedToolCalls = extractToolCallsFromUnknown(streamOutcome.streamMessage);
        if (requestedToolCalls.length > 0) {
          emitRequestedToolCallObservations(generation, requestedToolCalls);
        }
        const streamFailureMessage = streamOutcome.clientDisconnected
          ? "Client disconnected before stream completion."
          : streamOutcome.iterationErrorMessage
            ? `Streaming interrupted: ${streamOutcome.iterationErrorMessage}`
            : null;
        generation
          .update({
            output: streamOutput,
            ...(streamOutcome.streamUsage
              ? {
                  usageDetails: {
                    input: streamOutcome.streamUsage.prompt_tokens,
                    output: streamOutcome.streamUsage.completion_tokens,
                    total: streamOutcome.streamUsage.total_tokens
                  }
                }
              : {}),
            ...(streamOutcome.streamFinishReason ? { metadata: { finishReason: streamOutcome.streamFinishReason } } : {}),
            ...(streamFailureMessage ? { level: "ERROR", statusMessage: streamFailureMessage } : {})
          })
          .end();

        if (streamFailureMessage) {
          rootObservation.update({
            output: streamOutput,
            level: "ERROR",
            statusMessage: streamFailureMessage
          });
        } else {
          rootObservation.update({ output: streamOutput });
        }
        rootObservation.updateTrace({ output: streamOutput });
        if (streamFailureMessage) {
          writeLog("warn", "chat.request.stream_incomplete", {
            request_id: requestId,
            model: requestedModelId,
            stream: true,
            status_code: response.statusCode,
            duration_ms: Date.now() - startedAt,
            emitted_output: streamOutcome.emittedAnyOutput,
            tool_calls_requested: requestedToolCalls.length,
            message: streamFailureMessage
          });
        } else {
          writeLog("info", "chat.request.complete", {
            request_id: requestId,
            model: requestedModelId,
            stream: true,
            status_code: response.statusCode,
            duration_ms: Date.now() - startedAt,
            emitted_output: streamOutcome.emittedAnyOutput,
            tool_calls_requested: requestedToolCalls.length
          });
        }
        return;
      }

      const assistantMessage = await getFinalAssistantMessage(eventStream);
      if (!assistantMessage) {
        generation.update({ level: "ERROR", statusMessage: "No final assistant message returned." }).end();
        throw new Error("No final assistant message returned from upstream stream.");
      }

      const stopReason = getAssistantStopReason(assistantMessage);
      if (stopReason === "error" || stopReason === "aborted") {
        const message =
          getAssistantErrorMessage(assistantMessage) || `Upstream request failed with stop reason '${stopReason}'.`;
        generation.update({ level: "ERROR", statusMessage: message }).end();
        throw new UpstreamModelError(message);
      }

      const responseBody = assistantMessageToResponse(assistantMessage, requestedModelId, generateCompletionId());
      sendJson(response, 200, responseBody);

      const responseMessage = responseBody.choices[0]?.message;
      const requestedToolCalls = extractToolCallsFromUnknown(responseMessage);
      if (requestedToolCalls.length > 0) {
        emitRequestedToolCallObservations(generation, requestedToolCalls);
      }
      generation
        .update({
          output: responseMessage,
          usageDetails: {
            input: responseBody.usage.prompt_tokens,
            output: responseBody.usage.completion_tokens,
            total: responseBody.usage.total_tokens
          },
          ...(responseBody.choices[0]?.finish_reason ? { metadata: { finishReason: responseBody.choices[0].finish_reason } } : {})
        })
        .end();

      rootObservation.update({ output: responseMessage });
      rootObservation.updateTrace({ output: responseMessage });

      writeLog("info", "chat.request.complete", {
        request_id: requestId,
        model: requestedModelId,
        stream: false,
        status_code: 200,
        duration_ms: Date.now() - startedAt,
        finish_reason: responseBody.choices[0]?.finish_reason ?? null,
        prompt_tokens: responseBody.usage.prompt_tokens,
        completion_tokens: responseBody.usage.completion_tokens,
        tool_calls_requested: requestedToolCalls.length
      });
    };

    if (!sessionId && !userId) {
      await executeCompletion();
      return;
    }

    await propagateAttributes(
      {
        ...(sessionId ? { sessionId } : {}),
        ...(userId ? { userId } : {})
      },
      async () => {
        await executeCompletion();
      }
    );
  };

  const startContext = parentSpanContext ? { parentSpanContext } : undefined;

  try {
    if (toolContext.isAgentic) {
      if (startContext) {
        await startActiveObservation("chat.turn.agent", runWithRootObservation, { ...startContext, asType: "agent" });
        return;
      }
      await startActiveObservation("chat.turn.agent", runWithRootObservation, { asType: "agent" });
      return;
    }

    if (startContext) {
      await startActiveObservation("chat.turn", runWithRootObservation, startContext);
      return;
    }
    await startActiveObservation("chat.turn", runWithRootObservation);
  } finally {
    request.off("aborted", abortUpstream);
    response.off("close", abortUpstream);
  }
}

export async function handleRequest(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const requestId = getRequestId(request);
  const startedAt = Date.now();
  response.setHeader("x-request-id", requestId);

  let method = request.method ?? "GET";
  let pathname = request.url ?? "/";
  writeLog("info", "request.start", {
    request_id: requestId,
    method,
    path: pathname
  });

  try {
    const requestUrl = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
    pathname = requestUrl.pathname;
    method = request.method ?? "GET";

    if (method === "OPTIONS") {
      response.statusCode = 204;
      setCorsHeaders(response);
      response.end();
      writeLog("info", "request.complete", {
        request_id: requestId,
        method,
        path: pathname,
        status_code: 204,
        duration_ms: Date.now() - startedAt
      });
      return;
    }

    if (method === "GET" && pathname === "/v1/models") {
      validateApiKeyHeader(request);
      await handleModels(response, requestId);
      writeLog("info", "request.complete", {
        request_id: requestId,
        method,
        path: pathname,
        status_code: response.statusCode,
        duration_ms: Date.now() - startedAt
      });
      return;
    }

    if (method === "POST" && pathname === "/v1/chat/completions") {
      await handleChatCompletions(request, response, requestId);
      writeLog("info", "request.complete", {
        request_id: requestId,
        method,
        path: pathname,
        status_code: response.statusCode,
        duration_ms: Date.now() - startedAt
      });
      return;
    }

    sendOpenAIError(response, 404, "Route not found.", "invalid_request_error", "not_found");
    writeLog("warn", "request.complete.not_found", {
      request_id: requestId,
      method,
      path: pathname,
      status_code: 404,
      duration_ms: Date.now() - startedAt
    });
  } catch (error) {
    if (response.headersSent) {
      writeLog("warn", "request.error.headers_sent", {
        request_id: requestId,
        method,
        path: pathname,
        message: getErrorMessage(error)
      });
      if (!response.writableEnded) {
        response.end();
      }
      return;
    }
    handleRequestError(response, error, requestId);
  }
}
