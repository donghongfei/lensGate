import type { IncomingMessage, ServerResponse } from "node:http";
import { randomBytes } from "node:crypto";
import { streamSimple } from "@mariozechner/pi-ai";
import { getAccessToken, NotLoggedInError } from "./auth.js";
import { BODY_LIMIT_BYTES, CORS_ALLOW_ORIGIN, LOG_LEVEL, PROXY_API_KEY } from "./config.js";
import { endGeneration, startGeneration } from "./observe.js";
import {
  assistantMessageToResponse,
  createStreamState,
  eventToSSEChunks,
  generateCompletionId,
  ImageNotSupportedError,
  openAIRequestToContext
} from "./convert.js";
import { getAllCodexModels, modelIdFromUnknown, resolveCodexModel, UnknownModelError } from "./models.js";
import type { OpenAIChatRequest, OpenAIErrorResponse, OpenAIModelObject, OpenAIModelsResponse } from "./types.js";

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

function getRequestId(request: IncomingMessage): string {
  const headerValue = request.headers["x-request-id"];
  const candidate = Array.isArray(headerValue) ? headerValue[0] : headerValue;
  if (typeof candidate === "string") {
    const trimmed = candidate.trim();
    if (trimmed.length > 0 && trimmed.length <= 128) {
      return trimmed;
    }
  }
  return `req_${randomBytes(8).toString("hex")}`;
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
  requestId: string
): Promise<{ emittedAnyOutput: boolean; accumulatedText: string; streamFinishReason: string | null }> {
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

  response.on("close", () => {
    clientDisconnected = true;
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
    writeLog("warn", "chat.stream.iteration_error", {
      request_id: requestId,
      model: modelId,
      message: getErrorMessage(error)
    });
  } finally {
    if (!closed && !response.writableEnded) {
      response.write("data: [DONE]\n\n");
      response.end();
      closed = true;
    }
  }

  return { emittedAnyOutput: state.emittedAnyOutput, accumulatedText: state.accumulatedText, streamFinishReason: state.streamFinishReason };
}

function toOpenAIModelObject(model: unknown): OpenAIModelObject {
  const id = modelIdFromUnknown(model);
  return {
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "openai"
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
  const requestedModelId = body.model;
  const model = resolveCodexModel(requestedModelId);
  const context = openAIRequestToContext(body);
  const accessToken = await getAccessToken();

  const streamOptions: Record<string, unknown> = {
    apiKey: accessToken
  };
  if (typeof body.temperature === "number") {
    streamOptions.temperature = body.temperature;
  }
  if (typeof body.max_tokens === "number") {
    streamOptions.maxTokens = body.max_tokens;
  }
  if (typeof body.top_p === "number") {
    streamOptions.topP = body.top_p;
  }
  if (typeof body.frequency_penalty === "number") {
    streamOptions.frequencyPenalty = body.frequency_penalty;
  }
  if (typeof body.presence_penalty === "number") {
    streamOptions.presencePenalty = body.presence_penalty;
  }

  writeLog("info", "chat.request.start", {
    request_id: requestId,
    requested_model: body.model,
    stream: body.stream === true
  });

  const generation = startGeneration({
    traceId: requestId,
    model: body.model,
    messages: body.messages,
    temperature: body.temperature,
    maxTokens: body.max_tokens,
    topP: body.top_p,
    startedAt
  });

  const eventStream = streamSimple(model as never, context as never, streamOptions as never);

  if (body.stream === true) {
    const streamOutcome = await writeSSEStream(response, eventStream, requestedModelId, requestId);
    endGeneration(generation, {
      text: streamOutcome.accumulatedText,
      finishReason: streamOutcome.streamFinishReason ?? undefined
    });
    writeLog("info", "chat.request.complete", {
      request_id: requestId,
      model: requestedModelId,
      stream: true,
      status_code: response.statusCode,
      duration_ms: Date.now() - startedAt,
      emitted_output: streamOutcome.emittedAnyOutput
    });
    return;
  }

  const assistantMessage = await getFinalAssistantMessage(eventStream);
  if (!assistantMessage) {
    throw new Error("No final assistant message returned from upstream stream.");
  }

  const stopReason = getAssistantStopReason(assistantMessage);
  if (stopReason === "error" || stopReason === "aborted") {
    const message =
      getAssistantErrorMessage(assistantMessage) || `Upstream request failed with stop reason '${stopReason}'.`;
    throw new UpstreamModelError(message);
  }

  const responseBody = assistantMessageToResponse(assistantMessage, requestedModelId, generateCompletionId());
  sendJson(response, 200, responseBody);
  endGeneration(generation, {
    message: responseBody.choices[0]?.message,
    promptTokens: responseBody.usage.prompt_tokens,
    completionTokens: responseBody.usage.completion_tokens,
    totalTokens: responseBody.usage.total_tokens,
    finishReason: responseBody.choices[0]?.finish_reason ?? undefined
  });
  writeLog("info", "chat.request.complete", {
    request_id: requestId,
    model: requestedModelId,
    stream: false,
    status_code: 200,
    duration_ms: Date.now() - startedAt,
    finish_reason: responseBody.choices[0]?.finish_reason ?? null,
    prompt_tokens: responseBody.usage.prompt_tokens,
    completion_tokens: responseBody.usage.completion_tokens
  });
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
