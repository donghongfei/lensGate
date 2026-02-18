import type { IncomingMessage, ServerResponse } from "node:http";
import { streamSimple } from "@mariozechner/pi-ai";
import { getAccessToken, NotLoggedInError } from "./auth.js";
import {
  assistantMessageToResponse,
  createStreamState,
  eventToSSEChunks,
  generateCompletionId,
  ImageNotSupportedError,
  openAIRequestToContext
} from "./convert.js";
import { getAllCodexModels, modelIdFromUnknown, resolveCodexModel, resolveRequestedModelId } from "./models.js";
import type { OpenAIChatRequest, OpenAIErrorResponse, OpenAIModelObject, OpenAIModelsResponse } from "./types.js";

const BODY_LIMIT_BYTES = Number(process.env.BODY_LIMIT_BYTES ?? 1024 * 1024);
const CORS_ALLOW_ORIGIN = process.env.CORS_ALLOW_ORIGIN ?? "*";

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

async function writeSSEStream(response: ServerResponse, streamLike: unknown, modelId: string): Promise<void> {
  response.statusCode = 200;
  setCorsHeaders(response);
  response.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  response.setHeader("Cache-Control", "no-cache, no-transform");
  response.setHeader("Connection", "keep-alive");
  response.setHeader("X-Accel-Buffering", "no");
  response.flushHeaders?.();

  const state = createStreamState(modelId);
  let closed = false;

  try {
    for await (const event of asAsyncIterable(streamLike)) {
      const chunks = eventToSSEChunks(event, state);
      for (const chunk of chunks) {
        response.write(chunk);
      }
    }
  } catch {
    // Streaming response may already be partially written; finish with [DONE].
  } finally {
    if (!closed && !response.writableEnded) {
      response.write("data: [DONE]\n\n");
      response.end();
      closed = true;
    }
  }
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

function handleRequestError(response: ServerResponse, error: unknown): void {
  if (error instanceof NotLoggedInError) {
    sendOpenAIError(response, 401, error.message, "authentication_error", "not_logged_in");
    return;
  }

  if (error instanceof ImageNotSupportedError) {
    sendOpenAIError(response, 400, error.message, "invalid_request_error", "image_not_supported");
    return;
  }

  if (error instanceof InvalidRequestError) {
    sendOpenAIError(response, 400, error.message, "invalid_request_error", error.code);
    return;
  }

  if (error instanceof InvalidJSONError) {
    sendOpenAIError(response, 400, error.message, "invalid_request_error", "invalid_json");
    return;
  }

  if (error instanceof PayloadTooLargeError) {
    sendOpenAIError(response, 413, error.message, "invalid_request_error", "payload_too_large");
    return;
  }

  if (isRateLimitError(error)) {
    sendOpenAIError(response, 429, "Rate limit exceeded.", "rate_limit_error", "rate_limit");
    return;
  }

  const message = error instanceof Error ? error.message : "Unknown proxy error.";
  sendOpenAIError(response, 500, message, "server_error", "proxy_internal_error");
}

function validateApiKeyHeader(request: IncomingMessage): void {
  const expectedApiKey = process.env.PROXY_API_KEY;
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

async function handleModels(response: ServerResponse): Promise<void> {
  const models: OpenAIModelsResponse = {
    object: "list",
    data: getAllCodexModels().map((model) => toOpenAIModelObject(model))
  };
  sendJson(response, 200, models);
}

async function handleChatCompletions(request: IncomingMessage, response: ServerResponse): Promise<void> {
  validateApiKeyHeader(request);

  const body = await parseChatRequest(request);
  const requestedModelId = resolveRequestedModelId(body.model);
  const model = resolveCodexModel(body.model);
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

  const eventStream = streamSimple(model as never, context as never, streamOptions as never);

  if (body.stream === true) {
    await writeSSEStream(response, eventStream, requestedModelId);
    return;
  }

  const assistantMessage = await getFinalAssistantMessage(eventStream);
  if (!assistantMessage) {
    throw new Error("No final assistant message returned from upstream stream.");
  }

  const responseBody = assistantMessageToResponse(assistantMessage, requestedModelId, generateCompletionId());
  sendJson(response, 200, responseBody);
}

export async function handleRequest(request: IncomingMessage, response: ServerResponse): Promise<void> {
  try {
    const requestUrl = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
    const pathname = requestUrl.pathname;
    const method = request.method ?? "GET";

    if (method === "OPTIONS") {
      response.statusCode = 204;
      setCorsHeaders(response);
      response.end();
      return;
    }

    if (method === "GET" && pathname === "/v1/models") {
      await handleModels(response);
      return;
    }

    if (method === "POST" && pathname === "/v1/chat/completions") {
      await handleChatCompletions(request, response);
      return;
    }

    sendOpenAIError(response, 404, "Route not found.", "invalid_request_error", "not_found");
  } catch (error) {
    if (response.headersSent) {
      if (!response.writableEnded) {
        response.end();
      }
      return;
    }
    handleRequestError(response, error);
  }
}
