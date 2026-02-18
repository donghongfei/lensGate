import Langfuse, { LangfuseGenerationClient } from "langfuse";
import { LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY } from "./config.js";
import type { OpenAIMessage } from "./types.js";

let _langfuse: Langfuse | null = null;

export function getLangfuse(): Langfuse | null {
  if (!LANGFUSE_PUBLIC_KEY || !LANGFUSE_SECRET_KEY) {
    return null;
  }
  if (!_langfuse) {
    _langfuse = new Langfuse({
      publicKey: LANGFUSE_PUBLIC_KEY,
      secretKey: LANGFUSE_SECRET_KEY,
      ...(LANGFUSE_HOST ? { baseUrl: LANGFUSE_HOST } : {})
    });
  }
  return _langfuse;
}

export async function flushLangfuse(): Promise<void> {
  if (_langfuse) {
    await _langfuse.flushAsync();
  }
}

export interface GenerationParams {
  traceId: string;
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  startedAt: number;
}

export interface GenerationOutput {
  text?: string;
  message?: OpenAIMessage;
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  finishReason?: string;
}

export function startGeneration(params: GenerationParams): LangfuseGenerationClient | null {
  const lf = getLangfuse();
  if (!lf) return null;

  const modelParameters: Record<string, string | number | boolean | string[] | null> = {};
  if (params.temperature !== undefined) modelParameters.temperature = params.temperature;
  if (params.maxTokens !== undefined) modelParameters.maxTokens = params.maxTokens;
  if (params.topP !== undefined) modelParameters.topP = params.topP;

  const trace = lf.trace({ id: params.traceId, name: "chat" });
  return trace.generation({
    name: "chat_completion",
    model: params.model,
    input: params.messages,
    modelParameters,
    startTime: new Date(params.startedAt)
  });
}

export function endGeneration(generation: LangfuseGenerationClient | null, output: GenerationOutput): void {
  if (!generation) return;

  const hasUsage = output.promptTokens !== undefined && output.completionTokens !== undefined;

  generation.end({
    output: output.message ?? output.text ?? "",
    ...(hasUsage
      ? {
          usage: {
            promptTokens: output.promptTokens,
            completionTokens: output.completionTokens,
            totalTokens: output.totalTokens ?? (output.promptTokens ?? 0) + (output.completionTokens ?? 0)
          }
        }
      : {}),
    ...(output.finishReason ? { metadata: { finishReason: output.finishReason } } : {})
  });
}
