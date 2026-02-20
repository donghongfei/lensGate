export interface ExtractedToolCall {
  id: string;
  name: string;
  argumentsText: string;
}

interface ToolCallExtractionOptions {
  normalizeId?: (value: unknown) => string | null;
  normalizeName?: (value: unknown) => string | null;
  stringifyArguments?: (value: unknown) => string;
  includeFunctionId?: boolean;
  defaultName?: string;
  fallbackId?: (index: number) => string;
  fallbackArgumentsValue?: unknown;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function defaultStringifyArguments(value: unknown): string {
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

function collectToolCallCandidates(message: Record<string, unknown>): Record<string, unknown>[] {
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
  return source.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object");
}

export function extractNormalizedToolCalls(
  message: unknown,
  options: ToolCallExtractionOptions = {}
): ExtractedToolCall[] {
  if (!message || typeof message !== "object") {
    return [];
  }

  const record = message as Record<string, unknown>;
  const candidates = collectToolCallCandidates(record);
  const normalizeId = options.normalizeId ?? nonEmptyString;
  const normalizeName = options.normalizeName ?? nonEmptyString;
  const stringifyArguments = options.stringifyArguments ?? defaultStringifyArguments;
  const defaultName = options.defaultName ?? "tool";
  const fallbackId = options.fallbackId ?? ((index: number) => `call_${index}`);
  const includeFunctionId = options.includeFunctionId === true;
  const fallbackArgumentsValue = options.fallbackArgumentsValue ?? "";

  const calls: ExtractedToolCall[] = [];
  for (let i = 0; i < candidates.length; i += 1) {
    const itemRecord = candidates[i];
    const functionRecord =
      itemRecord.function && typeof itemRecord.function === "object"
        ? (itemRecord.function as Record<string, unknown>)
        : undefined;
    const id =
      normalizeId(itemRecord.id) ??
      normalizeId(itemRecord.toolCallId) ??
      (includeFunctionId ? normalizeId(functionRecord?.id) : null) ??
      fallbackId(i);
    const name =
      normalizeName(itemRecord.name) ??
      normalizeName(itemRecord.toolName) ??
      normalizeName(functionRecord?.name) ??
      defaultName;
    const argumentsText = stringifyArguments(
      itemRecord.arguments ?? functionRecord?.arguments ?? itemRecord.input ?? fallbackArgumentsValue
    );
    calls.push({
      id,
      name,
      argumentsText
    });
  }
  return calls;
}
