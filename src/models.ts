import { getModel, getModels } from "@mariozechner/pi-ai";

export const DEFAULT_MODEL_ID = "gpt-5.1";

const FALLBACK_MODEL_IDS = [
  "gpt-5.1",
  "gpt-5.1-codex-max",
  "gpt-5.1-codex-mini",
  "gpt-5.2",
  "gpt-5.2-codex",
  "gpt-5.3-codex",
  "gpt-5.3-codex-spark"
];

const MODEL_ALIAS_MAP: Record<string, string> = {
  "gpt-4o": "gpt-5.1",
  "gpt-4o-mini": "gpt-5.1-codex-mini",
  "gpt-4": "gpt-5.1",
  "gpt-4-turbo": "gpt-5.1",
  "gpt-3.5-turbo": "gpt-5.1-codex-mini",
  o1: "gpt-5.1",
  o3: "gpt-5.2",
  "o4-mini": "gpt-5.1-codex-mini"
};

function normalizeModelName(input: string | undefined): string {
  return (input ?? "").trim().toLowerCase();
}

function getModelId(model: unknown): string | null {
  if (!model || typeof model !== "object") {
    return null;
  }
  const id = (model as { id?: unknown }).id;
  return typeof id === "string" && id.length > 0 ? id : null;
}

function getModelProvider(model: unknown): string | null {
  if (!model || typeof model !== "object") {
    return null;
  }
  const provider = (model as { provider?: unknown }).provider;
  return typeof provider === "string" && provider.length > 0 ? provider : null;
}

function normalizeModelList(modelsValue: unknown): unknown[] {
  if (Array.isArray(modelsValue)) {
    return modelsValue;
  }
  if (modelsValue && typeof modelsValue === "object") {
    return Object.values(modelsValue as Record<string, unknown>);
  }
  return [];
}

function isCodexModel(model: unknown): boolean {
  const id = getModelId(model);
  if (!id) {
    return false;
  }
  const provider = getModelProvider(model);
  if (provider && provider.includes("openai-codex")) {
    return true;
  }
  return id.includes("codex") || FALLBACK_MODEL_IDS.includes(id);
}

function toModelOrFallback(id: string): unknown {
  try {
    return getModel(id);
  } catch {
    return {
      id,
      provider: "openai-codex-responses"
    };
  }
}

export function getAllCodexModels(): unknown[] {
  const models = normalizeModelList(getModels());
  const codexModels = models.filter((model) => isCodexModel(model));
  if (codexModels.length > 0) {
    return codexModels;
  }
  return FALLBACK_MODEL_IDS.map((id) => toModelOrFallback(id));
}

function getAvailableModelIds(): Set<string> {
  const ids = new Set<string>();
  for (const model of getAllCodexModels()) {
    const id = getModelId(model);
    if (id) {
      ids.add(id.toLowerCase());
    }
  }
  return ids;
}

function findActualModelId(normalizedId: string): string | null {
  for (const model of getAllCodexModels()) {
    const id = getModelId(model);
    if (id && id.toLowerCase() === normalizedId) {
      return id;
    }
  }
  return null;
}

export function resolveRequestedModelId(requested: string | undefined): string {
  const normalizedRequested = normalizeModelName(requested);
  const aliasResolved = MODEL_ALIAS_MAP[normalizedRequested] ?? normalizedRequested;

  if (!aliasResolved) {
    return DEFAULT_MODEL_ID;
  }

  const availableModelIds = getAvailableModelIds();
  if (availableModelIds.has(aliasResolved)) {
    return findActualModelId(aliasResolved) ?? aliasResolved;
  }

  return DEFAULT_MODEL_ID;
}

export function resolveCodexModel(requested: string | undefined): unknown {
  const resolvedModelId = resolveRequestedModelId(requested);
  try {
    return getModel(resolvedModelId);
  } catch {
    const fallback = getAllCodexModels().find((model) => {
      const id = getModelId(model);
      return id?.toLowerCase() === resolvedModelId.toLowerCase();
    });
    if (fallback) {
      return fallback;
    }
    return getModel(DEFAULT_MODEL_ID);
  }
}

export function modelIdFromUnknown(model: unknown): string {
  return getModelId(model) ?? DEFAULT_MODEL_ID;
}
