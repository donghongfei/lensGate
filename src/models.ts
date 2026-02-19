import { getModel, getModels } from "@mariozechner/pi-ai";

const CODEX_PROVIDER = "openai-codex" as const;

export class UnknownModelError extends Error {
  modelId: string;

  constructor(modelId: string) {
    super(
      modelId
        ? `The model '${modelId}' does not exist or you do not have access to it.`
        : "The requested model does not exist or you do not have access to it."
    );
    this.name = "UnknownModelError";
    this.modelId = modelId;
  }
}

function getModelId(model: unknown): string | null {
  if (!model || typeof model !== "object") {
    return null;
  }
  const id = (model as { id?: unknown }).id;
  return typeof id === "string" && id.length > 0 ? id : null;
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

export function getAllCodexModels(): unknown[] {
  return normalizeModelList(getModels(CODEX_PROVIDER));
}

export function resolveCodexModel(requested: string | undefined): unknown {
  const requestedModelId = (requested ?? "").trim();
  if (requestedModelId.length === 0) {
    throw new UnknownModelError("");
  }

  const resolvedModel = getModel(CODEX_PROVIDER, requestedModelId as never);
  if (!resolvedModel) {
    throw new UnknownModelError(requestedModelId);
  }
  return resolvedModel;
}

export function modelIdFromUnknown(model: unknown): string {
  const id = getModelId(model);
  if (!id) {
    throw new Error("Model object does not contain a valid 'id' field.");
  }
  return id;
}
