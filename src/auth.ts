import { AuthStorage } from "@mariozechner/pi-coding-agent";
import { AUTH_STORAGE_PATH } from "./config.js";

type AnyAuthStorage = {
  hasAuth?: (...args: unknown[]) => boolean;
  getApiKey?: (...args: unknown[]) => Promise<unknown> | unknown;
};

const authStorage = AuthStorage.create(AUTH_STORAGE_PATH) as unknown as AnyAuthStorage;

export class NotLoggedInError extends Error {
  constructor(message = "No OAuth credentials found. Run `pnpm run login` first.") {
    super(message);
    this.name = "NotLoggedInError";
  }
}

function tokenFromUnknown(value: unknown): string | null {
  if (typeof value === "string" && value.length > 0) {
    return value;
  }

  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Record<string, unknown>;
  const candidates = [record.accessToken, record.apiKey, record.token];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.length > 0) {
      return candidate;
    }
  }
  return null;
}

export function hasCredentials(): boolean {
  try {
    if (typeof authStorage.hasAuth !== "function") {
      return false;
    }

    return Boolean(authStorage.hasAuth("openai-codex"));
  } catch {
    return false;
  }
}

export async function getAccessToken(): Promise<string> {
  if (!hasCredentials()) {
    throw new NotLoggedInError();
  }

  if (typeof authStorage.getApiKey !== "function") {
    throw new Error("AuthStorage.getApiKey() is not available in this environment.");
  }

  const value = await authStorage.getApiKey("openai-codex");
  const accessToken = tokenFromUnknown(value);
  if (!accessToken) {
    throw new NotLoggedInError("OAuth credentials exist but no access token could be loaded.");
  }

  return accessToken;
}
