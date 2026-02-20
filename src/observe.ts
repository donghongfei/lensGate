import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY } from "./config.js";

let _sdk: NodeSDK | null = null;

function maskLangfusePayload(data: string): string {
  return data
    .replace(/\bsk-lf-[A-Za-z0-9_-]+\b/g, "[REDACTED_LANGFUSE_SECRET]")
    .replace(/\bpk-lf-[A-Za-z0-9_-]+\b/g, "[REDACTED_LANGFUSE_PUBLIC]")
    .replace(/\b(?:Bearer|bearer)\s+[A-Za-z0-9._-]+\b/g, "Bearer [REDACTED_TOKEN]")
    .replace(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g, "[REDACTED_EMAIL]");
}

export function initLangfuse(): void {
  if (!LANGFUSE_PUBLIC_KEY || !LANGFUSE_SECRET_KEY) return;

  _sdk = new NodeSDK({
    spanProcessors: [
      new LangfuseSpanProcessor({
        publicKey: LANGFUSE_PUBLIC_KEY,
        secretKey: LANGFUSE_SECRET_KEY,
        mask: ({ data }) => maskLangfusePayload(data),
        ...(LANGFUSE_BASE_URL ? { baseUrl: LANGFUSE_BASE_URL } : {})
      })
    ]
  });
  _sdk.start();
}

export async function shutdownLangfuse(): Promise<void> {
  if (_sdk) {
    await _sdk.shutdown();
    _sdk = null;
  }
}
