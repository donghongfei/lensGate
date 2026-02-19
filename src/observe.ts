import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY } from "./config.js";

let _sdk: NodeSDK | null = null;

export function initLangfuse(): void {
  if (!LANGFUSE_PUBLIC_KEY || !LANGFUSE_SECRET_KEY) return;

  _sdk = new NodeSDK({
    spanProcessors: [
      new LangfuseSpanProcessor({
        publicKey: LANGFUSE_PUBLIC_KEY,
        secretKey: LANGFUSE_SECRET_KEY,
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
