#!/usr/bin/env node
import { exec } from "node:child_process";
import { stdin as input, stdout as output } from "node:process";
import { createInterface } from "node:readline";
import { AuthStorage } from "@mariozechner/pi-coding-agent";

type AnyAuthStorage = {
  hasAuth?: (...args: unknown[]) => boolean;
  login?: (...args: unknown[]) => Promise<unknown>;
};

function ask(rl: ReturnType<typeof createInterface>, question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => resolve(answer));
  });
}

function maybeExtractUrl(payload: unknown): string | null {
  if (typeof payload === "string" && payload.startsWith("http")) {
    return payload;
  }
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const record = payload as Record<string, unknown>;
  const candidates = [record.url, record.authUrl, record.verificationUri, record.verification_url];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.startsWith("http")) {
      return candidate;
    }
  }

  return null;
}

function tryOpenBrowser(url: string): void {
  const platform = process.platform;
  const command =
    platform === "darwin" ? `open "${url}"` : platform === "win32" ? `start "" "${url}"` : `xdg-open "${url}"`;
  exec(command, () => {
    // Best effort only; login still works if browser auto-open fails.
  });
}

function stringifyProgress(event: unknown): string {
  if (typeof event === "string") {
    return event;
  }
  if (!event || typeof event !== "object") {
    return "";
  }

  const record = event as Record<string, unknown>;
  const candidates = [record.message, record.status, record.type];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.length > 0) {
      return candidate;
    }
  }
  return "";
}

async function main(): Promise<void> {
  const authStorage = AuthStorage.create() as unknown as AnyAuthStorage;
  const rl = createInterface({ input, output });

  try {
    const alreadyLoggedIn = Boolean(
      (typeof authStorage.hasAuth === "function" && authStorage.hasAuth("openai-codex")) ||
        (typeof authStorage.hasAuth === "function" && authStorage.hasAuth())
    );

    if (alreadyLoggedIn) {
      const answer = (await ask(rl, "Existing credentials found. Re-login? [y/N] ")).trim().toLowerCase();
      if (answer !== "y" && answer !== "yes") {
        process.stdout.write("Keeping existing credentials.\n");
        return;
      }
    }

    if (typeof authStorage.login !== "function") {
      throw new Error("AuthStorage.login() is not available.");
    }

    process.stdout.write("Starting OpenAI Codex OAuth login...\n");

    await authStorage.login("openai-codex", {
      onAuth: (payload: unknown) => {
        const url = maybeExtractUrl(payload);
        if (!url) {
          process.stdout.write("Authorization started.\n");
          return;
        }
        process.stdout.write(`Open this URL in your browser:\n${url}\n`);
        tryOpenBrowser(url);
      },
      onPrompt: async (payload: unknown) => {
        let promptText = "Paste authorization code:";
        if (typeof payload === "string" && payload.length > 0) {
          promptText = payload;
        } else if (payload && typeof payload === "object") {
          const record = payload as Record<string, unknown>;
          if (typeof record.prompt === "string" && record.prompt.length > 0) {
            promptText = record.prompt;
          }
        }
        const answer = await ask(rl, `${promptText} `);

        if (payload && typeof payload === "object") {
          const record = payload as Record<string, unknown>;
          const callbacks = [record.respond, record.resolve, record.callback];
          for (const callback of callbacks) {
            if (typeof callback === "function") {
              (callback as (value: string) => void)(answer);
              break;
            }
          }
        }

        return answer;
      },
      onProgress: (event: unknown) => {
        const message = stringifyProgress(event);
        if (message.length > 0) {
          process.stdout.write(`${message}\n`);
        }
      }
    });

    process.stdout.write("Login successful. You can now run: pnpm run start\n");
  } finally {
    rl.close();
  }
}

void main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Login failed: ${message}\n`);
  process.exitCode = 1;
});
