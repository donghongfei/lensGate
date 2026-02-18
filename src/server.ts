#!/usr/bin/env node
import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { HOST, PORT } from "./config.js";
import { handleRequest } from "./handler.js";

const port = PORT;
const host = HOST;

const server = createServer((request: IncomingMessage, response: ServerResponse) => {
  void handleRequest(request, response);
});

server.listen(port, host, () => {
  process.stdout.write(`OpenAI proxy listening on http://${host}:${port}\n`);
});

function shutdown(signal: string): void {
  process.stdout.write(`\nReceived ${signal}, shutting down...\n`);
  server.close(() => {
    process.exit(0);
  });
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
