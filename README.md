# OpenAI-Compatible OAuth Proxy

This package exposes a minimal OpenAI-compatible HTTP API on top of Codex OAuth credentials.

## Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions` (streaming and non-streaming)

## Relay Behavior

- No default prompt injection.
- No model alias mapping or model fallback.
- The proxy uses the exact `model` provided by the client.
- Unknown model IDs return `400` with code `unknown_model`.
- The proxy only does protocol conversion + logging.

## Quick Start

```bash
# already in project root: /Users/hongfei/Projects/lensGate
pnpm install
pnpm run build
pnpm run login
pnpm run start
```

Default server address:

- `http://127.0.0.1:3000`
- OpenAI base URL to configure in clients: `http://127.0.0.1:3000/v1`

## Environment Variables

- `HOST` default `127.0.0.1`
- `PORT` default `3000`
- `BODY_LIMIT_BYTES` default `1048576` (1 MiB)
- `CORS_ALLOW_ORIGIN` default `*`
- `PROXY_API_KEY` optional extra proxy-layer bearer key check
- `LOG_LEVEL` optional log level: `debug` | `info` | `warn` | `error` (default `info`)
