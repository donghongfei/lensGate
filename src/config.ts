import { homedir } from "node:os";
import { join } from "node:path";

/** Token 文件路径，默认 ~/.lensgate/agent/auth.json */
export const AUTH_STORAGE_PATH =
  process.env.AUTH_STORAGE_PATH ?? join(homedir(), ".lensgate", "agent", "auth.json");

/** HTTP 监听端口，默认 3000 */
export const PORT = Number(process.env.PORT ?? 3000);

/** HTTP 绑定地址，默认 127.0.0.1 */
export const HOST = process.env.HOST ?? "127.0.0.1";

/** 请求 Body 大小上限（字节），默认 4MB */
export const BODY_LIMIT_BYTES = Number(process.env.BODY_LIMIT_BYTES ?? 4 * 1024 * 1024);

/** CORS 允许的来源，默认 * */
export const CORS_ALLOW_ORIGIN = process.env.CORS_ALLOW_ORIGIN ?? "*";

/** 日志级别，默认 info */
export const LOG_LEVEL = process.env.LOG_LEVEL ?? "info";

/**
 * 可选：限制访问代理的 API Key。
 * 设置后客户端须在 Authorization: Bearer <key> 中提供此值。
 * 不设置则不校验，任何请求都放行。
 */
export const PROXY_API_KEY = process.env.PROXY_API_KEY ?? "";

/** Langfuse 公钥（不设置则不上报） */
export const LANGFUSE_PUBLIC_KEY = process.env.LANGFUSE_PUBLIC_KEY ?? "";

/** Langfuse 密钥 */
export const LANGFUSE_SECRET_KEY = process.env.LANGFUSE_SECRET_KEY ?? "";

/**
 * Langfuse 服务地址（自建实例时修改）
 * 默认 https://cloud.langfuse.com
 */
export const LANGFUSE_HOST = process.env.LANGFUSE_HOST ?? "";
