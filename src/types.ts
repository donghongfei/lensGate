export type OpenAIChatRole = "system" | "user" | "assistant" | "tool";

export interface OpenAITextContentPart {
  type: "text";
  text: string;
}

export interface OpenAIImageUrlContentPart {
  type: "image_url";
  image_url: {
    url: string;
  };
}

export type OpenAIContentPart = OpenAITextContentPart | OpenAIImageUrlContentPart;

export interface OpenAIToolCallFunction {
  name: string;
  arguments: string;
}

export interface OpenAIToolCall {
  id: string;
  type: "function";
  index?: number;
  function: OpenAIToolCallFunction;
}

export interface OpenAIMessage {
  role: OpenAIChatRole;
  content: string | OpenAIContentPart[] | null;
  name?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export type OpenAIToolChoice =
  | "none"
  | "auto"
  | "required"
  | {
      type: "function";
      function: {
        name: string;
      };
    };

export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIMessage[];
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  temperature?: number;
  max_tokens?: number;
  tools?: OpenAITool[];
  tool_choice?: OpenAIToolChoice;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  [key: string]: unknown;
}

export interface OpenAIChatResponseChoice {
  index: number;
  message: OpenAIMessage;
  logprobs: null;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
}

export interface OpenAICompletionTokensDetails {
  reasoning_tokens: number;
  audio_tokens: number;
  accepted_prediction_tokens: number;
  rejected_prediction_tokens: number;
}

export interface OpenAIPromptTokensDetails {
  cached_tokens: number;
  audio_tokens: number;
}

export interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  completion_tokens_details: OpenAICompletionTokensDetails;
  prompt_tokens_details: OpenAIPromptTokensDetails;
}

export interface OpenAIChatResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  system_fingerprint: string;
  choices: OpenAIChatResponseChoice[];
  usage: OpenAIUsage;
}

export interface OpenAIChatChunkChoice {
  index: number;
  delta: Partial<OpenAIMessage>;
  logprobs: null;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
}

export interface OpenAIChatChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  system_fingerprint: string;
  choices: OpenAIChatChunkChoice[];
  usage?: OpenAIUsage | null;
}

export interface OpenAIModelObject {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
  context_window?: number;
  max_tokens?: number;
}

export interface OpenAIModelsResponse {
  object: "list";
  data: OpenAIModelObject[];
}

export interface OpenAIErrorBody {
  message: string;
  type: string;
  param: string | null;
  code: string | null;
}

export interface OpenAIErrorResponse {
  error: OpenAIErrorBody;
}
