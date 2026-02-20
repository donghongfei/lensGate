export function parseToolArgumentsInput(argumentsText: string): unknown {
  const input = argumentsText.trim();
  if (!input) {
    return {};
  }
  try {
    return JSON.parse(input);
  } catch {
    return {
      raw: argumentsText
    };
  }
}
