import re
import time
import json
import hashlib
import os
from openai import OpenAI
from config import PROVIDERS, MODEL_ROLES, MAX_RETRIES, RETRY_DELAY_SEC, MAX_TOKENS, TEMPERATURE, CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)


class LLMClient:
    """
    Unified LLM client with:
    - Two-model strategy (reason vs code roles)
    - Automatic fallback to OpenRouter on Groq failure
    - Response caching to save API quota during dev
    - Thinking tag stripping for reasoning models
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._clients  = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            cfg = PROVIDERS[provider]
            self._clients[provider] = OpenAI(
                api_key=cfg["api_key"] or "none",
                base_url=cfg["base_url"],
            )
        return self._clients[provider]

    def _strip_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks emitted by reasoning models."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def _cache_key(self, messages: list, provider: str, model: str) -> str:
        payload = json.dumps(
            {"provider": provider, "model": model, "messages": messages},
            sort_keys=True
        )
        return hashlib.md5(payload.encode()).hexdigest()

    def _read_cache(self, key: str):
        path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _write_cache(self, key: str, response: str):
        path = os.path.join(CACHE_DIR, f"{key}.json")
        with open(path, "w") as f:
            json.dump({"response": response}, f)

    def _call_provider(self, provider: str, model: str, messages: list) -> str:
        """
        Single attempt to call one provider/model combo.
        Raises on any failure so the caller can retry or fallback.
        """
        client = self._get_client(provider)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        result = resp.choices[0].message.content.strip()
        return self._strip_thinking(result)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list,
        role: str = "code",
        system: str | None = None,
    ) -> str:
        """
        Send messages to the LLM.

        role="code"   → Groq Llama 70B  (fast, straightforward codegen)
        role="reason" → Groq QwQ-32B    (slower, better decisions)
        role="fallback" → OpenRouter    (when you explicitly want fallback)

        On any Groq failure, automatically retries then falls to OpenRouter.
        Returns the assistant reply as a plain string, thinking tags stripped.
        """
        if system:
            messages = [{"role": "system", "content": system}] + messages

        # Build attempt sequence: primary role first, then fallback
        primary  = MODEL_ROLES[role]
        fallback = MODEL_ROLES["fallback"]

        attempt_sequence = [
            (primary["provider"],  primary["model"]),
            (fallback["provider"], fallback["model"]),
        ]

        last_error = None

        for provider, model in attempt_sequence:
            # Cache check
            if self.use_cache:
                key    = self._cache_key(messages, provider, model)
                cached = self._read_cache(key)
                if cached:
                    print(f"[LLMClient] Cache hit — {provider}/{model}")
                    return cached["response"]

            for attempt in range(MAX_RETRIES):
                try:
                    print(f"[LLMClient] {provider}/{model} — attempt {attempt + 1}")
                    result = self._call_provider(provider, model, messages)

                    if self.use_cache:
                        self._write_cache(key, result)

                    return result

                except Exception as e:
                    last_error = str(e)
                    print(f"[LLMClient] Failed: {last_error}")

                    is_rate_limit = "429" in last_error or "rate" in last_error.lower()

                    if is_rate_limit:
                        wait = RETRY_DELAY_SEC * (attempt + 1) * 2
                        print(f"[LLMClient] Rate limited — waiting {wait}s")
                        time.sleep(wait)
                    elif attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY_SEC)
                    else:
                        print(f"[LLMClient] {provider} exhausted — trying fallback")
                        break

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}\n"
            "Check GROQ_API_KEY and OPENROUTER_API_KEY in your .env"
        )

    def chat_json(
        self,
        messages: list,
        role: str = "code",
        system: str | None = None,
    ) -> dict:
        """
        Same as chat() but parses the response as JSON.
        Strips markdown fences automatically.
        """
        raw = self.chat(messages, role=role, system=system)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            parts   = cleaned.split("```")
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            print(f"[LLMClient] JSON parse error: {e}\nRaw output:\n{raw}")
            
          
    def extract_code(self, text: str) -> str:
      """Extract code from markdown fences. Returns plain code string."""
      # Match ```python ... ``` or ``` ... ```
      match = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
      if match:
          return match.group(1).strip()
      # No fences — assume the whole response is code
      return text.strip()