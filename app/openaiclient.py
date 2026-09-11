import asyncio
from typing import AsyncGenerator, List, Optional

from openai import AsyncOpenAI, APIConnectionError, APIError, RateLimitError

from base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse

MAX_ATTEMPTS = 3


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        super().__init__(config)
        # AsyncOpenAI = versión no bloqueante del SDK. Usar la síncrona
        # dentro de una función async congelaría el event loop.
        self._client = AsyncOpenAI(api_key=api_key)

    def _to_payload(self, messages: List[ChatMessage]) -> list[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        payload = self._to_payload(messages)
        last_error = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=payload,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                choice = response.choices[0]
                usage = response.usage
                return ModelResponse(
                    content=choice.message.content or "",
                    provider="openai",
                    model=self.config.model,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    finish_reason=choice.finish_reason,
                )
            except RateLimitError as e:
                last_error = f"Rate limit alcanzado (intento {attempt}/{MAX_ATTEMPTS}): {e}"
                await asyncio.sleep(2 ** attempt)
            except (APIConnectionError, APIError) as e:
                last_error = f"Error de API OpenAI (intento {attempt}/{MAX_ATTEMPTS}): {e}"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                # Error no recuperable (p.ej. API key inválida): no reintentar.
                last_error = f"Error inesperado: {e}"
                break

        return ModelResponse(
            content="", provider="openai", model=self.config.model, error=last_error
        )

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        payload = self._to_payload(messages)
        try:
            stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=payload,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"[ERROR STREAM OPENAI]: {e}"