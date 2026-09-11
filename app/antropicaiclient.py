import asyncio
from typing import AsyncGenerator, List, Optional

from anthropic import AsyncAnthropic, APIConnectionError, APIError, RateLimitError

from base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Role

MAX_ATTEMPTS = 3


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        super().__init__(config)
        self._client = AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _split_system(messages: List[ChatMessage]) -> tuple[Optional[str], list[dict]]:
        # La API de Anthropic no acepta role="system" dentro de la lista de
        # mensajes: va aparte, como parámetro `system`.
        system_parts = [m.content for m in messages if m.role == Role.system]
        chat_messages = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.system
        ]
        system = "\n".join(system_parts) if system_parts else None
        return system, chat_messages

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        system, chat_messages = self._split_system(messages)
        last_error = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                kwargs = dict(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    messages=chat_messages,
                    timeout=self.config.timeout,
                )
                if system:
                    kwargs["system"] = system

                response = await self._client.messages.create(**kwargs)
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                usage = response.usage
                return ModelResponse(
                    content=text,
                    provider="anthropic",
                    model=self.config.model,
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                    finish_reason=response.stop_reason,
                )
            except RateLimitError as e:
                last_error = f"Rate limit alcanzado (intento {attempt}/{MAX_ATTEMPTS}): {e}"
                await asyncio.sleep(2 ** attempt)
            except (APIConnectionError, APIError) as e:
                last_error = f"Error de API Anthropic (intento {attempt}/{MAX_ATTEMPTS}): {e}"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = f"Error inesperado: {e}"
                break

        return ModelResponse(
            content="", provider="anthropic", model=self.config.model, error=last_error
        )

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        system, chat_messages = self._split_system(messages)
        try:
            kwargs = dict(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=chat_messages,
            )
            if system:
                kwargs["system"] = system

            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"[ERROR STREAM ANTHROPIC]: {e}"