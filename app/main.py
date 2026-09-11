import asyncio
import os

from dotenv import load_dotenv

from factory import create_client
from schemas import ChatMessage, ModelConfig, Role

load_dotenv()

QUESTION = "¿Qué es la entropía?"


def build_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=Role.system, content="Responde en español, de forma breve."),
        ChatMessage(role=Role.user, content=QUESTION),
    ]


async def run_normal(provider: str, model: str, api_key: str) -> None:
    config = ModelConfig(provider=provider, model=model)
    client = create_client(config, api_key=api_key)
    response = await client.generate(build_messages())

    print(f"\n--- {provider.upper()} (modo normal) ---")
    if response.ok:
        print(response.content)
        print(f"[tokens in/out: {response.input_tokens}/{response.output_tokens}]")
    else:
        print(f"Error controlado: {response.error}")


async def run_streaming(provider: str, model: str, api_key: str) -> None:
    config = ModelConfig(provider=provider, model=model)
    client = create_client(config, api_key=api_key)

    print(f"\n--- {provider.upper()} (modo streaming) ---")
    async for token in client.stream(build_messages()):
        print(token, end="", flush=True)
    print()


async def main() -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        await run_normal("openai", "gpt-4o-mini", openai_key)
        await run_streaming("openai", "gpt-4o-mini", openai_key)
    else:
        print("OPENAI_API_KEY no configurada, se omite prueba de OpenAI.")

    if anthropic_key:
        await run_normal("anthropic", "claude-haiku-4-5-20251001", anthropic_key)
        await run_streaming("anthropic", "claude-haiku-4-5-20251001", anthropic_key)
    else:
        print("ANTHROPIC_API_KEY no configurada, se omite prueba de Anthropic.")


if __name__ == "__main__":
    asyncio.run(main())