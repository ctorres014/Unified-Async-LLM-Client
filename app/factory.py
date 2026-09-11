from typing import Optional

from antropicaiclient import AnthropicClient
from base import BaseLLMClient
from openaiclient import OpenAIClient
from schemas import ModelConfig


def create_client(config: ModelConfig, api_key: Optional[str] = None) -> BaseLLMClient:
    """Punto único de instanciación: el resto del código programa contra
    BaseLLMClient y no le importa qué proveedor hay detrás."""
    if config.provider == "openai":
        return OpenAIClient(config, api_key=api_key)
    if config.provider == "anthropic":
        return AnthropicClient(config, api_key=api_key)
    raise ValueError(f"Proveedor no soportado: {config.provider}")