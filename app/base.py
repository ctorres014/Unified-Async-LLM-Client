"""
Clase base abstracta (equivalente a una interfaz/clase abstracta en C#,
tipo `interface ILLMClient` o `abstract class LLMClientBase`).
Cada proveedor concreto implementa estos dos métodos.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Llamada no bloqueante, respuesta completa de una sola vez."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Generador asíncrono: yield de cada token/fragmento a medida que llega."""
        raise NotImplementedError