"""
Estructuras de datos validadas con Pydantic.
Definir esto primero evita pasar dicts sueltos entre capas (el clásico
"error de diccionarios anidados" cuando vienes de C# y esperabas algo
parecido a un DTO/record con validación).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """Equivalente a un DTO inmutable: role + content, validado."""

    role: Role
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Configuración de la llamada al modelo (parámetros de inferencia)."""

    provider: Literal["openai", "anthropic"]
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout: float = Field(default=30.0, gt=0)


class ModelResponse(BaseModel):
    """Respuesta normalizada, igual sin importar el proveedor."""

    content: str
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None