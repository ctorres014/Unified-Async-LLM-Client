# Unified Async LLM Client

Cliente asíncrono e intercambiable para OpenAI y Anthropic.

## Estructura

```
schemas.py           # ChatMessage, ModelConfig, ModelResponse (Pydantic)
base.py              # BaseLLMClient (clase abstracta, como una interfaz en C#)
openaiclient.py      # OpenAIClient(BaseLLMClient)
antropicaiclient.py  # AnthropicClient(BaseLLMClient)
factory.py           # create_client(config) -> instancia el proveedor correcto
main.py              # script de validación (modo normal + streaming)
```

## Instalación

**macOS / Linux:**
 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # completar con tus API keys
```
 
**Windows (PowerShell):**
 
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env           # completar con tus API keys
```

## Ejecutar

El entorno virtual debe estar activado antes de ejecutar (ver sección Instalación).

```bash
# Desde la raíz del proyecto
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\Activate.ps1  # Windows (PowerShell)

cd app
python main.py
```

Prueba cada proveedor cuya API key esté presente en `.env`, primero en
modo normal (`generate`) y luego en streaming (`stream`).

## Uso programático

```python
from schemas import ChatMessage, ModelConfig, Role
from factory import create_client

config = ModelConfig(provider="openai", model="gpt-4o-mini")
client = create_client(config, api_key="sk-...")

messages = [ChatMessage(role=Role.user, content="Hola")]

response = await client.generate(messages)      # respuesta completa
async for token in client.stream(messages):      # tokens a medida que llegan
    print(token, end="")
```

## Notas de diseño

- **Intercambiabilidad**: `OpenAIClient` y `AnthropicClient` implementan la
  misma interfaz (`BaseLLMClient`). El resto del código nunca sabe (ni le
  importa) qué proveedor hay detrás.
- **Asincronía**: se usan `AsyncOpenAI` / `AsyncAnthropic`, nunca las
  versiones síncronas dentro de funciones `async`, para no bloquear el
  event loop.
- **Errores controlados**: `generate()` nunca lanza una excepción hacia
  arriba; reintenta 3 veces con backoff exponencial ante rate limit/errores
  de red, y en caso de fallo definitivo devuelve un `ModelResponse` con
  `error` seteado (`response.ok == False`) en vez de romper el programa.
- **Validación**: todos los datos de entrada/salida pasan por modelos
  Pydantic, evitando dicts sueltos sin forma garantizada.