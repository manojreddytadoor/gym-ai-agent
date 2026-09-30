"""Local model access through Ollama, with JSON-schema constrained output."""

import json
import os
import threading
from collections.abc import Generator

import ollama

DEFAULT_MODEL = "llama3.2"
PROGRESS_EVERY = 10  # chunks (roughly tokens) between progress events


class LLMError(Exception):
    """A problem with a message that can be shown to the user as-is."""


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self.client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST"))
        # On a CPU, Ollama runs one request at a time anyway; queueing here lets
        # the waiting coach say who it's waiting for instead of looking stuck.
        self._turn = threading.Lock()
        self._busy_with = ""

    def generate_json(self, messages: list[dict], schema: dict, *, status: str, caller: str
                      ) -> Generator[dict, None, dict]:
        """Stream a reply that must match `schema`, yielding status and progress events.

        Ollama constrains generation to the schema, so enums (the allowed foods
        and exercises) can't be violated. Use as `result = yield from ...`.
        """
        if not self._turn.acquire(blocking=False):
            yield {"type": "status", "text": f"Waiting for the {self._busy_with} to finish with the model"}
            self._turn.acquire()
        self._busy_with = caller
        try:
            yield {"type": "status", "text": status}
            return (yield from self._generate(messages, schema))
        finally:
            self._turn.release()

    def _generate(self, messages: list[dict], schema: dict) -> Generator[dict, None, dict]:
        for _ in range(2):
            parts: list[str] = []
            try:
                stream = self.client.chat(
                    model=self.model,
                    messages=messages,
                    format=schema,
                    stream=True,
                    # Keep the model in memory between steps; reloading it costs a minute on a CPU.
                    keep_alive="20m",
                    # num_predict stops a small model that rambles inside a text field.
                    options={"temperature": 0.4, "num_ctx": 8192, "num_predict": 4096},
                )
                for count, chunk in enumerate(stream, start=1):
                    parts.append(chunk.message.content or "")
                    if count % PROGRESS_EVERY == 0:
                        yield {"type": "progress", "tokens": count}
            except ollama.ResponseError as exc:
                if exc.status_code == 404:
                    raise LLMError(f'The model "{self.model}" isn\'t downloaded. Run: ollama pull {self.model}') from exc
                raise LLMError(f"Ollama error: {exc.error}") from exc
            except ConnectionError as exc:
                raise LLMError("Can't connect to Ollama. Make sure the Ollama app is running.") from exc
            try:
                return json.loads("".join(parts))
            except json.JSONDecodeError:
                continue  # truncated or malformed output; try once more
        raise LLMError("The model returned malformed output twice. Please try again.")

    def model_info(self) -> dict:
        """Name, size and ID of the local model, for checking it against ollama.com."""
        base = self.model.split(":")[0]
        page = f"https://ollama.com/{base}/tags" if "/" in base else f"https://ollama.com/library/{base}/tags"
        info: dict = {"model": self.model, "library_url": page}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        try:
            models = self.client.list().models
        except ConnectionError:
            info["error"] = "Ollama isn't running. Start the Ollama app."
            return info
        for m in models:
            if m.model == wanted:
                info["id"] = (m.digest or "")[:12]
                if m.details:
                    info["parameters"] = m.details.parameter_size
                    info["quantization"] = m.details.quantization_level
                return info
        info["error"] = f"Model not downloaded. Run: ollama pull {self.model}"
        return info
