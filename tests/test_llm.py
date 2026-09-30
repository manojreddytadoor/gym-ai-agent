"""OllamaLLM behaviour with a fake Ollama client."""

from types import SimpleNamespace

import ollama
import pytest

from app.llm import LLMError, OllamaLLM


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)

    def chat(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return iter(SimpleNamespace(message=SimpleNamespace(content=c)) for c in reply)


def make_llm(*replies) -> OllamaLLM:
    llm = OllamaLLM(model="test-model")
    llm.client = FakeClient(*replies)
    return llm


def drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def test_parses_streamed_json():
    events, result = drain(make_llm(['{"a"', ': 1}']).generate_json([], {}, status="Writing", caller="Diet Coach"))
    assert result == {"a": 1}
    assert events[0] == {"type": "status", "text": "Writing"}


def test_retries_once_on_malformed_output():
    _, result = drain(make_llm(['{"a": '], ['{"a": 2}']).generate_json([], {}, status="s", caller="c"))
    assert result == {"a": 2}


def test_missing_model_gives_pull_instructions():
    llm = make_llm(ollama.ResponseError("model not found", status_code=404))
    with pytest.raises(LLMError, match="ollama pull test-model"):
        drain(llm.generate_json([], {}, status="s", caller="c"))


def test_second_caller_waits_and_says_for_whom():
    llm = make_llm(['{"a": 1}'], ['{"b": 2}'])
    first = llm.generate_json([], {}, status="Writing the plan", caller="Workout Coach")
    assert next(first)["text"] == "Writing the plan"  # first caller now holds the model

    second = llm.generate_json([], {}, status="Writing the plan", caller="Diet Coach")
    assert next(second) == {"type": "status", "text": "Waiting for the Workout Coach to finish with the model"}

    assert drain(first)[1] == {"a": 1}  # releases the model
    events, result = drain(second)
    assert events[0] == {"type": "status", "text": "Writing the plan"}
    assert result == {"b": 2}
