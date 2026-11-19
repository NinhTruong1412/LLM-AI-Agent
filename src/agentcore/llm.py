"""A thin, provider-agnostic LLM interface.

Projects depend on ``LLMClient`` rather than an SDK directly, so tests can use
``FakeLLM`` and later projects (the routing gateway, the semantic cache) can
wrap any client without touching call sites.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 16000,
        effort: Effort | None = None,
    ) -> LLMResponse: ...


class AnthropicLLM:
    """Claude via the official Anthropic SDK.

    Refusal fallbacks are on by default: if the model declines, the API reruns
    the request on a fallback model it picks for the refusal category.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, fallbacks: bool = True) -> None:
        import anthropic  # optional dependency: pip install -e ".[llm]"

        self._client = anthropic.Anthropic()
        self.model = model
        self.fallbacks = fallbacks

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 16000,
        effort: Effort | None = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if self.fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        response = self._client.beta.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests. Returns ``responder(prompt)`` or a canned reply."""

    reply: str = "ok"
    responder: Callable[[str], str] | None = None
    calls: list[str] = field(default_factory=list)
    model: str = "fake"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 16000,
        effort: Effort | None = None,
    ) -> LLMResponse:
        self.calls.append(prompt)
        text = self.responder(prompt) if self.responder else self.reply
        return LLMResponse(text=text, model=self.model, stop_reason="end_turn")
