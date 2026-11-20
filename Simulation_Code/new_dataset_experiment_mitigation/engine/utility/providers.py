"""Small shared boundary for structured LLM calls.

The provider knows how to call an API and return validated JSON. Lusifer,
Direct LLM, the agentic response model, and future item-selection models own
their prompts, schemas, and semantic validation.

OpenAI and local Ollama are implemented behind the same small interface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any, Protocol
import urllib.error
import urllib.request

from openai import OpenAI


logger = logging.getLogger(__name__)

JSONValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
OllamaTransport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


class ProviderError(RuntimeError):
    """The provider call failed or exhausted its retries."""


class InvalidProviderResponse(ProviderError):
    """The provider returned output that could not be accepted."""


@dataclass(frozen=True)
class ProviderConfig:
    """Settings shared by profile generation and LLM item selection."""

    provider: str = "openai"
    model: str = "gpt-5.4-nano"
    max_output_tokens: int = 2056
    temperature: float | None = None
    reasoning_effort: str | None = "none"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 15.0
    retry_increase_factor: float = 4.0
    endpoint: str = "http://localhost:11434/api/chat"
    seed: int | None = None
    ollama_num_gpu: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-blank string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-blank string")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        if self.retry_increase_factor < 1:
            raise ValueError("retry_increase_factor must be at least 1")
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise ValueError("endpoint must be a non-blank string")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        if self.ollama_num_gpu is not None and (
            isinstance(self.ollama_num_gpu, bool)
            or not isinstance(self.ollama_num_gpu, int)
            or self.ollama_num_gpu < 0
        ):
            raise ValueError(
                "ollama_num_gpu must be a non-negative integer or None"
            )


@dataclass(frozen=True)
class ProviderResult:
    """One validated response and the small amount of metadata we use."""

    payload: Mapping[str, Any]
    response_id: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    attempts: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "usage", dict(self.usage))
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")


class JSONProvider(Protocol):
    """Minimal interface shared by all LLM-backed response models."""

    def semantic_settings(self) -> Mapping[str, Any]: ...

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        validator: JSONValidator,
        schema_name: str = "structured_response",
    ) -> ProviderResult: ...


class OpenAIProvider:
    """OpenAI Responses API implementation of :class:`JSONProvider`."""

    def __init__(
        self,
        *,
        config: ProviderConfig | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or ProviderConfig()
        if self.config.provider.lower() != "openai":
            raise ValueError("OpenAIProvider requires provider='openai'")
        self._api_key = api_key
        self._client = client
        self._sleep = sleep

    def semantic_settings(self) -> Mapping[str, Any]:
        """Return only settings that are safe to store in artifacts."""

        return {
            "provider": "openai",
            "api": "responses_v1",
            "model": self.config.model,
            "temperature": self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort,
            "max_output_tokens": self.config.max_output_tokens,
            "response_format": "strict_json_schema",
            "store": False,
            # OpenAI Responses requests currently do not receive the local
            # experiment seed. Do not imply deterministic provider sampling.
            "seed_supported": False,
        }

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        validator: JSONValidator,
        schema_name: str = "structured_response",
    ) -> ProviderResult:
        """Call OpenAI until success or the configured retry limit."""

        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt must be non-blank")
        if not schema_name.strip():
            raise ValueError("schema_name must be non-blank")

        request: dict[str, Any] = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(response_schema),
                }
            },
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
        }
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.config.reasoning_effort}

        total_attempts = self.config.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            started = time.monotonic()
            logger.info(
                "OpenAI request started schema=%s attempt=%s/%s timeout_seconds=%s",
                schema_name, attempt, total_attempts, self.config.timeout_seconds,
            )
            try:
                response = self._get_client().responses.create(**request)
            except Exception as exc:
                if _is_retryable(exc) and attempt < total_attempts:
                    logger.warning(
                        "OpenAI call failed on attempt %d/%d; retrying: %s",
                        attempt,
                        total_attempts,
                        exc,
                    )
                    self._sleep(_retry_delay(self.config, attempt))
                    continue
                logger.error(
                    "OpenAI call failed on attempt %d/%d: %s",
                    attempt,
                    total_attempts,
                    exc,
                )
                raise ProviderError(
                    f"OpenAI call failed after {attempt} attempt(s): {exc}"
                ) from exc

            try:
                result = _validated_result(
                    response=response,
                    validator=validator,
                    attempts=attempt,
                )
                logger.info(
                    "OpenAI request completed schema=%s attempt=%s elapsed_seconds=%.3f response_id=%s",
                    schema_name, attempt, time.monotonic() - started, result.response_id,
                )
                return result
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if attempt < total_attempts:
                    logger.warning(
                        "OpenAI returned invalid structured output on attempt "
                        "%d/%d; retrying: %s",
                        attempt,
                        total_attempts,
                        exc,
                    )
                    self._sleep(_retry_delay(self.config, attempt))
                    continue
                logger.error(
                    "OpenAI returned invalid structured output after %d "
                    "attempt(s): %s",
                    attempt,
                    exc,
                )
                raise InvalidProviderResponse(
                    "OpenAI returned invalid structured output after "
                    f"{attempt} attempt(s): {exc}"
                ) from exc

        raise AssertionError("configured attempt loop must return or raise")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not configured")
        # SDK retries are disabled so there is exactly one visible retry loop.
        self._client = OpenAI(
            api_key=api_key,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )
        return self._client


class OllamaProvider:
    """Local Ollama ``/api/chat`` implementation of :class:`JSONProvider`."""

    def __init__(
        self,
        *,
        config: ProviderConfig | None = None,
        transport: OllamaTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or ProviderConfig(provider="ollama")
        if self.config.provider.lower() != "ollama":
            raise ValueError("OllamaProvider requires provider='ollama'")
        self._transport = transport or _post_ollama_json
        self._sleep = sleep

    def semantic_settings(self) -> Mapping[str, Any]:
        """Return reproducibility-relevant settings without local secrets."""

        return {
            "provider": "ollama",
            "api": "chat_v1",
            "endpoint": self.config.endpoint,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "ollama_num_gpu": self.config.ollama_num_gpu,
            "max_output_tokens": self.config.max_output_tokens,
            "response_format": "json_schema",
            "stream": False,
            "think": False,
            "seed_supported": True,
        }

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        validator: JSONValidator,
        schema_name: str = "structured_response",
    ) -> ProviderResult:
        """Call Ollama until a valid structured response is returned."""

        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt must be non-blank")
        if not schema_name.strip():
            raise ValueError("schema_name must be non-blank")

        options: dict[str, Any] = {
            "temperature": (
                0.0
                if self.config.temperature is None
                else self.config.temperature
            ),
            "num_predict": self.config.max_output_tokens,
        }
        if self.config.seed is not None:
            options["seed"] = self.config.seed
        if self.config.ollama_num_gpu is not None:
            options["num_gpu"] = self.config.ollama_num_gpu
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": dict(response_schema),
            "think": False,
            "options": options,
        }

        total_attempts = self.config.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            started = time.monotonic()
            logger.info(
                "Ollama request started schema=%s attempt=%s/%s timeout_seconds=%s",
                schema_name, attempt, total_attempts, self.config.timeout_seconds,
            )
            try:
                response = self._transport(
                    self.config.endpoint,
                    request,
                    self.config.timeout_seconds,
                )
            except Exception as exc:
                if _is_retryable(exc) and attempt < total_attempts:
                    logger.warning(
                        "Ollama call failed on attempt %d/%d; retrying: %s",
                        attempt,
                        total_attempts,
                        exc,
                    )
                    self._sleep(_retry_delay(self.config, attempt))
                    continue
                logger.error(
                    "Ollama call failed on attempt %d/%d: %s",
                    attempt,
                    total_attempts,
                    exc,
                )
                raise ProviderError(
                    f"Ollama call failed after {attempt} attempt(s): {exc}"
                ) from exc

            try:
                result = _validated_ollama_result(
                    response=response,
                    validator=validator,
                    attempts=attempt,
                )
                logger.info(
                    "Ollama request completed schema=%s attempt=%s elapsed_seconds=%.3f response_id=%s",
                    schema_name, attempt, time.monotonic() - started, result.response_id,
                )
                return result
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if attempt < total_attempts:
                    logger.warning(
                        "Ollama returned invalid structured output on attempt "
                        "%d/%d; retrying: %s",
                        attempt,
                        total_attempts,
                        exc,
                    )
                    self._sleep(_retry_delay(self.config, attempt))
                    continue
                logger.error(
                    "Ollama returned invalid structured output after %d "
                    "attempt(s): %s",
                    attempt,
                    exc,
                )
                raise InvalidProviderResponse(
                    "Ollama returned invalid structured output after "
                    f"{attempt} attempt(s): {exc}"
                ) from exc

        raise AssertionError("configured attempt loop must return or raise")


def build_provider(
    config: ProviderConfig | None = None,
    *,
    api_key: str | None = None,
    client: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> JSONProvider:
    """Resolve the configured provider without model-specific logic."""

    resolved = config or ProviderConfig()
    if resolved.provider.lower() == "openai":
        return OpenAIProvider(
            config=resolved,
            api_key=api_key,
            client=client,
            sleep=sleep,
        )
    if resolved.provider.lower() == "ollama":
        return OllamaProvider(config=resolved, sleep=sleep)
    raise ValueError(f"unsupported provider: {resolved.provider!r}")


def _validated_result(
    *,
    response: Any,
    validator: JSONValidator,
    attempts: int,
) -> ProviderResult:
    status = getattr(response, "status", "completed")
    if status not in (None, "completed"):
        raise ValueError(f"response status is {status!r}")

    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("response contains no output text")

    payload = json.loads(output_text)
    if not isinstance(payload, Mapping):
        raise TypeError("response must contain one JSON object")
    validated = validator(payload)
    if not isinstance(validated, Mapping):
        raise TypeError("validator must return a mapping")

    response_id = getattr(response, "id", None)
    if response_id is not None:
        response_id = str(response_id)
    return ProviderResult(
        payload=dict(validated),
        response_id=response_id,
        usage=_as_dict(getattr(response, "usage", None)),
        attempts=attempts,
    )


def _validated_ollama_result(
    *,
    response: Mapping[str, Any],
    validator: JSONValidator,
    attempts: int,
) -> ProviderResult:
    if response.get("done_reason") == "length":
        raise ValueError("Ollama response reached its output limit")
    if response.get("done") is False:
        raise ValueError("Ollama returned an incomplete response")
    provider_error = response.get("error")
    if provider_error:
        raise ValueError(f"Ollama returned an error: {provider_error}")
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise TypeError("Ollama response contains no message")
    output_text = message.get("content")
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Ollama response contains no output text")

    payload = json.loads(output_text)
    if not isinstance(payload, Mapping):
        raise TypeError("response must contain one JSON object")
    validated = validator(payload)
    if not isinstance(validated, Mapping):
        raise TypeError("validator must return a mapping")

    input_tokens = _optional_non_negative_int(
        response.get("prompt_eval_count"), "prompt_eval_count"
    )
    output_tokens = _optional_non_negative_int(
        response.get("eval_count"), "eval_count"
    )
    usage: dict[str, Any] = {
        "provider_usage": {
            key: response[key]
            for key in (
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            )
            if key in response
        }
    }
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if input_tokens is not None and output_tokens is not None:
        usage["total_tokens"] = input_tokens + output_tokens
    usage["reasoning_tokens"] = 0
    total_duration = _optional_non_negative_int(
        response.get("total_duration"), "total_duration"
    )
    if total_duration is not None:
        usage["latency_ms"] = total_duration / 1_000_000

    return ProviderResult(
        payload=dict(validated),
        response_id=None,
        usage=usage,
        attempts=attempts,
    )


def _post_ollama_json(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("Ollama HTTP response must contain one JSON object")
    return dict(value)


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"Ollama {name} must be a non-negative integer")
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _retry_delay(config: ProviderConfig, failed_attempt: int) -> float:
    """Return exponential backoff after a failed one-based attempt."""

    delay = config.retry_delay_seconds * (
        config.retry_increase_factor ** (failed_attempt - 1)
    )
    logger.info("Provider retry backoff after attempt=%s delay_seconds=%s", failed_attempt, delay)
    return delay


def _is_retryable(error: Exception) -> bool:
    status_code = getattr(error, "status_code", getattr(error, "code", None))
    if isinstance(status_code, int):
        return status_code in {408, 409, 425, 429} or status_code >= 500
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    } or isinstance(error, (TimeoutError, urllib.error.URLError))


__all__ = [
    "InvalidProviderResponse",
    "JSONProvider",
    "JSONValidator",
    "OllamaProvider",
    "OllamaTransport",
    "OpenAIProvider",
    "ProviderConfig",
    "ProviderError",
    "ProviderResult",
    "build_provider",
]
