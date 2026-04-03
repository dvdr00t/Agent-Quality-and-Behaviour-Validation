"""Configuration for the retail support application."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PROVIDER = "openai"


@dataclass(frozen=True)
class SupportSettings:
    """Runtime configuration loaded from environment variables."""

    provider: str
    model_name: str
    temperature: float = 0.0
    request_timeout_seconds: float = 60.0
    max_retries: int = 2
    openai_api_key: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_deployment: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_base_url: str | None = None

    @classmethod
    def from_env(cls, model_name_override: str | None = None) -> "SupportSettings":
        provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
        temperature = _read_float("OPENAI_TEMPERATURE", 0.0)
        request_timeout_seconds = _read_float("OPENAI_TIMEOUT_SECONDS", 60.0)
        max_retries = _read_int("OPENAI_MAX_RETRIES", 2)
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_base_url = os.getenv("LANGFUSE_BASE_URL")

        if provider == "azure":
            azure_openai_api_key = _read_required_env("AZURE_OPENAI_API_KEY")
            azure_openai_endpoint = _read_required_env("AZURE_OPENAI_ENDPOINT")
            azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            azure_openai_deployment = model_name_override or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if not azure_openai_deployment:
                raise RuntimeError(
                    "AZURE_OPENAI_DEPLOYMENT is required when LLM_PROVIDER=azure."
                )

            return cls(
                provider=provider,
                model_name=azure_openai_deployment,
                temperature=temperature,
                request_timeout_seconds=request_timeout_seconds,
                max_retries=max_retries,
                azure_openai_api_key=azure_openai_api_key,
                azure_openai_endpoint=azure_openai_endpoint,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment=azure_openai_deployment,
                langfuse_secret_key=langfuse_secret_key,
                langfuse_public_key=langfuse_public_key,
                langfuse_base_url=langfuse_base_url,
            )

        if provider != "openai":
            raise RuntimeError("LLM_PROVIDER must be either 'openai' or 'azure'.")

        openai_api_key = _read_required_env("OPENAI_API_KEY")
        return cls(
            provider=provider,
            model_name=model_name_override or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            temperature=temperature,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            openai_api_key=openai_api_key,
            langfuse_secret_key=langfuse_secret_key,
            langfuse_public_key=langfuse_public_key,
            langfuse_base_url=langfuse_base_url,
        )


def _read_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float.") from exc


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer.") from exc
