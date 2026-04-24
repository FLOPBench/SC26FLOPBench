import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_core.runnables import ConfigurableField
from langchain_openai import AzureChatOpenAI, ChatOpenAI

def _env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first truthy environment variable from `keys` or the fallback."""
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return default


@dataclass
class OpenRouterLLMSettings:
    """Hyperparameters and credentials for an OpenRouter-backed ChatOpenAI."""

    model_name: str = "openai/gpt-5-mini"
    temperature: float = 0.2
    top_p: float = 0.1
    timeout: int = 120
    openai_api_key: Optional[str] = field(
        default_factory=lambda: _env_first("OPENROUTER_API_KEY", "OPENAI_API_KEY")
    )
    openai_api_base: str = field(
        default_factory=lambda: _env_first(
            "OPENROUTER_API_BASE", "OPENAI_API_BASE", default="https://openrouter.ai/api/v1"
        )
    )

    @property
    def init_kwargs(self) -> dict[str, object]:
        """Return kwargs that can be passed directly to `ChatOpenAI`."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "openai_api_key": self.openai_api_key,
            "openai_api_base": self.openai_api_base,
        }


@dataclass
class AzureLLMSettings:
    """Hyperparameters and credentials for an AzureChatOpenAI."""

    model_name: str = "gpt-5-mini"
    temperature: float = 1.0
    top_p: float = 1.0
    timeout: int = 120
    openai_api_key: Optional[str] = field(
        default_factory=lambda: _env_first("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    azure_endpoint: str = field(
        default_factory=lambda: _env_first(
            "AZURE_OPENAI_ENDPOINT",
            default="https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com",
        )
    )
    openai_api_version: str = field(
        default_factory=lambda: _env_first("OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION", default="2025-04-01-preview")
    )

    @property
    def init_kwargs(self) -> dict[str, object]:
        """Return kwargs that can be passed directly to `AzureChatOpenAI`."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "openai_api_key": self.openai_api_key,
            "azure_endpoint": self.azure_endpoint,
            "openai_api_version": self.openai_api_version,
        }


def _openrouter_configurable_fields() -> dict[str, ConfigurableField]:
    return {
        "model_name": ConfigurableField(id="opr_model"),
        "temperature": ConfigurableField(id="opr_temp"),
        "top_p": ConfigurableField(id="opr_top_p"),
        "openai_api_base": ConfigurableField(id="opr_provider_url"),
        "openai_api_key": ConfigurableField(id="opr_provider_api_key"),
        "request_timeout": ConfigurableField(id="opr_timeout"),
    }


def _azure_configurable_fields() -> dict[str, ConfigurableField]:
    return {
        "model_name": ConfigurableField(id="model"),
        "temperature": ConfigurableField(id="temp"),
        "top_p": ConfigurableField(id="top_p"),
        "azure_endpoint": ConfigurableField(id="provider_url"),
        "openai_api_key": ConfigurableField(id="provider_api_key"),
        "openai_api_version": ConfigurableField(id="api_version"),
        "request_timeout": ConfigurableField(id="timeout"),
    }


def build_openrouter_llm(settings: Optional[OpenRouterLLMSettings] = None) -> ChatOpenAI:
    """Return an OpenRouter-powered ChatOpenAI that can be reconfigured at runtime."""

    settings = settings or OpenRouterLLMSettings()
    llm = ChatOpenAI(**settings.init_kwargs)
    return llm.configurable_fields(**_openrouter_configurable_fields())


def build_azure_llm(settings: Optional[AzureLLMSettings] = None) -> AzureChatOpenAI:
    """Return an AzureChatOpenAI that can be reconfigured at runtime."""

    settings = settings or AzureLLMSettings()
    azure_llm = AzureChatOpenAI(**settings.init_kwargs)
    return azure_llm.configurable_fields(**_azure_configurable_fields())


LLMProvider = Literal["openrouter", "azure", "openai"]
_PROVIDER_ALIASES = {"openai": "openrouter", "openrouter": "openrouter", "azure": "azure"}


def _normalize_provider(provider: str) -> str:
    return _PROVIDER_ALIASES.get(provider, "openrouter")


def build_configurable_llm(
    *,
    default_provider: LLMProvider = "openrouter",
    openrouter_settings: Optional[OpenRouterLLMSettings] = None,
    azure_settings: Optional[AzureLLMSettings] = None,
) -> ChatOpenAI:
    """Return a top-level LLM that exposes OpenRouter + Azure alternatives."""

    openrouter_llm = build_openrouter_llm(openrouter_settings)
    azure_llm = build_azure_llm(azure_settings)
    provider = _normalize_provider(default_provider)
    return openrouter_llm.configurable_alternatives(
        ConfigurableField(id="llm_provider"),
        default_key=provider,
        azure=azure_llm,
    )
