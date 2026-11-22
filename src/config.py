"""
Configuration module for ncm-probe.
Reads required environment variables and fails fast with actionable errors.
"""

import os
import sys

def get_required_env(key: str, description: str = "") -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        desc = f" ({description})" if description else ""
        raise EnvironmentError(
            f"Missing required environment variable '{key}'{desc}. Please set it before running the pipeline."
        )
    return val.strip()

def get_optional_env(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    return val.strip()

# Lazy or fast-fail configuration checks
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
NCM_XSUM_REPO = os.getenv("NCM_XSUM_REPO", "").strip()
OPENAI_MODEL_EXTRACTOR = os.getenv("OPENAI_MODEL_EXTRACTOR", "gpt-4o-mini").strip()
OPENAI_MODEL_EXECUTOR = os.getenv("OPENAI_MODEL_EXECUTOR", "gpt-4o").strip()

def check_env_for_stage(stage_name: str) -> None:
    """Fails fast with clear instructions if keys needed for a stage are missing."""
    if stage_name in ("precheck", "execute"):
        if not TAVILY_API_KEY:
            raise EnvironmentError("TAVILY_API_KEY is required for search and precheck operations.")
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is required for LLM operations.")
    elif stage_name == "ncm_generate":
        if not NCM_XSUM_REPO:
            raise EnvironmentError(
                "NCM_XSUM_REPO environment variable is unset or ambiguous! "
                "Per research guidelines, never guess the NCM Hugging Face repo ID. Halting."
            )
    elif stage_name == "extract":
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is required for LLM extractor operations.")
