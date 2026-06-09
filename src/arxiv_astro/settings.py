from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEBUG = False


def parse_bool(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = enabled


def debug_log(message: str, **fields: Any) -> None:
    if not DEBUG:
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[DEBUG] {message}{suffix}", file=__import__("sys").stderr)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    output_dir: str = "data"
    request_timeout: float = 30.0
    llm_request_timeout: float = 180.0
    max_input_chars: int = 400000
    llm_max_output_tokens: int = 12000
    paper_interests: str = ""
    fetch_results: int = 100
    selection_max_input_chars: int = 220000
    selection_summary_max_chars: int = 4000
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", cls.base_url),
            model=os.getenv("DEEPSEEK_MODEL", cls.model),
            output_dir=os.getenv("OUTPUT_DIR", cls.output_dir),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", cls.request_timeout)),
            llm_request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", cls.llm_request_timeout)),
            max_input_chars=int(os.getenv("MAX_INPUT_CHARS", cls.max_input_chars)),
            llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", cls.llm_max_output_tokens)),
            paper_interests=os.getenv("PAPER_INTERESTS", cls.paper_interests),
            fetch_results=int(os.getenv("FETCH_RESULTS", cls.fetch_results)),
            selection_max_input_chars=int(os.getenv("SELECTION_MAX_INPUT_CHARS", cls.selection_max_input_chars)),
            selection_summary_max_chars=int(os.getenv("SELECTION_SUMMARY_MAX_CHARS", cls.selection_summary_max_chars)),
            debug=parse_bool(os.getenv("DEBUG")),
        )
        set_debug(settings.debug)
        return settings
